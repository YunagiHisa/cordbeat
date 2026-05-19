"""AI Backend abstraction layer."""

from __future__ import annotations

import base64
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from cordbeat.config import AIBackendConfig
from cordbeat.exceptions import AIBackendError
from cordbeat.tools.metrics import (
    LLM_GENERATE_LATENCY,
    LLM_GENERATE_TOTAL,
    inc_counter,
    time_block,
)

logger = logging.getLogger(__name__)


def _detect_image_mime(b64data: str) -> str:
    """Detect image MIME type from base64-encoded data magic bytes."""
    try:
        raw = base64.b64decode(b64data[:20] + "==")
        if raw[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if raw[:4] == b"\x89PNG":
            return "image/png"
        if raw[:4] in (b"GIF8", b"GIF9"):
            return "image/gif"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
    except Exception:
        pass
    return "image/jpeg"


class AIBackend(ABC):
    """Abstract interface for AI inference backends."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a text completion."""

    async def generate_with_vision(
        self,
        prompt: str,
        images: list[str],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a completion with image inputs.

        Default implementation falls back to text-only. Subclasses that support
        vision override this method.
        """
        logger.warning(
            "Vision not supported by this backend; falling back to text-only"
        )
        return await self.generate(prompt, system, temperature, max_tokens)

    async def aclose(self) -> None:
        """Close underlying resources. Subclasses may override."""

    async def generate_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response."""
        import re

        raw = await self.generate(prompt, system, temperature, max_tokens)
        if not raw:
            logger.warning(
                "generate_json: empty raw response (prompt=%d chars) — "
                "model may have hit context limit or returned only thinking tokens. "
                "For Qwen3/DeepSeek thinking models set "
                "ai_backend.enable_thinking: false in config.yaml.",
                len(prompt),
            )
        logger.debug("generate_json raw(%d chars): %.500s", len(raw), raw)
        # Strip <think>...</think> reasoning blocks (Qwen3, DeepSeek-R1, etc.)
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # If thinking model put ALL output inside <think> (e.g. JSON-only prompts),
        # fall back to extracting the outermost {...} from the raw response.
        if not text:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                text = m.group(0)
                logger.debug("generate_json: extracted JSON from think block")
        # Extract JSON from potential markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # skip ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        if not text:
            msg = (
                f"generate_json: no JSON content extracted from model response "
                f"(raw={len(raw)} chars). "
                "If using a thinking model, set ai_backend.enable_thinking: false"
            )
            raise json.JSONDecodeError(msg, "", 0)
        return dict(json.loads(text))


class OllamaBackend(AIBackend):
    """Ollama HTTP API backend."""

    def __init__(self, config: AIBackendConfig) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._model = config.model
        self._options = config.options
        self._client = httpx.AsyncClient(timeout=config.timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **self._options,
            },
        }
        if system:
            payload["system"] = system

        logger.debug(
            "ollama request: model=%s system=%d chars prompt=%d chars",
            self._model,
            len(system),
            len(prompt),
        )
        labels = {"backend": "ollama", "model": self._model}
        try:
            async with time_block(LLM_GENERATE_LATENCY, labels):
                resp = await self._client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ReadTimeout as exc:
            inc_counter(
                LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "timeout"}
            )
            logger.warning(
                "ollama request timed out after %.0fs (model=%s prompt=%d chars). "
                "Increase 'ai_backend.timeout' in config.yaml or use a faster model.",
                self._client.timeout.read or 0.0,
                self._model,
                len(prompt),
            )
            raise AIBackendError(
                f"Ollama request timed out after "
                f"{self._client.timeout.read or 0.0:.0f}s"
            ) from exc
        except Exception:
            inc_counter(LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "error"})
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "ok"})
        logger.debug("Ollama raw keys: %s", list(data.keys()))
        response_text = str(data.get("response", ""))
        logger.debug(
            "ollama response: %d chars: %.300s", len(response_text), response_text
        )
        return response_text

    async def generate_with_vision(
        self,
        prompt: str,
        images: list[str],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate using Ollama's chat API with image support (e.g. llava)."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        user_msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user_msg["images"] = images
        messages.append(user_msg)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **self._options,
            },
        }

        labels = {"backend": "ollama", "model": self._model}
        try:
            async with time_block(LLM_GENERATE_LATENCY, labels):
                resp = await self._client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ReadTimeout as exc:
            inc_counter(
                LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "timeout"}
            )
            logger.warning(
                "ollama vision request timed out after %.0fs "
                "(model=%s prompt=%d chars images=%d).",
                self._client.timeout.read or 0.0,
                self._model,
                len(prompt),
                len(images),
            )
            raise AIBackendError(
                f"Ollama vision request timed out after "
                f"{self._client.timeout.read or 0.0:.0f}s"
            ) from exc
        except Exception:
            inc_counter(LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "error"})
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "ok"})
        return str(data.get("message", {}).get("content", ""))


class OpenAICompatBackend(AIBackend):
    """OpenAI-compatible API backend (works with vLLM, LM Studio, etc.)."""

    def __init__(
        self, config: AIBackendConfig, extra_headers: dict[str, str] | None = None
    ) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._model = config.model
        self._api_key = config.options.get("api_key", "")
        # Qwen3 / DeepSeek-R1 thinking models: set enable_thinking: false in
        # ai.options to skip the <think> phase for JSON-mode requests.
        # Defaults to None (not sent) to avoid breaking non-thinking models.
        self._enable_thinking: bool | None = config.options.get("enable_thinking")
        # None = don't include max_tokens in requests (server uses its own default)
        self._max_tokens: int | None = config.max_tokens
        if (
            self._enable_thinking is True
            and self._max_tokens is not None
            and self._max_tokens < 2048
        ):
            logger.warning(
                "ai_backend.max_tokens=%d is very low for a thinking model. "
                "Qwen3/DeepSeek-R1 can consume 2000+ tokens in the thinking phase "
                "alone, leaving no budget for the actual response. "
                "Recommend setting 'ai_backend.max_tokens: 4096' or higher "
                "in config.yaml (or remove max_tokens to use the server default).",
                self._max_tokens,
            )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.AsyncClient(timeout=config.timeout, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        # When config.max_tokens is unset (None), we deliberately do NOT include
        # max_tokens in the request payload, letting the server's own default
        # (e.g. llama.cpp --n-predict) decide. Sending the caller's small default
        # (1024) would cap thinking models mid-reasoning and force the retry
        # path. Callers that genuinely need a tight cap should set
        # ai_backend.max_tokens explicitly in config.yaml.
        # NOTE: the `max_tokens` parameter is intentionally ignored here.
        _ = max_tokens
        effective_max_tokens: int | None = self._max_tokens
        messages: list[dict[str, str]] = []
        effective_system = system
        if self._enable_thinking is False:
            # Belt-and-suspenders: inject /no_think soft-switch into the system
            # message so Qwen3 disables thinking even if the server ignores
            # chat_template_kwargs (works across all llama.cpp versions).
            effective_system = (
                (system + "\n/no_think").lstrip() if system else "/no_think"
            )
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        logger.debug(
            "openai_compat request: model=%s system=%d chars prompt=%d chars",
            self._model,
            len(effective_system),
            len(prompt),
        )
        labels = {"backend": "openai_compat", "model": self._model}
        try:
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
            }
            if effective_max_tokens is not None:
                payload["max_tokens"] = effective_max_tokens
            if self._enable_thinking is not None:
                # llama.cpp passes template kwargs via chat_template_kwargs;
                # keep the legacy top-level field for other servers (vLLM etc.)
                payload["chat_template_kwargs"] = {
                    "enable_thinking": self._enable_thinking
                }
                payload["enable_thinking"] = self._enable_thinking
            async with time_block(LLM_GENERATE_LATENCY, labels):
                resp = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ReadTimeout as exc:
            inc_counter(
                LLM_GENERATE_TOTAL,
                {"backend": "openai_compat", "outcome": "timeout"},
            )
            logger.warning(
                "openai_compat request timed out after %.0fs "
                "(model=%s prompt=%d chars). Increase 'ai_backend.timeout' "
                "in config.yaml or use a faster model / smaller context.",
                self._client.timeout.read or 0.0,
                self._model,
                len(prompt),
            )
            raise AIBackendError(
                f"LLM request timed out after "
                f"{self._client.timeout.read or 0.0:.0f}s"
            ) from exc
        except Exception:
            inc_counter(
                LLM_GENERATE_TOTAL,
                {"backend": "openai_compat", "outcome": "error"},
            )
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "openai_compat", "outcome": "ok"})
        try:
            message = data["choices"][0]["message"]
            content = message.get("content")

            # Log reasoning_content (chain-of-thought) at DEBUG level so it
            # appears in logs when log.level=DEBUG without polluting responses.
            reasoning_content: str = message.get("reasoning_content") or ""
            if reasoning_content:
                logger.debug(
                    "openai_compat thinking (%d chars):\n%.2000s",
                    len(reasoning_content),
                    reasoning_content,
                )

            if not content:
                # Some thinking-model backends (llama.cpp + Qwen3/DeepSeek) return
                # reasoning_content with content=null or content="".
                # reasoning_content is internal chain-of-thought, NOT user-facing
                # output — using it as the response would leak raw thinking text.
                if reasoning_content:
                    # The model exhausted max_tokens during its thinking phase and
                    # produced no actual response (content=null).  Retry once with
                    # 2× the effective max_tokens to give the model budget for both
                    # thinking and the actual answer.
                    # When max_tokens was not set (server default), the server's
                    # --n-predict limit caused the exhaustion, so we must explicitly
                    # override it in the retry — fall back to 8192 as a safe minimum.
                    # Heavy-thinking models (Qwen3, DeepSeek) routinely use
                    # 3000-5000 chars in reasoning_content alone (~1000-1500 tokens),
                    # plus the actual answer — 8192 gives ample headroom.
                    _thinking_retry_fallback = 8192
                    if effective_max_tokens is not None:
                        retry_mt: int = max(
                            effective_max_tokens * 2, _thinking_retry_fallback
                        )
                    else:
                        retry_mt = _thinking_retry_fallback
                    logger.warning(
                        "openai_compat: content=null with reasoning_content=%d chars. "
                        "Model exhausted max_tokens in thinking phase. "
                        "Retrying with max_tokens=%d. "
                        "To avoid this retry, set 'ai_backend.max_tokens: 4096'"
                        " in config.yaml or increase --n-predict on your server.",
                        len(reasoning_content),
                        retry_mt,
                    )
                    retry_payload: dict[str, Any] = {
                        "model": self._model,
                        "messages": messages,
                        "temperature": temperature,
                        # Always set — avoids hitting the server default again.
                        "max_tokens": retry_mt,
                    }
                    if self._enable_thinking is not None:
                        retry_payload["chat_template_kwargs"] = {
                            "enable_thinking": self._enable_thinking
                        }
                        retry_payload["enable_thinking"] = self._enable_thinking
                    try:
                        async with time_block(LLM_GENERATE_LATENCY, labels):
                            retry_resp = await self._client.post(
                                f"{self._base_url}/chat/completions",
                                json=retry_payload,
                            )
                            retry_resp.raise_for_status()
                            retry_data = retry_resp.json()
                        retry_message = retry_data["choices"][0]["message"]
                        retry_content = retry_message.get("content") or ""
                        if retry_content:
                            result = re.sub(
                                r"<think>.*?</think>",
                                "",
                                str(retry_content),
                                flags=re.DOTALL,
                            ).strip() or str(retry_content)
                            logger.debug(
                                "openai_compat retry response: %d chars: %.300s",
                                len(result),
                                result,
                            )
                            return result
                        logger.warning(
                            "openai_compat retry also returned empty content. "
                            "Increase your llama.cpp server's --n-predict, or set "
                            "'ai_backend.max_tokens: 4096' in config.yaml."
                        )
                    except httpx.ReadTimeout:
                        # ReadTimeout on retry is expected when the model
                        # genuinely cannot finish in the configured timeout.
                        # Suppress the noisy traceback — the warning text is
                        # enough actionable signal for the user.
                        logger.warning(
                            "openai_compat retry timed out after %.0fs. "
                            "Increase 'ai_backend.timeout' in config.yaml "
                            "(currently %.0fs) or your llama.cpp server's "
                            "--n-predict.",
                            self._client.timeout.read or 0.0,
                            self._client.timeout.read or 0.0,
                        )
                    except Exception as _retry_err:
                        logger.warning(
                            "openai_compat retry request failed (%s). "
                            "If this is a ReadTimeout, increase 'ai_backend.timeout' "
                            "(default 300 s) or your llama.cpp server's --n-predict.",
                            type(_retry_err).__name__,
                            exc_info=True,
                        )
                    result = ""
                else:
                    logger.warning(
                        "OpenAI-compat backend returned empty content and no "
                        "reasoning_content; model may have emitted nothing"
                    )
                    result = ""
            else:
                raw_content = str(content)
                # Extract and log any inline <think>...</think> blocks before stripping.
                think_blocks = re.findall(
                    r"<think>(.*?)</think>", raw_content, flags=re.DOTALL
                )
                if think_blocks:
                    combined_thinking = "\n---\n".join(think_blocks)
                    logger.debug(
                        "openai_compat inline thinking (%d chars):\n%.2000s",
                        len(combined_thinking),
                        combined_thinking,
                    )
                stripped = re.sub(
                    r"<think>.*?</think>", "", raw_content, flags=re.DOTALL
                ).strip()
                if not stripped:
                    logger.warning(
                        "openai_compat: content was entirely <think> blocks; "
                        "set ai_backend.enable_thinking: false in config.yaml"
                    )
                    result = raw_content
                else:
                    result = stripped
            logger.debug(
                "openai_compat response: %d chars: %.300s", len(result), result
            )
            return result
        except (KeyError, IndexError) as exc:
            msg = f"Unexpected response format from {self._base_url}"
            raise AIBackendError(msg) from exc

    async def generate_with_vision(
        self,
        prompt: str,
        images: list[str],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate using OpenAI vision API (content array with image_url blocks)."""
        # Same max_tokens policy as generate(): only send if config explicitly set it.
        _ = max_tokens
        effective_max_tokens: int | None = self._max_tokens
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b64img in images:
            mime = _detect_image_mime(b64img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64img}"},
                }
            )
        messages.append({"role": "user", "content": content})

        labels = {"backend": "openai_compat", "model": self._model}
        try:
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
            }
            if effective_max_tokens is not None:
                payload["max_tokens"] = effective_max_tokens
            async with time_block(LLM_GENERATE_LATENCY, labels):
                resp = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ReadTimeout as exc:
            inc_counter(
                LLM_GENERATE_TOTAL,
                {"backend": "openai_compat", "outcome": "timeout"},
            )
            logger.warning(
                "openai_compat vision request timed out after %.0fs "
                "(model=%s prompt=%d chars images=%d). Increase "
                "'ai_backend.timeout' in config.yaml or use a faster model.",
                self._client.timeout.read or 0.0,
                self._model,
                len(prompt),
                len(images),
            )
            raise AIBackendError(
                f"LLM vision request timed out after "
                f"{self._client.timeout.read or 0.0:.0f}s"
            ) from exc
        except Exception:
            inc_counter(
                LLM_GENERATE_TOTAL,
                {"backend": "openai_compat", "outcome": "error"},
            )
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "openai_compat", "outcome": "ok"})
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError) as exc:
            msg = f"Unexpected vision response format from {self._base_url}"
            raise AIBackendError(msg) from exc


class AnthropicBackend(AIBackend):
    """Anthropic Claude API backend.

    Requires the ``anthropic`` extra: ``uv sync --extra anthropic``.
    """

    def __init__(self, config: AIBackendConfig) -> None:
        self._model = config.model or "claude-3-5-sonnet-20241022"
        self._api_key = config.options.get("api_key", "")
        self._base_url = config.base_url or ""
        self._timeout = config.timeout
        try:
            import anthropic as _anthropic  # noqa: PLC0415

            kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "timeout": self._timeout,
            }
            if self._base_url and self._base_url != "http://localhost:11434":
                kwargs["base_url"] = self._base_url
            self._client = _anthropic.AsyncAnthropic(**kwargs)
        except ImportError as exc:
            msg = (
                "anthropic package is required for the Anthropic backend. "
                "Install it with: uv sync --extra anthropic"
            )
            raise ImportError(msg) from exc

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        labels = {"backend": "anthropic", "model": self._model}
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        try:
            async with time_block(LLM_GENERATE_LATENCY, labels):
                response = await self._client.messages.create(**kwargs)
        except Exception:
            inc_counter(
                LLM_GENERATE_TOTAL, {"backend": "anthropic", "outcome": "error"}
            )
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "anthropic", "outcome": "ok"})
        block = response.content[0] if response.content else None
        if block is None or not hasattr(block, "text"):
            return ""
        return str(block.text)

    async def generate_with_vision(
        self,
        prompt: str,
        images: list[str],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate using Anthropic vision API (base64 image source blocks)."""
        content: list[Any] = []
        for b64img in images:
            mime = _detect_image_mime(b64img)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64img,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        labels = {"backend": "anthropic", "model": self._model}
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = system
        try:
            async with time_block(LLM_GENERATE_LATENCY, labels):
                response = await self._client.messages.create(**kwargs)
        except Exception:
            inc_counter(
                LLM_GENERATE_TOTAL, {"backend": "anthropic", "outcome": "error"}
            )
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "anthropic", "outcome": "ok"})
        block = response.content[0] if response.content else None
        if block is None or not hasattr(block, "text"):
            return ""
        return str(block.text)


def create_backend(config: AIBackendConfig) -> AIBackend:
    """Factory function to create the appropriate AI backend."""
    import dataclasses  # noqa: PLC0415

    backend: AIBackend
    backend_name: str
    match config.provider:
        case "ollama":
            backend = OllamaBackend(config)
            backend_name = "ollama"
        case "openai" | "openai_compat":
            backend = OpenAICompatBackend(config)
            backend_name = "openai_compat"
        case "openrouter":
            # OpenRouter is OpenAI-compatible; use its API URL + inject X-Title header.
            effective_url = (
                config.base_url
                if config.base_url and config.base_url != "http://localhost:11434"
                else "https://openrouter.ai/api/v1"
            )
            effective_cfg = dataclasses.replace(config, base_url=effective_url)
            backend = OpenAICompatBackend(
                effective_cfg, extra_headers={"X-Title": "CordBeat"}
            )
            backend_name = "openrouter"
        case "anthropic":
            backend = AnthropicBackend(config)
            backend_name = "anthropic"
        case _:
            msg = f"Unknown AI backend provider: {config.provider}"
            raise ValueError(msg)

    if config.cache.enabled:
        from .cache import CachingBackend  # noqa: PLC0415

        return CachingBackend(
            inner=backend,
            config=config.cache,
            model=config.model,
            backend_name=backend_name,
        )
    return backend
