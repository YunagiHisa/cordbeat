"""AI Backend abstraction layer."""

from __future__ import annotations

import base64
import contextlib
import contextvars
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

import httpx

from cordbeat.ai.reasoning import (
    _DEFAULT_REASONING_STRIP_TAGS,
    looks_like_reasoning_text,
    strip_thinking_text,
)
from cordbeat.config import AIBackendConfig
from cordbeat.exceptions import AIBackendError
from cordbeat.tools.metrics import (
    LLM_GENERATE_LATENCY,
    LLM_GENERATE_TOTAL,
    inc_counter,
    time_block,
)

logger = logging.getLogger(__name__)

_DEFAULT_REASONING_CONTENT_KEYS = ("reasoning_content",)
_API_DEFAULT_MAX_TOKENS = 1024
_STRICT_OPENAI_COMPAT = "strict_openai"
_LLAMA_CPP_COMPAT = "llama_cpp"
_VLLM_COMPAT = "vllm"
_OPENAI_COMPAT_MODES = {
    _STRICT_OPENAI_COMPAT,
    _LLAMA_CPP_COMPAT,
    _VLLM_COMPAT,
}
_LOCAL_OPENAI_COMPAT_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _resolve_configured_max_tokens(
    configured_max_tokens: int | None,
    requested_max_tokens: int,
) -> int:
    if (
        configured_max_tokens is not None
        and requested_max_tokens == _API_DEFAULT_MAX_TOKENS
    ):
        return configured_max_tokens
    return requested_max_tokens


# ── Voice-context contextvar ─────────────────────────────────────────
# Set by the engine at the start of message handling (via
# ``voice_context_scope(is_voice)``).  Backends that support per-context
# overrides (e.g. ``ai.options.voice_enable_thinking``) read this var
# inside their request-payload code.  Using a contextvar keeps the public
# ``generate(...)`` API unchanged while remaining async-task-safe.
_voice_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "cordbeat_voice_context",
    default=False,
)


@contextlib.contextmanager
def voice_context_scope(is_voice: bool) -> Any:
    """Set the voice-context flag for any backend calls inside the block."""
    token = _voice_context.set(bool(is_voice))
    try:
        yield
    finally:
        _voice_context.reset(token)


def is_voice_context() -> bool:
    """Return True if the current asyncio task is in a voice-context scope."""
    return _voice_context.get()


def _coerce_string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    items: tuple[Any, ...]
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, list | tuple):
        items = tuple(value)
    else:
        return default
    cleaned = tuple(str(item).strip() for item in items if str(item).strip())
    return cleaned or default


def _coerce_marker_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list | tuple):
        return ()
    pairs: list[tuple[str, str]] = []
    for item in value:
        start = ""
        end = ""
        if isinstance(item, dict):
            start = str(item.get("start", "")).strip()
            end = str(item.get("end", "")).strip()
        elif isinstance(item, list | tuple) and len(item) == 2:
            start = str(item[0]).strip()
            end = str(item[1]).strip()
        if start and end:
            pairs.append((start, end))
    return tuple(pairs)


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

    def _strip_reasoning_text(self, raw: str) -> str:
        return strip_thinking_text(raw)

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

    @abstractmethod
    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate from a messages list (role/content pairs).

        Used by the ReAct loop for multi-turn generation.
        """

    async def generate_chat_with_vision(
        self,
        messages: list[dict[str, Any]],
        images: list[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate from chat history with images attached to the last user turn."""

        logger.warning(
            "Multimodal chat not supported by this backend; falling back to text-only"
        )
        return await self.generate_chat(messages, temperature, max_tokens)

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
                "ai.options.enable_thinking: false in config.yaml.",
                len(prompt),
            )
        logger.debug("generate_json raw(%d chars): %.500s", len(raw), raw)
        # Strip reasoning blocks/fragments (Qwen3, DeepSeek-R1, etc.)
        text = self._strip_reasoning_text(raw)
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
                "If using a thinking model, set ai.options.enable_thinking: false"
            )
            raise json.JSONDecodeError(msg, "", 0)
        return dict(json.loads(text))


class OllamaBackend(AIBackend):
    """Ollama HTTP API backend."""

    def __init__(self, config: AIBackendConfig) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._model = config.model
        self._default_max_tokens = config.max_tokens
        options = config.options if isinstance(config.options, dict) else {}
        if not isinstance(config.options, dict):
            logger.warning(
                "ai_backend.options is %s (expected mapping); "
                "ignoring. Check config.yaml for missing space after a colon.",
                type(config.options).__name__,
            )
        self._options = options
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
        max_tokens = _resolve_configured_max_tokens(
            self._default_max_tokens,
            max_tokens,
        )
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
        max_tokens = _resolve_configured_max_tokens(
            self._default_max_tokens,
            max_tokens,
        )
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
        except Exception:
            inc_counter(LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "error"})
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "ok"})
        return str(data.get("message", {}).get("content", ""))

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        max_tokens = _resolve_configured_max_tokens(
            self._default_max_tokens,
            max_tokens,
        )
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
        except Exception:
            inc_counter(LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "error"})
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "ok"})
        return str(data.get("message", {}).get("content", ""))

    async def generate_chat_with_vision(
        self,
        messages: list[dict[str, Any]],
        images: list[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        enriched = [dict(message) for message in messages]
        for message in reversed(enriched):
            if message.get("role") == "user":
                message["images"] = images
                break
        return await self.generate_chat(enriched, temperature, max_tokens)


class OpenAICompatBackend(AIBackend):
    """OpenAI-compatible API backend (works with vLLM, LM Studio, etc.)."""

    def __init__(self, config: AIBackendConfig) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._model = config.model
        self._default_max_tokens = config.max_tokens
        options = config.options if isinstance(config.options, dict) else {}
        if not isinstance(config.options, dict):
            logger.warning(
                "ai_backend.options is %s (expected mapping); "
                "ignoring. Check config.yaml for missing space after a colon.",
                type(config.options).__name__,
            )
        self._api_key = options.get("api_key", "")
        self._compatibility_mode_value = self._resolve_compatibility_mode(
            options.get("compatibility_mode")
        )
        # Qwen3 / DeepSeek-R1 thinking models: set enable_thinking: false in
        # ai.options to skip the <think> phase for JSON-mode requests.
        # Defaults to None (not sent) to avoid breaking non-thinking models.
        self._enable_thinking: bool | None = options.get("enable_thinking")
        # Optional override for voice contexts (STT-originated messages).
        # When set, it replaces ``enable_thinking`` while
        # ``is_voice_context()`` is true so VC / voice-message replies stay
        # within real-time latency budgets.  None = use ``_enable_thinking``.
        self._voice_enable_thinking: bool | None = options.get("voice_enable_thinking")
        voice_max_tokens = options.get("voice_max_tokens")
        self._voice_max_tokens = (
            int(voice_max_tokens)
            if isinstance(voice_max_tokens, int)
            and not isinstance(voice_max_tokens, bool)
            and voice_max_tokens > 0
            else None
        )
        self._reasoning_content_keys = _coerce_string_tuple(
            options.get("reasoning_content_keys"),
            _DEFAULT_REASONING_CONTENT_KEYS,
        )
        self._reasoning_strip_tags = _coerce_string_tuple(
            options.get("reasoning_strip_tags"),
            _DEFAULT_REASONING_STRIP_TAGS,
        )
        self._reasoning_strip_markers = _coerce_marker_pairs(
            options.get("reasoning_strip_markers")
        )
        self._log_reasoning_content = bool(options.get("log_reasoning_content", False))
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.AsyncClient(timeout=config.timeout, headers=headers)
        logger.debug(
            "openai_compat compatibility_mode=%s base_url=%s",
            self._compatibility_mode(),
            self._base_url,
        )

    def _resolve_compatibility_mode(self, raw_mode: Any) -> str:
        if isinstance(raw_mode, str) and raw_mode.strip():
            mode = raw_mode.strip().lower()
            if mode in _OPENAI_COMPAT_MODES:
                return mode
            logger.warning(
                "Unknown openai_compat compatibility_mode=%r; using %s",
                raw_mode,
                _STRICT_OPENAI_COMPAT,
            )
            return _STRICT_OPENAI_COMPAT

        host = (urlparse(self._base_url).hostname or "").lower()
        if host in _LOCAL_OPENAI_COMPAT_HOSTS:
            return _LLAMA_CPP_COMPAT
        return _STRICT_OPENAI_COMPAT

    def _compatibility_mode(self) -> str:
        return self._compatibility_mode_value

    def _supports_chat_template_kwargs(self) -> bool:
        return self._compatibility_mode() in {_LLAMA_CPP_COMPAT, _VLLM_COMPAT}

    def _supports_top_level_enable_thinking(self) -> bool:
        return self._compatibility_mode() == _LLAMA_CPP_COMPAT

    def _supports_no_think_retry(self) -> bool:
        return self._compatibility_mode() in {_LLAMA_CPP_COMPAT, _VLLM_COMPAT}

    def _should_drop_reasoning_like_content(self) -> bool:
        return self._compatibility_mode() in {_LLAMA_CPP_COMPAT, _VLLM_COMPAT}

    def _apply_thinking_payload(
        self,
        payload: dict[str, Any],
        payload_thinking: bool | None,
    ) -> None:
        if payload_thinking is None:
            return
        if self._supports_chat_template_kwargs():
            payload["chat_template_kwargs"] = {"enable_thinking": payload_thinking}
        if self._supports_top_level_enable_thinking():
            payload["enable_thinking"] = payload_thinking

    @staticmethod
    def _raise_for_status_with_body(resp: httpx.Response) -> None:
        status_code = getattr(resp, "status_code", None)
        if isinstance(status_code, int) and status_code >= 400:
            logger.error("openai_compat error body: %s", resp.text[:4000])
        resp.raise_for_status()

    def _strip_reasoning_text(self, raw: str) -> str:
        return strip_thinking_text(
            raw,
            tags=self._reasoning_strip_tags,
            marker_pairs=self._reasoning_strip_markers,
        )

    def _extract_reasoning_content(self, message: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in self._reasoning_content_keys:
            value = message.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
            elif value:
                parts.append(json.dumps(value, ensure_ascii=False))
        return "\n---\n".join(parts)

    def _effective_enable_thinking(self) -> bool | None:
        """Return ``enable_thinking`` accounting for voice-context override."""
        if is_voice_context() and self._voice_enable_thinking is not None:
            return self._voice_enable_thinking
        return self._enable_thinking

    def _effective_max_tokens(self, requested_max_tokens: int) -> int:
        resolved = _resolve_configured_max_tokens(
            self._default_max_tokens,
            requested_max_tokens,
        )
        if is_voice_context() and self._voice_max_tokens is not None:
            return min(resolved, self._voice_max_tokens)
        return resolved

    @staticmethod
    def _messages_with_no_think(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        retry_messages = [dict(message) for message in messages]
        no_think_instruction = (
            "/no_think\n"
            "Return only the final user-facing answer. "
            "Do not include analysis, checklists, or private reasoning."
        )
        if retry_messages and retry_messages[0].get("role") == "system":
            retry_messages[0]["content"] = (
                f"{retry_messages[0].get('content', '')}\n{no_think_instruction}"
            ).strip()
        else:
            retry_messages.insert(
                0,
                {"role": "system", "content": no_think_instruction},
            )
        return retry_messages

    async def _retry_without_thinking(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        labels: dict[str, str],
        reason: str,
    ) -> str:
        if not self._supports_no_think_retry():
            logger.warning(
                "%s No-think retry disabled for compatibility_mode=%s.",
                reason,
                self._compatibility_mode(),
            )
            return ""
        logger.warning("%s Retrying with thinking disabled.", reason)
        retry_payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages_with_no_think(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self._apply_thinking_payload(retry_payload, False)
        try:
            async with time_block(LLM_GENERATE_LATENCY, labels):
                retry_resp = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=retry_payload,
                )
                self._raise_for_status_with_body(retry_resp)
                retry_data = retry_resp.json()
        except httpx.ReadTimeout:
            logger.warning(
                "openai_compat no-think retry timed out. "
                "Increase 'ai_backend.timeout' in config.yaml."
            )
            return ""
        except Exception:
            logger.warning(
                "openai_compat no-think retry failed",
                exc_info=True,
            )
            return ""

        try:
            retry_message = retry_data["choices"][0]["message"]
        except (KeyError, IndexError):
            logger.warning("openai_compat no-think retry returned bad format")
            return ""
        reasoning_content = self._extract_reasoning_content(retry_message)
        if reasoning_content and self._log_reasoning_content:
            logger.debug(
                "openai_compat retry thinking (%d chars):\n%.2000s",
                len(reasoning_content),
                reasoning_content,
            )
        retry_content = retry_message.get("content") or ""
        result = self._strip_reasoning_text(str(retry_content))
        if result and looks_like_reasoning_text(result):
            logger.warning(
                "openai_compat no-think retry still looked like reasoning; "
                "dropping response"
            )
            return ""
        return result

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        max_tokens = self._effective_max_tokens(max_tokens)
        messages: list[dict[str, str]] = []
        effective_thinking = self._effective_enable_thinking()
        effective_system = system
        if effective_thinking is False and self._supports_no_think_retry():
            # Belt-and-suspenders: inject /no_think soft-switch into the system
            # message so Qwen3 disables thinking even if the server ignores
            # chat_template_kwargs (works across all llama.cpp versions).
            effective_system = (
                (system + "\n/no_think").lstrip() if system else "/no_think"
            )
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})
        payload_thinking = (
            False if "/no_think" in effective_system.lower() else effective_thinking
        )

        logger.debug(
            "openai_compat request: model=%s system=%d chars prompt=%d chars"
            " voice_ctx=%s thinking=%s",
            self._model,
            len(effective_system),
            len(prompt),
            is_voice_context(),
            payload_thinking,
        )
        labels = {"backend": "openai_compat", "model": self._model}
        try:
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            self._apply_thinking_payload(payload, payload_thinking)
            async with time_block(LLM_GENERATE_LATENCY, labels):
                resp = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                )
                self._raise_for_status_with_body(resp)
                data = resp.json()
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

            # Log reasoning content at DEBUG level so it
            # appears in logs when log.level=DEBUG without polluting responses.
            reasoning_content = self._extract_reasoning_content(message)
            if reasoning_content and self._log_reasoning_content:
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
                # Return empty string; callers (engine, generate_json) handle empty.
                if reasoning_content:
                    # Model exhausted max_tokens during thinking phase. Retry
                    # once with thinking disabled and a larger budget (at
                    # least 8192 tokens) so the model produces a direct reply.
                    if self._supports_no_think_retry():
                        _thinking_retry_floor = 8192
                        retry_mt = max(max_tokens * 2, _thinking_retry_floor)
                        retry_result = await self._retry_without_thinking(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=retry_mt,
                            labels=labels,
                            reason=(
                                "openai_compat: content=null but "
                                f"reasoning_content={len(reasoning_content)} chars. "
                                "Model spent the whole token budget on thinking. "
                                "Set ai_backend.options.enable_thinking: false in "
                                "config.yaml to avoid these retries."
                            ),
                        )
                        if retry_result:
                            logger.debug(
                                "openai_compat retry response: %d chars: %.300s",
                                len(retry_result),
                                retry_result,
                            )
                            return retry_result
                        logger.warning(
                            "openai_compat retry also returned empty content."
                        )
                    else:
                        logger.warning(
                            "openai_compat: content=null but reasoning_content=%d "
                            "chars; compatibility_mode=%s does not support "
                            "no-think retry",
                            len(reasoning_content),
                            self._compatibility_mode(),
                        )
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
                if think_blocks and self._log_reasoning_content:
                    combined_thinking = "\n---\n".join(think_blocks)
                    logger.debug(
                        "openai_compat inline thinking (%d chars):\n%.2000s",
                        len(combined_thinking),
                        combined_thinking,
                    )
                stripped = self._strip_reasoning_text(raw_content)
                reasoning_like = looks_like_reasoning_text(stripped)
                result = ""
                if reasoning_like:
                    reason = (
                        "openai_compat: content looked like reasoning "
                        "or model-control output."
                    )
                    if self._supports_no_think_retry():
                        result = await self._retry_without_thinking(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max(max_tokens, 1024),
                            labels=labels,
                            reason=reason,
                        )
                    else:
                        logger.warning(
                            "openai_compat: reasoning-like output detected, but "
                            "compatibility_mode=%s does not support no-think retry; "
                            "keeping content",
                            self._compatibility_mode(),
                        )
                        result = stripped or raw_content
                    if not result and self._should_drop_reasoning_like_content():
                        logger.warning("openai_compat: dropping reasoning-like content")
                if not reasoning_like and not stripped:
                    logger.warning(
                        "openai_compat: content was entirely <think> blocks; "
                        "set ai.options.enable_thinking: false in config.yaml"
                    )
                    result = (
                        ""
                        if self._should_drop_reasoning_like_content()
                        else raw_content
                    )
                elif not reasoning_like:
                    result = result or stripped
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
        max_tokens = self._effective_max_tokens(max_tokens)
        messages: list[dict[str, Any]] = []
        effective_thinking = self._effective_enable_thinking()
        effective_system = system
        if effective_thinking is False and self._supports_no_think_retry():
            effective_system = (
                (system + "\n/no_think").lstrip() if system else "/no_think"
            )
        if effective_system:
            messages.append({"role": "system", "content": effective_system})

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
        payload_thinking = (
            False if "/no_think" in effective_system.lower() else effective_thinking
        )

        labels = {"backend": "openai_compat", "model": self._model}
        try:
            logger.debug(
                "openai_compat vision request: model=%s system=%d chars "
                "prompt=%d chars images=%d",
                self._model,
                len(effective_system),
                len(prompt),
                len(images),
            )
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            self._apply_thinking_payload(payload, payload_thinking)
            async with time_block(LLM_GENERATE_LATENCY, labels):
                resp = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                )
                self._raise_for_status_with_body(resp)
                data = resp.json()
        except Exception:
            inc_counter(
                LLM_GENERATE_TOTAL,
                {"backend": "openai_compat", "outcome": "error"},
            )
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "openai_compat", "outcome": "ok"})
        try:
            result = str(data["choices"][0]["message"]["content"])
            logger.debug(
                "openai_compat vision response: %d chars: %.300s",
                len(result),
                result,
            )
            return result
        except (KeyError, IndexError) as exc:
            msg = f"Unexpected vision response format from {self._base_url}"
            raise AIBackendError(msg) from exc

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        max_tokens = self._effective_max_tokens(max_tokens)
        labels = {"backend": "openai_compat", "model": self._model}
        effective_thinking = self._effective_enable_thinking()
        has_no_think = any(
            str(message.get("content", "")).lower().find("/no_think") >= 0
            for message in messages
            if message.get("role") == "system"
        )
        payload_thinking = False if has_no_think else effective_thinking
        try:
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            self._apply_thinking_payload(payload, payload_thinking)
            async with time_block(LLM_GENERATE_LATENCY, labels):
                resp = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                )
                self._raise_for_status_with_body(resp)
                data = resp.json()
        except Exception:
            inc_counter(
                LLM_GENERATE_TOTAL,
                {"backend": "openai_compat", "outcome": "error"},
            )
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "openai_compat", "outcome": "ok"})
        try:
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            raw_content = str(content)
            stripped = self._strip_reasoning_text(raw_content)
            if looks_like_reasoning_text(stripped):
                if not self._supports_no_think_retry():
                    logger.warning(
                        "openai_compat: reasoning-like output detected, but "
                        "compatibility_mode=%s does not support no-think retry; "
                        "keeping content",
                        self._compatibility_mode(),
                    )
                    return stripped or raw_content
                retry_messages: list[dict[str, str]] = [
                    {
                        "role": str(item.get("role", "user")),
                        "content": str(item.get("content", "")),
                    }
                    for item in messages
                ]
                return await self._retry_without_thinking(
                    messages=retry_messages,
                    temperature=temperature,
                    max_tokens=max(max_tokens, 1024),
                    labels=labels,
                    reason=(
                        "openai_compat chat content looked like reasoning "
                        "or model-control output."
                    ),
                )
            return stripped if stripped else raw_content
        except (KeyError, IndexError) as exc:
            msg = f"Unexpected response format from {self._base_url}"
            raise AIBackendError(msg) from exc

    async def generate_chat_with_vision(
        self,
        messages: list[dict[str, Any]],
        images: list[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        enriched = [dict(message) for message in messages]
        for message in reversed(enriched):
            if message.get("role") != "user":
                continue
            original = message.get("content", "")
            if isinstance(original, list):
                content = list(original)
            else:
                content = [{"type": "text", "text": str(original)}]
            for b64img in images:
                mime = _detect_image_mime(b64img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64img}"},
                    }
                )
            message["content"] = content
            break
        return await self.generate_chat(enriched, temperature, max_tokens)


def create_backend(config: AIBackendConfig) -> AIBackend:
    """Factory function to create the appropriate AI backend."""
    backend: AIBackend
    backend_name: str
    match config.provider:
        case "ollama":
            backend = OllamaBackend(config)
            backend_name = "ollama"
        case "openai" | "openai_compat":
            backend = OpenAICompatBackend(config)
            backend_name = "openai_compat"
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
