"""AI Backend abstraction layer."""

from __future__ import annotations

import base64
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from cordbeat.config import AIBackendConfig
from cordbeat.exceptions import AIBackendError
from cordbeat.metrics import (
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
        raw = await self.generate(prompt, system, temperature, max_tokens)
        # Extract JSON from potential markdown code blocks
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # skip ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
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
        logger.debug("Ollama response: %.200s", data.get("response", ""))
        return str(data.get("response", ""))

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
        except Exception:
            inc_counter(LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "error"})
            raise
        inc_counter(LLM_GENERATE_TOTAL, {"backend": "ollama", "outcome": "ok"})
        return str(data.get("message", {}).get("content", ""))


class OpenAICompatBackend(AIBackend):
    """OpenAI-compatible API backend (works with vLLM, LM Studio, etc.)."""

    def __init__(self, config: AIBackendConfig) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._model = config.model
        self._api_key = config.options.get("api_key", "")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        labels = {"backend": "openai_compat", "model": self._model}
        try:
            async with time_block(LLM_GENERATE_LATENCY, labels):
                resp = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json={
                        "model": self._model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
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
            async with time_block(LLM_GENERATE_LATENCY, labels):
                resp = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json={
                        "model": self._model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
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
        from cordbeat.llm_cache import CachingBackend  # noqa: PLC0415

        return CachingBackend(
            inner=backend,
            config=config.cache,
            model=config.model,
            backend_name=backend_name,
        )
    return backend
