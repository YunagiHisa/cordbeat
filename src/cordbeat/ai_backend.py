"""AI Backend abstraction layer."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from cordbeat.config import AIBackendConfig

logger = logging.getLogger(__name__)


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
        self._client = httpx.AsyncClient(timeout=120.0)

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

        resp = await self._client.post(
            f"{self._base_url}/api/generate",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.debug("Ollama raw keys: %s", list(data.keys()))
        logger.debug("Ollama response: %.200s", data.get("response", ""))
        return str(data.get("response", ""))


class OpenAICompatBackend(AIBackend):
    """OpenAI-compatible API backend (works with vLLM, LM Studio, etc.)."""

    def __init__(self, config: AIBackendConfig) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._model = config.model
        self._api_key = config.options.get("api_key", "")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.AsyncClient(timeout=120.0, headers=headers)

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

        resp = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError) as exc:
            msg = f"Unexpected response format from {self._base_url}"
            raise RuntimeError(msg) from exc


def create_backend(config: AIBackendConfig) -> AIBackend:
    """Factory function to create the appropriate AI backend."""
    match config.provider:
        case "ollama":
            return OllamaBackend(config)
        case "openai" | "openai_compat":
            return OpenAICompatBackend(config)
        case _:
            msg = f"Unknown AI backend provider: {config.provider}"
            raise ValueError(msg)
