"""Text-to-Speech backend abstraction for CordBeat.

To add a new TTS backend:
1. Subclass ``TTSBackend``, set ``content_type``, and implement ``synthesize()``.
2. Register the backend name in ``create_tts_backend()``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import httpx

from cordbeat.config import TTSConfig

logger = logging.getLogger(__name__)


class TTSBackend(ABC):
    """Abstract text-to-speech backend.

    ``content_type`` indicates the MIME type of bytes returned by
    ``synthesize()``.  Adapters use this to pick the right send method
    (e.g. ``send_voice`` for ``audio/ogg``, ``send_audio`` for ``audio/mpeg``).
    """

    content_type: str = "audio/mpeg"

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert *text* to audio bytes.

        Returns empty bytes on failure rather than raising so that the
        adapter can gracefully degrade to a plain-text reply.
        """


class EdgeTTSBackend(TTSBackend):
    """Microsoft Edge TTS via the *edge-tts* library (free, multilingual).

    Returns MP3 bytes (``content_type = "audio/mpeg"``).
    """

    content_type = "audio/mpeg"

    def __init__(self, config: TTSConfig) -> None:
        self._voice = config.voice or "en-US-AriaNeural"
        # Convert speed multiplier to ±% string expected by edge-tts
        pct = int((config.speed - 1.0) * 100)
        self._rate = f"{pct:+d}%"

    async def synthesize(self, text: str) -> bytes:
        try:
            import edge_tts
        except ImportError:
            logger.error(
                "edge-tts is not installed. Install with: uv sync --extra tts-edge"
            )
            return b""

        try:
            communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
            chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)
        except Exception:
            logger.exception("EdgeTTS synthesis failed")
            return b""


class OpenAITTS(TTSBackend):
    """OpenAI TTS API — returns ogg/opus (``content_type = "audio/ogg"``)."""

    content_type = "audio/ogg"

    def __init__(self, config: TTSConfig) -> None:
        self._api_key = config.api_key
        self._model = config.model or "tts-1"
        self._voice = config.voice or "alloy"
        self._speed = config.speed
        self._base_url = "https://api.openai.com"

    async def synthesize(self, text: str) -> bytes:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "response_format": "opus",
            "speed": self._speed,
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/v1/audio/speech",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                resp.raise_for_status()
                return bytes(resp.content)
        except Exception:
            logger.exception("OpenAI TTS synthesis failed")
            return b""


class OpenAICompatTTS(TTSBackend):
    """OpenAI-compatible TTS API (LocalAI, etc.) — ogg/opus output."""

    content_type = "audio/ogg"

    def __init__(self, config: TTSConfig) -> None:
        self._api_key = config.api_key
        self._model = config.model or "tts-1"
        self._voice = config.voice or "alloy"
        self._speed = config.speed
        self._base_url = config.api_url.rstrip("/")

    async def synthesize(self, text: str) -> bytes:
        if not self._base_url:
            logger.error("OpenAICompatTTS: tts.api_url is not configured")
            return b""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "response_format": "opus",
            "speed": self._speed,
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/v1/audio/speech",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                resp.raise_for_status()
                return bytes(resp.content)
        except Exception:
            logger.exception("OpenAICompatTTS synthesis failed")
            return b""


_BACKEND_MAP: dict[str, Callable[[TTSConfig], TTSBackend]] = {
    "edge_tts": EdgeTTSBackend,
    "openai": OpenAITTS,
    "openai_compat": OpenAICompatTTS,
}


def create_tts_backend(config: TTSConfig) -> TTSBackend:
    """Factory: create a TTS backend from *config*.

    Unsupported ``backend`` values fall back to ``edge_tts`` with an
    error log so the process continues rather than crashing on startup.
    """
    cls = _BACKEND_MAP.get(config.backend)
    if cls is None:
        logger.error(
            "Unknown TTS backend '%s'; falling back to edge_tts",
            config.backend,
        )
        cls = EdgeTTSBackend
    return cls(config)
