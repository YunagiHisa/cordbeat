"""Speech-to-Text backend abstraction for CordBeat.

To add a new STT backend:
1. Subclass ``STTBackend`` and implement ``transcribe()``.
2. Register the backend name in ``create_stt_backend()``.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from cordbeat.config import STTConfig

logger = logging.getLogger(__name__)


class STTBackend(ABC):
    """Abstract speech-to-text backend."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str = "") -> str:
        """Convert audio bytes to text.

        Returns an empty string on failure rather than raising, so the caller
        can gracefully degrade to a "could not transcribe" message.
        """


class WhisperLocalSTT(STTBackend):
    """Local Whisper inference via *faster-whisper* (CPU/CUDA).

    The model is loaded lazily on the first transcription call to avoid a
    long startup delay when the backend is instantiated.
    """

    def __init__(self, config: STTConfig) -> None:
        self._model_size = config.model or "base"
        self._language = config.language
        self._model: Any = None  # faster_whisper.WhisperModel, loaded lazily

    async def transcribe(self, audio_bytes: bytes, language: str = "") -> str:
        lang: str | None = language or self._language or None

        def _run() -> str:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                logger.error(
                    "faster-whisper is not installed. "
                    "Install with: uv sync --extra stt-local"
                )
                return ""

            if self._model is None:
                self._model = WhisperModel(self._model_size, device="cpu")

            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            try:
                segments, _ = self._model.transcribe(tmp_path, language=lang)
                return "".join(seg.text for seg in segments).strip()
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        return await asyncio.get_running_loop().run_in_executor(None, _run)


class WhisperOpenAISTT(STTBackend):
    """OpenAI Whisper STT via the official cloud API."""

    def __init__(self, config: STTConfig) -> None:
        self._api_key = config.api_key
        self._language = config.language
        self._base_url = "https://api.openai.com"

    async def transcribe(self, audio_bytes: bytes, language: str = "") -> str:
        lang = language or self._language
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data: dict[str, str] = {"model": "whisper-1"}
        if lang:
            data["language"] = lang
        files = {"file": ("audio.ogg", audio_bytes, "audio/ogg")}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=60.0,
                )
                resp.raise_for_status()
                return str(resp.json().get("text", "")).strip()
        except Exception:
            logger.exception("WhisperOpenAISTT transcription failed")
            return ""


class OpenAICompatSTT(STTBackend):
    """OpenAI-compatible STT API (whisper.cpp server, LocalAI, etc.)."""

    def __init__(self, config: STTConfig) -> None:
        self._base_url = config.api_url.rstrip("/")
        self._api_key = config.api_key
        self._language = config.language

    async def transcribe(self, audio_bytes: bytes, language: str = "") -> str:
        if not self._base_url:
            logger.error("OpenAICompatSTT: stt.api_url is not configured")
            return ""
        lang = language or self._language
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data: dict[str, str] = {"model": "whisper-1"}
        if lang:
            data["language"] = lang
        files = {"file": ("audio.ogg", audio_bytes, "audio/ogg")}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=60.0,
                )
                resp.raise_for_status()
                return str(resp.json().get("text", "")).strip()
        except Exception:
            logger.exception("OpenAICompatSTT transcription failed")
            return ""


_BACKEND_MAP: dict[str, Callable[[STTConfig], STTBackend]] = {
    "whisper_local": WhisperLocalSTT,
    "whisper_openai": WhisperOpenAISTT,
    "openai_compat": OpenAICompatSTT,
}


def create_stt_backend(config: STTConfig) -> STTBackend:
    """Factory: create an STT backend from *config*.

    Unsupported ``backend`` values fall back to ``whisper_openai`` with an
    error log so the process continues rather than crashing on startup.
    """
    cls = _BACKEND_MAP.get(config.backend)
    if cls is None:
        logger.error(
            "Unknown STT backend '%s'; falling back to whisper_openai",
            config.backend,
        )
        cls = WhisperOpenAISTT
    return cls(config)
