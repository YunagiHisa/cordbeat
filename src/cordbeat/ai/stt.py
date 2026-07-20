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

_LOCAL_WHISPER_MODEL_NAMES = frozenset(
    {
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
        "turbo",
    }
)
_DEFAULT_API_STT_MODEL = "whisper-1"


def _parse_device(device: str) -> tuple[str, int | None]:
    """Split a device string like "cuda:1" into (device, index).

    faster-whisper/CTranslate2 expects the device ("cpu"/"cuda"/"auto") and the
    index as separate arguments. A plain "cuda" or "cpu" yields index None.
    """
    device = (device or "cpu").strip() or "cpu"
    base, sep, idx = device.partition(":")
    if sep and idx.isdigit():
        return base or "cpu", int(idx)
    return device, None


def _resolve_api_stt_model(model: str) -> str:
    """Map local Whisper size defaults to the cloud-compatible API default."""

    normalized = (model or "").strip()
    if not normalized or normalized in _LOCAL_WHISPER_MODEL_NAMES:
        return _DEFAULT_API_STT_MODEL
    return normalized


class STTBackend(ABC):
    """Abstract speech-to-text backend."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str = "") -> str:
        """Convert audio bytes to text.

        Returns an empty string on failure rather than raising, so the caller
        can gracefully degrade to a "could not transcribe" message.
        """

    async def preload(self) -> None:
        """Prepare the backend so the first transcription is not delayed.

        Default is a no-op (cloud backends have nothing to warm up). Local
        model backends override this to download/load weights ahead of the
        first utterance, which otherwise stalls the whole VC pipeline.
        """
        return None


class WhisperLocalSTT(STTBackend):
    """Local Whisper inference via *faster-whisper* (CPU/CUDA).

    The model is loaded lazily on the first transcription call to avoid a
    long startup delay when the backend is instantiated.
    """

    def __init__(self, config: STTConfig) -> None:
        self._model_size = config.model or "base"
        self._language = config.language
        # Split "cuda:1" into device="cuda" + index=1 for CTranslate2, which
        # takes the index as a separate argument rather than in the string.
        device = (config.device or "cpu").strip()
        self._device, self._device_index = _parse_device(device)
        self._compute_type = (config.compute_type or "").strip()
        self._model: Any = None  # faster_whisper.WhisperModel, loaded lazily
        self._transcribe_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()

    def _load_model_sync(
        self,
        device: str,
        device_index: int | None,
        compute_type: str,
    ) -> Any:
        """Load (downloading if needed) the faster-whisper model. Blocking."""
        from faster_whisper import WhisperModel

        kwargs: dict[str, Any] = {"device": device}
        if device_index is not None:
            kwargs["device_index"] = device_index
        if compute_type:
            kwargs["compute_type"] = compute_type
        return WhisperModel(self._model_size, **kwargs)

    async def _ensure_model(self) -> Any:
        """Return the loaded model, loading it off-loop under a lock.

        The large-v3 weights are ~3 GB; loading them lazily inside the first
        transcription stalled the whole VC pipeline while the download ran.
        If a CUDA load fails (e.g. VRAM exhausted by the LLM), fall back to
        CPU so VC keeps working — slower — instead of failing every utterance.
        """
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:
                return self._model
            loop = asyncio.get_running_loop()
            device_desc = self._device
            if self._device_index is not None:
                device_desc = f"{self._device}:{self._device_index}"
            logger.info(
                "Loading faster-whisper model %r on %s (compute_type=%s) ...",
                self._model_size,
                device_desc,
                self._compute_type or "default",
            )
            try:
                self._model = await loop.run_in_executor(
                    None,
                    self._load_model_sync,
                    self._device,
                    self._device_index,
                    self._compute_type,
                )
            except ImportError:
                raise
            except Exception:
                if self._device == "cpu":
                    raise
                logger.warning(
                    "faster-whisper failed to load on %s; falling back to CPU. "
                    "VC transcription will work but be slower. Free GPU VRAM "
                    "(e.g. reduce the LLM context) to use the GPU.",
                    device_desc,
                    exc_info=True,
                )
                # CPU cannot use CUDA compute types like int8_float16; let
                # CTranslate2 pick its CPU default (int8) instead.
                self._model = await loop.run_in_executor(
                    None, self._load_model_sync, "cpu", None, ""
                )
                self._device, self._device_index, self._compute_type = "cpu", None, ""
            logger.info(
                "faster-whisper model %r ready on %s",
                self._model_size,
                self._device,
            )
        return self._model

    async def preload(self) -> None:
        """Download/load the model at startup instead of on first speech."""
        try:
            await self._ensure_model()
        except ImportError:
            logger.error(
                "faster-whisper is not installed. "
                "Install with: uv sync --extra stt-local"
            )
        except Exception:
            logger.warning(
                "Failed to preload faster-whisper model %r; it will be loaded "
                "on first use",
                self._model_size,
                exc_info=True,
            )

    async def transcribe(self, audio_bytes: bytes, language: str = "") -> str:
        lang: str | None = language or self._language or None

        try:
            model = await self._ensure_model()
        except ImportError:
            logger.error(
                "faster-whisper is not installed. "
                "Install with: uv sync --extra stt-local"
            )
            return ""

        def _run() -> str:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            try:
                segments, _ = model.transcribe(tmp_path, language=lang)
                return "".join(seg.text for seg in segments).strip()
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # faster-whisper model instances are not intended to serve overlapping
        # calls. Concurrent VC fragments caused severe CPU contention and made
        # sub-second clips take tens of seconds.
        async with self._transcribe_lock:
            return await asyncio.get_running_loop().run_in_executor(None, _run)


class WhisperOpenAISTT(STTBackend):
    """OpenAI Whisper STT via the official cloud API."""

    _DEFAULT_BASE_URL = "https://api.openai.com"

    def __init__(self, config: STTConfig) -> None:
        self._api_key = config.api_key
        self._language = config.language
        self._model = _resolve_api_stt_model(config.model)
        self._base_url = (config.base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self._timeout = config.timeout

    async def transcribe(self, audio_bytes: bytes, language: str = "") -> str:
        lang = language or self._language
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data: dict[str, str] = {"model": self._model}
        if lang:
            data["language"] = lang
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=self._timeout,
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
        self._model = _resolve_api_stt_model(config.model)
        self._timeout = config.timeout

    async def transcribe(self, audio_bytes: bytes, language: str = "") -> str:
        if not self._base_url:
            logger.error("OpenAICompatSTT: stt.api_url is not configured")
            return ""
        lang = language or self._language
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data: dict[str, str] = {"model": self._model}
        if lang:
            data["language"] = lang
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=self._timeout,
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
