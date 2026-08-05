"""Text-to-Speech backend abstraction for CordBeat.

To add a new TTS backend:
1. Subclass ``TTSBackend``, set ``content_type``, and implement ``synthesize()``.
2. Register the backend name in ``create_tts_backend()``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

import httpx

from cordbeat.ai.speech import (
    SpeechStyle,
    chunk_speech_text,
    compile_voice_prompt,
    merge_wav_chunks,
)
from cordbeat.config import RVCConfig, TTSConfig

if TYPE_CHECKING:
    from cordbeat.rvc_backend import RVCBackend

logger = logging.getLogger(__name__)


class TTSBackend(ABC):
    """Abstract text-to-speech backend.

    ``content_type`` indicates the MIME type of bytes returned by
    ``synthesize()``.  Adapters use this to pick the right send method
    (e.g. ``send_voice`` for ``audio/ogg``, ``send_audio`` for ``audio/mpeg``).
    """

    content_type: str = "audio/mpeg"
    supports_chunked_playback: bool = False
    playback_queue_size: int = 1

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        style: SpeechStyle | None = None,
    ) -> bytes:
        """Convert *text* to audio bytes.

        Returns empty bytes on failure rather than raising so that the
        adapter can gracefully degrade to a plain-text reply.
        """

    async def synthesize_chunks(
        self,
        text: str,
        *,
        style: SpeechStyle | None = None,
    ) -> AsyncIterator[bytes]:
        audio = await self.synthesize(text, style=style)
        if audio:
            yield audio

    async def preload(self) -> None:
        """Warm up or health-check a backend. Default is a no-op."""

    async def aclose(self) -> None:
        """Release persistent backend resources. Default is a no-op."""


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

    async def synthesize(
        self,
        text: str,
        *,
        style: SpeechStyle | None = None,
    ) -> bytes:
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
    _DEFAULT_BASE_URL = "https://api.openai.com"

    def __init__(self, config: TTSConfig) -> None:
        self._api_key = config.api_key
        self._model = config.model or "tts-1"
        self._voice = config.voice or "alloy"
        self._speed = config.speed
        self._base_url = (config.base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self._timeout = config.timeout

    async def synthesize(
        self,
        text: str,
        *,
        style: SpeechStyle | None = None,
    ) -> bytes:
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
                    timeout=self._timeout,
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
        self._timeout = config.timeout

    async def synthesize(
        self,
        text: str,
        *,
        style: SpeechStyle | None = None,
    ) -> bytes:
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
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                return bytes(resp.content)
        except Exception:
            logger.exception("OpenAICompatTTS synthesis failed")
            return b""


_VOICE_DESIGN_SYNTHESIS_SEMAPHORE = asyncio.Semaphore(1)


class VoiceDesignTTS(TTSBackend):
    """Prompt-capable OpenAI-compatible TTS with chunked WAV synthesis.

    CordBeat exposes generic voice-profile and speech-direction settings. The
    provider-specific request shape is intentionally confined to this adapter.
    """

    content_type = "audio/wav"
    supports_chunked_playback = True

    def __init__(self, config: TTSConfig) -> None:
        self._config = config
        self._base_url = config.api_url.rstrip("/")
        self._api_key = config.api_key
        self._model = config.model or "tts-1"
        self._response_format = config.response_format or "wav"
        self._timeout = max(1.0, float(config.timeout))
        options = config.backend_options
        self._num_steps = max(1, int(options.get("num_steps", 16)))
        self._retries = max(0, min(2, int(options.get("retries", 1))))
        self._voice = str(options.get("voice") or "none")
        self.playback_queue_size = max(1, int(config.streaming.max_queue_size))
        self._client = httpx.AsyncClient(
            base_url=self._base_url or "http://127.0.0.1",
            timeout=self._timeout,
        )
        self._reachable: bool | None = None

    async def preload(self) -> None:
        if not self._base_url:
            logger.error("VoiceDesignTTS: tts.api_url is not configured")
            self._reachable = False
            return
        try:
            response = await self._client.get(
                "/health",
                timeout=min(5.0, self._timeout),
            )
            response.raise_for_status()
            data = response.json()
            self._reachable = data.get("status") == "ok"
            if self._reachable:
                logger.info("Voice-design TTS API is healthy and ready")
            else:
                logger.warning("Voice-design TTS API returned an unhealthy status")
        except Exception:
            self._reachable = False
            logger.warning(
                "Voice-design TTS API is unavailable; voice replies will "
                "degrade safely",
                exc_info=True,
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def synthesize(
        self,
        text: str,
        *,
        style: SpeechStyle | None = None,
    ) -> bytes:
        chunks = [
            chunk
            async for chunk in self.synthesize_chunks(text, style=style)
            if chunk
        ]
        if not chunks:
            return b""
        try:
            return merge_wav_chunks(chunks)
        except (ValueError, wave.Error):
            logger.exception("Voice-design TTS returned incompatible WAV chunks")
            return b""

    async def synthesize_chunks(
        self,
        text: str,
        *,
        style: SpeechStyle | None = None,
    ) -> AsyncIterator[bytes]:
        if not self._base_url:
            logger.error("VoiceDesignTTS: tts.api_url is not configured")
            return
        streaming = self._config.streaming
        chunks = (
            chunk_speech_text(
                text,
                min_chars=streaming.chunk_min_chars,
                max_chars=streaming.chunk_max_chars,
            )
            if streaming.enabled
            else [text.strip()]
        )
        caption = compile_voice_prompt(self._config.voice_profile, style)
        for index, chunk in enumerate(chunks):
            if not chunk:
                continue
            audio = await self._synthesize_one(
                chunk,
                caption=caption,
                chunk_index=index,
            )
            if not audio:
                return
            yield audio

    async def _synthesize_one(
        self,
        text: str,
        *,
        caption: str,
        chunk_index: int,
    ) -> bytes:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "response_format": self._response_format,
            # Provider extension. It remains private to this transport adapter.
            "irodori": {
                "caption": caption,
                "num_steps": self._num_steps,
            },
        }
        for attempt in range(self._retries + 1):
            try:
                async with _VOICE_DESIGN_SYNTHESIS_SEMAPHORE:
                    response = await self._client.post(
                        "/v1/audio/speech",
                        json=payload,
                        headers=headers,
                    )
                if response.status_code >= 400:
                    detail = response.text[:500]
                    folded_detail = detail.casefold()
                    is_oom = (
                        "out of memory" in folded_detail
                        or "cuda oom" in folded_detail
                    )
                    if is_oom:
                        logger.error(
                            "Voice-design TTS CUDA OOM chunk_index=%d chunk_chars=%d",
                            chunk_index,
                            len(text),
                        )
                    if response.status_code < 500 or is_oom or attempt >= self._retries:
                        logger.error(
                            "Voice-design TTS HTTP %d chunk_index=%d detail=%.500s",
                            response.status_code,
                            chunk_index,
                            detail,
                        )
                        return b""
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                audio = bytes(response.content)
                if not self._valid_wav(audio):
                    logger.error(
                        "Voice-design TTS returned invalid WAV chunk_index=%d bytes=%d",
                        chunk_index,
                        len(audio),
                    )
                    return b""
                self._reachable = True
                return audio
            except asyncio.CancelledError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self._retries:
                    logger.warning(
                        "Voice-design TTS request failed chunk_index=%d: %s",
                        chunk_index,
                        exc,
                    )
                    self._reachable = False
                    return b""
                await asyncio.sleep(0.5 * (2**attempt))
            except Exception:
                logger.exception(
                    "Voice-design TTS synthesis failed chunk_index=%d", chunk_index
                )
                return b""
        return b""

    @staticmethod
    def _valid_wav(audio: bytes) -> bool:
        if not audio:
            return False
        try:
            with wave.open(io.BytesIO(audio), "rb") as reader:
                return reader.getnframes() > 0 and reader.getframerate() > 0
        except (EOFError, wave.Error):
            return False


_BACKEND_MAP: dict[str, Callable[[TTSConfig], TTSBackend]] = {
    "edge_tts": EdgeTTSBackend,
    "openai": OpenAITTS,
    "openai_compat": OpenAICompatTTS,
    "voice_design": VoiceDesignTTS,
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


class RVCWrappedTTS(TTSBackend):
    """Wraps any TTSBackend and passes its audio through RVC voice conversion."""

    def __init__(self, inner: TTSBackend, rvc: RVCBackend) -> None:
        self._inner = inner
        self._rvc = rvc

    @property
    def content_type(self) -> str:  # type: ignore[override]
        return self._inner.content_type

    async def synthesize(
        self,
        text: str,
        *,
        style: SpeechStyle | None = None,
    ) -> bytes:
        wav = await self._inner.synthesize(text, style=style)
        if not wav or not self._rvc.is_loaded():
            return wav
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._rvc.convert, wav)
        except Exception:
            logger.exception("RVC conversion failed; returning original audio")
            return wav

    @property
    def supports_chunked_playback(self) -> bool:  # type: ignore[override]
        return self._inner.supports_chunked_playback

    @property
    def playback_queue_size(self) -> int:  # type: ignore[override]
        return self._inner.playback_queue_size

    async def synthesize_chunks(
        self,
        text: str,
        *,
        style: SpeechStyle | None = None,
    ) -> AsyncIterator[bytes]:
        async for wav in self._inner.synthesize_chunks(text, style=style):
            if not wav or not self._rvc.is_loaded():
                if wav:
                    yield wav
                continue
            try:
                converted = await asyncio.get_running_loop().run_in_executor(
                    None, self._rvc.convert, wav
                )
                if converted:
                    yield converted
            except Exception:
                logger.exception("RVC conversion failed; returning original audio")
                yield wav

    async def preload(self) -> None:
        await self._inner.preload()

    async def aclose(self) -> None:
        await self._inner.aclose()


def create_tts_with_rvc(
    tts_config: TTSConfig, rvc_config: RVCConfig | None = None
) -> TTSBackend:
    """Factory: create a TTS backend, optionally wrapped with RVC."""
    backend = create_tts_backend(tts_config)
    if rvc_config is None or not rvc_config.enabled:
        return backend
    if not rvc_config.model_path:
        logger.warning("RVC is enabled but model_path is empty; using plain TTS")
        return backend
    try:
        from cordbeat.rvc_backend import RVCBackend

        rvc = RVCBackend()
        rvc.load(
            model_path=rvc_config.model_path,
            index_path=rvc_config.index_path or None,
            f0_up_key=rvc_config.f0_up_key,
            device=rvc_config.device or None,
        )
        return RVCWrappedTTS(backend, rvc)
    except Exception:
        logger.exception("Failed to initialise RVC backend; using plain TTS")
        return backend
