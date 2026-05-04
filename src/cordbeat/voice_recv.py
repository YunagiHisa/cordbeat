"""Discord Voice Channel receive + PCM processing for CordBeat.

Ports discord.ext.voice_recv (BasicSink) audio into per-user PCM buffers,
detects utterance boundaries via silence, converts to 16 kHz WAV, and
delivers the result to registered async callbacks.

Optional dependencies:
  - discord.py with opus support (discord extra)
  - discord-ext-voice-recv (discord-vc extra)
  - numpy + scipy (discord-vc extra, for high-quality PCM → WAV conversion)

The module imports cleanly even when none of the above are installed.
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import logging
import time
import wave
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # int16
FRAME_SIZE = 960  # 20 ms @ 48 kHz


def _is_non_speech_noise(
    pcm: bytes,
    noise_rms_threshold: float,
    noise_active_ratio: float,
) -> tuple[bool, float, float]:
    """Return (is_noise, rms, active_ratio) based on PCM energy distribution."""
    try:
        import numpy as np
    except ImportError:
        # Without numpy we cannot classify; treat everything as speech.
        return False, 0.0, 0.0

    if not pcm:
        return True, 0.0, 0.0

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size < FRAME_SIZE * CHANNELS:
        return True, 0.0, 0.0

    mono = samples.reshape(-1, CHANNELS).mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(mono))))

    frame_len = FRAME_SIZE
    usable = (mono.size // frame_len) * frame_len
    if usable <= 0:
        return True, rms, 0.0

    frames = mono[:usable].reshape(-1, frame_len)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    active = float(np.mean(frame_rms >= noise_rms_threshold))

    return rms < noise_rms_threshold or active < noise_active_ratio, rms, active


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Convert 48 kHz stereo int16 PCM to a 16 kHz mono WAV suitable for Whisper.

    Processing chain (when numpy + scipy are available):
    1. Stereo → mono (L+R average)
    2. DC offset removal
    3. 80 Hz high-pass filter (low-frequency noise suppression)
    4. Peak normalisation to −3 dBFS
    5. 48 kHz → 16 kHz anti-alias downsampling

    Falls back to a minimal WAV wrapper when scipy is not installed.
    """
    try:
        import numpy as np
        from scipy.signal import butter, resample_poly, sosfilt
    except ImportError:
        # Graceful degradation: return raw stereo 48 kHz WAV.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)
        return buf.getvalue()

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    stereo = samples.reshape(-1, 2)
    mono = stereo.mean(axis=1)
    mono -= mono.mean()

    sos = butter(4, 80, btype="high", fs=SAMPLE_RATE, output="sos")
    mono = sosfilt(sos, mono)

    peak = float(np.max(np.abs(mono)))
    if peak > 100:
        target_peak = 32768 * 0.708  # −3 dBFS
        gain = min(target_peak / peak, 10.0)
        mono *= gain

    downsampled = resample_poly(mono, up=1, down=3)
    downsampled = np.clip(downsampled, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(16000)
        wf.writeframes(downsampled.tobytes())
    return buf.getvalue()


class _OpusDecoder:
    """ctypes-based Opus decoder with packet loss concealment (PLC)."""

    def __init__(self) -> None:
        import discord.opus as _opus  # type: ignore[import-not-found]

        if not _opus.is_loaded():
            _opus._load_default()  # type: ignore[attr-defined]
        self._lib = _opus._lib  # type: ignore[attr-defined]

        self._lib.opus_decoder_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.opus_decoder_create.restype = ctypes.c_void_p
        self._lib.opus_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.opus_decode.restype = ctypes.c_int
        self._lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        self._lib.opus_decoder_destroy.restype = None

        err = ctypes.c_int(0)
        self._decoder = self._lib.opus_decoder_create(
            ctypes.c_int(SAMPLE_RATE),
            ctypes.c_int(CHANNELS),
            ctypes.byref(err),
        )
        if err.value != 0 or not self._decoder:
            raise RuntimeError(f"opus_decoder_create failed: error={err.value}")

    def decode(self, data: bytes) -> bytes | None:
        """Decode an Opus frame to PCM; returns None on failure."""
        try:
            pcm_buf = (ctypes.c_int16 * (FRAME_SIZE * CHANNELS))()
            data_buf = ctypes.create_string_buffer(data)
            result = self._lib.opus_decode(
                self._decoder,
                ctypes.cast(data_buf, ctypes.POINTER(ctypes.c_char)),
                ctypes.c_int32(len(data)),
                pcm_buf,
                ctypes.c_int(FRAME_SIZE),
                ctypes.c_int(0),
            )
            if result < 0:
                return None
            return bytes(pcm_buf)[: result * CHANNELS * SAMPLE_WIDTH]
        except Exception:
            return None

    def decode_plc(self) -> bytes | None:
        """Packet-loss concealment: synthesise audio from previous frame state."""
        try:
            pcm_buf = (ctypes.c_int16 * (FRAME_SIZE * CHANNELS))()
            result = self._lib.opus_decode(
                self._decoder,
                None,
                ctypes.c_int32(0),
                pcm_buf,
                ctypes.c_int(FRAME_SIZE),
                ctypes.c_int(0),
            )
            if result < 0:
                return None
            return bytes(pcm_buf)[: result * CHANNELS * SAMPLE_WIDTH]
        except Exception:
            return None

    def __del__(self) -> None:
        if hasattr(self, "_decoder") and self._decoder and hasattr(self, "_lib"):
            try:
                self._lib.opus_decoder_destroy(self._decoder)
            except Exception:
                pass


class VoiceReceiver:
    """Receive voice audio from a ``VoiceRecvClient``, buffer per-user PCM,
    detect utterance boundaries, and deliver WAV data to async callbacks.

    Usage::

        receiver = VoiceReceiver(voice_client)
        receiver.on_speech_end(my_callback)  # async def cb(user_id, wav_bytes)
        await receiver.start()
        ...
        await receiver.stop()

    Requires ``discord-ext-voice-recv`` (``uv sync --extra discord-vc``).
    DAVE (Secure Frames) encryption is handled transparently when the
    ``davey`` package is available.
    """

    def __init__(
        self,
        voice_client: Any,
        *,
        silence_sec: float = 0.8,
        min_speech_sec: float = 0.5,
        noise_rms_threshold: float = 50.0,
        noise_active_ratio: float = 0.15,
        speech_queue_max: int = 5,
    ) -> None:
        self.vc = voice_client
        self._decoder = _OpusDecoder()
        self._buffers: dict[int, bytearray] = defaultdict(bytearray)
        self._last_time: dict[int, float] = {}
        self._running = False
        self._callbacks: list[Any] = []
        self._silence_sec = silence_sec
        self._min_speech_sec = min_speech_sec
        self._noise_rms_threshold = noise_rms_threshold
        self._noise_active_ratio = noise_active_ratio
        self._speech_queue_max = speech_queue_max
        # Diagnostics
        self._pkt_count = 0
        self._decode_ok = 0
        self._decode_fail = 0
        self._dave_ok = 0
        self._dave_fail = 0
        self._dave_session: Any = None
        self._dave_media_audio: Any = None
        self._dave_skip = False

    # ── Public API ──────────────────────────────────────────────────────────

    def on_speech_end(self, callback: Any) -> None:
        """Register an async callback.

        Signature: ``async def callback(user_id: int, wav_data: bytes) -> None``
        """
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start receiving audio. The voice client must already be connected."""
        self._running = True
        self._init_dave()

        try:
            from discord.ext.voice_recv import (
                BasicSink,  # type: ignore[import-not-found]
            )
        except ImportError:
            logger.error(
                "discord-ext-voice-recv is not installed. "
                "Install with: uv sync --extra discord-vc"
            )
            return

        sink = BasicSink(self._on_voice_data, decode=False)
        self.vc.listen(sink)
        logger.info("Voice receive started (BasicSink decode=False, manual Opus)")
        asyncio.create_task(self._silence_detector())

    async def stop(self) -> None:
        """Stop receiving audio and clear buffers."""
        self._running = False
        try:
            self.vc.stop_listening()
        except Exception as exc:
            logger.debug("stop_listening skipped: %s", exc)
        self._buffers.clear()
        self._last_time.clear()

    # ── BasicSink callback ───────────────────────────────────────────────────

    def _on_voice_data(self, user: Any, data: Any) -> None:
        try:
            self._on_voice_data_inner(user, data)
        except Exception:
            if not hasattr(self, "_cb_err_logged"):
                self._cb_err_logged = True
                logger.exception("Voice data callback error")

    def _on_voice_data_inner(self, user: Any, data: Any) -> None:
        if not self._running:
            return

        self._pkt_count += 1

        if user is None:
            if not hasattr(self, "_null_user_logged"):
                self._null_user_logged = True
                logger.warning(
                    "Received voice packet with user=None (SSRC mapping incomplete)"
                )
            return
        if getattr(user, "bot", False):
            return

        opus_data: bytes = data.opus
        if not opus_data:
            return

        opus_data = self._dave_decrypt(user.id, opus_data)

        pcm = self._decoder.decode(opus_data)
        if pcm is None:
            self._decode_fail += 1
            if self._decode_fail <= 5:
                logger.warning(
                    "Opus decode failed #%d: len=%d hex=%s",
                    self._decode_fail,
                    len(opus_data),
                    opus_data[:16].hex(),
                )
            pcm = self._decoder.decode_plc()
            if pcm is None:
                pcm = b"\x00" * (FRAME_SIZE * CHANNELS * SAMPLE_WIDTH)
        else:
            self._decode_ok += 1

        self._buffers[user.id].extend(pcm)
        self._last_time[user.id] = time.monotonic()

    # ── DAVE encryption ──────────────────────────────────────────────────────

    def _init_dave(self) -> None:
        conn = getattr(self.vc, "_connection", None)
        dave = getattr(conn, "dave_session", None) if conn else None
        can_encrypt = getattr(conn, "can_encrypt", False) if conn else False

        logger.info("DAVE: session=%s, can_encrypt=%s", dave is not None, can_encrypt)

        if dave and hasattr(dave, "decrypt"):
            self._dave_session = dave
            try:
                import davey  # type: ignore[import-not-found]

                mt_cls = getattr(davey, "MediaType", None)
                if mt_cls:
                    for name in ("Audio", "AUDIO", "audio"):
                        val = getattr(mt_cls, name, None)
                        if val is not None:
                            self._dave_media_audio = val
                            break
                logger.info("DAVE MediaType.Audio = %s", self._dave_media_audio)
            except ImportError:
                logger.info("davey not installed; DAVE decryption disabled")
                self._dave_skip = True

    def _refresh_dave(self) -> None:
        conn = getattr(self.vc, "_connection", None)
        dave = getattr(conn, "dave_session", None) if conn else None
        if dave and dave is not self._dave_session and hasattr(dave, "decrypt"):
            self._dave_session = dave
            self._dave_skip = False
            logger.info("DAVE session refreshed (rekey detected)")

    def _dave_decrypt(self, user_id: int, data: bytes) -> bytes:
        if self._dave_skip or not self._dave_session or not self._dave_media_audio:
            return data
        try:
            result = self._dave_session.decrypt(user_id, self._dave_media_audio, data)
            if result is not None:
                self._dave_ok += 1
                if self._dave_ok == 1:
                    logger.info(
                        "DAVE decrypt success: %d B → %d B", len(data), len(result)
                    )
                return result  # type: ignore[return-value]
        except Exception:
            self._dave_fail += 1
            if self._dave_fail <= 5:
                logger.warning("DAVE decrypt failed #%d", self._dave_fail)
        return data

    # ── Silence detection ─────────────────────────────────────────────────────

    async def _silence_detector(self) -> None:
        _stats_tick = 0
        while self._running:
            now = time.monotonic()
            finished: list[tuple[int, bytes]] = []

            _stats_tick += 1
            if _stats_tick % 100 == 0 and self._pkt_count > 0:
                logger.info(
                    "VC stats: pkt=%d opus_ok=%d opus_fail=%d dave_ok=%d dave_fail=%d",
                    self._pkt_count,
                    self._decode_ok,
                    self._decode_fail,
                    self._dave_ok,
                    self._dave_fail,
                )
                self._refresh_dave()

            for user_id in list(self._last_time):
                last_t = self._last_time.get(user_id, now)
                if now - last_t < self._silence_sec:
                    continue
                if user_id not in self._buffers:
                    continue

                pcm = bytes(self._buffers.pop(user_id, b""))
                self._last_time.pop(user_id, None)

                duration = len(pcm) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
                if duration < self._min_speech_sec:
                    logger.debug("Skipping short utterance (%.1f s)", duration)
                    continue

                is_noise, rms, active_ratio = _is_non_speech_noise(
                    pcm, self._noise_rms_threshold, self._noise_active_ratio
                )
                if is_noise:
                    logger.debug(
                        "Skipping non-speech noise: user=%d duration=%.1f s "
                        "rms=%.1f active_ratio=%.2f",
                        user_id,
                        duration,
                        rms,
                        active_ratio,
                    )
                    continue

                wav = _pcm_to_wav(pcm)
                logger.info(
                    "Speech end: user=%d duration=%.1f s wav=%d bytes",
                    user_id,
                    duration,
                    len(wav),
                )
                finished.append((user_id, wav))

            for user_id, wav_data in finished:
                for cb in self._callbacks:
                    try:
                        await cb(user_id, wav_data)
                    except Exception:
                        logger.exception("Speech callback error for user %d", user_id)

            await asyncio.sleep(0.1)
