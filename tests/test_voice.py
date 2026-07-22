"""Tests for E-2 voice STT/TTS backends and adapter integration."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cordbeat.ai.stt import (
    OpenAICompatSTT,
    WhisperLocalSTT,
    WhisperOpenAISTT,
    create_stt_backend,
)
from cordbeat.ai.tts import (
    EdgeTTSBackend,
    OpenAICompatTTS,
    OpenAITTS,
    create_tts_backend,
)
from cordbeat.config import STTConfig, TTSConfig

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_stt_config_defaults() -> None:
    cfg = STTConfig()
    assert cfg.enabled is False
    assert cfg.backend == "whisper_openai"
    assert cfg.model == "base"
    assert cfg.device == ""
    assert cfg.batch_size == 1


def test_whisper_local_uses_configured_device() -> None:
    backend = WhisperLocalSTT(STTConfig(backend="whisper_local", device="cuda"))
    assert backend._device == "cuda"


def test_whisper_local_defaults_to_cpu_device() -> None:
    backend = WhisperLocalSTT(STTConfig(backend="whisper_local"))
    assert backend._device == "cpu"
    assert backend._device_index is None


def test_whisper_local_parses_device_index() -> None:
    backend = WhisperLocalSTT(
        STTConfig(backend="whisper_local", device="cuda:1")
    )
    assert backend._device == "cuda"
    assert backend._device_index == 1


def test_whisper_local_model_kwargs_include_compute_type_and_index() -> None:
    """VRAM-constrained GPUs need int8 quantization and a pinned device."""
    backend = WhisperLocalSTT(
        STTConfig(
            backend="whisper_local",
            model="large-v3",
            device="cuda:1",
            compute_type="int8_float16",
        )
    )
    captured: dict[str, Any] = {}

    class FakeWhisperModel:
        def __init__(self, model_size: str, **kwargs: Any) -> None:
            captured["model_size"] = model_size
            captured.update(kwargs)

    fake_module = MagicMock()
    fake_module.WhisperModel = FakeWhisperModel
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        backend._load_model_sync("cuda", 1, "int8_float16")

    assert captured["model_size"] == "large-v3"
    assert captured["device"] == "cuda"
    assert captured["device_index"] == 1
    assert captured["compute_type"] == "int8_float16"


def test_whisper_local_uses_configured_batch_size() -> None:
    backend = WhisperLocalSTT(
        STTConfig(backend="whisper_local", batch_size=1)
    )
    assert backend._batch_size == 1


def test_whisper_local_clamps_invalid_batch_size() -> None:
    backend = WhisperLocalSTT(
        STTConfig(backend="whisper_local", batch_size=0)
    )
    assert backend._batch_size == 1


def test_whisper_local_omits_compute_type_when_unset() -> None:
    backend = WhisperLocalSTT(STTConfig(backend="whisper_local", device="cpu"))
    captured: dict[str, Any] = {}

    class FakeWhisperModel:
        def __init__(self, model_size: str, **kwargs: Any) -> None:
            captured.update(kwargs)

    fake_module = MagicMock()
    fake_module.WhisperModel = FakeWhisperModel
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        backend._load_model_sync("cpu", None, "")

    assert "compute_type" not in captured
    assert "device_index" not in captured
    assert captured["device"] == "cpu"


async def test_whisper_local_preload_loads_model_once() -> None:
    """preload() loads the model at startup so the first utterance is not
    stalled behind a multi-gigabyte download (production VC incident)."""
    backend = WhisperLocalSTT(STTConfig(backend="whisper_local", model="large-v3"))
    fake_segment = MagicMock()
    fake_segment.text = "hi"
    fake_model = MagicMock()
    fake_model.transcribe = MagicMock(return_value=([fake_segment], None))
    load_calls = 0

    def fake_load(device: str, index: object, compute_type: str) -> object:
        nonlocal load_calls
        load_calls += 1
        return fake_model

    backend._load_model_sync = fake_load  # type: ignore[method-assign]

    await backend.preload()
    assert backend._model is fake_model
    assert load_calls == 1

    # A later transcribe reuses the preloaded model instead of loading again.
    with patch("tempfile.NamedTemporaryFile"), patch("cordbeat.ai.stt.Path"):
        result = await backend.transcribe(b"wav")

    assert result == "hi"
    assert load_calls == 1


async def test_whisper_local_preload_failure_is_non_fatal() -> None:
    backend = WhisperLocalSTT(STTConfig(backend="whisper_local"))

    def boom(device: str, index: object, compute_type: str) -> object:
        raise RuntimeError("no weights")

    backend._load_model_sync = boom  # type: ignore[method-assign]

    # Must not raise; model stays unloaded for a later lazy retry.
    await backend.preload()
    assert backend._model is None


async def test_whisper_local_falls_back_to_cpu_on_cuda_oom() -> None:
    """When the GPU is full, STT must degrade to CPU instead of failing every
    utterance (production: LLM filled VRAM, whisper load hit CUDA OOM)."""
    backend = WhisperLocalSTT(
        STTConfig(
            backend="whisper_local",
            device="cuda:1",
            compute_type="int8_float16",
        )
    )
    cpu_model = MagicMock()
    calls: list[tuple[str, object, str]] = []

    def fake_load(device: str, index: object, compute_type: str) -> object:
        calls.append((device, index, compute_type))
        if device != "cpu":
            raise RuntimeError("CUDA failed with error out of memory")
        return cpu_model

    backend._load_model_sync = fake_load  # type: ignore[method-assign]

    model = await backend._ensure_model()

    assert model is cpu_model
    # First tried cuda:1 with the CUDA compute type, then fell back to CPU
    # without the CUDA-only compute type.
    assert calls[0] == ("cuda", 1, "int8_float16")
    assert calls[1] == ("cpu", None, "")
    # State updated so later loads/logs reflect the CPU fallback.
    assert backend._device == "cpu"
    assert backend._device_index is None
    assert backend._compute_type == ""


async def test_whisper_local_cpu_load_failure_still_raises() -> None:
    """A genuine CPU load failure has no fallback and must propagate."""
    backend = WhisperLocalSTT(STTConfig(backend="whisper_local", device="cpu"))

    def boom(device: str, index: object, compute_type: str) -> object:
        raise RuntimeError("no weights")

    backend._load_model_sync = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await backend._ensure_model()


def test_tts_config_defaults() -> None:
    cfg = TTSConfig()
    assert cfg.enabled is False
    assert cfg.backend == "edge_tts"
    assert cfg.speed == 1.0


# ---------------------------------------------------------------------------
# create_stt_backend factory
# ---------------------------------------------------------------------------


def test_create_stt_backend_whisper_local() -> None:
    cfg = STTConfig(backend="whisper_local")
    backend = create_stt_backend(cfg)
    assert isinstance(backend, WhisperLocalSTT)


def test_create_stt_backend_whisper_openai() -> None:
    cfg = STTConfig(backend="whisper_openai")
    backend = create_stt_backend(cfg)
    assert isinstance(backend, WhisperOpenAISTT)


def test_create_stt_backend_openai_compat() -> None:
    cfg = STTConfig(backend="openai_compat")
    backend = create_stt_backend(cfg)
    assert isinstance(backend, OpenAICompatSTT)


def test_create_stt_backend_unknown_falls_back(caplog: Any) -> None:
    cfg = STTConfig(backend="unknown_backend")
    with caplog.at_level("ERROR"):
        backend = create_stt_backend(cfg)
    assert isinstance(backend, WhisperOpenAISTT)
    assert "Unknown STT backend" in caplog.text


# ---------------------------------------------------------------------------
# create_tts_backend factory
# ---------------------------------------------------------------------------


def test_create_tts_backend_edge() -> None:
    cfg = TTSConfig(backend="edge_tts")
    backend = create_tts_backend(cfg)
    assert isinstance(backend, EdgeTTSBackend)


def test_create_tts_backend_openai() -> None:
    cfg = TTSConfig(backend="openai")
    backend = create_tts_backend(cfg)
    assert isinstance(backend, OpenAITTS)


def test_create_tts_backend_openai_compat() -> None:
    cfg = TTSConfig(backend="openai_compat")
    backend = create_tts_backend(cfg)
    assert isinstance(backend, OpenAICompatTTS)


def test_create_tts_backend_unknown_falls_back(caplog: Any) -> None:
    cfg = TTSConfig(backend="nope")
    with caplog.at_level("ERROR"):
        backend = create_tts_backend(cfg)
    assert isinstance(backend, EdgeTTSBackend)
    assert "Unknown TTS backend" in caplog.text


# ---------------------------------------------------------------------------
# WhisperOpenAISTT
# ---------------------------------------------------------------------------


async def test_whisper_openai_stt_transcribes() -> None:
    cfg = STTConfig(backend="whisper_openai", api_key="key123")
    stt = WhisperOpenAISTT(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"text": "hello world"}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        result = await stt.transcribe(b"fake_audio", language="en")

    assert result == "hello world"
    files = mock_client.post.call_args.kwargs["files"]
    assert files["file"][0] == "audio.wav"
    assert files["file"][2] == "audio/wav"


async def test_whisper_openai_stt_uses_configured_api_model() -> None:
    cfg = STTConfig(
        backend="whisper_openai",
        api_key="key123",
        model="gpt-4o-transcribe",
    )
    stt = WhisperOpenAISTT(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"text": "hello world"}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        await stt.transcribe(b"fake_audio", language="en")

    data = mock_client.post.call_args.kwargs["data"]
    assert data["model"] == "gpt-4o-transcribe"


async def test_whisper_openai_stt_returns_empty_on_error() -> None:
    cfg = STTConfig(backend="whisper_openai", api_key="key")
    stt = WhisperOpenAISTT(cfg)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        mock_client_cls.return_value = mock_client

        result = await stt.transcribe(b"fake_audio")

    assert result == ""


# ---------------------------------------------------------------------------
# OpenAICompatSTT
# ---------------------------------------------------------------------------


async def test_openai_compat_stt_no_url_returns_empty(caplog: Any) -> None:
    cfg = STTConfig(backend="openai_compat", api_url="")
    stt = OpenAICompatSTT(cfg)
    with caplog.at_level("ERROR"):
        result = await stt.transcribe(b"audio")
    assert result == ""
    assert "api_url is not configured" in caplog.text


async def test_openai_compat_stt_transcribes() -> None:
    cfg = STTConfig(backend="openai_compat", api_url="http://localhost:8080")
    stt = OpenAICompatSTT(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"text": "  transcribed text  "}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        result = await stt.transcribe(b"audio")

    assert result == "transcribed text"
    files = mock_client.post.call_args.kwargs["files"]
    assert files["file"][0] == "audio.wav"
    assert files["file"][2] == "audio/wav"


async def test_openai_compat_stt_uses_configured_api_model() -> None:
    cfg = STTConfig(
        backend="openai_compat",
        api_url="http://localhost:8080",
        model="custom-whisper",
    )
    stt = OpenAICompatSTT(cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"text": "text"}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        await stt.transcribe(b"audio")

    data = mock_client.post.call_args.kwargs["data"]
    assert data["model"] == "custom-whisper"


# ---------------------------------------------------------------------------
# WhisperLocalSTT
# ---------------------------------------------------------------------------


async def test_whisper_local_stt_returns_empty_when_not_installed() -> None:
    cfg = STTConfig(backend="whisper_local", model="base")
    stt = WhisperLocalSTT(cfg)

    with patch.dict("sys.modules", {"faster_whisper": None}):
        result = await stt.transcribe(b"audio_data")

    assert result == ""


async def test_whisper_local_stt_serializes_concurrent_transcriptions() -> None:
    cfg = STTConfig(backend="whisper_local", model="base")
    stt = WhisperLocalSTT(cfg)
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    class FakeModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def transcribe(
            self, path: str, language: str | None = None
        ) -> tuple[list[Any], None]:
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            segment = MagicMock(text=" hello")
            return [segment], None

    fake_module = MagicMock(WhisperModel=FakeModel)
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        results = await asyncio.gather(
            stt.transcribe(b"first"),
            stt.transcribe(b"second"),
            stt.transcribe(b"third"),
        )

    assert results == ["hello", "hello", "hello"]
    assert max_active == 1


# ---------------------------------------------------------------------------
# OpenAI TTS
# ---------------------------------------------------------------------------


async def test_openai_tts_synthesizes() -> None:
    cfg = TTSConfig(backend="openai", api_key="key", voice="alloy")
    tts = OpenAITTS(cfg)

    mock_resp = MagicMock()
    mock_resp.content = b"ogg_audio_data"
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        result = await tts.synthesize("hello")

    assert result == b"ogg_audio_data"
    assert tts.content_type == "audio/ogg"


async def test_openai_tts_returns_empty_on_error() -> None:
    cfg = TTSConfig(backend="openai", api_key="key")
    tts = OpenAITTS(cfg)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=Exception("error"))
        mock_client_cls.return_value = mock_client

        result = await tts.synthesize("hello")

    assert result == b""


# ---------------------------------------------------------------------------
# OpenAICompatTTS
# ---------------------------------------------------------------------------


async def test_openai_compat_tts_no_url_returns_empty(caplog: Any) -> None:
    cfg = TTSConfig(backend="openai_compat", api_url="")
    tts = OpenAICompatTTS(cfg)
    with caplog.at_level("ERROR"):
        result = await tts.synthesize("hello")
    assert result == b""
    assert "api_url is not configured" in caplog.text


async def test_openai_compat_tts_synthesizes() -> None:
    cfg = TTSConfig(backend="openai_compat", api_url="http://localhost:8080")
    tts = OpenAICompatTTS(cfg)

    mock_resp = MagicMock()
    mock_resp.content = b"audio_bytes"
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        result = await tts.synthesize("hello world")

    assert result == b"audio_bytes"


# ---------------------------------------------------------------------------
# EdgeTTS
# ---------------------------------------------------------------------------


async def test_edge_tts_synthesizes() -> None:
    cfg = TTSConfig(backend="edge_tts", voice="ja-JP-NanamiNeural", speed=1.0)
    tts = EdgeTTSBackend(cfg)

    async def fake_stream() -> Any:
        yield {"type": "audio", "data": b"chunk1"}
        yield {"type": "audio", "data": b"chunk2"}
        yield {"type": "WordBoundary", "data": b"ignored"}

    mock_communicate = MagicMock()
    mock_communicate.stream.return_value = fake_stream()
    mock_edge_tts = MagicMock()
    mock_edge_tts.Communicate.return_value = mock_communicate

    with patch.dict("sys.modules", {"edge_tts": mock_edge_tts}):
        result = await tts.synthesize("Hello")

    assert result == b"chunk1chunk2"
    assert tts.content_type == "audio/mpeg"


async def test_edge_tts_returns_empty_when_not_installed() -> None:
    cfg = TTSConfig(backend="edge_tts")
    tts = EdgeTTSBackend(cfg)

    with patch.dict("sys.modules", {"edge_tts": None}):
        result = await tts.synthesize("hello")

    assert result == b""


# ---------------------------------------------------------------------------
# EdgeTTS speed conversion
# ---------------------------------------------------------------------------


def test_edge_tts_rate_positive() -> None:
    cfg = TTSConfig(backend="edge_tts", speed=1.25)
    tts = EdgeTTSBackend(cfg)
    assert tts._rate == "+25%"


def test_edge_tts_rate_negative() -> None:
    cfg = TTSConfig(backend="edge_tts", speed=0.75)
    tts = EdgeTTSBackend(cfg)
    assert tts._rate == "-25%"


def test_edge_tts_rate_neutral() -> None:
    cfg = TTSConfig(backend="edge_tts", speed=1.0)
    tts = EdgeTTSBackend(cfg)
    assert tts._rate == "+0%"


# ---------------------------------------------------------------------------
# Config loading round-trip
# ---------------------------------------------------------------------------


def test_load_config_stt_tts_defaults(tmp_path: Any) -> None:
    """load_config returns STTConfig/TTSConfig with defaults when not set."""
    from cordbeat.config import load_config

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.stt.enabled is False
    assert cfg.tts.enabled is False


def test_load_config_stt_tts_from_yaml(tmp_path: Any) -> None:
    from cordbeat.config import load_config

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        (
            "stt:\n  enabled: true\n  backend: openai_compat\n"
            "tts:\n  enabled: true\n  backend: openai\n"
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.stt.enabled is True
    assert cfg.stt.backend == "openai_compat"
    assert cfg.tts.enabled is True
    assert cfg.tts.backend == "openai"


# ---------------------------------------------------------------------------
# Adapter runner passes stt/tts configs
# ---------------------------------------------------------------------------


async def test_adapter_runner_passes_voice_configs(tmp_path: Any) -> None:
    """_run_adapter passes STTConfig/TTSConfig to TelegramAdapter."""
    from cordbeat.adapters.runner import _run_adapter

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        (
            "adapters:\n  telegram:\n    enabled: true\n    token: test_token\n"
            "stt:\n  enabled: true\ntts:\n  enabled: true\n"
        ),
        encoding="utf-8",
    )

    mock_instance = AsyncMock()
    mock_instance.start = AsyncMock(return_value=None)
    mock_instance.stop = AsyncMock(return_value=None)

    with patch(
        "cordbeat.adapters.telegram.TelegramAdapter",
        return_value=mock_instance,
    ) as mock_cls:
        await _run_adapter("telegram", str(cfg_path))

    # Verify STT/TTS configs were passed as keyword args
    call_kwargs = mock_cls.call_args
    assert call_kwargs is not None
    _, kwargs = call_kwargs
    assert "stt_config" in kwargs
    assert kwargs["stt_config"].enabled is True
    assert "tts_config" in kwargs
    assert kwargs["tts_config"].enabled is True
