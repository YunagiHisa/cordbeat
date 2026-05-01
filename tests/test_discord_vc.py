"""Tests for Discord VC (voice channel) and RVC (voice conversion) support.

All external dependencies (discord.py, torch, numpy, etc.) are mocked so
these tests run in the default dev environment without extras installed.
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

from cordbeat.config import AdapterConfig, RVCConfig, TTSConfig

if TYPE_CHECKING:
    from cordbeat.adapters.discord import DiscordAdapter

# ---------------------------------------------------------------------------
# VoiceReceiver
# ---------------------------------------------------------------------------


class TestVoiceReceiver:
    def test_construction_stores_params(self) -> None:
        with patch("cordbeat.voice_recv._OpusDecoder"):
            from cordbeat.voice_recv import VoiceReceiver

            vc = MagicMock()
            receiver = VoiceReceiver(
                vc,
                silence_sec=1.2,
                min_speech_sec=0.4,
                noise_rms_threshold=60.0,
                noise_active_ratio=0.2,
                speech_queue_max=3,
            )
            assert receiver._silence_sec == 1.2
            assert receiver._min_speech_sec == 0.4
            assert receiver._noise_rms_threshold == 60.0
            assert receiver._noise_active_ratio == 0.2
            assert receiver._speech_queue_max == 3
            assert not receiver._running
            assert receiver.vc is vc

    def test_on_speech_end_registers_callback(self) -> None:
        with patch("cordbeat.voice_recv._OpusDecoder"):
            from cordbeat.voice_recv import VoiceReceiver

            receiver = VoiceReceiver(MagicMock())
            cb = AsyncMock()
            receiver.on_speech_end(cb)
            assert cb in receiver._callbacks

    async def test_start_without_voice_recv_installed(self) -> None:
        with patch("cordbeat.voice_recv._OpusDecoder"):
            from cordbeat.voice_recv import VoiceReceiver

            vc = MagicMock()
            receiver = VoiceReceiver(vc)

            # Simulate discord.ext.voice_recv not installed
            with patch.dict("sys.modules", {"discord.ext.voice_recv": None}):
                try:
                    await receiver.start()
                except Exception:
                    pass
                # Running flag was set even if BasicSink import fails
                assert receiver._running

    async def test_stop_clears_buffers(self) -> None:
        with patch("cordbeat.voice_recv._OpusDecoder"):
            from cordbeat.voice_recv import VoiceReceiver

            vc = MagicMock()
            receiver = VoiceReceiver(vc)
            receiver._running = True
            receiver._buffers[123] = bytearray(b"pcm")
            receiver._last_time[123] = 1.0

            await receiver.stop()

            assert not receiver._running
            assert 123 not in receiver._buffers
            assert 123 not in receiver._last_time

    def test_bot_packets_ignored(self) -> None:
        with patch("cordbeat.voice_recv._OpusDecoder"):
            from cordbeat.voice_recv import VoiceReceiver

            receiver = VoiceReceiver(MagicMock())
            receiver._running = True

            bot_user = MagicMock()
            bot_user.bot = True
            bot_user.id = 9999
            data = MagicMock()
            data.opus = b"\x01" * 10
            receiver._on_voice_data(bot_user, data)

            # No buffer should be created for a bot user
            assert 9999 not in receiver._buffers

    def test_is_non_speech_noise_no_numpy(self) -> None:
        from cordbeat.voice_recv import _is_non_speech_noise

        with patch("builtins.__import__", side_effect=ImportError):
            is_noise, rms, active = _is_non_speech_noise(b"\x00" * 100, 50.0, 0.15)
        assert not is_noise  # unknown → treat as speech


# ---------------------------------------------------------------------------
# RVCBackend
# ---------------------------------------------------------------------------


class TestRVCBackend:
    def test_not_loaded_by_default(self) -> None:
        from cordbeat.rvc_backend import RVCBackend

        backend = RVCBackend()
        assert not backend.is_loaded()

    def test_unload_when_not_loaded(self) -> None:
        from cordbeat.rvc_backend import RVCBackend

        backend = RVCBackend()
        backend.unload()  # should not raise
        assert not backend.is_loaded()

    def test_load_without_torch_logs_error(self) -> None:
        import cordbeat.rvc_backend as rvc_mod

        orig = rvc_mod._TORCH_AVAILABLE
        try:
            rvc_mod._TORCH_AVAILABLE = False
            backend = rvc_mod.RVCBackend()
            with patch.object(rvc_mod.logger, "error") as mock_err:
                backend.load("nonexistent.pth")
                mock_err.assert_called_once()
            assert not backend.is_loaded()
        finally:
            rvc_mod._TORCH_AVAILABLE = orig

    def test_load_missing_model_file(self) -> None:
        import cordbeat.rvc_backend as rvc_mod

        orig = rvc_mod._TORCH_AVAILABLE
        try:
            rvc_mod._TORCH_AVAILABLE = True
            backend = rvc_mod.RVCBackend()
            with patch("cordbeat.rvc_backend.logger") as mock_log:
                backend.load("/nonexistent/path/model.pth")
                mock_log.error.assert_called()
            assert not backend.is_loaded()
        finally:
            rvc_mod._TORCH_AVAILABLE = orig

    def test_convert_passthrough_when_not_loaded(self) -> None:
        from cordbeat.rvc_backend import RVCBackend

        backend = RVCBackend()
        wav = b"RIFF...."
        result = backend.convert(wav)
        assert result == wav  # passthrough

    def test_f0_to_coarse_import_guard(self) -> None:
        """_f0_to_coarse should fail gracefully if numpy raises."""
        from cordbeat.rvc_backend import _f0_to_coarse

        try:
            import numpy as np

            f0 = np.zeros(10, dtype=np.float32)
            coarse = _f0_to_coarse(f0)
            assert coarse.shape == (10,)
        except ImportError:
            pass  # numpy not installed; skip


# ---------------------------------------------------------------------------
# RVCWrappedTTS
# ---------------------------------------------------------------------------


class TestRVCWrappedTTS:
    async def test_passthrough_when_rvc_not_loaded(self) -> None:
        from cordbeat.ai.tts import RVCWrappedTTS

        inner = AsyncMock()
        inner.synthesize = AsyncMock(return_value=b"audio_bytes")
        inner.content_type = "audio/mpeg"

        rvc = MagicMock()
        rvc.is_loaded.return_value = False

        wrapped = RVCWrappedTTS(inner, rvc)
        assert wrapped.content_type == "audio/mpeg"

        result = await wrapped.synthesize("hello world")
        assert result == b"audio_bytes"
        rvc.convert.assert_not_called()

    async def test_applies_rvc_when_loaded(self) -> None:
        from cordbeat.ai.tts import RVCWrappedTTS

        inner = AsyncMock()
        inner.synthesize = AsyncMock(return_value=b"tts_wav")
        inner.content_type = "audio/wav"

        rvc = MagicMock()
        rvc.is_loaded.return_value = True
        rvc.convert.return_value = b"rvc_wav"

        wrapped = RVCWrappedTTS(inner, rvc)
        result = await wrapped.synthesize("hi")
        rvc.convert.assert_called_once_with(b"tts_wav")
        assert result == b"rvc_wav"

    async def test_fallback_on_rvc_error(self) -> None:
        from cordbeat.ai.tts import RVCWrappedTTS

        inner = AsyncMock()
        inner.synthesize = AsyncMock(return_value=b"original")
        inner.content_type = "audio/wav"

        rvc = MagicMock()
        rvc.is_loaded.return_value = True
        rvc.convert.side_effect = RuntimeError("GPU OOM")

        wrapped = RVCWrappedTTS(inner, rvc)
        result = await wrapped.synthesize("test")
        # Should fall back to original TTS output
        assert result == b"original"

    async def test_empty_audio_skips_rvc(self) -> None:
        from cordbeat.ai.tts import RVCWrappedTTS

        inner = AsyncMock()
        inner.synthesize = AsyncMock(return_value=b"")
        inner.content_type = "audio/wav"

        rvc = MagicMock()
        rvc.is_loaded.return_value = True

        wrapped = RVCWrappedTTS(inner, rvc)
        result = await wrapped.synthesize("")
        rvc.convert.assert_not_called()
        assert result == b""


# ---------------------------------------------------------------------------
# DiscordAdapter VC methods
# ---------------------------------------------------------------------------


class TestDiscordAdapterVC:
    def _make_adapter(self) -> DiscordAdapter:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test-token"})
        adapter = DiscordAdapter(config)
        return adapter

    async def test_on_vc_speech_no_ws(self) -> None:
        adapter = self._make_adapter()
        adapter._ws = None
        with patch("cordbeat.adapters.discord.logger") as mock_log:
            await adapter._on_vc_speech(111, 222, b"wav")
            mock_log.warning.assert_called()

    async def test_on_vc_speech_no_stt(self) -> None:
        adapter = self._make_adapter()
        adapter._ws = AsyncMock()
        adapter._stt = None
        await adapter._on_vc_speech(111, 222, b"wav")
        # WS should not receive any message
        adapter._ws.send.assert_not_called()

    async def test_on_vc_speech_empty_transcription(self) -> None:
        adapter = self._make_adapter()
        adapter._ws = AsyncMock()
        adapter._stt = AsyncMock()
        adapter._stt.transcribe = AsyncMock(return_value="")
        await adapter._on_vc_speech(111, 222, b"wav")
        adapter._ws.send.assert_not_called()

    async def test_on_vc_speech_forwards_transcription(self) -> None:
        import json

        adapter = self._make_adapter()
        adapter._ws = AsyncMock()
        adapter._stt = AsyncMock()
        adapter._stt.transcribe = AsyncMock(return_value="hello world")
        await adapter._on_vc_speech(111, 222, b"wav")

        adapter._ws.send.assert_called_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["content"] == "hello world"
        assert payload["platform_user_id"] == "222"
        assert payload["metadata"]["via_vc"] is True

    async def test_on_vc_speech_updates_vc_user_guild(self) -> None:
        adapter = self._make_adapter()
        adapter._ws = AsyncMock()
        adapter._stt = AsyncMock()
        adapter._stt.transcribe = AsyncMock(return_value="test")
        await adapter._on_vc_speech(555, 777, b"wav")
        assert adapter._vc_user_guild["777"] == 555

    async def test_on_vc_speech_muted_guild(self) -> None:
        adapter = self._make_adapter()
        adapter._ws = AsyncMock()
        adapter._stt = AsyncMock()
        adapter._vc_muted.add(111)
        await adapter._on_vc_speech(111, 222, b"wav")
        adapter._ws.send.assert_not_called()

    async def test_dispatch_routes_to_vc(self) -> None:
        adapter = self._make_adapter()
        adapter._vc_user_guild["42"] = 999
        adapter._vc_receivers[999] = MagicMock()
        adapter._speak_in_vc = AsyncMock()  # type: ignore[method-assign]

        await adapter._dispatch_core_message("42", "hello from core", [])
        adapter._speak_in_vc.assert_called_once_with(999, "hello from core")

    async def test_dispatch_falls_through_to_text_when_no_vc(self) -> None:
        adapter = self._make_adapter()
        adapter._send_to_discord = AsyncMock()  # type: ignore[method-assign]

        await adapter._dispatch_core_message("99", "text reply", [])
        adapter._send_to_discord.assert_called_once_with("99", "text reply", [])

    async def test_speak_in_vc_no_tts(self) -> None:
        adapter = self._make_adapter()
        adapter._tts = None
        # Should return silently
        await adapter._speak_in_vc(111, "test")

    async def test_speak_in_vc_no_bot(self) -> None:
        adapter = self._make_adapter()
        adapter._tts = AsyncMock()
        adapter._bot = None
        await adapter._speak_in_vc(111, "test")

    async def test_speak_in_vc_guild_not_found(self) -> None:
        adapter = self._make_adapter()
        adapter._tts = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_guild.return_value = None
        await adapter._speak_in_vc(111, "test")
        adapter._tts.synthesize.assert_not_called()

    async def test_speak_in_vc_not_connected(self) -> None:
        adapter = self._make_adapter()
        adapter._tts = AsyncMock()
        guild_mock = MagicMock()
        guild_mock.voice_client = None
        adapter._bot = MagicMock()
        adapter._bot.get_guild.return_value = guild_mock
        await adapter._speak_in_vc(111, "test")
        adapter._tts.synthesize.assert_not_called()

    async def test_speak_in_vc_plays_audio(self) -> None:
        import sys

        adapter = self._make_adapter()
        adapter._tts = AsyncMock()
        adapter._tts.synthesize = AsyncMock(return_value=b"RIFF....")

        vc_mock = MagicMock()
        vc_mock.is_connected.return_value = True
        vc_mock.is_playing.return_value = False

        guild_mock = MagicMock()
        guild_mock.voice_client = vc_mock

        adapter._bot = MagicMock()
        adapter._bot.get_guild.return_value = guild_mock

        discord_mock = MagicMock()
        ffmpeg_source = MagicMock()
        discord_mock.FFmpegPCMAudio.return_value = ffmpeg_source

        old = sys.modules.get("discord")
        sys.modules["discord"] = discord_mock
        try:
            await adapter._speak_in_vc(111, "hello")
        finally:
            if old is None:
                sys.modules.pop("discord", None)
            else:
                sys.modules["discord"] = old

        vc_mock.play.assert_called_once_with(ffmpeg_source)


# ---------------------------------------------------------------------------
# Slash command handlers
# ---------------------------------------------------------------------------


class TestSlashCommands:
    def _make_adapter(self) -> DiscordAdapter:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test-token"})
        return DiscordAdapter(config)

    async def test_handle_join_user_not_in_vc(self) -> None:
        adapter = self._make_adapter()
        interaction = MagicMock()
        interaction.user.voice = None
        interaction.response = AsyncMock()
        interaction.response.send_message = AsyncMock()

        await adapter._handle_join(interaction)
        interaction.response.send_message.assert_called_once()
        msg: str = interaction.response.send_message.call_args[0][0]
        assert "voice channel" in msg.lower()

    async def test_handle_join_voice_recv_not_installed(self) -> None:
        adapter = self._make_adapter()
        interaction = MagicMock()
        interaction.user.voice.channel = MagicMock()
        interaction.guild_id = 123
        interaction.response = AsyncMock()
        interaction.response.send_message = AsyncMock()

        with patch.dict(
            "sys.modules",
            {"discord.ext.voice_recv": None},
        ):
            real_import = builtins.__import__

            def mock_import(name: str, *args: object, **kwargs: object) -> object:
                if name == "discord.ext.voice_recv":
                    raise ImportError("not installed")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                await adapter._handle_join(interaction)

        interaction.response.send_message.assert_called()
    async def test_handle_leave_cleans_up(self) -> None:
        adapter = self._make_adapter()
        guild_id = 456
        adapter._vc_user_guild["123"] = guild_id
        adapter._vc_muted.add(guild_id)

        mock_receiver = AsyncMock()
        adapter._vc_receivers[guild_id] = mock_receiver

        guild_mock = MagicMock()
        vc_mock = AsyncMock()
        guild_mock.voice_client = vc_mock
        adapter._bot = MagicMock()
        adapter._bot.get_guild.return_value = guild_mock

        interaction = MagicMock()
        interaction.guild_id = guild_id
        interaction.response = AsyncMock()
        interaction.response.send_message = AsyncMock()

        await adapter._handle_leave(interaction)

        mock_receiver.stop.assert_called_once()
        vc_mock.disconnect.assert_called_once()
        assert guild_id not in adapter._vc_receivers
        assert "123" not in adapter._vc_user_guild
        assert guild_id not in adapter._vc_muted

    async def test_handle_mute_toggle(self) -> None:
        adapter = self._make_adapter()
        guild_id = 789
        interaction = MagicMock()
        interaction.guild_id = guild_id
        interaction.response = AsyncMock()
        interaction.response.send_message = AsyncMock()

        # First call: mutes
        await adapter._handle_mute(interaction)
        assert guild_id in adapter._vc_muted

        # Second call: unmutes
        await adapter._handle_mute(interaction)
        assert guild_id not in adapter._vc_muted


# ---------------------------------------------------------------------------
# Config — RVCConfig dataclass
# ---------------------------------------------------------------------------


class TestRVCConfig:
    def test_defaults(self) -> None:
        cfg = RVCConfig()
        assert not cfg.enabled
        assert cfg.model_path == ""
        assert cfg.index_path == ""
        assert cfg.f0_up_key == 0
        assert cfg.device == ""

    def test_custom_values(self) -> None:
        cfg = RVCConfig(enabled=True, model_path="model.pth", f0_up_key=6)
        assert cfg.enabled
        assert cfg.model_path == "model.pth"
        assert cfg.f0_up_key == 6

    def test_config_has_rvc_field(self) -> None:
        from cordbeat.config import Config

        cfg = Config()
        assert hasattr(cfg, "rvc")
        assert isinstance(cfg.rvc, RVCConfig)

    def test_load_config_rvc_section(self, tmp_path: Any) -> None:  # type: ignore[misc]
        import yaml

        from cordbeat.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {"rvc": {"enabled": True, "model_path": "model.pth", "f0_up_key": 12}}
            )
        )
        cfg = load_config(config_file)
        assert cfg.rvc.enabled
        assert cfg.rvc.f0_up_key == 12

    def test_resolve_rvc_paths(self, tmp_path: Any) -> None:  # type: ignore[misc]
        import yaml

        from cordbeat.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "rvc": {
                        "enabled": True,
                        "model_path": "models/rvc.pth",
                        "index_path": "models/rvc.index",
                    }
                }
            )
        )
        cfg = load_config(config_file)
        # Paths should be absolute after resolution
        from pathlib import Path

        assert Path(cfg.rvc.model_path).is_absolute()
        assert Path(cfg.rvc.index_path).is_absolute()


# ---------------------------------------------------------------------------
# create_tts_with_rvc factory
# ---------------------------------------------------------------------------


class TestCreateTTSWithRVC:
    def test_returns_plain_when_rvc_disabled(self) -> None:
        from cordbeat.ai.tts import create_tts_with_rvc

        tts_cfg = TTSConfig(backend="edge_tts")
        rvc_cfg = RVCConfig(enabled=False)
        backend = create_tts_with_rvc(tts_cfg, rvc_cfg)
        from cordbeat.ai.tts import RVCWrappedTTS

        assert not isinstance(backend, RVCWrappedTTS)

    def test_returns_plain_when_rvc_config_none(self) -> None:
        from cordbeat.ai.tts import create_tts_with_rvc

        tts_cfg = TTSConfig(backend="edge_tts")
        backend = create_tts_with_rvc(tts_cfg, None)
        from cordbeat.ai.tts import RVCWrappedTTS

        assert not isinstance(backend, RVCWrappedTTS)

    def test_returns_plain_when_no_model_path(self) -> None:
        from cordbeat.ai.tts import create_tts_with_rvc

        tts_cfg = TTSConfig(backend="edge_tts")
        rvc_cfg = RVCConfig(enabled=True, model_path="")
        backend = create_tts_with_rvc(tts_cfg, rvc_cfg)
        from cordbeat.ai.tts import RVCWrappedTTS

        assert not isinstance(backend, RVCWrappedTTS)

    def test_falls_back_on_rvc_init_error(self) -> None:
        from cordbeat.ai.tts import RVCWrappedTTS, create_tts_with_rvc

        tts_cfg = TTSConfig(backend="edge_tts")
        rvc_cfg = RVCConfig(enabled=True, model_path="model.pth")

        with patch(
            "cordbeat.rvc_backend.RVCBackend",
            side_effect=RuntimeError("torch error"),
        ):
            backend = create_tts_with_rvc(tts_cfg, rvc_cfg)

        assert not isinstance(backend, RVCWrappedTTS)
