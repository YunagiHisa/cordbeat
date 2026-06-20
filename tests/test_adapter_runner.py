"""Tests for adapter runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cordbeat.adapters.runner import _resolve_adapter_config_path, _run_adapter


@pytest.fixture
def config_path(tmp_path: Path) -> str:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
        "adapters:\n"
        "  discord:\n    enabled: true\n"
        "    options:\n      token: fake\n"
        "  telegram:\n    enabled: true\n"
        "    options:\n      token: fake\n"
        "  disabled:\n    enabled: false\n",
        encoding="utf-8",
    )
    return str(cfg)


class TestRunAdapter:
    async def test_unknown_adapter(self, config_path: str) -> None:
        """Unknown adapter name logs error and returns."""
        await _run_adapter("nonexistent", config_path)

    async def test_disabled_adapter(self, tmp_path: Path) -> None:
        """Disabled adapter returns early."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
            "adapters:\n  discord:\n    enabled: false\n",
            encoding="utf-8",
        )
        await _run_adapter("discord", str(cfg))

    async def test_missing_adapter_config_uses_defaults(self, tmp_path: Path) -> None:
        """Adapter not in config gets default AdapterConfig."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n",
            encoding="utf-8",
        )
        mock_adapter = AsyncMock()
        with patch(
            "cordbeat.adapters.discord.DiscordAdapter",
            return_value=mock_adapter,
        ):
            with patch(
                "cordbeat.adapters.runner.DiscordAdapter",
                create=True,
                return_value=mock_adapter,
            ):
                await _run_adapter("discord", str(cfg))
                mock_adapter.start.assert_called_once()
                mock_adapter.stop.assert_called_once()

    async def test_discord_adapter_created(self, config_path: str) -> None:
        """Discord adapter is instantiated and started."""
        mock_adapter = AsyncMock()
        with patch(
            "cordbeat.adapters.discord.DiscordAdapter",
            return_value=mock_adapter,
        ):
            with patch(
                "cordbeat.adapters.runner.DiscordAdapter",
                create=True,
                return_value=mock_adapter,
            ):
                await _run_adapter("discord", config_path)
                mock_adapter.start.assert_called_once()
                mock_adapter.stop.assert_called_once()

    async def test_telegram_adapter_created(self, config_path: str) -> None:
        """Telegram adapter is instantiated and started."""
        mock_adapter = AsyncMock()
        with patch(
            "cordbeat.adapters.telegram.TelegramAdapter",
            return_value=mock_adapter,
        ):
            with patch(
                "cordbeat.adapters.runner.TelegramAdapter",
                create=True,
                return_value=mock_adapter,
            ):
                await _run_adapter("telegram", config_path)
                mock_adapter.start.assert_called_once()
                mock_adapter.stop.assert_called_once()

    @pytest.mark.parametrize(
        "adapter_name,module_path,class_name",
        [
            ("slack", "cordbeat.adapters.slack", "SlackAdapter"),
            ("line", "cordbeat.adapters.line", "LineAdapter"),
            ("whatsapp", "cordbeat.adapters.whatsapp", "WhatsAppAdapter"),
            ("signal", "cordbeat.adapters.signal", "SignalAdapter"),
        ],
    )
    async def test_v1_adapter_created(
        self,
        tmp_path: Path,
        adapter_name: str,
        module_path: str,
        class_name: str,
    ) -> None:
        """Optional scaffold adapters are instantiated and started."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
            f"adapters:\n  {adapter_name}:\n    enabled: true\n",
            encoding="utf-8",
        )
        mock_adapter = AsyncMock()
        with patch(f"{module_path}.{class_name}", return_value=mock_adapter):
            await _run_adapter(adapter_name, str(cfg))
            mock_adapter.start.assert_called_once()
            mock_adapter.stop.assert_called_once()

    async def test_discord_loads_soul_name(self, tmp_path: Path) -> None:
        """The Discord branch reads the character name from soul.yaml."""
        soul_dir = tmp_path / "soul"
        soul_dir.mkdir()
        (soul_dir / "soul.yaml").write_text(
            "identity:\n  name: Aria\n", encoding="utf-8"
        )
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
            f"soul:\n  soul_dir: {soul_dir.as_posix()}\n"
            "adapters:\n  discord:\n    enabled: true\n",
            encoding="utf-8",
        )
        mock_adapter = AsyncMock()
        captured = {}
        with patch(
            "cordbeat.adapters.discord.DiscordAdapter",
            side_effect=lambda *a, **kw: captured.update(kw) or mock_adapter,
        ):
            await _run_adapter("discord", str(cfg))
        assert captured["soul_name"] == "Aria"

    async def test_discord_soul_load_failure_is_tolerated(self, tmp_path: Path) -> None:
        """A malformed soul.yaml leaves soul_name empty without raising."""
        soul_dir = tmp_path / "soul"
        soul_dir.mkdir()
        (soul_dir / "soul.yaml").write_text("identity: [unterminated", encoding="utf-8")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
            f"soul:\n  soul_dir: {soul_dir.as_posix()}\n"
            "adapters:\n  discord:\n    enabled: true\n",
            encoding="utf-8",
        )
        mock_adapter = AsyncMock()
        captured = {}
        with patch(
            "cordbeat.adapters.discord.DiscordAdapter",
            side_effect=lambda *a, **kw: captured.update(kw) or mock_adapter,
        ):
            await _run_adapter("discord", str(cfg))
        assert captured["soul_name"] == ""

    async def test_judge_backend_initialized_and_closed(self, tmp_path: Path) -> None:
        """An ai_decision config wires up and later closes the judge backend."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
            "ai_decision:\n  provider: ollama\n"
            "  options:\n    judge_max_tokens: 100\n    judge_temperature: 0.5\n"
            "adapters:\n  signal:\n    enabled: true\n",
            encoding="utf-8",
        )
        judge = AsyncMock()
        mock_adapter = AsyncMock()
        with (
            patch("cordbeat.ai.backend.create_backend", return_value=judge) as mk,
            patch("cordbeat.adapters._utils.set_judge_backend") as set_judge,
            patch("cordbeat.adapters.signal.SignalAdapter", return_value=mock_adapter),
        ):
            await _run_adapter("signal", str(cfg))

        mk.assert_called_once()
        # First call wires the backend with parsed options; final call clears it.
        first_kwargs = set_judge.call_args_list[0].kwargs
        assert first_kwargs == {"max_tokens": 100, "temperature": 0.5}
        judge.aclose.assert_awaited_once()
        assert set_judge.call_args_list[-1].args == (None,)

    async def test_judge_backend_invalid_options_warn_and_default(
        self, tmp_path: Path
    ) -> None:
        """Non-numeric judge options fall back to None instead of crashing."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
            "ai_decision:\n  provider: ollama\n"
            "  options:\n    judge_max_tokens: not-a-number\n"
            "    judge_temperature: also-bad\n"
            "adapters:\n  signal:\n    enabled: true\n",
            encoding="utf-8",
        )
        with (
            patch("cordbeat.ai.backend.create_backend", return_value=AsyncMock()),
            patch("cordbeat.adapters._utils.set_judge_backend") as set_judge,
            patch("cordbeat.adapters.signal.SignalAdapter", return_value=AsyncMock()),
        ):
            await _run_adapter("signal", str(cfg))
        assert set_judge.call_args_list[0].kwargs == {
            "max_tokens": None,
            "temperature": None,
        }

    async def test_judge_backend_init_failure_is_tolerated(
        self, tmp_path: Path
    ) -> None:
        """If the judge backend can't be built, the adapter still starts."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
            "ai_decision:\n  provider: ollama\n"
            "adapters:\n  signal:\n    enabled: true\n",
            encoding="utf-8",
        )
        mock_adapter = AsyncMock()
        with (
            patch(
                "cordbeat.ai.backend.create_backend",
                side_effect=RuntimeError("no backend"),
            ),
            patch("cordbeat.adapters.signal.SignalAdapter", return_value=mock_adapter),
        ):
            await _run_adapter("signal", str(cfg))
        mock_adapter.start.assert_awaited_once()

    async def test_file_logging_handler_added(self, tmp_path: Path) -> None:
        """A configured log file produces a per-adapter rotating log file."""
        log_file = tmp_path / "cordbeat.log"
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
            f"log:\n  file: {log_file.as_posix()}\n"
            "adapters:\n  signal:\n    enabled: true\n",
            encoding="utf-8",
        )
        with patch("cordbeat.adapters.signal.SignalAdapter", return_value=AsyncMock()):
            await _run_adapter("signal", str(cfg))
        # The runner derives "<stem>-<adapter><suffix>" next to the core log.
        assert (tmp_path / "cordbeat-signal.log").exists()

    async def test_gateway_auth_token_propagates_to_adapter(
        self, tmp_path: Path
    ) -> None:
        """A gateway auth_token is copied onto an adapter that lacks one."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n  auth_token: secret\n"
            "adapters:\n  signal:\n    enabled: true\n",
            encoding="utf-8",
        )
        captured = {}
        with patch(
            "cordbeat.adapters.signal.SignalAdapter",
            side_effect=lambda cfg, **kw: captured.update(cfg=cfg) or AsyncMock(),
        ):
            await _run_adapter("signal", str(cfg))
        assert captured["cfg"].auth_token == "secret"

    async def test_judge_backend_close_failure_is_tolerated(
        self, tmp_path: Path
    ) -> None:
        """An error while closing the judge backend is logged, not raised."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
            "ai_decision:\n  provider: ollama\n"
            "adapters:\n  signal:\n    enabled: true\n",
            encoding="utf-8",
        )
        judge = AsyncMock()
        judge.aclose.side_effect = RuntimeError("close failed")
        with (
            patch("cordbeat.ai.backend.create_backend", return_value=judge),
            patch("cordbeat.adapters._utils.set_judge_backend"),
            patch("cordbeat.adapters.signal.SignalAdapter", return_value=AsyncMock()),
        ):
            await _run_adapter("signal", str(cfg))  # must not raise
        judge.aclose.assert_awaited_once()

    async def test_start_keyboard_interrupt_still_stops(self, tmp_path: Path) -> None:
        """A KeyboardInterrupt during start is swallowed and stop() runs."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  host: 127.0.0.1\n  port: 8765\n"
            "adapters:\n  signal:\n    enabled: true\n",
            encoding="utf-8",
        )
        mock_adapter = AsyncMock()
        mock_adapter.start.side_effect = KeyboardInterrupt
        with patch("cordbeat.adapters.signal.SignalAdapter", return_value=mock_adapter):
            await _run_adapter("signal", str(cfg))
        mock_adapter.stop.assert_awaited_once()


class TestCliEntryPoints:
    @pytest.mark.parametrize(
        "func_name,adapter_name",
        [
            ("discord_cli", "discord"),
            ("telegram_cli", "telegram"),
            ("slack_cli", "slack"),
            ("line_cli", "line"),
            ("whatsapp_cli", "whatsapp"),
            ("signal_cli", "signal"),
        ],
    )
    def test_cli_runs_adapter(self, func_name: str, adapter_name: str) -> None:
        from cordbeat.adapters import runner

        with (
            patch.object(
                runner, "_resolve_adapter_config_path", return_value="cfg.yaml"
            ),
            patch.object(runner.asyncio, "run") as run_mock,
        ):
            getattr(runner, func_name)()

        run_mock.assert_called_once()
        # The coroutine handed to asyncio.run targets the right adapter.
        coro = run_mock.call_args[0][0]
        assert coro.cr_frame.f_locals["adapter_name"] == adapter_name
        coro.close()  # avoid "coroutine was never awaited"

    @pytest.mark.parametrize(
        "func_name",
        [
            "discord_cli",
            "telegram_cli",
            "slack_cli",
            "line_cli",
            "whatsapp_cli",
            "signal_cli",
        ],
    )
    def test_cli_swallows_keyboard_interrupt(self, func_name: str) -> None:
        from cordbeat.adapters import runner

        with (
            patch.object(
                runner, "_resolve_adapter_config_path", return_value="cfg.yaml"
            ),
            patch.object(
                runner.asyncio, "run", side_effect=KeyboardInterrupt
            ) as run_mock,
        ):
            getattr(runner, func_name)()  # must not raise
        run_mock.call_args[0][0].close()


class TestResolveAdapterConfigPath:
    def test_explicit_argv_path_wins(self) -> None:
        with patch("cordbeat.adapters.runner.sys.argv", ["prog", "/custom/cfg.yaml"]):
            assert _resolve_adapter_config_path() == "/custom/cfg.yaml"

    def test_home_config_used_when_present(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text("x: 1\n", encoding="utf-8")
        with (
            patch("cordbeat.adapters.runner.sys.argv", ["prog"]),
            patch("cordbeat.adapters.runner.cordbeat_home", return_value=tmp_path),
        ):
            assert _resolve_adapter_config_path() == str(tmp_path / "config.yaml")

    def test_cwd_config_used_as_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()  # no config.yaml here
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        (cwd / "config.yaml").write_text("x: 1\n", encoding="utf-8")
        monkeypatch.chdir(cwd)
        with (
            patch("cordbeat.adapters.runner.sys.argv", ["prog"]),
            patch("cordbeat.adapters.runner.cordbeat_home", return_value=home),
        ):
            assert _resolve_adapter_config_path() == "config.yaml"

    def test_default_home_path_when_nothing_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        empty_cwd = tmp_path / "empty"
        empty_cwd.mkdir()  # ensure no config.yaml in the working directory
        monkeypatch.chdir(empty_cwd)
        with (
            patch("cordbeat.adapters.runner.sys.argv", ["prog"]),
            patch("cordbeat.adapters.runner.cordbeat_home", return_value=home),
        ):
            assert _resolve_adapter_config_path() == str(home / "config.yaml")
