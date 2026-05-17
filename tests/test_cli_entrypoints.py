"""Tests that CLI entry points handle KeyboardInterrupt cleanly (no traceback)."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestMainCliKeyboardInterrupt:
    """cli() and cli_chat() must exit cleanly on Ctrl+C (no traceback)."""

    def test_cli_keyboard_interrupt_is_swallowed(self) -> None:
        """cordbeat (cli) exits cleanly on KeyboardInterrupt."""
        import sys

        from cordbeat.main import cli

        with (
            patch("cordbeat.main._resolve_config_path", return_value="config.yaml"),
            patch(
                "cordbeat.main.asyncio.run",
                side_effect=KeyboardInterrupt,
            ),
            patch.object(sys, "argv", ["cordbeat"]),
        ):
            # Must not raise
            cli()

    def test_cli_chat_keyboard_interrupt_is_swallowed(self) -> None:
        """cordbeat-chat exits cleanly on KeyboardInterrupt.

        cli_chat() probes for a running server then delegates to cli_main()
        which internally swallows KeyboardInterrupt.  We verify the whole
        chain exits without raising.
        """
        from unittest.mock import MagicMock

        from cordbeat.main import cli_chat

        mock_cfg = MagicMock()
        mock_cfg.gateway.host = "localhost"
        mock_cfg.gateway.port = 8765

        with (
            patch("cordbeat.main._resolve_config_path", return_value="config.yaml"),
            patch("cordbeat.main.load_config", return_value=mock_cfg),
            # Pretend server is already running so _start_server_background is skipped
            patch("cordbeat.main._probe_ws", return_value=True),
            # cli_main swallows KeyboardInterrupt internally; mock it as a no-op
            patch("cordbeat.adapters.cli.cli_main"),
        ):
            cli_chat()


class TestAdapterRunnerKeyboardInterrupt:
    """All adapter CLI entry points must swallow KeyboardInterrupt."""

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
    def test_adapter_keyboard_interrupt_is_swallowed(self, func_name: str) -> None:
        import importlib
        import sys

        runner = importlib.import_module("cordbeat.adapters.runner")
        func = getattr(runner, func_name)

        with (
            patch.object(sys, "argv", ["cordbeat-adapter", "config.yaml"]),
            patch(
                "cordbeat.adapters.runner.asyncio.run",
                side_effect=KeyboardInterrupt,
            ),
        ):
            func()  # Must not raise


class TestWizardKeyboardInterrupt:
    """cordbeat-init must swallow KeyboardInterrupt after starting main."""

    def test_cordbeat_init_keyboard_interrupt_is_swallowed(self) -> None:
        from cordbeat.tools.wizard import cordbeat_init_cli

        # asyncio is imported locally inside cordbeat_init_cli, so patch at the source
        with (
            patch("cordbeat.tools.wizard._check_required_deps"),
            patch(
                "cordbeat.tools.wizard.run_wizard",
                return_value=("/tmp/config.yaml", False),
            ),
            patch("builtins.input", return_value="y"),
            patch("asyncio.run", side_effect=KeyboardInterrupt),
        ):
            cordbeat_init_cli()  # Must not raise
