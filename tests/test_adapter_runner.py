"""Tests for adapter runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cordbeat.adapter_runner import _run_adapter


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
            "cordbeat.discord_adapter.DiscordAdapter",
            return_value=mock_adapter,
        ):
            with patch(
                "cordbeat.adapter_runner.DiscordAdapter",
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
            "cordbeat.discord_adapter.DiscordAdapter",
            return_value=mock_adapter,
        ):
            with patch(
                "cordbeat.adapter_runner.DiscordAdapter",
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
            "cordbeat.telegram_adapter.TelegramAdapter",
            return_value=mock_adapter,
        ):
            with patch(
                "cordbeat.adapter_runner.TelegramAdapter",
                create=True,
                return_value=mock_adapter,
            ):
                await _run_adapter("telegram", config_path)
                mock_adapter.start.assert_called_once()
                mock_adapter.stop.assert_called_once()

    @pytest.mark.parametrize(
        "adapter_name,module_path,class_name",
        [
            ("slack", "cordbeat.slack_adapter", "SlackAdapter"),
            ("line", "cordbeat.line_adapter", "LineAdapter"),
            ("whatsapp", "cordbeat.whatsapp_adapter", "WhatsAppAdapter"),
            ("signal", "cordbeat.signal_adapter", "SignalAdapter"),
        ],
    )
    async def test_v1_adapter_created(
        self,
        tmp_path: Path,
        adapter_name: str,
        module_path: str,
        class_name: str,
    ) -> None:
        """v1.0+ scaffold adapters are instantiated and started."""
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
