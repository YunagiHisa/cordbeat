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
