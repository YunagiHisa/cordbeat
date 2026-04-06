"""Tests for platform adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cordbeat.config import AdapterConfig


class TestDiscordAdapter:
    def test_import(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        assert DiscordAdapter is not None

    def test_init(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        config = AdapterConfig(
            core_ws_url="ws://localhost:8765",
            options={"token": "test-token"},
        )
        adapter = DiscordAdapter(config)
        assert adapter._token == "test-token"
        assert adapter._ws_url == "ws://localhost:8765"

    async def test_start_without_token_logs_error(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        config = AdapterConfig(options={})
        adapter = DiscordAdapter(config)

        with patch("cordbeat.discord_adapter.logger") as mock_logger:
            # Patch discord import to succeed
            with patch.dict("sys.modules", {"discord": MagicMock()}):
                await adapter.start()
            mock_logger.error.assert_called()

    async def test_start_without_discord_installed(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)

        with (
            patch("cordbeat.discord_adapter.logger") as mock_logger,
            patch.dict("sys.modules", {"discord": None}),
            patch("builtins.__import__", side_effect=ImportError),
        ):
            await adapter.start()
            mock_logger.error.assert_called()

    async def test_forward_to_core_no_ws(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        adapter._ws = None

        # Should not raise, just log warning
        with patch("cordbeat.discord_adapter.logger") as mock_logger:
            await adapter._forward_to_core(
                MagicMock(
                    author=MagicMock(id=123, display_name="Test"),
                    content="hello",
                    channel=MagicMock(id=456),
                    guild=None,
                )
            )
            mock_logger.warning.assert_called()


class TestTelegramAdapter:
    def test_import(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        assert TelegramAdapter is not None

    def test_init(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(
            core_ws_url="ws://localhost:8765",
            options={"token": "test-telegram-token"},
        )
        adapter = TelegramAdapter(config)
        assert adapter._token == "test-telegram-token"
        assert adapter._ws_url == "ws://localhost:8765"

    async def test_start_without_token_logs_error(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(options={})
        adapter = TelegramAdapter(config)

        with patch("cordbeat.telegram_adapter.logger") as mock_logger:
            # Patch telegram import to succeed
            telegram_mock = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "telegram": telegram_mock,
                    "telegram.ext": telegram_mock,
                },
            ):
                await adapter.start()
            mock_logger.error.assert_called()

    async def test_send_to_telegram_no_app(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._app = None

        # Should not raise
        await adapter._send_to_telegram("123", "hello")

    async def test_chat_map_stores_ids(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)

        # Simulate message storing chat_id
        adapter._chat_map["12345"] = 67890
        assert adapter._chat_map["12345"] == 67890


class TestAdapterRunner:
    async def test_unknown_adapter(self) -> None:
        from cordbeat.adapter_runner import _run_adapter

        with patch("cordbeat.adapter_runner.logger") as mock_logger:
            await _run_adapter("unknown", "config.yaml")
            mock_logger.error.assert_called()

    async def test_disabled_adapter(self) -> None:
        from cordbeat.adapter_runner import _run_adapter

        config_mock = MagicMock()
        config_mock.adapters = {
            "discord": AdapterConfig(enabled=False),
        }
        with (
            patch("cordbeat.adapter_runner.load_config", return_value=config_mock),
            patch("cordbeat.adapter_runner.logger") as mock_logger,
        ):
            await _run_adapter("discord", "config.yaml")
            mock_logger.info.assert_called()
