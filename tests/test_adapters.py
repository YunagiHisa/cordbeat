"""Tests for platform adapters."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from cordbeat.config import AdapterConfig, LogConfig


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

    async def test_forward_to_core_sends_payload(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        adapter._ws = AsyncMock()

        message = MagicMock(
            author=MagicMock(id=123, display_name="Alice"),
            content="Hello!",
            channel=MagicMock(id=456),
            guild=MagicMock(id=789),
        )
        await adapter._forward_to_core(message)

        adapter._ws.send.assert_awaited_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["type"] == "message"
        assert payload["adapter_id"] == "discord"
        assert payload["platform_user_id"] == "123"
        assert payload["content"] == "Hello!"
        # User channel should be cached
        assert adapter._user_channels["123"] == 456

    async def test_forward_to_core_ws_error(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        adapter._ws = AsyncMock()
        adapter._ws.send = AsyncMock(side_effect=RuntimeError("ws closed"))

        message = MagicMock(
            author=MagicMock(id=123, display_name="Alice"),
            content="Hello!",
            channel=MagicMock(id=456),
            guild=None,
        )
        with patch("cordbeat.discord_adapter.logger"):
            await adapter._forward_to_core(message)
        # Should not raise

    async def test_stop(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        adapter._running = True
        adapter._ws = AsyncMock()
        adapter._bot = AsyncMock()

        await adapter.stop()
        assert adapter._running is False
        adapter._ws.close.assert_awaited_once()
        adapter._bot.close.assert_awaited_once()

    async def test_send_to_discord_no_bot(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        adapter._bot = None

        # Should not raise
        await adapter._send_to_discord("123", "hello")

    async def test_send_to_discord_via_channel(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        mock_channel = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel = MagicMock(return_value=mock_channel)
        adapter._user_channels["123"] = 456

        await adapter._send_to_discord("123", "hello")
        mock_channel.send.assert_awaited_once_with("hello")

    async def test_send_to_discord_fallback_dm(self) -> None:
        from cordbeat.discord_adapter import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        mock_user = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel = MagicMock(return_value=None)
        adapter._bot.fetch_user = AsyncMock(return_value=mock_user)
        adapter._user_channels["123"] = 456

        await adapter._send_to_discord("123", "hello")
        mock_user.send.assert_awaited_once_with("hello")


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

    async def test_forward_to_core_sends_payload(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._ws = AsyncMock()

        await adapter._forward_to_core(
            "user42",
            "Hello!",
            display_name="Alice",
            chat_id=999,
        )

        adapter._ws.send.assert_awaited_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["type"] == "message"
        assert payload["adapter_id"] == "telegram"
        assert payload["platform_user_id"] == "user42"
        assert payload["content"] == "Hello!"

    async def test_forward_to_core_no_ws(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._ws = None

        with patch("cordbeat.telegram_adapter.logger") as mock_logger:
            await adapter._forward_to_core("user1", "hello")
            mock_logger.warning.assert_called()

    async def test_forward_to_core_ws_error(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._ws = AsyncMock()
        adapter._ws.send = AsyncMock(side_effect=RuntimeError("ws fail"))

        with patch("cordbeat.telegram_adapter.logger"):
            await adapter._forward_to_core("user1", "hello")
        # Should not raise

    async def test_stop(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._running = True
        adapter._ws = AsyncMock()
        adapter._app = MagicMock()
        adapter._app.updater = MagicMock()
        adapter._app.updater.stop = AsyncMock()
        adapter._app.stop = AsyncMock()
        adapter._app.shutdown = AsyncMock()

        await adapter.stop()
        assert adapter._running is False
        adapter._ws.close.assert_awaited_once()

    async def test_send_to_telegram_with_chat_map(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._app = MagicMock()
        adapter._app.bot = MagicMock()
        adapter._app.bot.send_message = AsyncMock()
        adapter._chat_map["user1"] = 12345

        await adapter._send_to_telegram("user1", "hello")
        adapter._app.bot.send_message.assert_awaited_once_with(
            chat_id=12345,
            text="hello",
        )

    async def test_send_to_telegram_dm_fallback(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._app = MagicMock()
        adapter._app.bot = MagicMock()
        adapter._app.bot.send_message = AsyncMock()
        # No entry in chat_map — should use user_id as int

        await adapter._send_to_telegram("67890", "hello")
        adapter._app.bot.send_message.assert_awaited_once_with(
            chat_id=67890,
            text="hello",
        )

    async def test_send_to_telegram_invalid_user_id(self) -> None:
        from cordbeat.telegram_adapter import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._app = MagicMock()
        adapter._app.bot = MagicMock()
        adapter._app.bot.send_message = AsyncMock()
        # No chat_map entry and non-numeric user_id

        with patch("cordbeat.telegram_adapter.logger") as mock_logger:
            await adapter._send_to_telegram("not_a_number", "hello")
            mock_logger.warning.assert_called()
        adapter._app.bot.send_message.assert_not_awaited()


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
        config_mock.log = LogConfig()
        with (
            patch("cordbeat.adapter_runner.load_config", return_value=config_mock),
            patch("cordbeat.adapter_runner.logger") as mock_logger,
        ):
            await _run_adapter("discord", "config.yaml")
            mock_logger.info.assert_called()
