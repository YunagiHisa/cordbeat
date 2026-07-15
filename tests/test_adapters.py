"""Tests for platform adapters."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from cordbeat.config import AdapterConfig, LogConfig


async def _start_telegram_and_capture(adapter: Any) -> tuple[Any, dict[str, Any]]:
    """Run ``TelegramAdapter.start`` against a mocked SDK and return the app
    plus the handler callables it registered (message / command / callback).

    The real message-handling logic lives in closures defined inside
    ``start``; the only way to exercise them is to build the app with a
    mocked ``python-telegram-bot`` and pull the callbacks back out of the
    recorded ``Handler(...)`` constructor calls.
    """
    import asyncio

    telegram_mock = MagicMock()
    app = MagicMock()
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.updater.start_polling = AsyncMock()
    app.bot.set_my_commands = AsyncMock()
    builder = telegram_mock.ApplicationBuilder.return_value
    builder.token.return_value.build.return_value = app
    adapter._connect_to_core = AsyncMock()  # don't open a real socket

    with patch.dict(
        "sys.modules",
        {"telegram": telegram_mock, "telegram.ext": telegram_mock},
    ):
        await adapter.start()
        await asyncio.sleep(0)  # let the create_task(_connect_to_core) settle

    handlers = {
        "message": telegram_mock.MessageHandler.call_args.args[1],
        "command": telegram_mock.CommandHandler.call_args.args[1],
        "callback": telegram_mock.CallbackQueryHandler.call_args.args[0],
    }
    return app, handlers


async def _start_discord_and_capture(adapter: Any) -> tuple[Any, dict[str, Any]]:
    """Run ``DiscordAdapter.start`` against a mocked discord.py and return the
    bot plus the ``@bot.event`` handlers it registered (on_ready/on_message/...).

    discord.py registers gateway events via ``@bot.event``; with a mocked
    client the decorated function is passed straight to ``bot.event``, so we
    recover the closures from its recorded calls.
    """
    discord_mock = MagicMock()
    bot = MagicMock()
    bot.start = AsyncMock()
    discord_mock.Client.return_value = bot
    tree = discord_mock.app_commands.CommandTree.return_value
    tree.sync = AsyncMock(return_value=[1, 2])
    adapter._connect_to_core = AsyncMock()

    with patch.dict("sys.modules", {"discord": discord_mock}):
        await adapter.start()

    events = {c.args[0].__name__: c.args[0] for c in bot.event.call_args_list}
    return bot, events


class TestDiscordAdapter:
    def test_import(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        assert DiscordAdapter is not None

    def test_core_slash_command_names_cover_core_commands(self) -> None:
        from cordbeat.adapters.discord import _CORE_SLASH_COMMAND_NAMES

        assert set(_CORE_SLASH_COMMAND_NAMES) == {
            "approve",
            "reject",
            "proposals",
            "link",
            "link-confirm",
            "unlink",
            "name",
            "quiet",
            "prefer",
            "draw",
        }

    def test_init(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(
            core_ws_url="ws://localhost:8765",
            options={"token": "test-token"},
        )
        adapter = DiscordAdapter(config)
        assert adapter._token == "test-token"
        assert adapter._ws_url == "ws://localhost:8765"

    async def test_start_without_token_logs_error(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={})
        adapter = DiscordAdapter(config)

        with patch("cordbeat.adapters.discord.logger") as mock_logger:
            # Patch discord import to succeed
            with patch.dict("sys.modules", {"discord": MagicMock()}):
                await adapter.start()
            mock_logger.error.assert_called()

    async def test_start_without_discord_installed(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)

        with (
            patch("cordbeat.adapters.discord.logger") as mock_logger,
            patch.dict("sys.modules", {"discord": None}),
            patch("builtins.__import__", side_effect=ImportError),
        ):
            await adapter.start()
            mock_logger.error.assert_called()

    async def test_forward_to_core_no_ws(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        adapter._ws = None

        # Should not raise; message is buffered in the outbox for resend
        await adapter._forward_to_core(
            MagicMock(
                author=MagicMock(id=123, display_name="Test"),
                content="hello",
                channel=MagicMock(id=456),
                guild=None,
            )
        )
        assert len(adapter._outbox) == 1
        payload = json.loads(adapter._outbox[0])
        assert payload["content"] == "hello"

    async def test_forward_to_core_sends_payload(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        adapter._ws = AsyncMock()

        message = MagicMock(
            author=MagicMock(id=123, display_name="Alice"),
            content="Hello!",
            channel=MagicMock(id=456),
            guild=MagicMock(id=789),
            created_at=datetime(2026, 7, 14, 14, 55, tzinfo=UTC),
        )
        await adapter._forward_to_core(message)

        adapter._ws.send.assert_awaited_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["type"] == "message"
        assert payload["adapter_id"] == "discord"
        assert payload["platform_user_id"] == "123"
        assert payload["content"] == "Hello!"
        assert payload["timestamp"] == "2026-07-14T14:55:00+00:00"
        assert payload["metadata"]["channel_is_public"] is True
        # User channel should be cached
        assert adapter._user_channels["123"] == 456

    async def test_dm_includes_mutual_guild_ids_for_private_read_context(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        adapter = DiscordAdapter(AdapterConfig(options={"token": "test"}))
        adapter._ws = AsyncMock()
        guild = MagicMock(id=789)
        guild.get_member.return_value = MagicMock()
        adapter._bot = MagicMock(guilds=[guild])
        message = MagicMock(
            author=MagicMock(id=123, display_name="Alice"),
            content="Hello privately",
            channel=MagicMock(id=456),
            guild=None,
            created_at=datetime(2026, 7, 14, 14, 55, tzinfo=UTC),
        )

        await adapter._forward_to_core(message)

        payload = json.loads(adapter._ws.send.call_args.args[0])
        assert payload["metadata"]["is_dm"] is True
        assert payload["metadata"]["mutual_guild_ids"] == ["789"]
        assert payload["metadata"]["channel_is_public"] is False

    async def test_forward_to_core_includes_discord_reply_context(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        adapter._ws = AsyncMock()

        referenced = MagicMock(
            id=777,
            content="Earlier message",
            author=MagicMock(display_name="Bob", name="bob", bot=False),
            attachments=[],
        )
        message = MagicMock(
            author=MagicMock(id=123, display_name="Alice"),
            content="What about this?",
            channel=MagicMock(id=456),
            guild=MagicMock(id=789),
            attachments=[],
            reference=MagicMock(resolved=referenced),
        )

        await adapter._forward_to_core(message)

        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["metadata"]["reply_context"] == {
            "author": "Bob",
            "content": "Earlier message",
            "message_id": "777",
            "is_bot": False,
            "image_count": 0,
        }

    async def test_forward_to_core_fetches_unresolved_discord_reply(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        adapter._ws = AsyncMock()
        referenced = MagicMock(
            id=778,
            content="Fetched earlier message",
            author=MagicMock(display_name="Bob", name="bob", bot=True),
            attachments=[],
        )
        channel = MagicMock(id=456)
        channel.fetch_message = AsyncMock(return_value=referenced)
        message = MagicMock(
            author=MagicMock(id=123, display_name="Alice"),
            content="Explain this",
            channel=channel,
            guild=MagicMock(id=789),
            attachments=[],
            reference=MagicMock(resolved=None, message_id=778),
        )

        await adapter._forward_to_core(message)

        channel.fetch_message.assert_awaited_once_with(778)
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["metadata"]["reply_context"]["content"] == (
            "Fetched earlier message"
        )
        assert payload["metadata"]["reply_context"]["is_bot"] is True

    async def test_forward_to_core_ws_error(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

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
        with patch("cordbeat.adapters.discord.logger"):
            await adapter._forward_to_core(message)
        # Should not raise

    async def test_stop(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

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
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        adapter._bot = None

        # Should not raise
        await adapter._send_to_discord("123", "hello")

    async def test_send_to_discord_via_channel(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        mock_channel = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel = MagicMock(return_value=mock_channel)
        adapter._user_channels["123"] = 456

        await adapter._send_to_discord("123", "hello")
        mock_channel.send.assert_awaited_once_with("hello", files=[])

    async def test_send_to_discord_fallback_dm(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        mock_user = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel = MagicMock(return_value=None)
        adapter._bot.fetch_channel = AsyncMock(return_value=None)
        adapter._bot.fetch_user = AsyncMock(return_value=mock_user)
        adapter._user_channels["123"] = 456

        await adapter._send_to_discord("123", "hello")
        mock_user.send.assert_awaited_once_with("hello", files=[])

    async def test_send_to_discord_channel_failure_falls_back_to_dm(self) -> None:
        """Channel send failures should still try DM when fallback is allowed."""
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        mock_channel = MagicMock()
        mock_channel.send = AsyncMock(side_effect=RuntimeError("missing permissions"))
        mock_user = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel = MagicMock(return_value=mock_channel)
        adapter._bot.fetch_user = AsyncMock(return_value=mock_user)
        adapter._user_channels["123"] = 456

        await adapter._send_to_discord("123", "hello")

        mock_channel.send.assert_awaited_once_with("hello", files=[])
        mock_user.send.assert_awaited_once_with("hello", files=[])
        assert "123" not in adapter._user_channels

    async def test_send_to_discord_metadata_channel_id_overrides_cache(self) -> None:
        """Core-supplied channel_id in metadata wins over the in-memory cache."""
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        pinned_channel = AsyncMock()
        cached_channel = AsyncMock()

        def _get(cid: int) -> Any:
            return pinned_channel if cid == 999 else cached_channel

        adapter._bot = MagicMock()
        adapter._bot.get_channel = MagicMock(side_effect=_get)
        adapter._user_channels["123"] = 456  # stale/wrong cache

        await adapter._send_to_discord(
            "123", "hi", metadata={"channel_id": "999", "is_dm": False}
        )
        pinned_channel.send.assert_awaited_once_with("hi", files=[])
        cached_channel.send.assert_not_called()

    async def test_send_to_discord_no_dm_fallback_when_disallowed(self) -> None:
        """allow_dm_fallback=False prevents DM as the last-resort path."""
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        mock_user = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel = MagicMock(return_value=None)
        adapter._bot.fetch_channel = AsyncMock(return_value=None)
        adapter._bot.fetch_user = AsyncMock(return_value=mock_user)
        # No cache, no metadata.channel_id → previously would DM.

        await adapter._send_to_discord(
            "123", "hi", metadata={"allow_dm_fallback": False}
        )
        mock_user.send.assert_not_awaited()

    async def test_send_to_discord_channel_failure_respects_no_dm_fallback(
        self,
    ) -> None:
        """allow_dm_fallback=False prevents DM after channel send failures too."""
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        mock_channel = MagicMock()
        mock_channel.send = AsyncMock(side_effect=RuntimeError("missing permissions"))
        mock_user = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel = MagicMock(return_value=mock_channel)
        adapter._bot.fetch_user = AsyncMock(return_value=mock_user)
        adapter._user_channels["123"] = 456

        await adapter._send_to_discord(
            "123", "hi", metadata={"allow_dm_fallback": False}
        )

        mock_channel.send.assert_awaited_once_with("hi", files=[])
        mock_user.send.assert_not_awaited()

    async def test_send_to_discord_channel_message_does_not_dm_fallback(
        self,
    ) -> None:
        """Channel-originated replies must not fallback to DM implicitly."""
        from cordbeat.adapters.discord import DiscordAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = DiscordAdapter(config)
        mock_channel = MagicMock()
        mock_channel.send = AsyncMock(side_effect=RuntimeError("missing permissions"))
        mock_user = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel = MagicMock(return_value=mock_channel)
        adapter._bot.fetch_user = AsyncMock(return_value=mock_user)

        await adapter._send_to_discord(
            "123", "hi", metadata={"channel_id": "456", "is_dm": False}
        )

        mock_channel.send.assert_awaited_once_with("hi", files=[])
        mock_user.send.assert_not_awaited()


class TestDiscordAdapterInternals:
    """Direct unit tests for DiscordAdapter helper methods and handlers."""

    def _adapter(self, **options: Any) -> Any:
        from cordbeat.adapters.discord import DiscordAdapter

        opts = {"token": "t", **options}
        return DiscordAdapter(AdapterConfig(options=opts))

    # ── init option parsing ──────────────────────────────────────────
    def test_invalid_vc_activation_mode_defaults_to_hybrid(self) -> None:
        with patch("cordbeat.adapters.discord.logger") as log:
            adapter = self._adapter(vc_activation_mode="bogus")
        assert adapter._vc_activation_mode == "hybrid"
        log.warning.assert_called()

    def test_string_wake_word_is_wrapped(self) -> None:
        adapter = self._adapter(vc_activation_phrases="hey bot")
        assert "hey bot" in adapter._vc_wake_words

    def test_invalid_wake_words_use_defaults(self) -> None:
        with patch("cordbeat.adapters.discord.logger"):
            adapter = self._adapter(vc_wake_words=123)
        assert adapter._vc_wake_words  # falls back to defaults, non-empty

    def test_init_creates_stt_and_tts_backends(self) -> None:
        from cordbeat.adapters.discord import DiscordAdapter
        from cordbeat.config import STTConfig, TTSConfig

        with (
            patch("cordbeat.ai.stt.create_stt_backend", return_value="STT") as make_stt,
            patch(
                "cordbeat.ai.tts.create_tts_with_rvc", return_value="TTS"
            ) as make_tts,
        ):
            adapter = DiscordAdapter(
                AdapterConfig(options={"token": "t"}),
                stt_config=STTConfig(enabled=True),
                tts_config=TTSConfig(enabled=True),
            )
        make_stt.assert_called_once()
        make_tts.assert_called_once()
        assert adapter._stt == "STT"
        assert adapter._tts == "TTS"

    # ── small helpers ────────────────────────────────────────────────
    def test_remember_user_channel_evicts_oldest(self) -> None:
        from cordbeat.adapters import discord as dmod

        adapter = self._adapter()
        with patch.object(dmod, "_USER_CHANNEL_CACHE_MAX", 2):
            adapter._remember_user_channel("u1", 1)
            adapter._remember_user_channel("u2", 2)
            adapter._remember_user_channel("u3", 3)  # evicts u1
        assert "u1" not in adapter._user_channels
        assert set(adapter._user_channels) == {"u2", "u3"}

    def test_cache_pending_skill_confirm_ignores_empty_id(self) -> None:
        adapter = self._adapter()
        adapter._cache_pending_skill_confirm(
            platform_user_id="u", proposal_id="", skill_name="s", skill_params={}
        )
        assert adapter._pending_skill_confirms == {}

    def test_cache_pending_skill_confirm_evicts_oldest(self) -> None:
        from cordbeat.adapters import discord as dmod

        adapter = self._adapter()
        with patch.object(dmod, "_PENDING_SKILL_CONFIRM_MAX", 2):
            for i in range(3):
                adapter._cache_pending_skill_confirm(
                    platform_user_id="u",
                    proposal_id=f"p{i}",
                    skill_name="s",
                    skill_params={},
                )
        assert "p0" not in adapter._pending_skill_confirms

    def test_pending_proposal_choices_filters_and_caps(self) -> None:
        adapter = self._adapter()
        for i in range(30):
            adapter._cache_pending_skill_confirm(
                platform_user_id="u1",
                proposal_id=f"prop{i}",
                skill_name="draw",
                skill_params={},
            )
        adapter._cache_pending_skill_confirm(
            platform_user_id="other",
            proposal_id="x",
            skill_name="draw",
            skill_params={},
        )
        choices = adapter._pending_proposal_choices("u1")
        assert len(choices) == 25  # capped
        assert all(value.startswith("prop") for _, value in choices)
        # Filtering by current substring narrows results.
        narrowed = adapter._pending_proposal_choices("u1", current="prop1")
        assert narrowed and all("prop1" in v for _, v in narrowed)

    def test_cache_pending_proposals_from_text(self) -> None:
        adapter = self._adapter()
        proposal_id = "076527a0-845b-4f5a-ac6d-12472b281eb2"

        adapter._cache_pending_proposals_from_text(
            "u1",
            "\n".join(
                [
                    "📋 Pending proposals:",
                    "  • Change traits",
                    f"    approve: /approve {proposal_id}",
                    f"    reject:  /reject {proposal_id}",
                ]
            ),
        )

        choices = adapter._pending_proposal_choices("u1", current="0765")
        assert choices == [(f"{proposal_id[:8]}… proposal", proposal_id)]

    def test_mark_proposal_action_sent_removes_entry(self) -> None:
        adapter = self._adapter()
        adapter._pending_skill_confirms["abc"] = {"skill_name": "s"}
        adapter._mark_proposal_action_sent("/approve abc")
        assert "abc" not in adapter._pending_skill_confirms

    def test_core_command_payload_shape(self) -> None:
        adapter = self._adapter()
        payload = json.loads(adapter._core_command_payload("/proposals", "u9"))
        assert payload["adapter_id"] == "discord"
        assert payload["content"] == "/proposals"
        assert payload["platform_user_id"] == "u9"

    # ── slash command forwarding ─────────────────────────────────────
    async def test_forward_core_command_interaction_no_ws(self) -> None:
        adapter = self._adapter()
        adapter._ws = None
        interaction = MagicMock()
        interaction.response.send_message = AsyncMock()
        await adapter._forward_core_command_interaction(interaction, "/proposals")
        interaction.response.send_message.assert_awaited_once()
        assert interaction.response.send_message.call_args.kwargs["ephemeral"] is True

    async def test_forward_core_command_interaction_sends(self) -> None:
        adapter = self._adapter()
        adapter._ws = AsyncMock()
        interaction = MagicMock()
        interaction.user.id = 55
        interaction.channel_id = 99
        interaction.response.defer = AsyncMock()
        interaction.followup.send = AsyncMock()
        await adapter._forward_core_command_interaction(interaction, "/link")
        interaction.response.defer.assert_awaited_once_with(
            ephemeral=True,
            thinking=True,
        )
        adapter._ws.send.assert_awaited_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["content"] == "/link"
        assert adapter._user_channels["55"] == 99
        interaction.followup.send.assert_awaited_once_with(
            "Command sent to CordBeat.",
            ephemeral=True,
        )

    async def test_forward_core_command_interaction_acknowledges_send_failure(
        self,
    ) -> None:
        adapter = self._adapter()
        adapter._ws = AsyncMock()
        adapter._ws.send = AsyncMock(side_effect=RuntimeError("closed"))
        interaction = MagicMock()
        interaction.user.id = 55
        interaction.channel_id = 99
        interaction.response.defer = AsyncMock()
        interaction.followup.send = AsyncMock()

        with patch("cordbeat.adapters.discord.logger"):
            await adapter._forward_core_command_interaction(interaction, "/link")

        interaction.response.defer.assert_awaited_once_with(
            ephemeral=True,
            thinking=True,
        )
        interaction.followup.send.assert_awaited_once()
        assert "Failed to send command" in interaction.followup.send.call_args.args[0]

    # ── dispatch routing ─────────────────────────────────────────────
    async def test_dispatch_core_message_plain_text(self) -> None:
        adapter = self._adapter()
        adapter._send_to_discord = AsyncMock()  # type: ignore[assignment]
        await adapter._dispatch_core_message("u1", "hello", [])
        adapter._send_to_discord.assert_awaited_once_with(
            "u1", "hello", [], metadata=None
        )

    async def test_dispatch_core_message_vc_invalid_guild_id(self) -> None:
        adapter = self._adapter()
        with patch("cordbeat.adapters.discord.logger") as log:
            await adapter._dispatch_core_message(
                "u1", "hi", [], metadata={"via_vc": True, "guild_id": "nope"}
            )
        log.warning.assert_called()

    async def test_dispatch_core_message_vc_stale_session_dropped(self) -> None:
        adapter = self._adapter()
        adapter._vc_session_ids[7] = "current"
        with patch("cordbeat.adapters.discord.logger") as log:
            await adapter._dispatch_core_message(
                "u1",
                "hi",
                [],
                metadata={"via_vc": True, "guild_id": "7", "vc_session_id": "stale"},
            )
        log.info.assert_called()

    # ── _send_to_discord ─────────────────────────────────────────────
    async def test_send_to_discord_splits_long_content(self) -> None:
        adapter = self._adapter()
        channel = MagicMock()
        channel.send = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel.return_value = channel
        adapter._user_channels["u1"] = 123
        long_text = "x" * 4500  # exceeds the 2000-char Discord limit

        await adapter._send_to_discord("u1", long_text, [])

        assert channel.send.await_count >= 3  # chunked into multiple messages

    async def test_send_to_discord_handles_bad_image_b64(self) -> None:
        adapter = self._adapter()
        channel = MagicMock()
        channel.send = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel.return_value = channel
        adapter._user_channels["u1"] = 123
        with patch("cordbeat.adapters.discord.logger") as log:
            await adapter._send_to_discord("u1", "caption", ["!!not-base64!!"])
        log.exception.assert_called()  # decode failure logged, still sends text

    # ── _forward_to_core filter ──────────────────────────────────────
    async def test_forward_to_core_respects_filter_rejection(self) -> None:
        adapter = self._adapter()
        adapter._send_to_core = AsyncMock()  # type: ignore[assignment]
        adapter._filter.should_respond_async = AsyncMock(  # type: ignore[attr-defined]
            return_value=False
        )
        message = MagicMock()
        message.author.id = 1
        message.author.bot = False
        message.channel.id = 2
        message.guild = None
        message.content = "ignored"
        await adapter._forward_to_core(message)
        adapter._send_to_core.assert_not_awaited()


class TestDiscordAdapterStartHandlers:
    """Drive DiscordAdapter.start() with a mocked discord.py SDK."""

    def _adapter(self, **options: Any) -> Any:
        from cordbeat.adapters.discord import DiscordAdapter

        return DiscordAdapter(AdapterConfig(options={"token": "t", **options}))

    async def test_start_registers_events_and_starts_bot(self) -> None:
        adapter = self._adapter()
        bot, events = await _start_discord_and_capture(adapter)
        assert adapter._running is True
        bot.start.assert_awaited_once_with("t")
        assert {"on_ready", "on_message", "on_voice_state_update"} <= set(events)

    async def test_start_without_token_logs_error(self) -> None:
        adapter = self._adapter()
        adapter._token = ""
        with patch("cordbeat.adapters.discord.logger") as log:
            bot, events = await _start_discord_and_capture(adapter)
        log.error.assert_called()

    async def test_on_message_forwards_user_message(self) -> None:
        adapter = self._adapter()
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]
        bot, events = await _start_discord_and_capture(adapter)

        message = MagicMock()
        message.author = MagicMock()
        message.author.bot = False
        bot.user = MagicMock()  # author != bot.user
        await events["on_message"](message)
        adapter._forward_to_core.assert_awaited_once_with(message)

    async def test_on_message_ignores_bot_authors(self) -> None:
        adapter = self._adapter()
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]
        bot, events = await _start_discord_and_capture(adapter)

        bot_msg = MagicMock()
        bot_msg.author.bot = True
        bot_msg.author = MagicMock(bot=True)
        await events["on_message"](bot_msg)
        adapter._forward_to_core.assert_not_awaited()

    async def test_on_ready_syncs_commands_globally(self) -> None:
        adapter = self._adapter()
        bot, events = await _start_discord_and_capture(adapter)
        # on_ready triggers a global tree sync and starts the core connection.
        await events["on_ready"]()
        adapter._connect_to_core.assert_called()

    async def test_on_voice_state_update_cleans_up_on_disconnect(self) -> None:
        adapter = self._adapter()
        adapter._cleanup_vc_state = AsyncMock()  # type: ignore[assignment]
        bot, events = await _start_discord_and_capture(adapter)
        bot.user = MagicMock()
        bot.user.id = 42

        member = MagicMock()
        member.id = 42  # the bot itself
        member.guild.id = 7
        before = MagicMock()
        before.channel = MagicMock()  # was in a channel
        after = MagicMock()
        after.channel = None  # now disconnected
        with patch("cordbeat.adapters.discord.logger"):
            await events["on_voice_state_update"](member, before, after)
        adapter._cleanup_vc_state.assert_awaited_once_with(7)


def _discord_ui_mock() -> MagicMock:
    """A discord mock whose ``ui.View`` is subclassable and ``ui.button`` is a
    pass-through decorator, so the inner SkillConfirmView class body executes."""
    d = MagicMock()

    class _View:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.timeout = k.get("timeout")

        def stop(self) -> None:
            pass

    class _Embed:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.title = k.get("title", "")
            self.description = k.get("description", "")
            self.color = k.get("color")
            self.fields: list[tuple[str, str, bool]] = []
            self.footer = MagicMock()
            self.footer.text = ""

        def add_field(self, *, name: str, value: str, inline: bool) -> None:
            self.fields.append((name, value, inline))

        def set_footer(self, *, text: str) -> None:
            self.footer.text = text

    d.ui.View = _View
    d.ui.button = lambda **kw: lambda fn: fn
    d.Embed = _Embed
    return d


class TestDiscordSkillConfirm:
    def _adapter(self) -> Any:
        from cordbeat.adapters.discord import DiscordAdapter

        return DiscordAdapter(AdapterConfig(options={"token": "t"}))

    async def test_sends_embed_and_buttons_drive_core_commands(self) -> None:
        adapter = self._adapter()
        channel = MagicMock()
        channel.send = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel.return_value = channel
        adapter._user_channels["u1"] = 123
        adapter._ws = AsyncMock()
        data = {
            "content": "fallback",
            "metadata": {
                "proposal_id": "p1",
                "skill_name": "draw",
                "skill_params": {"a": "b"},
            },
        }
        with patch.dict("sys.modules", {"discord": _discord_ui_mock()}):
            await adapter._dispatch_skill_confirm("u1", data)
            channel.send.assert_awaited_once()
            view = channel.send.call_args.kwargs["view"]
            embed = channel.send.call_args.kwargs["embed"]
            assert view.timeout is None

            inter = MagicMock()
            inter.user.id = "u1"
            inter.message.embeds = [embed]
            inter.response.defer = AsyncMock()
            inter.edit_original_response = AsyncMock()
            await view.allow_once(inter, MagicMock())
            await view.deny(inter, MagicMock())

        sent = [
            json.loads(c.args[0])["content"] for c in adapter._ws.send.call_args_list
        ]
        assert "/approve p1" in sent
        assert "/reject p1" in sent
        payloads = [json.loads(c.args[0]) for c in adapter._ws.send.call_args_list]
        assert {p["platform_user_id"] for p in payloads} == {"u1"}
        assert inter.response.defer.await_count == 2
        assert inter.edit_original_response.await_count == 2

    async def test_skill_confirm_rejects_non_owner_without_core_send(self) -> None:
        adapter = self._adapter()
        channel = MagicMock()
        channel.send = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel.return_value = channel
        adapter._user_channels["owner"] = 123
        adapter._ws = AsyncMock()
        data = {
            "content": "fallback",
            "metadata": {"proposal_id": "p-owner", "skill_name": "draw"},
        }
        with patch.dict("sys.modules", {"discord": _discord_ui_mock()}):
            await adapter._dispatch_skill_confirm("owner", data)
            view = channel.send.call_args.kwargs["view"]
            embed = channel.send.call_args.kwargs["embed"]

            inter = MagicMock()
            inter.user.id = "intruder"
            inter.message.embeds = [embed]
            inter.response.defer = AsyncMock()
            inter.followup.send = AsyncMock()
            inter.edit_original_response = AsyncMock()
            await view.allow_once(inter, MagicMock())

        adapter._ws.send.assert_not_awaited()
        inter.followup.send.assert_awaited_once_with(
            "This approval belongs to another user.",
            ephemeral=True,
        )
        inter.edit_original_response.assert_not_awaited()

    async def test_skill_confirm_cache_miss_uses_neutral_sent_copy(self) -> None:
        adapter = self._adapter()
        adapter._ws = AsyncMock()
        discord = _discord_ui_mock()
        view = adapter._make_skill_confirm_view(discord)
        embed = discord.Embed(description="Approve `draw`?")
        embed.set_footer(text="Proposal ID: p-miss")
        inter = MagicMock()
        inter.user.id = "u1"
        inter.message.embeds = [embed]
        inter.response.defer = AsyncMock()
        inter.edit_original_response = AsyncMock()

        await view.allow_once(inter, MagicMock())

        adapter._ws.send.assert_awaited_once()
        content = inter.edit_original_response.call_args.kwargs["content"]
        assert "Sent approve request" in content
        assert "see the bot's reply" in content

    async def test_button_acknowledges_before_core_send_failure(self) -> None:
        adapter = self._adapter()
        channel = MagicMock()
        channel.send = AsyncMock()
        adapter._bot = MagicMock()
        adapter._bot.get_channel.return_value = channel
        adapter._user_channels["u1"] = 123
        adapter._ws = AsyncMock()
        adapter._ws.send = AsyncMock(side_effect=RuntimeError("closed"))
        data = {
            "content": "fallback",
            "metadata": {
                "proposal_id": "p1",
                "skill_name": "draw",
                "skill_params": {"a": "b"},
            },
        }

        with (
            patch.dict("sys.modules", {"discord": _discord_ui_mock()}),
            patch("cordbeat.adapters.discord.logger"),
        ):
            await adapter._dispatch_skill_confirm("u1", data)
            view = channel.send.call_args.kwargs["view"]
            embed = channel.send.call_args.kwargs["embed"]

            inter = MagicMock()
            inter.user.id = "u1"
            inter.message.embeds = [embed]
            inter.response.defer = AsyncMock()
            inter.edit_original_response = AsyncMock()
            await view.allow_once(inter, MagicMock())

        inter.response.defer.assert_awaited_once()
        inter.edit_original_response.assert_awaited_once()
        content = inter.edit_original_response.call_args.kwargs["content"]
        assert "Failed to send approval" in content

    async def test_persistent_view_handles_old_button_from_embed_footer(self) -> None:
        adapter = self._adapter()
        adapter._ws = AsyncMock()
        discord = _discord_ui_mock()
        view = adapter._make_skill_confirm_view(discord)

        embed = discord.Embed(
            title="Skill Execution Required",
            description="**`search`** wants to run",
        )
        embed.set_footer(text="Proposal ID: old-prop")
        inter = MagicMock()
        inter.user.id = 77
        inter.message.embeds = [embed]
        inter.response.defer = AsyncMock()
        inter.edit_original_response = AsyncMock()

        await view.allow_once(inter, MagicMock())

        payload = json.loads(adapter._ws.send.call_args.args[0])
        assert payload["content"] == "/approve old-prop"
        assert payload["platform_user_id"] == "77"

    async def test_falls_back_to_text_when_no_channel(self) -> None:
        adapter = self._adapter()
        adapter._bot = MagicMock()
        adapter._dispatch_core_message = AsyncMock()  # type: ignore[assignment]
        data = {"content": "hi", "metadata": {"proposal_id": "p", "skill_name": "s"}}
        with patch("cordbeat.adapters.discord.logger"):
            await adapter._dispatch_skill_confirm("unknown-user", data)
        adapter._dispatch_core_message.assert_awaited_once()

    async def test_falls_back_when_send_fails(self) -> None:
        adapter = self._adapter()
        channel = MagicMock()
        channel.send = AsyncMock(side_effect=RuntimeError("boom"))
        adapter._bot = MagicMock()
        adapter._bot.get_channel.return_value = channel
        adapter._user_channels["u1"] = 123
        adapter._dispatch_core_message = AsyncMock()  # type: ignore[assignment]
        data = {"content": "hi", "metadata": {"proposal_id": "p", "skill_name": "s"}}
        with (
            patch.dict("sys.modules", {"discord": _discord_ui_mock()}),
            patch("cordbeat.adapters.discord.logger"),
        ):
            await adapter._dispatch_skill_confirm("u1", data)
        adapter._dispatch_core_message.assert_awaited_once()


class TestDiscordForwardToCore:
    def _adapter(self) -> Any:
        from cordbeat.adapters.discord import DiscordAdapter

        adapter = DiscordAdapter(AdapterConfig(options={"token": "t"}))
        adapter._bot = MagicMock()
        adapter._bot.user = MagicMock()
        adapter._bot.user.mentioned_in.return_value = False
        adapter._send_to_core = AsyncMock()  # type: ignore[assignment]
        adapter._filter.should_respond_async = AsyncMock(  # type: ignore[attr-defined]
            return_value=True
        )
        return adapter

    def _dm_message(self) -> MagicMock:
        message = MagicMock()
        message.author.id = 1
        message.author.bot = False
        message.author.display_name = "Alice"
        message.channel.id = 2
        message.guild = None
        message.content = "hi"
        message.attachments = []
        message.reference = None
        return message

    async def test_downloads_image_attachment(self) -> None:
        adapter = self._adapter()
        adapter._stt = None
        att = MagicMock()
        att.content_type = "image/png"
        att.size = 1000
        att.url = "https://example.com/x.png"
        message = self._dm_message()
        message.attachments = [att]

        resp = MagicMock()
        resp.content = b"imgbytes"
        resp.raise_for_status = MagicMock()
        client = MagicMock()
        client.get = AsyncMock(return_value=resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("cordbeat.adapters.discord.httpx.AsyncClient", return_value=client):
            await adapter._forward_to_core(message)

        payload = json.loads(adapter._send_to_core.call_args[0][0])
        assert len(payload["images"]) == 1

    async def test_skips_oversized_image_attachment(self) -> None:
        adapter = self._adapter()
        adapter._stt = None
        att = MagicMock()
        att.content_type = "image/png"
        att.size = 999_999_999  # over the limit
        att.url = "https://example.com/big.png"
        message = self._dm_message()
        message.attachments = [att]

        with patch("cordbeat.adapters.discord.logger") as log:
            await adapter._forward_to_core(message)
        log.warning.assert_called()
        payload = json.loads(adapter._send_to_core.call_args[0][0])
        assert payload["images"] == []

    async def test_transcribes_audio_attachment(self) -> None:
        adapter = self._adapter()
        adapter._stt = MagicMock()
        adapter._stt.transcribe = AsyncMock(return_value="spoken words")
        att = MagicMock()
        att.content_type = "audio/ogg"
        att.size = 1000
        att.url = "https://example.com/a.ogg"
        message = self._dm_message()
        message.content = ""
        message.attachments = [att]

        resp = MagicMock()
        resp.content = b"audio"
        resp.raise_for_status = MagicMock()
        client = MagicMock()
        client.get = AsyncMock(return_value=resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("cordbeat.adapters.discord.httpx.AsyncClient", return_value=client):
            await adapter._forward_to_core(message)

        payload = json.loads(adapter._send_to_core.call_args[0][0])
        assert payload["content"] == "spoken words"
        assert payload["is_voice"] is True


class TestTelegramAdapter:
    def test_import(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        assert TelegramAdapter is not None

    def test_core_bot_commands_cover_core_commands(self) -> None:
        from cordbeat.adapters.telegram import _CORE_BOT_COMMANDS

        command_names = {name for name, _ in _CORE_BOT_COMMANDS}
        assert command_names == {
            "approve",
            "reject",
            "proposals",
            "link",
            "link_confirm",
            "unlink",
            "name",
            "quiet",
            "prefer",
            "draw",
        }
        assert all(re.fullmatch(r"[a-z0-9_]{1,32}", name) for name in command_names)

    def test_normalize_telegram_command_strips_bot_suffix(self) -> None:
        from cordbeat.adapters.telegram import _normalize_telegram_command_text

        assert (
            _normalize_telegram_command_text("/approve@CordBeatBot abc-123")
            == "/approve abc-123"
        )
        assert _normalize_telegram_command_text("/proposals") == "/proposals"
        assert (
            _normalize_telegram_command_text("/link_confirm@CordBeatBot abc-123")
            == "/link-confirm abc-123"
        )

    def test_init(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        config = AdapterConfig(
            core_ws_url="ws://localhost:8765",
            options={"token": "test-telegram-token"},
        )
        adapter = TelegramAdapter(config)
        assert adapter._token == "test-telegram-token"
        assert adapter._ws_url == "ws://localhost:8765"

    async def test_start_without_token_logs_error(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        config = AdapterConfig(options={})
        adapter = TelegramAdapter(config)

        with patch("cordbeat.adapters.telegram.logger") as mock_logger:
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
        from cordbeat.adapters.telegram import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._app = None

        # Should not raise
        await adapter._send_to_telegram("123", "hello")

    async def test_forward_to_core_sends_payload(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._ws = AsyncMock()

        await adapter._forward_to_core(
            "user42",
            "Hello!",
            display_name="Alice",
            chat_id=999,
            sent_at=datetime(2026, 7, 14, 14, 56, tzinfo=UTC),
        )

        adapter._ws.send.assert_awaited_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["type"] == "message"
        assert payload["adapter_id"] == "telegram"
        assert payload["platform_user_id"] == "user42"
        assert payload["content"] == "Hello!"
        assert payload["timestamp"] == "2026-07-14T14:56:00+00:00"

    async def test_forward_to_core_includes_telegram_reply_context(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._ws = AsyncMock()
        reply_context = {
            "author": "Bob",
            "content": "Earlier message",
            "message_id": "42",
            "is_bot": False,
            "image_count": 1,
        }

        await adapter._forward_to_core(
            "user42",
            "What about this?",
            display_name="Alice",
            chat_id=999,
            images=["image"],
            reply_context=reply_context,
        )

        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["metadata"]["reply_context"] == reply_context
        assert payload["images"] == ["image"]

    async def test_forward_to_core_no_ws(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._ws = None

        # Should not raise; message is buffered in the outbox for resend
        await adapter._forward_to_core("user1", "hello")
        assert len(adapter._outbox) == 1
        payload = json.loads(adapter._outbox[0])
        assert payload["content"] == "hello"

    async def test_forward_to_core_ws_error(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._ws = AsyncMock()
        adapter._ws.send = AsyncMock(side_effect=RuntimeError("ws fail"))

        with patch("cordbeat.adapters.telegram.logger"):
            await adapter._forward_to_core("user1", "hello")
        # Should not raise

    async def test_stop(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

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
        from cordbeat.adapters.telegram import TelegramAdapter

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

    async def test_send_to_telegram_splits_long_content(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "test"}))
        adapter._app = MagicMock()
        adapter._app.bot.send_message = AsyncMock()
        adapter._chat_map["user1"] = 12345
        long_text = "x" * 5000

        await adapter._send_to_telegram("user1", long_text)

        assert adapter._app.bot.send_message.await_count == 2
        sent = [
            call.kwargs["text"]
            for call in adapter._app.bot.send_message.await_args_list
        ]
        assert "".join(sent) == long_text
        assert all(len(chunk) <= 4096 for chunk in sent)

    async def test_send_to_telegram_dm_fallback(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

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
        from cordbeat.adapters.telegram import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._app = MagicMock()
        adapter._app.bot = MagicMock()
        adapter._app.bot.send_message = AsyncMock()
        # No chat_map entry and non-numeric user_id

        with patch("cordbeat.adapters.telegram.logger") as mock_logger:
            await adapter._send_to_telegram("not_a_number", "hello")
            mock_logger.warning.assert_called()
        adapter._app.bot.send_message.assert_not_awaited()

    async def test_dispatch_skill_confirm_sends_inline_keyboard(self) -> None:
        """Telegram skill confirm should send an InlineKeyboardMarkup message."""
        from cordbeat.adapters.telegram import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._app = MagicMock()
        adapter._app.bot = MagicMock()
        adapter._app.bot.send_message = AsyncMock()
        adapter._chat_map["user1"] = 12345

        tg_mock = MagicMock()
        # InlineKeyboardButton / InlineKeyboardMarkup just need to be callable
        tg_mock.InlineKeyboardButton = MagicMock(side_effect=lambda text, **kw: text)
        tg_mock.InlineKeyboardMarkup = MagicMock(return_value="keyboard")

        data = {
            "content": "🔧 run shell_exec?",
            "metadata": {
                "proposal_id": "abc-123",
                "skill_name": "shell_exec",
                "skill_params": {"path_name": "value_with_underscores"},
            },
        }

        with patch.dict(
            "sys.modules",
            {"telegram": tg_mock, "telegram.ext": MagicMock()},
        ):
            await adapter._dispatch_skill_confirm("user1", data)

        adapter._app.bot.send_message.assert_awaited_once()
        call_kwargs = adapter._app.bot.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == 12345
        assert "shell_exec" in call_kwargs["text"]
        assert "path_name: value_with_underscores" in call_kwargs["text"]
        assert "parse_mode" not in call_kwargs
        assert call_kwargs["reply_markup"] == "keyboard"

    async def test_dispatch_skill_confirm_fallback_no_chat_map(self) -> None:
        """Skill confirm should fall back to text if chat_id is unknown."""
        from cordbeat.adapters.telegram import TelegramAdapter

        config = AdapterConfig(options={"token": "test"})
        adapter = TelegramAdapter(config)
        adapter._app = MagicMock()
        adapter._app.bot = MagicMock()
        adapter._app.bot.send_message = AsyncMock()
        # No chat_map entry and non-numeric user_id

        data = {
            "content": "🔧 run shell_exec?",
            "metadata": {
                "proposal_id": "abc-123",
                "skill_name": "shell_exec",
                "skill_params": {},
            },
        }

        # Should fall back to super()._dispatch_skill_confirm -> _dispatch_core_message
        adapter._dispatch_core_message = AsyncMock()  # type: ignore[assignment]
        with patch("cordbeat.adapters.telegram.logger"):
            await adapter._dispatch_skill_confirm("not-a-number", data)

        adapter._dispatch_core_message.assert_awaited()

    async def test_dispatch_skill_confirm_fallback_on_import_error(self) -> None:
        """If python-telegram-bot is missing, fall back to a plain message."""
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "test"}))
        adapter._app = MagicMock()
        adapter._chat_map["user1"] = 12345
        adapter._dispatch_core_message = AsyncMock()  # type: ignore[assignment]

        data = {
            "content": "🔧 run shell_exec?",
            "metadata": {"proposal_id": "abc-123", "skill_name": "shell_exec"},
        }

        # ``from telegram import InlineKeyboard...`` raises ImportError when the
        # module entry is None, exercising the except-ImportError fallback.
        with (
            patch.dict("sys.modules", {"telegram": None}),
            patch("cordbeat.adapters.telegram.logger"),
        ):
            await adapter._dispatch_skill_confirm("user1", data)

        adapter._dispatch_core_message.assert_awaited_once()

    async def test_dispatch_skill_confirm_no_app_falls_back(self) -> None:
        """With no Telegram app yet, skill confirm defers to the base dispatch."""
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "test"}))
        adapter._app = None
        adapter._dispatch_core_message = AsyncMock()  # type: ignore[assignment]

        await adapter._dispatch_skill_confirm(
            "user1", {"content": "x", "metadata": {"proposal_id": "p"}}
        )
        adapter._dispatch_core_message.assert_awaited_once()

    async def test_dispatch_skill_confirm_handles_send_failure(self) -> None:
        """A send error during skill confirm falls back to the base dispatch."""
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "test"}))
        adapter._app = MagicMock()
        adapter._app.bot.send_message = AsyncMock(side_effect=RuntimeError("nope"))
        adapter._chat_map["user1"] = 12345
        adapter._dispatch_core_message = AsyncMock()  # type: ignore[assignment]

        tg_mock = MagicMock()
        tg_mock.InlineKeyboardButton = MagicMock(side_effect=lambda text, **kw: text)
        tg_mock.InlineKeyboardMarkup = MagicMock(return_value="keyboard")
        data = {"content": "x", "metadata": {"proposal_id": "p", "skill_name": "s"}}

        with (
            patch.dict("sys.modules", {"telegram": tg_mock, "telegram.ext": tg_mock}),
            patch("cordbeat.adapters.telegram.logger"),
        ):
            await adapter._dispatch_skill_confirm("user1", data)

        adapter._dispatch_core_message.assert_awaited_once()

    def test_init_creates_stt_and_tts_backends_when_enabled(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter
        from cordbeat.config import STTConfig, TTSConfig

        stt_backend = object()
        tts_backend = object()
        with (
            patch(
                "cordbeat.ai.stt.create_stt_backend", return_value=stt_backend
            ) as make_stt,
            patch(
                "cordbeat.ai.tts.create_tts_backend", return_value=tts_backend
            ) as make_tts,
        ):
            adapter = TelegramAdapter(
                AdapterConfig(options={"token": "t"}),
                stt_config=STTConfig(enabled=True),
                tts_config=TTSConfig(enabled=True),
            )
        make_stt.assert_called_once()
        make_tts.assert_called_once()
        assert adapter._stt is stt_backend
        assert adapter._tts is tts_backend

    async def test_dispatch_core_message_routes_to_voice(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._tts = MagicMock()
        adapter._voice_users.add("u1")
        adapter._send_voice_to_telegram = AsyncMock()  # type: ignore[assignment]

        await adapter._dispatch_core_message("u1", "hi", [])
        adapter._send_voice_to_telegram.assert_awaited_once_with("u1", "hi")

    async def test_dispatch_core_message_routes_to_images(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._send_images_to_telegram = AsyncMock()  # type: ignore[assignment]

        await adapter._dispatch_core_message("u1", "caption", ["img1"])
        adapter._send_images_to_telegram.assert_awaited_once_with(
            "u1", "caption", ["img1"]
        )

    async def test_dispatch_core_message_routes_to_text(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._send_to_telegram = AsyncMock()  # type: ignore[assignment]

        await adapter._dispatch_core_message("u1", "plain", [])
        adapter._send_to_telegram.assert_awaited_once_with("u1", "plain")

    async def test_send_voice_ogg_uses_send_voice(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._app.bot.send_voice = AsyncMock()
        adapter._tts = MagicMock()
        adapter._tts.synthesize = AsyncMock(return_value=b"audio")
        adapter._tts.content_type = "audio/ogg"
        adapter._chat_map["u1"] = 555

        await adapter._send_voice_to_telegram("u1", "hello")
        adapter._app.bot.send_voice.assert_awaited_once()
        assert adapter._app.bot.send_voice.call_args.kwargs["chat_id"] == 555

    async def test_send_voice_mp3_uses_send_audio(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._app.bot.send_audio = AsyncMock()
        adapter._tts = MagicMock()
        adapter._tts.synthesize = AsyncMock(return_value=b"audio")
        adapter._tts.content_type = "audio/mpeg"
        adapter._chat_map["u1"] = 777

        await adapter._send_voice_to_telegram("u1", "hello")
        adapter._app.bot.send_audio.assert_awaited_once()

    async def test_send_voice_empty_audio_falls_back_to_text(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._tts = MagicMock()
        adapter._tts.synthesize = AsyncMock(return_value=b"")
        adapter._chat_map["u1"] = 1
        adapter._send_to_telegram = AsyncMock()  # type: ignore[assignment]

        await adapter._send_voice_to_telegram("u1", "hello")
        adapter._send_to_telegram.assert_awaited_once_with("u1", "hello")

    async def test_send_voice_no_app_is_noop(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = None
        adapter._tts = MagicMock()
        # Should not raise despite tts being set.
        await adapter._send_voice_to_telegram("u1", "hello")

    async def test_send_images_success_caption_on_first_only(self) -> None:
        import base64

        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._app.bot.send_photo = AsyncMock()
        adapter._chat_map["u1"] = 42
        b64 = base64.b64encode(b"png-bytes").decode("ascii")

        await adapter._send_images_to_telegram("u1", "look", [b64, b64])

        assert adapter._app.bot.send_photo.await_count == 2
        first_kwargs = adapter._app.bot.send_photo.call_args_list[0].kwargs
        second_kwargs = adapter._app.bot.send_photo.call_args_list[1].kwargs
        assert first_kwargs["caption"] == "look"
        assert second_kwargs["caption"] is None

    async def test_send_images_long_caption_sent_as_text(self) -> None:
        import base64

        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._app.bot.send_photo = AsyncMock()
        adapter._app.bot.send_message = AsyncMock()
        adapter._chat_map["u1"] = 42
        b64 = base64.b64encode(b"png-bytes").decode("ascii")
        caption = "c" * 1200

        await adapter._send_images_to_telegram("u1", caption, [b64])

        adapter._app.bot.send_photo.assert_awaited_once()
        assert adapter._app.bot.send_photo.call_args.kwargs["caption"] is None
        adapter._app.bot.send_message.assert_awaited_once_with(
            chat_id=42,
            text=caption,
        )

    async def test_send_images_all_fail_falls_back_to_text(self) -> None:
        import base64

        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._app.bot.send_photo = AsyncMock(side_effect=RuntimeError("boom"))
        adapter._chat_map["u1"] = 42
        adapter._send_to_telegram = AsyncMock()  # type: ignore[assignment]
        b64 = base64.b64encode(b"png-bytes").decode("ascii")

        with patch("cordbeat.adapters.telegram.logger"):
            await adapter._send_images_to_telegram("u1", "cap", [b64])
        adapter._send_to_telegram.assert_awaited_once_with("u1", "cap")

    async def test_send_images_invalid_user_falls_back_to_text(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._send_to_telegram = AsyncMock()  # type: ignore[assignment]
        # No chat_map entry and a non-numeric user id.
        await adapter._send_images_to_telegram("nope", "cap", ["x"])
        adapter._send_to_telegram.assert_awaited_once_with("nope", "cap")

    async def test_start_and_stop_typing_manage_task(self) -> None:
        import asyncio

        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._app.bot.send_chat_action = AsyncMock()

        adapter._start_typing(123)
        assert 123 in adapter._typing_tasks
        first_task = adapter._typing_tasks[123]

        # Starting again replaces the previous task with a fresh one.
        adapter._start_typing(123)
        assert adapter._typing_tasks[123] is not first_task

        adapter._stop_typing(123)
        assert 123 not in adapter._typing_tasks
        # Let the event loop process the cancellations cleanly.
        await asyncio.sleep(0)
        assert first_task.cancelled()

    async def test_start_registers_handlers_and_starts_polling(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        app, handlers = await _start_telegram_and_capture(adapter)

        assert adapter._running is True
        app.initialize.assert_awaited_once()
        app.start.assert_awaited_once()
        app.updater.start_polling.assert_awaited_once()
        assert callable(handlers["message"])
        assert callable(handlers["command"])
        assert callable(handlers["callback"])

    async def test_message_handler_forwards_text_message(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]

        update = MagicMock()
        msg = update.message
        msg.text = "hello bot"
        msg.caption = None
        msg.photo = []
        msg.document = None
        msg.voice = None
        msg.reply_to_message = None
        msg.chat_id = 999
        msg.chat.type = "private"
        update.effective_user.id = 555
        update.effective_user.full_name = "Alice"
        update.effective_user.username = "alice"
        context = MagicMock()
        context.bot.username = "CordBeatBot"

        await handlers["message"](update, context)

        adapter._forward_to_core.assert_awaited_once()
        args, kwargs = adapter._forward_to_core.call_args
        assert args[0] == "555"
        assert args[1] == "hello bot"
        assert kwargs["chat_id"] == 999
        # The chat_id is remembered for reply routing.
        assert adapter._chat_map["555"] == 999

    async def test_message_handler_ignores_message_without_user(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]

        update = MagicMock()
        update.effective_user = None
        await handlers["message"](update, MagicMock())
        adapter._forward_to_core.assert_not_awaited()

    async def test_message_handler_downloads_photo(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]

        photo = MagicMock()
        photo.file_size = 1024
        photo.file_id = "photo-1"
        file_obj = MagicMock()
        file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(b"imgdata"))

        update = MagicMock()
        msg = update.message
        msg.text = None
        msg.caption = "a photo"
        msg.photo = [photo]
        msg.document = None
        msg.voice = None
        msg.reply_to_message = None
        msg.chat_id = 12
        msg.chat.type = "private"
        update.effective_user.id = 1
        update.effective_user.full_name = "Alice"
        context = MagicMock()
        context.bot.username = "CordBeatBot"
        context.bot.get_file = AsyncMock(return_value=file_obj)

        await handlers["message"](update, context)

        context.bot.get_file.assert_awaited_once_with("photo-1")
        kwargs = adapter._forward_to_core.call_args.kwargs
        assert len(kwargs["images"]) == 1  # base64-encoded download

    async def test_message_handler_transcribes_voice_via_stt(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]
        adapter._stt = MagicMock()
        adapter._stt.transcribe = AsyncMock(return_value="hello from voice")

        file_obj = MagicMock()
        file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(b"ogg"))

        update = MagicMock()
        msg = update.message
        msg.text = None
        msg.caption = None
        msg.photo = []
        msg.document = None
        msg.voice = MagicMock(file_id="voice-1")
        msg.reply_to_message = None
        msg.chat_id = 12
        msg.chat.type = "private"
        update.effective_user.id = 2
        update.effective_user.full_name = "Bob"
        context = MagicMock()
        context.bot.username = "CordBeatBot"
        context.bot.get_file = AsyncMock(return_value=file_obj)

        await handlers["message"](update, context)

        adapter._stt.transcribe.assert_awaited_once()
        args, kwargs = adapter._forward_to_core.call_args
        assert args[1] == "hello from voice"
        assert kwargs["is_voice"] is True
        assert "2" in adapter._voice_users

    async def test_message_handler_captures_reply_context(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]

        reply = MagicMock()
        reply.photo = []
        reply.document = None
        reply.text = "earlier message"
        reply.caption = None
        reply.message_id = 77
        reply.from_user.full_name = "Carol"
        reply.from_user.username = "carol"
        reply.from_user.is_bot = False

        update = MagicMock()
        msg = update.message
        msg.text = "replying now"
        msg.caption = None
        msg.photo = []
        msg.document = None
        msg.voice = None
        msg.reply_to_message = reply
        msg.chat_id = 12
        msg.chat.type = "private"
        update.effective_user.id = 3
        update.effective_user.full_name = "Dan"
        context = MagicMock()
        context.bot.username = "CordBeatBot"

        await handlers["message"](update, context)

        reply_context = adapter._forward_to_core.call_args.kwargs["reply_context"]
        assert reply_context["author"] == "Carol"
        assert reply_context["content"] == "earlier message"
        assert reply_context["message_id"] == "77"

    async def test_command_handler_forwards_normalized_command(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]

        update = MagicMock()
        update.message.text = "/approve@CordBeatBot abc-123"
        update.message.chat_id = 7
        update.effective_user.id = 42
        update.effective_user.full_name = "Bob"
        update.effective_user.username = "bob"

        await handlers["command"](update, MagicMock())

        adapter._forward_to_core.assert_awaited_once()
        args, kwargs = adapter._forward_to_core.call_args
        assert args[0] == "42"
        assert args[1] == "/approve abc-123"

    async def test_callback_handler_approves_skill_confirm(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._ws = AsyncMock()

        update = MagicMock()
        query = update.callback_query
        query.data = "skill_confirm:approve:prop-9"
        query.answer = AsyncMock()
        query.edit_message_reply_markup = AsyncMock()
        query.message.reply_text = AsyncMock()
        update.effective_user.id = 5

        await handlers["callback"](update, MagicMock())

        adapter._ws.send.assert_awaited_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["content"] == "/approve prop-9"
        assert payload["platform_user_id"] == "5"
        query.answer.assert_awaited_once()

    async def test_callback_handler_rejects_non_owner_without_core_send(
        self,
    ) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._ws = AsyncMock()
        adapter._pending_skill_confirm_owners["prop-9"] = "5"

        update = MagicMock()
        query = update.callback_query
        query.data = "skill_confirm:approve:prop-9"
        query.answer = AsyncMock()
        query.edit_message_reply_markup = AsyncMock()
        query.message.reply_text = AsyncMock()
        update.effective_user.id = 6

        await handlers["callback"](update, MagicMock())

        adapter._ws.send.assert_not_awaited()
        query.answer.assert_awaited_once_with(
            "This approval isn't yours",
            show_alert=True,
        )
        query.edit_message_reply_markup.assert_not_awaited()
        query.message.reply_text.assert_not_awaited()

    async def test_callback_handler_clears_owner_even_if_edit_fails(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._ws = AsyncMock()
        adapter._pending_skill_confirm_owners["prop-9"] = "5"

        update = MagicMock()
        query = update.callback_query
        query.data = "skill_confirm:approve:prop-9"
        query.answer = AsyncMock()
        query.edit_message_reply_markup = AsyncMock(
            side_effect=RuntimeError("message too old")
        )
        query.message.reply_text = AsyncMock()
        update.effective_user.id = 5

        await handlers["callback"](update, MagicMock())

        adapter._ws.send.assert_awaited_once()
        assert "prop-9" not in adapter._pending_skill_confirm_owners

    async def test_dispatch_skill_confirm_owner_cache_is_bounded(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._app.bot = MagicMock()
        adapter._app.bot.send_message = AsyncMock()
        adapter._chat_map["user1"] = 12345

        tg_mock = MagicMock()
        tg_mock.InlineKeyboardButton = MagicMock(side_effect=lambda text, **kw: text)
        tg_mock.InlineKeyboardMarkup = MagicMock(return_value="keyboard")

        with (
            patch.dict(
                "sys.modules",
                {"telegram": tg_mock, "telegram.ext": MagicMock()},
            ),
            patch("cordbeat.adapters.telegram._PENDING_SKILL_CONFIRM_MAX", 3),
        ):
            for i in range(4):
                data = {
                    "content": "🔧 run?",
                    "metadata": {"proposal_id": f"prop-{i}", "skill_name": "s"},
                }
                await adapter._dispatch_skill_confirm("user1", data)

        assert len(adapter._pending_skill_confirm_owners) == 3
        assert "prop-0" not in adapter._pending_skill_confirm_owners
        assert "prop-3" in adapter._pending_skill_confirm_owners

    async def test_callback_handler_acknowledges_before_core_send_failure(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        calls: list[str] = []

        async def answer() -> None:
            calls.append("answer")

        async def send(payload: str) -> None:
            calls.append("send")
            raise RuntimeError("closed")

        adapter._ws = AsyncMock()
        adapter._ws.send = AsyncMock(side_effect=send)

        update = MagicMock()
        query = update.callback_query
        query.data = "skill_confirm:approve:prop-9"
        query.answer = AsyncMock(side_effect=answer)
        query.edit_message_reply_markup = AsyncMock()
        query.message.reply_text = AsyncMock()
        update.effective_user.id = 5

        with patch("cordbeat.adapters.telegram.logger"):
            await handlers["callback"](update, MagicMock())

        assert calls == ["answer", "send"]
        query.message.reply_text.assert_awaited_once()
        assert (
            "Failed to send approval"
            in query.message.reply_text.call_args.args[0]
        )

    async def test_callback_handler_ignores_unmatched_data(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._ws = AsyncMock()

        update = MagicMock()
        query = update.callback_query
        query.data = "something_else"
        query.answer = AsyncMock()

        await handlers["callback"](update, MagicMock())

        query.answer.assert_awaited_once()
        adapter._ws.send.assert_not_awaited()

    async def test_message_handler_skips_oversized_photo(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]

        photo = MagicMock()
        photo.file_size = 50 * 1024 * 1024  # over the 10 MiB cap
        update = MagicMock()
        msg = update.message
        msg.text = "big pic"
        msg.caption = None
        msg.photo = [photo]
        msg.document = None
        msg.voice = None
        msg.reply_to_message = None
        msg.chat_id = 1
        msg.chat.type = "private"
        update.effective_user.id = 1
        context = MagicMock()
        context.bot.username = "CordBeatBot"
        context.bot.get_file = AsyncMock()

        with patch("cordbeat.adapters.telegram.logger") as mock_logger:
            await handlers["message"](update, context)

        context.bot.get_file.assert_not_awaited()  # never downloaded
        assert adapter._forward_to_core.call_args.kwargs["images"] == []
        mock_logger.warning.assert_called()

    async def test_message_handler_downloads_document_image(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]

        doc = MagicMock()
        doc.mime_type = "image/png"
        doc.file_size = 1024
        doc.file_id = "doc-1"
        file_obj = MagicMock()
        file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(b"png"))

        update = MagicMock()
        msg = update.message
        msg.text = "see doc"
        msg.caption = None
        msg.photo = []
        msg.document = doc
        msg.voice = None
        msg.reply_to_message = None
        msg.chat_id = 1
        msg.chat.type = "private"
        update.effective_user.id = 1
        context = MagicMock()
        context.bot.username = "CordBeatBot"
        context.bot.get_file = AsyncMock(return_value=file_obj)

        await handlers["message"](update, context)

        context.bot.get_file.assert_awaited_once_with("doc-1")
        assert len(adapter._forward_to_core.call_args.kwargs["images"]) == 1

    async def test_message_handler_voice_without_stt_keeps_text(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]
        adapter._stt = None  # STT not configured

        update = MagicMock()
        msg = update.message
        msg.text = "caption only"
        msg.caption = None
        msg.photo = []
        msg.document = None
        msg.voice = MagicMock(file_id="v")
        msg.reply_to_message = None
        msg.chat_id = 1
        msg.chat.type = "private"
        update.effective_user.id = 1
        context = MagicMock()
        context.bot.username = "CordBeatBot"

        await handlers["message"](update, context)

        # No transcription happened; the original text is forwarded as-is.
        assert adapter._forward_to_core.call_args.args[1] == "caption only"
        assert adapter._forward_to_core.call_args.kwargs["is_voice"] is False

    async def test_message_handler_voice_empty_transcription(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        _, handlers = await _start_telegram_and_capture(adapter)
        adapter._forward_to_core = AsyncMock()  # type: ignore[assignment]
        adapter._stt = MagicMock()
        adapter._stt.transcribe = AsyncMock(return_value="")  # empty result
        file_obj = MagicMock()
        file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(b"ogg"))

        update = MagicMock()
        msg = update.message
        msg.text = "fallback text"
        msg.caption = None
        msg.photo = []
        msg.document = None
        msg.voice = MagicMock(file_id="v")
        msg.reply_to_message = None
        msg.chat_id = 1
        msg.chat.type = "private"
        update.effective_user.id = 9
        context = MagicMock()
        context.bot.username = "CordBeatBot"
        context.bot.get_file = AsyncMock(return_value=file_obj)

        with patch("cordbeat.adapters.telegram.logger") as mock_logger:
            await handlers["message"](update, context)

        assert adapter._forward_to_core.call_args.args[1] == "fallback text"
        assert "9" not in adapter._voice_users
        mock_logger.warning.assert_called()

    async def test_send_to_telegram_logs_send_failure(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._app.bot.send_message = AsyncMock(side_effect=RuntimeError("down"))
        adapter._chat_map["u1"] = 5

        with patch("cordbeat.adapters.telegram.logger") as mock_logger:
            await adapter._send_to_telegram("u1", "hi")
        mock_logger.exception.assert_called()

    async def test_send_voice_invalid_user_falls_back_to_text(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._tts = MagicMock()
        adapter._send_to_telegram = AsyncMock()  # type: ignore[assignment]
        # No chat_map entry and a non-numeric id forces the text fallback.
        await adapter._send_voice_to_telegram("nope", "hi")
        adapter._send_to_telegram.assert_awaited_once_with("nope", "hi")

    async def test_send_voice_synthesis_error_falls_back_to_text(self) -> None:
        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._tts = MagicMock()
        adapter._tts.synthesize = AsyncMock(side_effect=RuntimeError("tts down"))
        adapter._chat_map["u1"] = 5
        adapter._send_to_telegram = AsyncMock()  # type: ignore[assignment]

        with patch("cordbeat.adapters.telegram.logger") as mock_logger:
            await adapter._send_voice_to_telegram("u1", "hi")
        adapter._send_to_telegram.assert_awaited_once_with("u1", "hi")
        mock_logger.exception.assert_called()

    async def test_keep_typing_sends_chat_action_then_cancels(self) -> None:
        import asyncio

        from cordbeat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(AdapterConfig(options={"token": "t"}))
        adapter._app = MagicMock()
        adapter._app.bot.send_chat_action = AsyncMock()

        # First sleep raises CancelledError so the loop runs exactly once.
        with (
            patch.dict("sys.modules", {"telegram": MagicMock()}),
            patch(
                "cordbeat.adapters.telegram.asyncio.sleep",
                AsyncMock(side_effect=asyncio.CancelledError),
            ),
        ):
            await adapter._keep_typing_telegram(123)

        adapter._app.bot.send_chat_action.assert_awaited_once()

    async def test_unknown_adapter(self) -> None:
        from cordbeat.adapters.runner import _run_adapter

        with patch("cordbeat.adapters.runner.logger") as mock_logger:
            await _run_adapter("unknown", "config.yaml")
            mock_logger.error.assert_called()

    async def test_disabled_adapter(self) -> None:
        from cordbeat.adapters.runner import _run_adapter

        config_mock = MagicMock()
        config_mock.adapters = {
            "discord": AdapterConfig(enabled=False),
        }
        config_mock.log = LogConfig()
        with (
            patch("cordbeat.adapters.runner.load_config", return_value=config_mock),
            patch("cordbeat.adapters.runner.logger") as mock_logger,
        ):
            await _run_adapter("discord", "config.yaml")
            mock_logger.info.assert_called()


class TestSlackAdapter:
    def test_import(self) -> None:
        from cordbeat.adapters.slack import SlackAdapter

        assert SlackAdapter is not None

    async def test_forward_to_core_includes_channel_scope(self) -> None:
        from cordbeat.adapters.slack import SlackAdapter

        config = AdapterConfig(options={})
        adapter = SlackAdapter(config)
        adapter._ws = AsyncMock()

        await adapter._forward_to_core(
            user_id="U123",
            text="hello",
            channel="C999",
            channel_type="channel",
            sent_at=datetime(2026, 7, 14, 14, 57, tzinfo=UTC),
        )

        adapter._ws.send.assert_awaited_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["adapter_id"] == "slack"
        assert payload["metadata"]["channel_id"] == "C999"
        assert payload["metadata"]["is_dm"] is False
        assert payload["timestamp"] == "2026-07-14T14:57:00+00:00"

    async def test_send_to_slack_metadata_channel_overrides_cache(self) -> None:
        from cordbeat.adapters.slack import SlackAdapter

        config = AdapterConfig(options={})
        adapter = SlackAdapter(config)
        adapter._web_client = AsyncMock()
        adapter._user_channels["U123"] = "Cold"

        await adapter._send_to_slack(
            "U123",
            "hello",
            metadata={"channel_id": "Cnew"},
        )

        adapter._web_client.chat_postMessage.assert_awaited_once_with(
            channel="Cnew",
            text="hello",
        )

    async def test_send_to_slack_dm_fallback_when_metadata_missing(self) -> None:
        from cordbeat.adapters.slack import SlackAdapter

        adapter = SlackAdapter(AdapterConfig(options={}))
        adapter._web_client = AsyncMock()
        adapter._web_client.conversations_open.return_value = {
            "channel": {"id": "Dnew"}
        }

        await adapter._send_to_slack("U123", "hello")

        adapter._web_client.conversations_open.assert_awaited_once_with(users="U123")
        adapter._web_client.chat_postMessage.assert_awaited_once_with(
            channel="Dnew",
            text="hello",
        )

    async def test_send_to_slack_respects_disabled_dm_fallback(self) -> None:
        from cordbeat.adapters.slack import SlackAdapter

        adapter = SlackAdapter(AdapterConfig(options={}))
        adapter._web_client = AsyncMock()

        await adapter._send_to_slack(
            "U123",
            "hello",
            metadata={"allow_dm_fallback": False},
        )

        adapter._web_client.conversations_open.assert_not_awaited()
        adapter._web_client.chat_postMessage.assert_not_awaited()


class TestLineAdapter:
    async def test_forward_to_core_includes_channel_scope(self) -> None:
        from cordbeat.adapters.line import LineAdapter

        config = AdapterConfig(options={})
        adapter = LineAdapter(config)
        adapter._ws = AsyncMock()

        await adapter._forward_to_core(
            user_id="Uline",
            text="hello",
            is_group=True,
            channel_id="Gline",
            sent_at=datetime(2026, 7, 14, 14, 58, tzinfo=UTC),
        )

        adapter._ws.send.assert_awaited_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["adapter_id"] == "line"
        assert payload["metadata"]["channel_id"] == "Gline"
        assert payload["metadata"]["is_dm"] is False
        assert payload["timestamp"] == "2026-07-14T14:58:00+00:00"

    async def test_line_group_without_keywords_is_not_mentioned(self) -> None:
        from cordbeat.adapters.line import LineAdapter

        config = AdapterConfig(options={"respond_mode": "mention_only"})
        adapter = LineAdapter(config)
        adapter._ws = AsyncMock()

        with patch("cordbeat.adapters._utils._read_soul_keywords", return_value=[]):
            await adapter._forward_to_core(
                user_id="Uline",
                text="hello",
                is_group=True,
                channel_id="Gline",
            )

        adapter._ws.send.assert_not_awaited()


class TestWhatsAppAdapter:
    async def test_forward_to_core_marks_dm_scope(self) -> None:
        from cordbeat.adapters._utils import MAX_INBOUND_TEXT_CHARS
        from cordbeat.adapters.whatsapp import WhatsAppAdapter

        config = AdapterConfig(options={})
        adapter = WhatsAppAdapter(config)
        adapter._ws = AsyncMock()

        await adapter._forward_to_core(
            user_id="15551234567",
            text="x" * (MAX_INBOUND_TEXT_CHARS + 10),
            sent_at=datetime(2026, 7, 14, 14, 59, tzinfo=UTC),
        )

        adapter._ws.send.assert_awaited_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["adapter_id"] == "whatsapp"
        assert len(payload["content"]) == MAX_INBOUND_TEXT_CHARS
        assert payload["metadata"]["channel_id"] == "15551234567"
        assert payload["metadata"]["is_dm"] is True
        assert payload["timestamp"] == "2026-07-14T14:59:00+00:00"


class TestSignalAdapter:
    async def test_poll_backoff_grows_and_resets_after_success(self) -> None:
        from cordbeat.adapters.signal import SignalAdapter

        config = AdapterConfig(options={"poll_interval": 2})
        adapter = SignalAdapter(config)
        adapter._running = True
        adapter._rpc = AsyncMock(
            side_effect=[RuntimeError("down"), RuntimeError("still down"), []]
        )
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)
            if len(delays) == 3:
                adapter._running = False

        with patch("cordbeat.adapters.signal.asyncio.sleep", side_effect=record_sleep):
            await adapter._poll_loop()

        assert delays == [4.0, 8.0, 2.0]

    async def test_forward_to_core_marks_dm_scope(self) -> None:
        from cordbeat.adapters.signal import SignalAdapter

        config = AdapterConfig(options={})
        adapter = SignalAdapter(config)
        adapter._ws = AsyncMock()

        await adapter._forward_to_core(
            user_id="+15551234567",
            text="hello",
            sent_at=datetime(2026, 7, 14, 15, 0, tzinfo=UTC),
        )

        adapter._ws.send.assert_awaited_once()
        payload = json.loads(adapter._ws.send.call_args[0][0])
        assert payload["adapter_id"] == "signal"
        assert payload["metadata"]["channel_id"] == "+15551234567"
        assert payload["metadata"]["is_dm"] is True
        assert payload["timestamp"] == "2026-07-14T15:00:00+00:00"
