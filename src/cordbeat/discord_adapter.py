"""Discord adapter — bridges Discord bot to CordBeat Core via WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

import websockets

from cordbeat.config import AdapterConfig

logger = logging.getLogger(__name__)

ADAPTER_ID = "discord"
_MAX_BACKOFF = 60


class DiscordAdapter:
    """Discord bot that forwards messages to CordBeat Core via WebSocket."""

    def __init__(self, config: AdapterConfig) -> None:
        self._config = config
        self._ws_url = config.core_ws_url
        self._token: str = config.options.get("token", "")
        self._bot: Any = None
        self._ws: Any = None
        self._running = False

    async def start(self) -> None:
        try:
            import discord  # noqa: F811
        except ImportError:
            logger.error(
                "discord.py is not installed. Install with: uv sync --extra discord"
            )
            return

        if not self._token:
            logger.error(
                "Discord bot token not configured in adapters.discord.options.token"
            )
            return

        intents = discord.Intents.default()
        intents.message_content = True
        self._bot = discord.Client(intents=intents)
        self._running = True

        @self._bot.event
        async def on_ready() -> None:
            logger.info("Discord bot logged in as %s", self._bot.user)
            asyncio.create_task(self._connect_to_core())

        @self._bot.event
        async def on_message(message: discord.Message) -> None:
            if message.author == self._bot.user:
                return
            if message.author.bot:
                return
            await self._forward_to_core(message)

        try:
            await self._bot.start(self._token)
        except Exception:
            logger.exception("Discord bot failed to start")

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._bot:
            await self._bot.close()

    async def _connect_to_core(self) -> None:
        backoff = 1
        while self._running:
            try:
                self._ws = await websockets.connect(self._ws_url)
                await self._ws.send(json.dumps({"adapter_id": ADAPTER_ID}))
                ack = json.loads(await self._ws.recv())
                logger.info("Connected to Core: %s", ack.get("content", "OK"))
                backoff = 1

                # Listen for messages from Core
                await self._listen_core()
            except Exception:
                if not self._running:
                    break
                logger.warning(
                    "Core connection failed, retrying in %ds...",
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    async def _listen_core(self) -> None:
        try:
            async for raw in self._ws:
                data = json.loads(raw)
                msg_type = data.get("type", "")
                content = data.get("content", "")
                platform_user_id = data.get("platform_user_id", "")

                if msg_type in ("message", "heartbeat_message", "error"):
                    await self._send_to_discord(platform_user_id, content)
        except websockets.ConnectionClosed:
            logger.info("Core connection closed")

    async def _forward_to_core(self, message: Any) -> None:
        if self._ws is None:
            logger.warning("Not connected to Core, dropping message")
            return

        payload = json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": str(message.author.id),
                "content": message.content,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "channel_id": str(message.channel.id),
                    "guild_id": str(message.guild.id) if message.guild else "",
                    "display_name": message.author.display_name,
                },
            }
        )
        try:
            await self._ws.send(payload)
        except Exception:
            logger.exception("Failed to forward message to Core")

    async def _send_to_discord(
        self,
        platform_user_id: str,
        content: str,
    ) -> None:
        if not self._bot or not platform_user_id:
            return

        try:
            user = await self._bot.fetch_user(int(platform_user_id))
            if user:
                await user.send(content)
        except Exception:
            logger.exception(
                "Failed to send message to Discord user %s",
                platform_user_id,
            )
