"""Telegram adapter — bridges Telegram bot to CordBeat Core via WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

import websockets

from cordbeat.config import AdapterConfig

logger = logging.getLogger(__name__)

ADAPTER_ID = "telegram"
_MAX_BACKOFF = 60


class TelegramAdapter:
    """Telegram bot that forwards messages to CordBeat Core via WebSocket."""

    def __init__(self, config: AdapterConfig) -> None:
        self._config = config
        self._ws_url = config.core_ws_url
        self._token: str = config.options.get("token", "")
        self._app: Any = None
        self._ws: Any = None
        self._running = False
        # Map platform_user_id → chat_id for reply routing
        self._chat_map: dict[str, int] = {}

    async def start(self) -> None:
        try:
            from telegram import Update
            from telegram.ext import (
                ApplicationBuilder,
                MessageHandler,
                filters,
            )
        except ImportError:
            logger.error(
                "python-telegram-bot is not installed. "
                "Install with: uv sync --extra telegram"
            )
            return

        if not self._token:
            logger.error(
                "Telegram bot token not configured in adapters.telegram.options.token"
            )
            return

        self._running = True
        self._app = ApplicationBuilder().token(self._token).build()

        async def handle_message(
            update: Update,
            context: Any,
        ) -> None:
            if update.message and update.message.text:
                user = update.effective_user
                if user is None:
                    return
                user_id = str(user.id)
                chat_id = update.message.chat_id
                self._chat_map[user_id] = chat_id
                await self._forward_to_core(
                    user_id,
                    update.message.text,
                    display_name=user.full_name or user.username or "",
                    chat_id=chat_id,
                )

        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )

        # Connect to Core in background
        asyncio.create_task(self._connect_to_core())

        # Start polling (non-blocking)
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        logger.info("Telegram adapter started")

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def _connect_to_core(self) -> None:
        backoff = 1
        while self._running:
            try:
                self._ws = await websockets.connect(self._ws_url)
                await self._ws.send(json.dumps({"adapter_id": ADAPTER_ID}))
                ack = json.loads(await self._ws.recv())
                logger.info("Connected to Core: %s", ack.get("content", "OK"))
                backoff = 1
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
                    await self._send_to_telegram(
                        platform_user_id,
                        content,
                    )
        except websockets.ConnectionClosed:
            logger.info("Core connection closed")

    async def _forward_to_core(
        self,
        user_id: str,
        text: str,
        *,
        display_name: str = "",
        chat_id: int = 0,
    ) -> None:
        if self._ws is None:
            logger.warning("Not connected to Core, dropping message")
            return

        payload = json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": user_id,
                "content": text,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "chat_id": str(chat_id),
                    "display_name": display_name,
                },
            }
        )
        try:
            await self._ws.send(payload)
        except Exception:
            logger.exception("Failed to forward message to Core")

    async def _send_to_telegram(
        self,
        platform_user_id: str,
        content: str,
    ) -> None:
        if not self._app or not platform_user_id:
            return

        chat_id = self._chat_map.get(platform_user_id)
        if chat_id is None:
            # Fall back to using user_id as chat_id (DM)
            try:
                chat_id = int(platform_user_id)
            except ValueError:
                logger.warning(
                    "No chat_id for Telegram user %s",
                    platform_user_id,
                )
                return

        try:
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=content,
            )
        except Exception:
            logger.exception(
                "Failed to send message to Telegram chat %s",
                chat_id,
            )
