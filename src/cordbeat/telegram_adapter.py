"""Telegram adapter — bridges Telegram bot to CordBeat Core via WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import UTC, datetime
from typing import Any

from cordbeat.config import AdapterConfig
from cordbeat.gateway import RetryableConnection

logger = logging.getLogger(__name__)

ADAPTER_ID = "telegram"


class TelegramAdapter(RetryableConnection):
    """Telegram bot that forwards messages to CordBeat Core via WebSocket."""

    adapter_id = ADAPTER_ID

    def __init__(self, config: AdapterConfig) -> None:
        self._config = config
        self._ws_url = config.core_ws_url
        self._token: str = config.options.get("token", "")
        self._app: Any = None
        self._ws: Any = None
        self._running = False
        self._max_backoff = config.reconnect_max_backoff
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
            msg = update.message
            if not msg:
                return
            user = update.effective_user
            if user is None:
                return
            user_id = str(user.id)
            chat_id = msg.chat_id
            self._chat_map[user_id] = chat_id
            display_name = user.full_name or user.username or ""

            _image_size_limit = 10 * 1024 * 1024  # 10 MB

            # Download photo if present (pick highest resolution).
            images: list[str] = []
            if msg.photo:
                try:
                    photo = msg.photo[-1]
                    if (photo.file_size or 0) <= _image_size_limit:
                        file = await context.bot.get_file(photo.file_id)
                        raw = await file.download_as_bytearray()
                        images.append(base64.b64encode(bytes(raw)).decode("ascii"))
                    else:
                        logger.warning(
                            "Skipping oversized Telegram photo (%d bytes)",
                            photo.file_size,
                        )
                except Exception:
                    logger.warning("Failed to download Telegram photo")

            # Also handle images sent as documents (no compression).
            if not images and msg.document:
                doc = msg.document
                mime = (doc.mime_type or "").startswith("image/")
                size_ok = (doc.file_size or 0) <= _image_size_limit
                if mime and size_ok:
                    try:
                        file = await context.bot.get_file(doc.file_id)
                        raw = await file.download_as_bytearray()
                        images.append(base64.b64encode(bytes(raw)).decode("ascii"))
                    except Exception:
                        logger.warning("Failed to download Telegram document image")

            text = msg.text or msg.caption or ""
            await self._forward_to_core(
                user_id,
                text,
                display_name=display_name,
                chat_id=chat_id,
                images=images,
            )

        self._app.add_handler(
            MessageHandler(
                (filters.TEXT | filters.PHOTO | filters.Document.IMAGE)
                & ~filters.COMMAND,
                handle_message,
            )
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

    async def _dispatch_core_message(self, platform_user_id: str, content: str) -> None:
        await self._send_to_telegram(platform_user_id, content)

    async def _forward_to_core(
        self,
        user_id: str,
        text: str,
        *,
        display_name: str = "",
        chat_id: int = 0,
        images: list[str] | None = None,
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
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "images": images or [],
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
