"""Discord adapter — bridges Discord bot to CordBeat Core via WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import OrderedDict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from cordbeat.config import AdapterConfig, STTConfig, TTSConfig
from cordbeat.gateway import RetryableConnection

if TYPE_CHECKING:
    from cordbeat.stt_backend import STTBackend
    from cordbeat.tts_backend import TTSBackend

logger = logging.getLogger(__name__)

ADAPTER_ID = "discord"

# Upper bound on cached user → channel entries. Prevents unbounded growth
# over long-running bot uptimes; oldest entries are evicted on insertion.
_USER_CHANNEL_CACHE_MAX = 10_000


class DiscordAdapter(RetryableConnection):
    """Discord bot that forwards messages to CordBeat Core via WebSocket."""

    adapter_id = ADAPTER_ID

    def __init__(
        self,
        config: AdapterConfig,
        *,
        stt_config: STTConfig | None = None,
        tts_config: TTSConfig | None = None,
    ) -> None:
        self._config = config
        self._ws_url = config.core_ws_url
        self._token: str = config.options.get("token", "")
        self._bot: Any = None
        self._ws: Any = None
        self._running = False
        self._max_backoff = config.reconnect_max_backoff
        # Bounded LRU of last channel per user for guild replies. Inserts
        # beyond the cap evict the least-recently-used entry so that the
        # process does not grow without bound over time.
        self._user_channels: OrderedDict[str, int] = OrderedDict()

        self._stt: STTBackend | None = None
        # TTS stored for future VC support; not used in text channels yet.
        self._tts: TTSBackend | None = None

        if stt_config is not None and stt_config.enabled:
            from cordbeat.stt_backend import create_stt_backend

            self._stt = create_stt_backend(stt_config)

        if tts_config is not None and tts_config.enabled:
            from cordbeat.tts_backend import create_tts_backend

            self._tts = create_tts_backend(tts_config)

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

    async def _dispatch_core_message(self, platform_user_id: str, content: str) -> None:
        await self._send_to_discord(platform_user_id, content)

    async def _forward_to_core(self, message: Any) -> None:
        if self._ws is None:
            logger.warning("Not connected to Core, dropping message")
            return

        user_id = str(message.author.id)
        channel_id = message.channel.id

        # Cache channel for reply routing (LRU-bounded).
        self._user_channels[user_id] = channel_id
        self._user_channels.move_to_end(user_id)
        if len(self._user_channels) > _USER_CHANNEL_CACHE_MAX:
            self._user_channels.popitem(last=False)

        # Collect image attachments (base64-encoded), cap at 4 images / 10 MB each.
        _max_images = 4
        _image_size_limit = 10 * 1024 * 1024  # 10 MB
        _audio_size_limit = 25 * 1024 * 1024  # 25 MB

        async def _fetch_attachment_images(src_msg: Any, buf: list[str]) -> None:
            for att in getattr(src_msg, "attachments", []):
                if len(buf) >= _max_images:
                    break
                ct = getattr(att, "content_type", "") or ""
                if not ct.startswith("image/"):
                    continue
                size = getattr(att, "size", 0) or 0
                if size > _image_size_limit:
                    logger.warning(
                        "Skipping oversized Discord image attachment: %d bytes", size
                    )
                    continue
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(str(att.url))
                        resp.raise_for_status()
                        buf.append(base64.b64encode(resp.content).decode("ascii"))
                except Exception:
                    logger.warning(
                        "Failed to download Discord image attachment: %s", att.url
                    )

        images: list[str] = []
        await _fetch_attachment_images(message, images)

        # Also check 1-level parent message (reply chain): image may live there.
        ref = getattr(message, "reference", None)
        if ref is not None and len(images) < _max_images:
            ref_msg = getattr(ref, "resolved", None)
            if ref_msg is not None:
                await _fetch_attachment_images(ref_msg, images)

        # STT: transcribe audio attachments when the backend is configured.
        content = message.content
        if self._stt is not None:
            for att in getattr(message, "attachments", []):
                ct = getattr(att, "content_type", "") or ""
                if not ct.startswith("audio/"):
                    continue
                size = getattr(att, "size", 0) or 0
                if size > _audio_size_limit:
                    logger.warning(
                        "Skipping oversized Discord audio attachment: %d bytes", size
                    )
                    continue
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(str(att.url))
                        resp.raise_for_status()
                        transcribed = await self._stt.transcribe(resp.content)
                    if transcribed:
                        content = (
                            f"{content} {transcribed}".strip()
                            if content
                            else transcribed
                        )
                        logger.debug("Discord STT transcribed: %.100s", transcribed)
                except Exception:
                    logger.warning(
                        "Failed to transcribe Discord audio attachment: %s", att.url
                    )

        payload = json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": str(message.author.id),
                "content": content,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "images": images,
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
            # Try guild channel first, fall back to DM
            channel_id = self._user_channels.get(platform_user_id)
            if channel_id:
                channel = self._bot.get_channel(channel_id)
                if channel:
                    await channel.send(content)
                    return

            # Fallback: DM the user
            user = await self._bot.fetch_user(int(platform_user_id))
            if user:
                await user.send(content)
        except Exception:
            logger.exception(
                "Failed to send message to Discord user %s",
                platform_user_id,
            )
