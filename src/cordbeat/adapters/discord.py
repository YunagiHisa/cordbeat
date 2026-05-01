"""Discord adapter — bridges Discord bot to CordBeat Core via WebSocket."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from collections import OrderedDict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from cordbeat.config import AdapterConfig, RVCConfig, STTConfig, TTSConfig
from cordbeat.core.gateway import RetryableConnection

if TYPE_CHECKING:
    from cordbeat.ai.stt import STTBackend
    from cordbeat.ai.tts import TTSBackend

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
        rvc_config: RVCConfig | None = None,
    ) -> None:
        self._config = config
        self._ws_url = config.core_ws_url
        self._token: str = config.options.get("token", "")
        self._bot: Any = None
        self._ws: Any = None
        self._running = False
        self._max_backoff = config.reconnect_max_backoff
        self._user_channels: OrderedDict[str, int] = OrderedDict()

        # Voice channel state
        self._vc_receivers: dict[int, Any] = {}
        self._vc_muted: set[int] = set()
        self._vc_user_guild: dict[str, int] = {}

        self._stt: STTBackend | None = None
        self._tts: TTSBackend | None = None

        if stt_config is not None and stt_config.enabled:
            from cordbeat.ai.stt import create_stt_backend

            self._stt = create_stt_backend(stt_config)

        if tts_config is not None and tts_config.enabled:
            from cordbeat.ai.tts import create_tts_with_rvc

            self._tts = create_tts_with_rvc(tts_config, rvc_config)

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
        intents.voice_states = True
        self._bot = discord.Client(intents=intents)
        self._running = True

        self._tree = discord.app_commands.CommandTree(self._bot)

        @self._tree.command(name="join", description="Join your current voice channel")
        async def join_cmd(interaction: Any) -> None:
            await self._handle_join(interaction)

        @self._tree.command(name="leave", description="Leave the voice channel")
        async def leave_cmd(interaction: Any) -> None:
            await self._handle_leave(interaction)

        @self._tree.command(name="mute", description="Toggle voice mute on/off")
        async def mute_cmd(interaction: Any) -> None:
            await self._handle_mute(interaction)

        @self._bot.event
        async def on_ready() -> None:
            logger.info("Discord bot logged in as %s", self._bot.user)
            try:
                await self._tree.sync()
                logger.info("Slash commands synced globally")
            except Exception:
                logger.exception("Failed to sync slash commands")
            asyncio.create_task(self._connect_to_core())

        @self._bot.event
        async def on_interaction(interaction: Any) -> None:
            try:
                await self._tree.on_interaction(interaction)
            except Exception:
                logger.exception("Interaction handling error")

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
        guild_id = self._vc_user_guild.get(platform_user_id)
        if guild_id is not None and guild_id in self._vc_receivers:
            await self._speak_in_vc(guild_id, content)
            return
        await self._send_to_discord(platform_user_id, content)

    async def _forward_to_core(self, message: Any) -> None:
        if self._ws is None:
            logger.warning("Not connected to Core, dropping message")
            return

        user_id = str(message.author.id)
        channel_id = message.channel.id

        self._user_channels[user_id] = channel_id
        self._user_channels.move_to_end(user_id)
        if len(self._user_channels) > _USER_CHANNEL_CACHE_MAX:
            self._user_channels.popitem(last=False)

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

        ref = getattr(message, "reference", None)
        if ref is not None and len(images) < _max_images:
            ref_msg = getattr(ref, "resolved", None)
            if ref_msg is not None:
                await _fetch_attachment_images(ref_msg, images)

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
            channel_id = self._user_channels.get(platform_user_id)
            if channel_id:
                channel = self._bot.get_channel(channel_id)
                if channel:
                    await channel.send(content)
                    return

            user = await self._bot.fetch_user(int(platform_user_id))
            if user:
                await user.send(content)
        except Exception:
            logger.exception(
                "Failed to send message to Discord user %s",
                platform_user_id,
            )

    # ── Voice channel helpers ────────────────────────────────────────────────

    async def _on_vc_speech(
        self, guild_id: int, user_id: int, wav_data: bytes
    ) -> None:
        """Called by VoiceReceiver when a user finishes speaking."""
        if guild_id in self._vc_muted:
            return
        if self._ws is None:
            logger.warning(
                "Not connected to Core; dropping VC speech from user %d", user_id
            )
            return
        if self._stt is None:
            logger.debug("STT not configured; ignoring VC speech")
            return

        try:
            transcribed = await self._stt.transcribe(wav_data)
        except Exception:
            logger.exception("STT transcription error for VC user %d", user_id)
            return

        if not transcribed or not transcribed.strip():
            return

        self._vc_user_guild[str(user_id)] = guild_id

        payload = json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": str(user_id),
                "content": transcribed,
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": {
                    "guild_id": str(guild_id),
                    "channel_id": "vc",
                    "via_vc": True,
                },
            }
        )
        try:
            await self._ws.send(payload)
        except Exception:
            logger.exception("Failed to forward VC speech to Core")

    async def _speak_in_vc(self, guild_id: int, text: str) -> None:
        """Synthesise *text* to audio and play it in the guild's voice channel."""
        if not self._tts or not self._bot:
            return

        guild = self._bot.get_guild(guild_id)
        if guild is None:
            return

        vc = guild.voice_client
        if not vc or not vc.is_connected():
            return

        try:
            audio = await self._tts.synthesize(text)
        except Exception:
            logger.exception("TTS synthesis failed for VC guild %d", guild_id)
            return

        if not audio:
            return

        try:
            import discord

            source = discord.FFmpegPCMAudio(io.BytesIO(audio), pipe=True)
            if vc.is_playing():
                vc.stop()
            vc.play(source)
        except Exception:
            logger.exception("Failed to play TTS audio in VC guild %d", guild_id)

    async def _handle_join(self, interaction: Any) -> None:
        """Slash command: join the user's voice channel."""
        user = interaction.user
        voice_state = getattr(user, "voice", None)
        if voice_state is None or voice_state.channel is None:
            await interaction.response.send_message(
                "❌ You must be in a voice channel first.", ephemeral=True
            )
            return

        channel = voice_state.channel
        guild_id: int = interaction.guild_id

        try:
            from discord.ext.voice_recv import (
                VoiceRecvClient,  # type: ignore[import-not-found]
            )
        except ImportError:
            await interaction.response.send_message(
                "❌ Voice receive support is not installed "
                "(run `uv sync --extra discord-vc`).",
                ephemeral=True,
            )
            return

        from cordbeat.voice_recv import VoiceReceiver

        try:
            vc = await channel.connect(cls=VoiceRecvClient)
        except Exception:
            logger.exception("Failed to connect to voice channel %s", channel)
            await interaction.response.send_message(
                "❌ Failed to join the voice channel.", ephemeral=True
            )
            return

        receiver = VoiceReceiver(vc)
        receiver.on_speech_end(
            lambda uid, wav: asyncio.ensure_future(
                self._on_vc_speech(guild_id, uid, wav)
            )
        )
        await receiver.start()
        self._vc_receivers[guild_id] = receiver

        await interaction.response.send_message(
            f"✅ Joined **{channel.name}**. Listening…", ephemeral=True
        )
        logger.info("Joined VC guild=%d channel=%s", guild_id, channel.name)

    async def _handle_leave(self, interaction: Any) -> None:
        """Slash command: leave the voice channel."""
        guild_id: int = interaction.guild_id
        receiver = self._vc_receivers.pop(guild_id, None)
        if receiver is not None:
            await receiver.stop()

        guild = self._bot.get_guild(guild_id) if self._bot else None
        if guild and guild.voice_client:
            await guild.voice_client.disconnect()

        stale = [uid for uid, gid in self._vc_user_guild.items() if gid == guild_id]
        for uid in stale:
            del self._vc_user_guild[uid]
        self._vc_muted.discard(guild_id)

        await interaction.response.send_message(
            "👋 Left the voice channel.", ephemeral=True
        )
        logger.info("Left VC guild=%d", guild_id)

    async def _handle_mute(self, interaction: Any) -> None:
        """Slash command: toggle voice mute."""
        guild_id: int = interaction.guild_id
        if guild_id in self._vc_muted:
            self._vc_muted.discard(guild_id)
            await interaction.response.send_message("🔊 Unmuted.", ephemeral=True)
        else:
            self._vc_muted.add(guild_id)
            await interaction.response.send_message("🔇 Muted.", ephemeral=True)
