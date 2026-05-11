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

from cordbeat.adapters._utils import _read_soul_keywords
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
        self._auth_token = config.auth_token
        self._token: str = config.options.get("token", "")
        self._bot: Any = None
        self._ws: Any = None
        self._running = False
        self._max_backoff = config.reconnect_max_backoff
        self._user_channels: OrderedDict[str, int] = OrderedDict()

        # Typing indicator tasks: channel_id → asyncio.Task
        self._typing_tasks: dict[int, asyncio.Task[None]] = {}

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
                guild_id = self._config.options.get("guild_id")
                if guild_id:
                    guild = discord.Object(id=int(guild_id))
                    # Copy global commands to this guild for instant sync
                    self._tree.copy_global_to(guild=guild)
                    synced = await self._tree.sync(guild=guild)
                    logger.info(
                        "Slash commands synced to guild %s (%d command(s))",
                        guild_id,
                        len(synced),
                    )
                else:
                    synced = await self._tree.sync()
                    logger.info(
                        "Slash commands synced globally (%d command(s)) — "
                        "may take up to 1 hour to propagate.  "
                        "Set adapters.discord.options.guild_id for instant dev sync.",
                        len(synced),
                    )
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

    def _matches_ai_keywords(self, message: Any) -> bool:
        """Return True if the message content contains any configured keyword.

        Used for ``respond_mode: ai_decision``.  Keywords are case-insensitive.
        Priority: 1) ``ai_decision_keywords`` in config, 2) AI name from soul.yaml,
        3) bot's Discord display name as last resort.
        """
        opts = self._config.options
        keywords: list[str] = [str(k) for k in opts.get("ai_decision_keywords", [])]
        if not keywords:
            keywords = _read_soul_keywords()
        if not keywords and self._bot is not None and self._bot.user is not None:
            name: str = self._bot.user.display_name or self._bot.user.name or ""
            if name:
                keywords = [name]
        if not keywords:
            return True  # no filter available → allow
        text = (message.content or "").lower()
        return any(kw.lower() in text for kw in keywords)

    async def _dispatch_core_message(
        self, platform_user_id: str, content: str, images: list[str]
    ) -> None:
        guild_id = self._vc_user_guild.get(platform_user_id)
        if guild_id is not None and guild_id in self._vc_receivers:
            await self._speak_in_vc(guild_id, content)
            return
        await self._send_to_discord(platform_user_id, content, images)

    async def _forward_to_core(self, message: Any) -> None:
        if self._ws is None:
            logger.warning("Not connected to Core, dropping message")
            return

        user_id = str(message.author.id)
        channel_id: int = message.channel.id

        # ── E-4 response filtering ────────────────────────────────────
        opts = self._config.options
        user_blocklist: list[str] = [str(u) for u in opts.get("user_blocklist", [])]
        if user_id in user_blocklist:
            return

        is_guild_channel = message.guild is not None  # False = DM
        channel_whitelist: list[int] = [
            int(c) for c in opts.get("channel_whitelist", [])
        ]
        channel_blacklist: list[int] = [
            int(c) for c in opts.get("channel_blacklist", [])
        ]
        # channel_whitelist/blacklist apply only to guild channels; DMs always bypass.
        if is_guild_channel:
            if channel_whitelist and channel_id not in channel_whitelist:
                return
            if channel_id in channel_blacklist:
                return

        respond_mode: str = opts.get("respond_mode", "all")
        if respond_mode == "mention_only":
            bot_mentioned = (
                self._bot is not None
                and self._bot.user is not None
                and self._bot.user.mentioned_in(message)
            )
            if is_guild_channel and not bot_mentioned:
                return
        elif respond_mode == "ai_decision":
            if is_guild_channel and not self._matches_ai_keywords(message):
                return
        # ─────────────────────────────────────────────────────────────

        self._user_channels[user_id] = channel_id
        self._user_channels.move_to_end(user_id)
        if len(self._user_channels) > _USER_CHANNEL_CACHE_MAX:
            self._user_channels.popitem(last=False)

        # Show typing indicator while core is processing
        self._start_typing(channel_id, message.channel)

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

    def _start_typing(self, channel_id: int, channel: Any) -> None:
        """Start a background typing indicator loop for the given channel."""
        existing = self._typing_tasks.pop(channel_id, None)
        if existing is not None:
            existing.cancel()
        self._typing_tasks[channel_id] = asyncio.create_task(self._keep_typing(channel))

    def _stop_typing(self, channel_id: int) -> None:
        """Cancel the typing indicator for the given channel."""
        task = self._typing_tasks.pop(channel_id, None)
        if task is not None:
            task.cancel()

    @staticmethod
    async def _keep_typing(channel: Any) -> None:
        """Show Discord typing indicator until this task is cancelled.

        Uses ``channel.typing()`` context manager (discord.py v2 API).
        The context manager handles repeated sends internally; we just wait
        inside it until the task is cancelled by ``_stop_typing``.
        """
        try:
            async with channel.typing():
                # Wait indefinitely — cancelled by _stop_typing() when reply
                # is ready, which propagates CancelledError out of here.
                await asyncio.get_event_loop().create_future()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug(
                "typing indicator failed for channel %s",
                getattr(channel, "id", channel),
                exc_info=True,
            )

    async def _send_to_discord(
        self,
        platform_user_id: str,
        content: str,
        images: list[str] | None = None,
    ) -> None:
        if not self._bot or not platform_user_id:
            return

        discord_files: list[Any] = []
        if images:
            try:
                import base64  # noqa: PLC0415
                import io as _io  # noqa: PLC0415

                import discord  # noqa: PLC0415

                for idx, b64 in enumerate(images):
                    raw = base64.b64decode(b64)
                    discord_files.append(
                        discord.File(_io.BytesIO(raw), filename=f"draw_{idx + 1}.png")
                    )
            except Exception:
                logger.exception("Failed to decode images for Discord")

        # Discord 400s on empty string content; use None to allow files-only messages.
        # Discord enforces a 2000-character limit per message — split long content.
        discord_limit = 2000
        chunks: list[str] = []
        if content:
            remaining = content
            while len(remaining) > discord_limit:
                # Prefer splitting on a newline boundary within the limit
                split_at = remaining.rfind("\n", 0, discord_limit)
                if split_at <= 0:
                    split_at = remaining.rfind(" ", 0, discord_limit)
                if split_at <= 0:
                    split_at = discord_limit
                chunks.append(remaining[:split_at])
                remaining = remaining[split_at:].lstrip("\n")
            if remaining:
                chunks.append(remaining)

        if not chunks and not discord_files:
            return

        try:
            channel_id = self._user_channels.get(platform_user_id)
            if channel_id:
                self._stop_typing(channel_id)
                # get_channel() only checks the cache; fall back to fetch_channel()
                # to reliably reach channels not in the bot's in-memory cache.
                channel = self._bot.get_channel(
                    channel_id
                ) or await self._bot.fetch_channel(channel_id)
                if channel:
                    for i, chunk in enumerate(chunks):
                        # Attach files to the last chunk (or first if no text)
                        files = discord_files if i == len(chunks) - 1 else []
                        await channel.send(chunk, files=files)
                    if not chunks:
                        await channel.send(None, files=discord_files)
                    return

            # No channel mapping found → fall back to DM
            user = await self._bot.fetch_user(int(platform_user_id))
            if user:
                for i, chunk in enumerate(chunks):
                    files = discord_files if i == len(chunks) - 1 else []
                    await user.send(chunk, files=files)
                if not chunks:
                    await user.send(None, files=discord_files)
        except Exception:
            logger.exception(
                "Failed to send message to Discord user %s",
                platform_user_id,
            )

    # ── Voice channel helpers ────────────────────────────────────────────────

    async def _on_vc_speech(self, guild_id: int, user_id: int, wav_data: bytes) -> None:
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
