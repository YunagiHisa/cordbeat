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

from cordbeat.adapters._utils import AdapterFilter
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
_PENDING_SKILL_CONFIRM_MAX = 1_000
_MAX_IMAGES_PER_MESSAGE = 4
_IMAGE_SIZE_LIMIT_BYTES = 10 * 1024 * 1024
_AUDIO_SIZE_LIMIT_BYTES = 25 * 1024 * 1024
_DISCORD_MESSAGE_LIMIT = 2000
_REQUIRED_VOICE_PERMISSIONS = (
    ("view_channel", "View Channel"),
    ("connect", "Connect"),
)
_CORE_SLASH_COMMAND_NAMES = (
    "approve",
    "reject",
    "proposals",
    "link",
    "unlink",
    "name",
    "quiet",
    "prefer",
    "draw",
)


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
        self._pending_skill_confirms: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._filter = AdapterFilter.from_options(config.options)

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

        async def proposal_id_autocomplete(interaction: Any, current: str) -> list[Any]:
            choices = self._pending_proposal_choices(str(interaction.user.id), current)
            return [
                discord.app_commands.Choice(name=name, value=value)
                for name, value in choices
            ]

        @self._tree.command(
            name="approve",
            description="Approve a pending CordBeat proposal",
        )
        @discord.app_commands.describe(proposal_id="Pending proposal to approve")
        @discord.app_commands.autocomplete(proposal_id=proposal_id_autocomplete)
        async def approve_cmd(interaction: Any, proposal_id: str) -> None:
            await self._forward_core_command_interaction(
                interaction, f"/approve {proposal_id}"
            )

        @self._tree.command(
            name="reject",
            description="Reject a pending CordBeat proposal",
        )
        @discord.app_commands.describe(proposal_id="Pending proposal to reject")
        @discord.app_commands.autocomplete(proposal_id=proposal_id_autocomplete)
        async def reject_cmd(interaction: Any, proposal_id: str) -> None:
            await self._forward_core_command_interaction(
                interaction, f"/reject {proposal_id}"
            )

        @self._tree.command(
            name="proposals",
            description="List pending CordBeat proposals",
        )
        async def proposals_cmd(interaction: Any) -> None:
            await self._forward_core_command_interaction(interaction, "/proposals")

        @self._tree.command(
            name="link",
            description="Generate a CordBeat cross-platform link token",
        )
        async def link_cmd(interaction: Any) -> None:
            await self._forward_core_command_interaction(interaction, "/link")

        @self._tree.command(
            name="unlink",
            description="Unlink one platform from your CordBeat account",
        )
        @discord.app_commands.describe(platform="Platform to unlink, e.g. telegram")
        async def unlink_cmd(interaction: Any, platform: str) -> None:
            await self._forward_core_command_interaction(
                interaction, f"/unlink {platform}"
            )

        @self._tree.command(
            name="name",
            description="Update CordBeat's displayed character name",
        )
        @discord.app_commands.describe(name="New character name")
        async def name_cmd(interaction: Any, name: str) -> None:
            await self._forward_core_command_interaction(interaction, f"/name {name}")

        @self._tree.command(
            name="quiet",
            description="Set heartbeat quiet hours",
        )
        @discord.app_commands.describe(
            start="Quiet start time, e.g. 01:00",
            end="Quiet end time, e.g. 07:00",
        )
        async def quiet_cmd(interaction: Any, start: str, end: str) -> None:
            await self._forward_core_command_interaction(
                interaction, f"/quiet {start} {end}"
            )

        @self._tree.command(
            name="prefer",
            description="Set preferred platform for heartbeat replies",
        )
        @discord.app_commands.describe(platform="Platform name, or clear")
        async def prefer_cmd(interaction: Any, platform: str = "") -> None:
            command = f"/prefer {platform}".strip()
            await self._forward_core_command_interaction(interaction, command)

        @self._tree.command(
            name="draw",
            description="Run Draw DSL and return an image",
        )
        @discord.app_commands.describe(commands="Draw DSL commands")
        async def draw_cmd(interaction: Any, commands: str) -> None:
            await self._forward_core_command_interaction(
                interaction, f"/draw {commands}"
            )

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
                    # Clear any previously registered global commands to avoid
                    # duplicate slash commands when switching from global → guild sync
                    self._tree.clear_commands(guild=None)
                    await self._tree.sync()
                    logger.debug("Cleared global slash commands (guild_id is set)")
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

        # CommandTree is auto-registered with the Client in discord.py 2.x.
        # No manual on_interaction handler is needed — the tree processes slash
        # commands internally via Client._handle_interaction.

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

    async def _dispatch_core_message(
        self,
        platform_user_id: str,
        content: str,
        images: list[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        guild_id = self._vc_user_guild.get(platform_user_id)
        if guild_id is not None and guild_id in self._vc_receivers:
            if await self._speak_in_vc(guild_id, content):
                return
        await self._send_to_discord(
            platform_user_id, content, images, metadata=metadata
        )

    async def _dispatch_skill_confirm(
        self, platform_user_id: str, data: dict[str, Any]
    ) -> None:
        """Send a rich skill-confirmation embed with interactive buttons to Discord."""
        import discord  # noqa: PLC0415

        meta = data.get("metadata") or {}
        proposal_id: str = str(meta.get("proposal_id", ""))
        skill_name: str = str(meta.get("skill_name", "unknown"))
        skill_params: dict[str, Any] = meta.get("skill_params") or {}
        self._cache_pending_skill_confirm(
            platform_user_id=platform_user_id,
            proposal_id=proposal_id,
            skill_name=skill_name,
            skill_params=skill_params,
        )

        channel_id = self._user_channels.get(platform_user_id)
        if channel_id is None:
            logger.warning(
                "No cached channel for user %s; falling back to text", platform_user_id
            )
            content = data.get("content", "")
            await self._dispatch_core_message(platform_user_id, content, [])
            return

        if self._bot is None:
            return
        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except Exception:
                logger.warning("Cannot fetch channel %s for skill confirm", channel_id)
                return

        embed = discord.Embed(
            title="🔧 Skill Execution Required",
            description=(
                f"**`{skill_name}`** wants to run with the following parameters:"
            ),
            color=discord.Color.orange(),
        )
        if skill_params:
            params_text = "\n".join(
                f"• **{k}**: `{v}`" for k, v in skill_params.items()
            )
            embed.add_field(name="Parameters", value=params_text, inline=False)
        embed.set_footer(text=f"Proposal ID: {proposal_id}")

        ws_ref = self._ws

        adapter = self

        class SkillConfirmView(discord.ui.View):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__(timeout=300)

            @discord.ui.button(label="✅ Allow Once", style=discord.ButtonStyle.success)  # type: ignore[misc]
            async def allow_once(
                self,
                interaction: discord.Interaction,
                button: discord.ui.Button[Any],  # type: ignore[type-arg]
            ) -> None:
                if ws_ref is not None:
                    command = f"/approve {proposal_id}"
                    adapter._mark_proposal_action_sent(command)
                    await ws_ref.send(
                        adapter._core_command_payload(command, platform_user_id)
                    )
                await interaction.response.edit_message(
                    content=f"✅ Approved once: `{skill_name}`", embed=None, view=None
                )
                self.stop()

            @discord.ui.button(label="❌ Deny", style=discord.ButtonStyle.danger)  # type: ignore[misc]
            async def deny(
                self,
                interaction: discord.Interaction,
                button: discord.ui.Button[Any],  # type: ignore[type-arg]
            ) -> None:
                if ws_ref is not None:
                    command = f"/reject {proposal_id}"
                    adapter._mark_proposal_action_sent(command)
                    await ws_ref.send(
                        adapter._core_command_payload(command, platform_user_id)
                    )
                await interaction.response.edit_message(
                    content=f"❌ Denied: `{skill_name}`", embed=None, view=None
                )
                self.stop()

        try:
            await channel.send(embed=embed, view=SkillConfirmView())
        except Exception:
            logger.warning(
                "Failed to send skill-confirm embed; falling back to text",
                exc_info=True,
            )
            content = data.get("content", "")
            await self._dispatch_core_message(platform_user_id, content, [])

    async def _forward_to_core(self, message: Any) -> None:
        if self._ws is None:
            logger.warning("Not connected to Core, dropping message")
            return

        user_id = str(message.author.id)
        channel_id: int = message.channel.id
        is_guild = message.guild is not None  # False = DM

        # ── E-4 response filtering ────────────────────────────────────
        bot_mentioned = (
            is_guild
            and self._bot is not None
            and self._bot.user is not None
            and self._bot.user.mentioned_in(message)
        )
        bot_name: str = (
            (self._bot.user.display_name or self._bot.user.name or "")
            if self._bot is not None and self._bot.user is not None
            else ""
        )
        if not await self._filter.should_respond_async(
            user_id=user_id,
            channel_id=str(channel_id),
            is_dm=not is_guild,
            is_mentioned=bot_mentioned,
            text=message.content or "",
            extra_keywords=[bot_name] if bot_name else None,
        ):
            return
        # ─────────────────────────────────────────────────────────────

        self._remember_user_channel(user_id, channel_id)

        # Show typing indicator while core is processing
        self._start_typing(channel_id, message.channel)

        async def _fetch_attachment_images(src_msg: Any, buf: list[str]) -> None:
            for att in getattr(src_msg, "attachments", []):
                if len(buf) >= _MAX_IMAGES_PER_MESSAGE:
                    break
                ct = getattr(att, "content_type", "") or ""
                if not ct.startswith("image/"):
                    continue
                size = getattr(att, "size", 0) or 0
                if size > _IMAGE_SIZE_LIMIT_BYTES:
                    logger.warning(
                        "Skipping oversized Discord image attachment: %d bytes", size
                    )
                    continue
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(str(att.url))
                        resp.raise_for_status()
                        buf.append(base64.b64encode(resp.content).decode("ascii"))
                        logger.debug(
                            "Downloaded Discord image attachment: %s (%d bytes)",
                            getattr(att, "filename", att.url),
                            len(resp.content),
                        )
                except Exception:
                    logger.warning(
                        "Failed to download Discord image attachment: %s", att.url
                    )

        images: list[str] = []
        await _fetch_attachment_images(message, images)

        ref = getattr(message, "reference", None)
        if ref is not None and len(images) < _MAX_IMAGES_PER_MESSAGE:
            ref_msg = getattr(ref, "resolved", None)
            if ref_msg is not None:
                await _fetch_attachment_images(ref_msg, images)

        content = message.content
        is_voice = False
        if self._stt is not None:
            for att in getattr(message, "attachments", []):
                ct = getattr(att, "content_type", "") or ""
                if not ct.startswith("audio/"):
                    continue
                size = getattr(att, "size", 0) or 0
                if size > _AUDIO_SIZE_LIMIT_BYTES:
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
                        is_voice = True
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
                "is_voice": is_voice,
                "metadata": {
                    "channel_id": str(message.channel.id),
                    "guild_id": str(message.guild.id) if message.guild else "",
                    "is_dm": message.guild is None,
                    "display_name": message.author.display_name,
                },
            }
        )
        try:
            logger.debug(
                "Forwarding Discord message to Core: payload=%d bytes, images=%d",
                len(payload.encode("utf-8")),
                len(images),
            )
            await self._ws.send(payload)
        except Exception:
            logger.exception("Failed to forward message to Core")

    def _remember_user_channel(self, platform_user_id: str, channel_id: int) -> None:
        self._user_channels[platform_user_id] = channel_id
        self._user_channels.move_to_end(platform_user_id)
        if len(self._user_channels) > _USER_CHANNEL_CACHE_MAX:
            self._user_channels.popitem(last=False)

    def _cache_pending_skill_confirm(
        self,
        *,
        platform_user_id: str,
        proposal_id: str,
        skill_name: str,
        skill_params: dict[str, Any],
    ) -> None:
        if not proposal_id:
            return
        self._pending_skill_confirms[proposal_id] = {
            "platform_user_id": platform_user_id,
            "skill_name": skill_name,
            "skill_params": skill_params,
        }
        self._pending_skill_confirms.move_to_end(proposal_id)
        if len(self._pending_skill_confirms) > _PENDING_SKILL_CONFIRM_MAX:
            self._pending_skill_confirms.popitem(last=False)

    def _pending_proposal_choices(
        self, platform_user_id: str, current: str = ""
    ) -> list[tuple[str, str]]:
        current_lower = current.lower().strip()
        choices: list[tuple[str, str]] = []
        for proposal_id, meta in reversed(self._pending_skill_confirms.items()):
            if meta.get("platform_user_id") != platform_user_id:
                continue
            skill_name = str(meta.get("skill_name") or "unknown")
            haystack = f"{proposal_id} {skill_name}".lower()
            if current_lower and current_lower not in haystack:
                continue
            label = f"{proposal_id[:8]}… {skill_name}"
            choices.append((label[:100], proposal_id))
            if len(choices) >= 25:
                break
        return choices

    def _mark_proposal_action_sent(self, command: str) -> None:
        parts = command.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() in {"/approve", "/reject"}:
            self._pending_skill_confirms.pop(parts[1].strip(), None)

    def _core_command_payload(self, command: str, platform_user_id: str) -> str:
        return json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": platform_user_id,
                "content": command,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
        )

    async def _forward_core_command_interaction(
        self, interaction: Any, command: str
    ) -> None:
        if self._ws is None:
            await interaction.response.send_message(
                "CordBeat Core is not connected.", ephemeral=True
            )
            return

        platform_user_id = str(interaction.user.id)
        channel_id = getattr(interaction, "channel_id", None)
        if channel_id is not None:
            self._remember_user_channel(platform_user_id, int(channel_id))

        self._mark_proposal_action_sent(command)
        await self._ws.send(self._core_command_payload(command, platform_user_id))
        await interaction.response.send_message(
            "CordBeatにコマンドを送ったよ。", ephemeral=True
        )

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
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._bot or not platform_user_id:
            return

        decoded_images: list[tuple[bytes, str]] = []
        if images:
            try:
                for idx, b64 in enumerate(images):
                    raw = base64.b64decode(b64)
                    decoded_images.append((raw, f"draw_{idx + 1}.png"))
            except Exception:
                logger.exception("Failed to decode images for Discord")

        def _make_discord_files() -> list[Any]:
            if not decoded_images:
                return []
            import discord  # noqa: PLC0415

            return [
                discord.File(io.BytesIO(raw), filename=filename)
                for raw, filename in decoded_images
            ]

        # Discord 400s on empty string content; use None to allow files-only messages.
        # Discord enforces a 2000-character limit per message — split long content.
        chunks: list[str] = []
        if content:
            remaining = content
            while len(remaining) > _DISCORD_MESSAGE_LIMIT:
                # Prefer splitting on a newline boundary within the limit
                split_at = remaining.rfind("\n", 0, _DISCORD_MESSAGE_LIMIT)
                if split_at <= 0:
                    split_at = remaining.rfind(" ", 0, _DISCORD_MESSAGE_LIMIT)
                if split_at <= 0:
                    split_at = _DISCORD_MESSAGE_LIMIT
                chunks.append(remaining[:split_at])
                remaining = remaining[split_at:].lstrip("\n")
            if remaining:
                chunks.append(remaining)

        if not chunks and not decoded_images:
            return

        async def _send_chunks(target: Any) -> None:
            if chunks:
                for i, chunk in enumerate(chunks):
                    # Attach files to the last chunk (or first if no text).
                    files = _make_discord_files() if i == len(chunks) - 1 else []
                    await target.send(chunk, files=files)
                return
            await target.send(None, files=_make_discord_files())

        # Routing precedence:
        #   1. Core-supplied metadata.channel_id (Heartbeat consults
        #      ``user_channels`` table and pins the channel explicitly).
        #   2. In-memory cache populated by recent inbound messages.
        #   3. DM fallback (only when ``allow_dm_fallback`` is true; the
        #      Heartbeat sets it to false to honour ``dm_policy``).
        hinted: int | None = None
        if metadata:
            raw_cid = metadata.get("channel_id")
            if raw_cid:
                try:
                    hinted = int(raw_cid)
                except (TypeError, ValueError):
                    hinted = None
        cached = self._user_channels.get(platform_user_id)
        channel_id = hinted or cached
        allow_dm_fallback = True
        if metadata is not None:
            if "allow_dm_fallback" in metadata:
                allow_dm_fallback = bool(metadata.get("allow_dm_fallback"))
            elif metadata.get("is_dm") is False:
                allow_dm_fallback = False

        if channel_id:
            try:
                self._stop_typing(channel_id)
                # get_channel() only checks the cache; fall back to fetch_channel()
                # to reliably reach channels not in the bot's in-memory cache.
                channel = self._bot.get_channel(
                    channel_id
                ) or await self._bot.fetch_channel(channel_id)
                if channel:
                    await _send_chunks(channel)
                    return
            except Exception:
                if not hinted and cached == channel_id:
                    self._user_channels.pop(platform_user_id, None)
                if not allow_dm_fallback:
                    logger.exception(
                        "Failed to send Discord channel message for user %s; "
                        "DM fallback disabled",
                        platform_user_id,
                    )
                    return
                logger.warning(
                    "Failed to send Discord channel message for user %s; "
                    "trying DM fallback",
                    platform_user_id,
                    exc_info=True,
                )

        if not allow_dm_fallback:
            logger.info(
                "No known channel for user %s and DM fallback disabled; skipping send",
                platform_user_id,
            )
            return

        try:
            user = await self._bot.fetch_user(int(platform_user_id))
            if user:
                await _send_chunks(user)
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
                "is_voice": True,
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

    async def _speak_in_vc(self, guild_id: int, text: str) -> bool:
        """Synthesise *text* to audio and play it in the guild's voice channel."""
        if not self._tts or not self._bot:
            return False

        guild = self._bot.get_guild(guild_id)
        if guild is None:
            return False

        vc = guild.voice_client
        if not vc or not vc.is_connected():
            return False

        try:
            audio = await self._tts.synthesize(text)
        except Exception:
            logger.exception("TTS synthesis failed for VC guild %d", guild_id)
            return False

        if not audio:
            return False

        try:
            import discord

            source = discord.FFmpegPCMAudio(io.BytesIO(audio), pipe=True)
            if vc.is_playing():
                vc.stop()
            vc.play(source)
            return True
        except Exception:
            logger.exception("Failed to play TTS audio in VC guild %d", guild_id)
            return False

    def _voice_join_message(self, channel_name: str) -> str:
        if self._stt is None:
            return (
                f"⚠️ Joined **{channel_name}**, but STT is disabled. "
                "I can receive audio packets but cannot understand speech."
            )
        if self._tts is None:
            return (
                f"✅ Joined **{channel_name}**. Listening… "
                "TTS is disabled, so replies will be sent as text."
            )
        return f"✅ Joined **{channel_name}**. Listening and ready to talk."

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
        guild = getattr(interaction, "guild", None)
        bot_member = getattr(guild, "me", None)
        if bot_member is not None:
            permissions = channel.permissions_for(bot_member)
            missing_permissions = [
                label
                for attribute, label in _REQUIRED_VOICE_PERMISSIONS
                if not getattr(permissions, attribute, False)
            ]
            if missing_permissions:
                missing = ", ".join(f"**{name}**" for name in missing_permissions)
                await interaction.response.send_message(
                    f"❌ I need the {missing} permission(s) in **{channel.name}**.",
                    ephemeral=True,
                )
                return

        await interaction.response.defer(ephemeral=True, thinking=True)

        try:
            from discord.ext.voice_recv import (
                VoiceRecvClient,  # type: ignore[import-not-found]
            )
        except ImportError:
            await interaction.followup.send(
                "❌ Voice receive support is not installed "
                "(run `uv sync --extra discord` to reinstall with voice support).",
                ephemeral=True,
            )
            return

        from cordbeat.voice_recv import VoiceReceiver

        try:
            vc = await channel.connect(cls=VoiceRecvClient)
        except TimeoutError:
            logger.exception("Timed out connecting to voice channel %s", channel)
            await interaction.followup.send(
                "❌ Timed out joining the voice channel. Check that the bot has "
                "**View Channel** and **Connect** permissions.",
                ephemeral=True,
            )
            return
        except Exception:
            logger.exception("Failed to connect to voice channel %s", channel)
            await interaction.followup.send(
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

        await interaction.followup.send(
            self._voice_join_message(channel.name), ephemeral=True
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
