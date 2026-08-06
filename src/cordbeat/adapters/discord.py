"""Discord adapter — bridges Discord bot to CordBeat Core via WebSocket."""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
import logging
import re
import unicodedata
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import TYPE_CHECKING, Any

import httpx

from cordbeat.adapters._utils import (
    AdapterFilter,
    get_judge_backend,
    judge_yes_no,
    split_message,
)
from cordbeat.ai.prompt import format_skill_params_for_display
from cordbeat.ai.speech import SpeechStyle
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
# One video per message, and a limit well under the gateway's frame cap:
# base64 inflates by 4/3, so 20 MiB of video is ~27 MiB on the wire.
_MAX_VIDEOS_PER_MESSAGE = 1
_VIDEO_SIZE_LIMIT_BYTES = 20 * 1024 * 1024
_DISCORD_MESSAGE_LIMIT = 2000
_VC_BUFFER_MAX_FRAGMENTS = 5
_VC_CONTEXT_MAX_LINES_DEFAULT = 8
_VC_FOLLOWUP_SECONDS_DEFAULT = 20.0
_VC_PENDING_TIMEOUT_SECONDS_DEFAULT = 120.0
_VC_BARGE_IN_SPEECH_SECONDS_DEFAULT = 0.24
_VC_BARGE_IN_RMS_THRESHOLD_DEFAULT = 50.0
_VC_DEFERRED_HEARTBEAT_MAX = 10
_VC_WAKE_WORDS_DEFAULT = ("cordbeat",)
_VC_ACTIVATION_MODES = frozenset({"always", "hybrid", "wake_phrase"})
_SKILL_CONFIRM_APPROVE_CUSTOM_ID = "cordbeat:skill_confirm:approve"
_SKILL_CONFIRM_DENY_CUSTOM_ID = "cordbeat:skill_confirm:deny"
_PROPOSAL_ID_FOOTER_PREFIX = "Proposal ID:"
_APPROVE_COMMAND_RE = re.compile(r"(?:^|\s)/approve\s+([^\s`]+)")
_REQUIRED_VOICE_PERMISSIONS = (
    ("view_channel", "View Channel"),
    ("connect", "Connect"),
)


@dataclass
class _VCChunkSession:
    generation_id: str
    queue: asyncio.Queue[bytes | None]
    runner: asyncio.Task[None] | None = None
    producer: asyncio.Task[None] | None = None
_CORE_SLASH_COMMAND_NAMES = (
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
)


def _normalize_vc_wake_text(text: str) -> str:
    """Normalize common STT spelling variations for wake-word matching."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        chr(ord(char) - 0x60) if "\u30a1" <= char <= "\u30f6" else char
        for char in normalized
    )
    return "".join(char for char in normalized if char.isalnum())


def _bounded_float_option(
    options: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
) -> float:
    try:
        return max(minimum, float(options.get(key, default)))
    except (TypeError, ValueError):
        logger.warning("Invalid Discord option %s; using %s", key, default)
        return default


def _bounded_int_option(
    options: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
) -> int:
    try:
        return max(minimum, int(options.get(key, default)))
    except (TypeError, ValueError):
        logger.warning("Invalid Discord option %s; using %s", key, default)
        return default


def _bool_option(
    options: dict[str, Any],
    key: str,
    default: bool,
) -> bool:
    """Read a boolean adapter option without treating ``"false"`` as true."""

    value = options.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    logger.warning("Invalid Discord option %s; using %s", key, default)
    return default


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
        soul_name: str = "",
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
        self._skill_confirm_view_registered = False
        self._filter = AdapterFilter.from_options(config.options)

        # Typing indicator tasks: channel_id → asyncio.Task
        self._typing_tasks: dict[int, asyncio.Task[None]] = {}

        # Voice channel state
        self._vc_receivers: dict[int, Any] = {}
        self._vc_muted: set[int] = set()
        self._vc_user_guild: dict[str, int] = {}
        self._vc_pending_guilds: set[int] = set()
        self._vc_pending_since: dict[int, float] = {}
        self._vc_buffered_speech: dict[int, list[str]] = {}
        self._vc_room_context: dict[int, deque[str]] = {}
        self._vc_followup_until: dict[int, float] = {}
        self._vc_session_ids: dict[int, str] = {}
        self._vc_pause_heartbeat = _bool_option(
            config.options,
            "vc_pause_heartbeat",
            True,
        )
        self._vc_barge_in = _bool_option(config.options, "vc_barge_in", True)
        self._vc_barge_in_speech_seconds = _bounded_float_option(
            config.options,
            "vc_barge_in_speech_seconds",
            _VC_BARGE_IN_SPEECH_SECONDS_DEFAULT,
            minimum=0.02,
        )
        self._vc_barge_in_rms_threshold = _bounded_float_option(
            config.options,
            "vc_barge_in_rms_threshold",
            _VC_BARGE_IN_RMS_THRESHOLD_DEFAULT,
            minimum=0.0,
        )
        self._vc_deferred_heartbeats: list[dict[str, Any]] = []
        self._vc_pending_speech: str | None = None
        self._vc_pending_speech_guild_id: int | None = None
        # Strong reference so the replay task cannot be garbage-collected
        # mid-flight; only one pending replay exists at a time.
        self._vc_pending_speech_task: asyncio.Task[bool] | None = None
        self._vc_chunk_sessions: dict[int, _VCChunkSession] = {}
        self._vc_chunk_playback_locks: dict[int, asyncio.Lock] = {}
        self._vc_participation_judge_lock = asyncio.Lock()
        raw_activation_mode = str(
            config.options.get("vc_activation_mode", "hybrid")
        ).strip()
        self._vc_activation_mode = (
            raw_activation_mode
            if raw_activation_mode in _VC_ACTIVATION_MODES
            else "hybrid"
        )
        if self._vc_activation_mode != raw_activation_mode:
            logger.warning(
                "Invalid Discord option vc_activation_mode=%r; using hybrid",
                raw_activation_mode,
            )
        raw_wake_words = config.options.get(
            "vc_activation_phrases",
            config.options.get("vc_wake_words", _VC_WAKE_WORDS_DEFAULT),
        )
        if isinstance(raw_wake_words, str):
            raw_wake_words = [raw_wake_words]
        elif not isinstance(raw_wake_words, list | tuple | set):
            logger.warning("Invalid Discord option vc_wake_words; using defaults")
            raw_wake_words = _VC_WAKE_WORDS_DEFAULT
        configured_phrases = tuple(
            str(word).strip().casefold() for word in raw_wake_words if str(word).strip()
        )
        soul_phrase = soul_name.strip().casefold()
        activation_phrases = (
            (*configured_phrases, soul_phrase) if soul_phrase else configured_phrases
        )
        self._vc_wake_words = (
            tuple(dict.fromkeys(activation_phrases)) or _VC_WAKE_WORDS_DEFAULT
        )
        self._vc_normalized_wake_words = tuple(
            normalized
            for word in self._vc_wake_words
            if (normalized := _normalize_vc_wake_text(word))
        )
        self._vc_followup_seconds = _bounded_float_option(
            config.options,
            "vc_followup_seconds",
            _VC_FOLLOWUP_SECONDS_DEFAULT,
            minimum=0.0,
        )
        self._vc_context_max_lines = _bounded_int_option(
            config.options,
            "vc_context_max_lines",
            _VC_CONTEXT_MAX_LINES_DEFAULT,
            minimum=1,
        )
        self._vc_pending_timeout_seconds = _bounded_float_option(
            config.options,
            "vc_pending_timeout_seconds",
            _VC_PENDING_TIMEOUT_SECONDS_DEFAULT,
            minimum=1.0,
        )

        self._stt: STTBackend | None = None
        self._tts: TTSBackend | None = None
        self._stt_preload_task: asyncio.Future[None] | None = None

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

        # Warm up the local STT model in the background so the first VC
        # utterance is not stalled behind a multi-gigabyte model download.
        if self._stt is not None:
            self._stt_preload_task = asyncio.ensure_future(self._stt.preload())
        if self._tts is not None:
            preload = getattr(self._tts, "preload", None)
            if callable(preload):
                await preload()

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
            name="link-confirm",
            description="Confirm a CordBeat cross-platform link token",
        )
        @discord.app_commands.describe(token="Link token generated on another platform")
        async def link_confirm_cmd(interaction: Any, token: str) -> None:
            await self._forward_core_command_interaction(
                interaction, f"/link-confirm {token}"
            )

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
            if not self._skill_confirm_view_registered:
                try:
                    self._bot.add_view(self._make_skill_confirm_view(discord))
                    self._skill_confirm_view_registered = True
                    logger.info("Registered persistent Discord skill-confirm view")
                except Exception:
                    logger.warning(
                        "Failed to register persistent skill-confirm view",
                        exc_info=True,
                    )
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

        @self._bot.event
        async def on_message(message: discord.Message) -> None:
            if message.author == self._bot.user:
                return
            if message.author.bot:
                return
            await self._forward_to_core(message)

        @self._bot.event
        async def on_voice_state_update(
            member: discord.Member,
            before: discord.VoiceState,
            after: discord.VoiceState,
        ) -> None:
            if (
                self._bot.user is not None
                and member.id == self._bot.user.id
                and before.channel is not None
                and after.channel is None
            ):
                await self._cleanup_vc_state(member.guild.id)
                logger.warning(
                    "Discord externally disconnected bot from VC guild=%d",
                    member.guild.id,
                )

        try:
            await self._bot.start(self._token)
        except Exception:
            logger.exception("Discord bot failed to start")

    async def stop(self) -> None:
        self._running = False
        await self._cancel_all_chunked_vc_speech()
        if self._ws:
            await self._ws.close()
        if self._bot:
            await self._bot.close()
        if self._tts is not None:
            close = getattr(self._tts, "aclose", None)
            if callable(close):
                await close()

    async def _dispatch_core_message(
        self,
        platform_user_id: str,
        content: str,
        images: list[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if (
            self._vc_pause_heartbeat
            and metadata
            and metadata.get("source") == "heartbeat"
            and not metadata.get("reminder")
            and self._vc_receivers
        ):
            # Core has already recorded this message as sent, so dropping it
            # here would desync the bot's memory from what the user actually
            # saw. Defer delivery until the bot leaves every voice channel.
            self._defer_heartbeat_during_vc(platform_user_id, content, metadata)
            return
        self._cache_pending_proposals_from_text(platform_user_id, content)
        if metadata and metadata.get("via_vc"):
            raw_guild_id = metadata.get("guild_id")
            try:
                guild_id = int(raw_guild_id)
            except (TypeError, ValueError):
                logger.warning("VC reply missing valid guild_id; dropping")
                return

            if metadata.get("vc_session_id") != self._vc_session_ids.get(guild_id):
                logger.info("Dropping stale VC reply for guild=%d", guild_id)
                return

            spoken = False
            if guild_id in self._vc_receivers:
                style = SpeechStyle.from_metadata(metadata)
                if style is None:
                    spoken = await self._speak_in_vc(guild_id, content)
                else:
                    spoken = await self._speak_in_vc(
                        guild_id,
                        content,
                        style=style,
                    )
            if spoken:
                self._vc_followup_until[guild_id] = (
                    monotonic() + self._vc_followup_seconds
                )
                await self._flush_buffered_vc_speech(guild_id)
                return

            self._vc_pending_guilds.discard(guild_id)
            self._vc_pending_since.pop(guild_id, None)
            self._vc_buffered_speech.pop(guild_id, None)
            logger.warning(
                "VC reply could not be delivered for guild %d; skipping DM fallback",
                guild_id,
            )
            return

        await self._send_to_discord(
            platform_user_id, content, images, metadata=metadata
        )

    def _defer_heartbeat_during_vc(
        self,
        platform_user_id: str,
        content: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Hold a heartbeat message until the bot has left every VC."""
        if len(self._vc_deferred_heartbeats) >= _VC_DEFERRED_HEARTBEAT_MAX:
            dropped = self._vc_deferred_heartbeats.pop(0)
            logger.warning(
                "Deferred-heartbeat queue full; dropping oldest for user=%s",
                dropped.get("platform_user_id"),
            )
        self._vc_deferred_heartbeats.append(
            {
                "platform_user_id": platform_user_id,
                "content": content,
                "metadata": dict(metadata or {}),
            }
        )
        logger.info(
            "Deferring Discord HEARTBEAT while bot is in VC guilds=%s "
            "(queued=%d)",
            sorted(self._vc_receivers),
            len(self._vc_deferred_heartbeats),
        )

    async def _flush_deferred_heartbeats(self) -> None:
        """Deliver heartbeats that were held back while the bot was in VC."""
        if self._vc_receivers or not self._vc_deferred_heartbeats:
            return
        pending = self._vc_deferred_heartbeats
        self._vc_deferred_heartbeats = []
        for item in pending:
            try:
                self._cache_pending_proposals_from_text(
                    item["platform_user_id"], item["content"]
                )
                await self._send_to_discord(
                    item["platform_user_id"],
                    item["content"],
                    [],
                    metadata=item["metadata"],
                )
            except Exception:
                logger.exception(
                    "Failed to deliver deferred heartbeat to user=%s",
                    item.get("platform_user_id"),
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
            # Redacted, truncated display: raw values can contain whole file
            # bodies (update_skill_file) or credentials, and embed fields are
            # capped at 1024 characters anyway.
            params_text = format_skill_params_for_display(skill_params)[:1000]
            embed.add_field(name="Parameters", value=params_text, inline=False)
        embed.set_footer(text=f"Proposal ID: {proposal_id}")

        try:
            await channel.send(embed=embed, view=self._make_skill_confirm_view(discord))
        except Exception:
            logger.warning(
                "Failed to send skill-confirm embed; falling back to text",
                exc_info=True,
            )
            content = data.get("content", "")
            await self._dispatch_core_message(platform_user_id, content, [])

    def _proposal_id_from_interaction(self, interaction: Any) -> str:
        message = getattr(interaction, "message", None)
        embeds = getattr(message, "embeds", None) or []
        for embed in embeds:
            footer = getattr(embed, "footer", None)
            text = str(getattr(footer, "text", "") or "")
            if text.startswith(_PROPOSAL_ID_FOOTER_PREFIX):
                return text.removeprefix(_PROPOSAL_ID_FOOTER_PREFIX).strip()
        return ""

    def _skill_name_from_interaction(self, interaction: Any) -> str:
        message = getattr(interaction, "message", None)
        embeds = getattr(message, "embeds", None) or []
        for embed in embeds:
            description = str(getattr(embed, "description", "") or "")
            match = re.search(r"`([^`]+)`", description)
            if match:
                return match.group(1)
        return "skill"

    def _make_skill_confirm_view(self, discord: Any) -> Any:
        adapter = self

        class SkillConfirmView(discord.ui.View):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__(timeout=None)

            async def _maybe_await(self, value: Any) -> None:
                if inspect.isawaitable(value):
                    await value

            async def _acknowledge(self, interaction: Any) -> bool:
                """Acknowledge Discord quickly so button clicks do not time out."""
                try:
                    await self._maybe_await(interaction.response.defer())
                    return True
                except Exception:
                    logger.warning(
                        "Failed to defer Discord skill-confirm interaction",
                        exc_info=True,
                    )
                    return False

            async def _finish(
                self,
                interaction: Any,
                *,
                content: str,
                deferred: bool,
            ) -> None:
                try:
                    if deferred:
                        await self._maybe_await(
                            interaction.edit_original_response(
                                content=content,
                                embed=None,
                                view=None,
                            )
                        )
                    else:
                        await self._maybe_await(
                            interaction.response.edit_message(
                                content=content,
                                embed=None,
                                view=None,
                            )
                        )
                except Exception:
                    logger.warning(
                        "Failed to update Discord skill-confirm interaction",
                        exc_info=True,
                    )

            async def _send_ephemeral(self, interaction: Any, content: str) -> None:
                try:
                    followup = getattr(interaction, "followup", None)
                    if followup is not None and hasattr(followup, "send"):
                        await self._maybe_await(
                            followup.send(content, ephemeral=True)
                        )
                    else:
                        await self._maybe_await(
                            interaction.response.send_message(
                                content,
                                ephemeral=True,
                            )
                        )
                except Exception:
                    logger.warning(
                        "Failed to send Discord skill-confirm notice",
                        exc_info=True,
                    )

            async def _send_command(
                self,
                interaction: Any,
                *,
                action: str,
            ) -> None:
                deferred = await self._acknowledge(interaction)
                proposal_id = adapter._proposal_id_from_interaction(interaction)
                skill_name = adapter._skill_name_from_interaction(interaction)
                if not proposal_id:
                    await self._finish(
                        interaction,
                        content=(
                            "⚠️ This approval button is missing its proposal ID. "
                            "Use /proposals to see active approvals."
                        ),
                        deferred=deferred,
                    )
                    return
                if adapter._ws is None:
                    await self._finish(
                        interaction,
                        content="⚠️ CordBeat Core is not connected.",
                        deferred=deferred,
                    )
                    return
                command = f"/{action} {proposal_id}"
                platform_user_id = str(getattr(interaction.user, "id", ""))
                cached = adapter._pending_skill_confirms.get(proposal_id)
                if (
                    cached is not None
                    and cached.get("platform_user_id") != platform_user_id
                ):
                    await self._send_ephemeral(
                        interaction,
                        "This approval belongs to another user.",
                    )
                    return
                try:
                    adapter._mark_proposal_action_sent(command)
                    await adapter._ws.send(
                        adapter._core_command_payload(command, platform_user_id)
                    )
                except Exception:
                    logger.warning(
                        "Failed to forward Discord skill-confirm command",
                        exc_info=True,
                    )
                    await self._finish(
                        interaction,
                        content="⚠️ Failed to send approval to CordBeat Core.",
                        deferred=deferred,
                    )
                    return
                if cached is None:
                    verb = "approve" if action == "approve" else "deny"
                    await self._finish(
                        interaction,
                        content=(
                            f"📨 Sent {verb} request: `{skill_name}` — "
                            "see the bot's reply for the result."
                        ),
                        deferred=deferred,
                    )
                    self.stop()
                    return
                verb = "Approved once" if action == "approve" else "Denied"
                icon = "✅" if action == "approve" else "❌"
                await self._finish(
                    interaction,
                    content=f"{icon} {verb}: `{skill_name}`",
                    deferred=deferred,
                )
                self.stop()

            @discord.ui.button(  # type: ignore[misc]
                label="✅ Allow Once",
                style=discord.ButtonStyle.success,
                custom_id=_SKILL_CONFIRM_APPROVE_CUSTOM_ID,
            )
            async def allow_once(
                self,
                interaction: Any,
                button: Any,
            ) -> None:
                await self._send_command(interaction, action="approve")

            @discord.ui.button(  # type: ignore[misc]
                label="❌ Deny",
                style=discord.ButtonStyle.danger,
                custom_id=_SKILL_CONFIRM_DENY_CUSTOM_ID,
            )
            async def deny(
                self,
                interaction: Any,
                button: Any,
            ) -> None:
                await self._send_command(interaction, action="reject")

        return SkillConfirmView()

    async def _forward_to_core(self, message: Any) -> None:
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

        async def _fetch_attachments(
            src_msg: Any,
            buf: list[str],
            *,
            mime_prefix: str,
            max_items: int,
            size_limit: int,
        ) -> None:
            kind = mime_prefix.rstrip("/")
            for att in getattr(src_msg, "attachments", []):
                if len(buf) >= max_items:
                    break
                ct = getattr(att, "content_type", "") or ""
                if not ct.startswith(mime_prefix):
                    continue
                size = getattr(att, "size", 0) or 0
                if size > size_limit:
                    logger.warning(
                        "Skipping oversized Discord %s attachment: %d bytes",
                        kind,
                        size,
                    )
                    continue
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(str(att.url))
                        resp.raise_for_status()
                        buf.append(base64.b64encode(resp.content).decode("ascii"))
                        logger.debug(
                            "Downloaded Discord %s attachment: %s (%d bytes)",
                            kind,
                            getattr(att, "filename", att.url),
                            len(resp.content),
                        )
                except Exception:
                    logger.warning(
                        "Failed to download Discord %s attachment: %s", kind, att.url
                    )

        async def _fetch_attachment_images(src_msg: Any, buf: list[str]) -> None:
            await _fetch_attachments(
                src_msg,
                buf,
                mime_prefix="image/",
                max_items=_MAX_IMAGES_PER_MESSAGE,
                size_limit=_IMAGE_SIZE_LIMIT_BYTES,
            )

        images: list[str] = []
        await _fetch_attachment_images(message, images)

        # Videos are large, so only collect them when Core said it will
        # actually send them to the model.
        videos: list[str] = []
        if self.core_supports("video_input"):
            await _fetch_attachments(
                message,
                videos,
                mime_prefix="video/",
                max_items=_MAX_VIDEOS_PER_MESSAGE,
                size_limit=_VIDEO_SIZE_LIMIT_BYTES,
            )

        ref = getattr(message, "reference", None)
        reply_context: dict[str, Any] | None = None
        if ref is not None:
            ref_msg = getattr(ref, "resolved", None)
            if ref_msg is None:
                message_id = getattr(ref, "message_id", None)
                if isinstance(message_id, int):
                    try:
                        ref_msg = await message.channel.fetch_message(message_id)
                    except Exception:
                        logger.debug(
                            "Failed to fetch unresolved Discord reply message %s",
                            message_id,
                        )
            if ref_msg is not None:
                reply_image_start = len(images)
                await _fetch_attachment_images(ref_msg, images)
                reply_author = getattr(ref_msg, "author", None)
                author_name = getattr(reply_author, "display_name", None) or getattr(
                    reply_author, "name", None
                )
                reply_content = getattr(ref_msg, "content", "")
                reply_message_id = getattr(ref_msg, "id", "")
                if not isinstance(author_name, str):
                    author_name = ""
                if not isinstance(reply_content, str):
                    reply_content = ""
                if not isinstance(reply_message_id, int | str):
                    reply_message_id = ""
                reply_image_count = len(images) - reply_image_start
                if (
                    author_name
                    or reply_content
                    or reply_message_id
                    or reply_image_count
                ):
                    reply_context = {
                        "author": author_name,
                        "content": reply_content,
                        "message_id": str(reply_message_id),
                        "is_bot": bool(getattr(reply_author, "bot", False)),
                        "image_count": reply_image_count,
                    }

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

        platform_created_at = getattr(message, "created_at", None)
        sent_at = (
            platform_created_at
            if isinstance(platform_created_at, datetime)
            else datetime.now(tz=UTC)
        )
        guild = message.guild
        channel = message.channel
        channel_is_public = False
        mutual_guild_ids: list[str] = []
        if guild is not None:
            try:
                channel_type = str(getattr(channel, "type", ""))
                channel_is_public = (
                    channel_type != "private_thread"
                    and bool(channel.permissions_for(guild.default_role).view_channel)
                )
            except Exception:
                logger.debug("Could not determine Discord channel visibility")
        elif self._bot is not None:
            try:
                mutual_guild_ids = [
                    str(candidate.id)
                    for candidate in self._bot.guilds
                    if candidate.get_member(message.author.id) is not None
                ]
            except Exception:
                logger.debug("Could not determine mutual Discord guilds")

        payload = json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": str(message.author.id),
                "content": content,
                "timestamp": sent_at.isoformat(),
                "images": images,
                "videos": videos,
                "is_voice": is_voice,
                "metadata": {
                    "channel_id": str(message.channel.id),
                    "channel_name": str(getattr(message.channel, "name", "") or ""),
                    "message_id": str(getattr(message, "id", "") or ""),
                    "guild_id": str(message.guild.id) if message.guild else "",
                    "guild_name": str(getattr(message.guild, "name", "") or ""),
                    "is_dm": message.guild is None,
                    "channel_is_public": channel_is_public,
                    "mutual_guild_ids": mutual_guild_ids,
                    "display_name": message.author.display_name,
                    **({"reply_context": reply_context} if reply_context else {}),
                },
            }
        )
        try:
            logger.debug(
                "Forwarding Discord message to Core: payload=%d bytes, images=%d",
                len(payload.encode("utf-8")),
                len(images),
            )
            # Buffered for resend if Core is unreachable (bounded outbox).
            await self._send_to_core(payload)
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
        self._cache_pending_proposal(
            platform_user_id=platform_user_id,
            proposal_id=proposal_id,
            label=skill_name,
            skill_params=skill_params,
        )

    def _cache_pending_proposal(
        self,
        *,
        platform_user_id: str,
        proposal_id: str,
        label: str,
        skill_params: dict[str, Any] | None = None,
    ) -> None:
        if not proposal_id:
            return
        self._pending_skill_confirms[proposal_id] = {
            "platform_user_id": platform_user_id,
            "skill_name": label or "proposal",
            "skill_params": skill_params or {},
        }
        self._pending_skill_confirms.move_to_end(proposal_id)
        if len(self._pending_skill_confirms) > _PENDING_SKILL_CONFIRM_MAX:
            self._pending_skill_confirms.popitem(last=False)

    def _cache_pending_proposals_from_text(
        self,
        platform_user_id: str,
        content: str,
    ) -> None:
        for match in _APPROVE_COMMAND_RE.finditer(content):
            self._cache_pending_proposal(
                platform_user_id=platform_user_id,
                proposal_id=match.group(1),
                label="proposal",
            )

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
        async def maybe_await(value: Any) -> None:
            if inspect.isawaitable(value):
                await value

        if self._ws is None:
            await maybe_await(
                interaction.response.send_message(
                    "CordBeat Core is not connected.", ephemeral=True
                )
            )
            return

        deferred = False
        try:
            await maybe_await(interaction.response.defer(ephemeral=True, thinking=True))
            deferred = True
        except Exception:
            logger.warning("Failed to defer Discord command interaction", exc_info=True)

        platform_user_id = str(interaction.user.id)
        channel_id = getattr(interaction, "channel_id", None)
        if channel_id is not None:
            self._remember_user_channel(platform_user_id, int(channel_id))

        try:
            self._mark_proposal_action_sent(command)
            await self._ws.send(self._core_command_payload(command, platform_user_id))
        except Exception:
            logger.warning("Failed to forward Discord command to Core", exc_info=True)
            if deferred:
                await maybe_await(
                    interaction.followup.send(
                        "Failed to send command to CordBeat Core.", ephemeral=True
                    )
                )
            else:
                await maybe_await(
                    interaction.response.send_message(
                        "Failed to send command to CordBeat Core.", ephemeral=True
                    )
                )
            return

        if deferred:
            await maybe_await(
                interaction.followup.send("Command sent to CordBeat.", ephemeral=True)
            )
        else:
            await maybe_await(
                interaction.response.send_message(
                    "Command sent to CordBeat.", ephemeral=True
                )
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
        chunks = split_message(content, _DISCORD_MESSAGE_LIMIT)

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

        started_at = monotonic()
        try:
            transcribed = await self._stt.transcribe(wav_data)
        except Exception:
            logger.exception("STT transcription error for VC user %d", user_id)
            return
        logger.debug(
            "VC STT completed guild=%d user=%d audio_bytes=%d text_chars=%d "
            "elapsed=%.3fs",
            guild_id,
            user_id,
            len(wav_data),
            len(transcribed or ""),
            monotonic() - started_at,
        )

        if not transcribed or not transcribed.strip():
            return

        platform_user_id = str(user_id)
        self._vc_user_guild[platform_user_id] = guild_id
        if guild_id not in self._vc_receivers:
            logger.debug("Dropping STT result after VC disconnect guild=%d", guild_id)
            return

        speaker_name = self._vc_speaker_name(guild_id, user_id)
        speech_line = f"{speaker_name}: {transcribed.strip()}"
        if self._vc_reply_pending(guild_id):
            buffered = self._vc_buffered_speech.setdefault(guild_id, [])
            buffered.append(speech_line)
            if len(buffered) > _VC_BUFFER_MAX_FRAGMENTS:
                del buffered[:-_VC_BUFFER_MAX_FRAGMENTS]
            logger.debug(
                "Buffered shared VC speech while reply is pending guild=%d "
                "fragments=%d",
                guild_id,
                len(buffered),
            )
            return

        room_context = self._vc_room_context.setdefault(
            guild_id,
            deque(maxlen=self._vc_context_max_lines),
        )
        room_context.append(speech_line)
        wake_detected = self._vc_has_wake_word(transcribed)
        followup_active = monotonic() <= self._vc_followup_until.get(guild_id, 0.0)
        should_activate = wake_detected or followup_active
        if not should_activate and self._vc_activation_mode == "always":
            should_activate = True
        if (
            not should_activate
            and self._vc_activation_mode == "hybrid"
            and get_judge_backend() is not None
        ):
            should_activate = await self._vc_should_join_conversation(
                speech_line,
                room_context,
            )
        if not should_activate:
            logger.debug(
                "Ignoring shared VC speech activation_mode=%s guild=%d user=%d "
                "phrases=%s",
                self._vc_activation_mode,
                guild_id,
                user_id,
                self._vc_wake_words,
            )
            return

        transcript = "\n".join(room_context)
        room_context.clear()
        await self._forward_vc_transcript(guild_id, transcript)

    async def _on_vc_speech_start(self, guild_id: int, user_id: int) -> None:
        """Stop an in-flight VC reply when a participant starts speaking."""
        if not self._vc_barge_in or self._bot is None:
            return
        guild = self._bot.get_guild(guild_id)
        vc = guild.voice_client if guild is not None else None
        if not vc or not vc.is_connected():
            return

        chunked = guild_id in self._vc_chunk_sessions
        playing = bool(vc.is_playing())
        if not chunked and not playing:
            return

        # An interruption is an explicit continuation of the active voice
        # turn, even if a long spoken reply outlived the normal follow-up window.
        self._vc_followup_until[guild_id] = monotonic() + self._vc_followup_seconds

        if self._vc_pending_speech_guild_id == guild_id:
            self._vc_pending_speech = None
            self._vc_pending_speech_guild_id = None
        pending_task = self._vc_pending_speech_task
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
        self._vc_pending_speech_task = None

        # Stop FFmpeg and cancel the chunk producer. Doing only one would let
        # the next already-synthesised chunk resume after the interruption.
        if playing:
            try:
                vc.stop()
            except Exception:
                logger.debug("VC playback stop failed during barge-in", exc_info=True)
        if chunked:
            await self._cancel_chunked_vc_speech(guild_id)

        logger.info(
            "Interrupted VC TTS on participant speech guild=%d user=%d chunked=%s",
            guild_id,
            user_id,
            chunked,
        )

    async def _vc_should_join_conversation(
        self,
        speech_line: str,
        room_context: deque[str],
    ) -> bool:
        """Use the lightweight judge to decide whether inactive VC chat invites us."""

        if self._vc_participation_judge_lock.locked():
            logger.debug("Skipping VC participation judge while previous call runs")
            return False
        async with self._vc_participation_judge_lock:
            context = "\n".join(list(room_context)[-5:-1])
            phrases = ", ".join(self._vc_wake_words)
            prompt = (
                "Decide whether the latest line clearly invites the voice AI to join. "
                "Say yes only for a direct question/request to the AI or an explicit "
                "invitation for its opinion. Say no for human-to-human chatter, even "
                "when it contains a question.\n"
                f"AI names: {phrases}\n"
                f"Recent room transcript:\n{context}\n"
                f"Latest line: {speech_line}\n"
                "Answer:"
            )
            started_at = monotonic()
            decision = await judge_yes_no(prompt, fail_open=False)
            logger.debug(
                "VC participation judge decision=%s elapsed=%.3fs line=%r",
                decision,
                monotonic() - started_at,
                speech_line[:120],
            )
            return decision

    async def _forward_vc_transcript(self, guild_id: int, transcribed: str) -> None:
        """Forward one consolidated VC turn to Core."""
        if self._ws is None:
            return
        session_id = self._vc_session_ids.get(guild_id)
        if session_id is None:
            return
        payload = json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": f"vc:{guild_id}",
                "content": transcribed,
                "timestamp": datetime.now(UTC).isoformat(),
                "is_voice": True,
                "metadata": {
                    "guild_id": str(guild_id),
                    "channel_id": "vc",
                    "via_vc": True,
                    "shared_voice": True,
                    "ephemeral": True,
                    "vc_session_id": session_id,
                    "display_name": "shared Discord voice channel",
                },
            }
        )
        self._vc_pending_guilds.add(guild_id)
        self._vc_pending_since[guild_id] = monotonic()
        try:
            await self._ws.send(payload)
        except Exception:
            self._vc_pending_guilds.discard(guild_id)
            self._vc_pending_since.pop(guild_id, None)
            logger.exception("Failed to forward VC speech to Core")

    async def _flush_buffered_vc_speech(self, guild_id: int) -> None:
        """Retain pending-room speech and only send an explicit wake request."""
        self._vc_pending_guilds.discard(guild_id)
        self._vc_pending_since.pop(guild_id, None)
        buffered = self._vc_buffered_speech.pop(guild_id, [])
        if not buffered:
            return
        room_context = self._vc_room_context.setdefault(
            guild_id,
            deque(maxlen=self._vc_context_max_lines),
        )
        room_context.extend(buffered)
        if not any(self._vc_line_has_wake_word(line) for line in buffered):
            return
        transcript = "\n".join(room_context)
        room_context.clear()
        await self._forward_vc_transcript(guild_id, transcript)

    def _vc_reply_pending(self, guild_id: int) -> bool:
        if guild_id not in self._vc_pending_guilds:
            return False
        started_at = self._vc_pending_since.get(guild_id)
        if (
            started_at is not None
            and monotonic() - started_at <= self._vc_pending_timeout_seconds
        ):
            return True
        self._vc_pending_guilds.discard(guild_id)
        self._vc_pending_since.pop(guild_id, None)
        self._vc_buffered_speech.pop(guild_id, None)
        logger.warning(
            "VC reply wait timed out for guild=%d; accepting new speech",
            guild_id,
        )
        return False

    def _vc_has_wake_word(self, transcribed: str) -> bool:
        text = _normalize_vc_wake_text(transcribed)
        return any(word in text for word in self._vc_normalized_wake_words)

    def _vc_line_has_wake_word(self, speech_line: str) -> bool:
        _, separator, transcribed = speech_line.partition(": ")
        return bool(separator) and self._vc_has_wake_word(transcribed)

    def _vc_speaker_name(self, guild_id: int, user_id: int) -> str:
        if self._bot is not None:
            guild = self._bot.get_guild(guild_id)
            member = guild.get_member(user_id) if guild is not None else None
            if member is not None:
                name = str(getattr(member, "display_name", "") or "").strip()
                if name:
                    return name.replace("\n", " ")[:80]
        return f"participant-{user_id}"

    async def _speak_in_vc(
        self,
        guild_id: int,
        text: str,
        *,
        style: SpeechStyle | None = None,
    ) -> bool:
        """Synthesise *text* to audio and play it in the guild's voice channel."""
        if not self._tts or not self._bot:
            return False

        guild = self._bot.get_guild(guild_id)
        if guild is None:
            return False

        vc = guild.voice_client
        if not vc or not vc.is_connected():
            return False
        if self._tts.supports_chunked_playback is True:
            return self._start_chunked_vc_speech(guild_id, text, style=style)
        if vc.is_playing():
            self._queue_pending_vc_speech(guild_id, text)
            return True

        started_at = monotonic()
        try:
            audio = await self._tts.synthesize(text)
        except Exception:
            logger.exception("TTS synthesis failed for VC guild %d", guild_id)
            return False
        logger.debug(
            "VC TTS completed guild=%d text_chars=%d audio_bytes=%d elapsed=%.3fs",
            guild_id,
            len(text),
            len(audio or b""),
            monotonic() - started_at,
        )

        if not audio:
            return False

        if vc.is_playing():
            self._queue_pending_vc_speech(guild_id, text)
            return True

        try:
            import discord

            source = discord.FFmpegPCMAudio(io.BytesIO(audio), pipe=True)
            loop = asyncio.get_running_loop()

            def after_playback(error: Exception | None) -> None:
                loop.call_soon_threadsafe(
                    self._on_vc_playback_finished,
                    guild_id,
                    error,
                )

            vc.play(source, after=after_playback)
            logger.debug("Started VC TTS playback guild=%d", guild_id)
            return True
        except Exception:
            logger.exception("Failed to play TTS audio in VC guild %d", guild_id)
            return False

    def _start_chunked_vc_speech(
        self,
        guild_id: int,
        text: str,
        *,
        style: SpeechStyle | None,
    ) -> bool:
        """Start producer/consumer pseudo-streaming for one VC response."""

        if self._tts is None or not text.strip():
            return False
        previous = self._vc_chunk_sessions.get(guild_id)
        if previous is not None:
            if previous.producer is not None:
                previous.producer.cancel()
            while not previous.queue.empty():
                try:
                    previous.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                previous.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        generation_id = uuid.uuid4().hex
        queue_size = max(1, int(self._tts.playback_queue_size))
        session = _VCChunkSession(
            generation_id=generation_id,
            queue=asyncio.Queue(maxsize=queue_size),
        )
        self._vc_chunk_sessions[guild_id] = session
        session.producer = asyncio.create_task(
            self._produce_vc_audio_chunks(guild_id, session, text, style)
        )
        session.runner = asyncio.create_task(
            self._consume_vc_audio_chunks(guild_id, session)
        )
        logger.debug(
            "Started chunked VC TTS guild=%d generation=%s queue_size=%d",
            guild_id,
            generation_id,
            queue_size,
        )
        return True

    async def _produce_vc_audio_chunks(
        self,
        guild_id: int,
        session: _VCChunkSession,
        text: str,
        style: SpeechStyle | None,
    ) -> None:
        assert self._tts is not None
        try:
            async for audio in self._tts.synthesize_chunks(text, style=style):
                current = self._vc_chunk_sessions.get(guild_id)
                if current is not session:
                    return
                await session.queue.put(audio)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Chunked VC TTS generation failed guild=%d generation=%s",
                guild_id,
                session.generation_id,
            )
        finally:
            if self._vc_chunk_sessions.get(guild_id) is session:
                await session.queue.put(None)

    async def _consume_vc_audio_chunks(
        self,
        guild_id: int,
        session: _VCChunkSession,
    ) -> None:
        lock = self._vc_chunk_playback_locks.setdefault(guild_id, asyncio.Lock())
        try:
            async with lock:
                while self._vc_chunk_sessions.get(guild_id) is session:
                    audio = await session.queue.get()
                    if audio is None:
                        break
                    if not await self._play_vc_audio_chunk(
                        guild_id,
                        session,
                        audio,
                    ):
                        break
        except asyncio.CancelledError:
            raise
        finally:
            if session.producer is not None and not session.producer.done():
                session.producer.cancel()
                await asyncio.gather(session.producer, return_exceptions=True)
            if self._vc_chunk_sessions.get(guild_id) is session:
                self._vc_chunk_sessions.pop(guild_id, None)

    async def _play_vc_audio_chunk(
        self,
        guild_id: int,
        session: _VCChunkSession,
        audio: bytes,
    ) -> bool:
        if self._bot is None:
            return False
        guild = self._bot.get_guild(guild_id)
        vc = guild.voice_client if guild is not None else None
        if not vc or not vc.is_connected():
            return False
        while vc.is_playing():
            if self._vc_chunk_sessions.get(guild_id) is not session:
                return False
            await asyncio.sleep(0.05)
        if self._vc_chunk_sessions.get(guild_id) is not session:
            return False
        try:
            import discord

            source = discord.FFmpegPCMAudio(io.BytesIO(audio), pipe=True)
            loop = asyncio.get_running_loop()
            completed: asyncio.Future[Exception | None] = loop.create_future()

            def after_playback(error: Exception | None) -> None:
                def resolve() -> None:
                    if not completed.done():
                        completed.set_result(error)

                loop.call_soon_threadsafe(resolve)

            vc.play(source, after=after_playback)
            error = await completed
            if error is not None:
                logger.warning(
                    "VC chunk playback failed guild=%d: %s", guild_id, error
                )
                return False
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to play VC audio chunk guild=%d", guild_id)
            return False

    async def _cancel_chunked_vc_speech(self, guild_id: int) -> None:
        session = self._vc_chunk_sessions.pop(guild_id, None)
        if session is None:
            return
        tasks = [
            task
            for task in (session.producer, session.runner)
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._vc_chunk_playback_locks.pop(guild_id, None)

    async def _cancel_all_chunked_vc_speech(self) -> None:
        for guild_id in list(self._vc_chunk_sessions):
            await self._cancel_chunked_vc_speech(guild_id)

    def _queue_pending_vc_speech(self, guild_id: int, text: str) -> None:
        if self._vc_pending_speech is not None:
            logger.debug(
                "Replacing queued VC speech guild=%s with latest guild=%d",
                self._vc_pending_speech_guild_id,
                guild_id,
            )
        self._vc_pending_speech = text
        self._vc_pending_speech_guild_id = guild_id

    def _on_vc_playback_finished(
        self,
        guild_id: int,
        error: Exception | None,
    ) -> None:
        """Run on the event loop after discord.py's playback callback."""
        if error is not None:
            logger.warning("VC playback failed for guild=%d: %s", guild_id, error)
        if (
            self._vc_pending_speech is None
            or self._vc_pending_speech_guild_id != guild_id
        ):
            return
        text = self._vc_pending_speech
        self._vc_pending_speech = None
        self._vc_pending_speech_guild_id = None
        self._vc_pending_speech_task = asyncio.create_task(
            self._speak_in_vc(guild_id, text)
        )

    def _voice_join_message(self, channel_name: str) -> str:
        if self._stt is None:
            return (
                f"⚠️ Joined **{channel_name}**, but STT is disabled. "
                "I can receive audio packets but cannot understand speech."
            )
        if self._tts is None:
            return (
                f"⚠️ Joined **{channel_name}** and listening, but TTS is disabled. "
                "I cannot safely deliver VC replies without speech output."
            )
        wake_words = ", ".join(self._vc_wake_words)
        return (
            f"✅ Joined **{channel_name}** in **{self._vc_activation_mode}** mode. "
            f"Say one of: **{wake_words}** to invite me directly. "
            "Recent room context is temporary."
        )

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

        receiver = VoiceReceiver(
            vc,
            speech_start_sec=self._vc_barge_in_speech_seconds,
            speech_start_rms_threshold=self._vc_barge_in_rms_threshold,
        )
        if self._vc_barge_in:
            receiver.on_speech_start(
                lambda uid: self._on_vc_speech_start(guild_id, uid)
            )
        receiver.on_speech_end(
            lambda uid, wav: asyncio.ensure_future(
                self._on_vc_speech(guild_id, uid, wav)
            )
        )
        await receiver.start()
        self._vc_receivers[guild_id] = receiver
        self._vc_session_ids[guild_id] = uuid.uuid4().hex
        self._vc_pending_guilds.discard(guild_id)
        self._vc_pending_since.pop(guild_id, None)
        self._vc_buffered_speech.pop(guild_id, None)
        self._vc_room_context[guild_id] = deque(maxlen=self._vc_context_max_lines)
        self._vc_followup_until.pop(guild_id, None)

        await interaction.followup.send(
            self._voice_join_message(channel.name), ephemeral=True
        )
        logger.info("Joined VC guild=%d channel=%s", guild_id, channel.name)

    async def _handle_leave(self, interaction: Any) -> None:
        """Slash command: leave the voice channel."""
        guild_id: int = interaction.guild_id
        await self._cleanup_vc_state(guild_id)

        guild = self._bot.get_guild(guild_id) if self._bot else None
        if guild and guild.voice_client:
            await guild.voice_client.disconnect()

        await interaction.response.send_message(
            "👋 Left the voice channel.", ephemeral=True
        )
        logger.info("Left VC guild=%d", guild_id)

    async def _cleanup_vc_state(self, guild_id: int) -> None:
        """Stop receiving and clear shared-room state for one VC guild."""
        await self._cancel_chunked_vc_speech(guild_id)
        receiver = self._vc_receivers.pop(guild_id, None)
        if receiver is not None:
            try:
                await receiver.stop()
            except Exception:
                logger.exception("Failed to stop VC receiver for guild=%d", guild_id)

        stale = [uid for uid, gid in self._vc_user_guild.items() if gid == guild_id]
        for uid in stale:
            del self._vc_user_guild[uid]
        self._vc_pending_guilds.discard(guild_id)
        self._vc_pending_since.pop(guild_id, None)
        self._vc_buffered_speech.pop(guild_id, None)
        self._vc_room_context.pop(guild_id, None)
        self._vc_followup_until.pop(guild_id, None)
        self._vc_session_ids.pop(guild_id, None)
        self._vc_muted.discard(guild_id)
        if self._vc_pending_speech_guild_id == guild_id:
            self._vc_pending_speech = None
            self._vc_pending_speech_guild_id = None
        await self._flush_deferred_heartbeats()

    async def _handle_mute(self, interaction: Any) -> None:
        """Slash command: toggle voice mute."""
        guild_id: int = interaction.guild_id
        if guild_id in self._vc_muted:
            self._vc_muted.discard(guild_id)
            await interaction.response.send_message("🔊 Unmuted.", ephemeral=True)
        else:
            self._vc_muted.add(guild_id)
            await interaction.response.send_message("🔇 Muted.", ephemeral=True)
