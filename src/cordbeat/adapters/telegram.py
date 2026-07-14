"""Telegram adapter — bridges Telegram bot to CordBeat Core via WebSocket."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from collections import OrderedDict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cordbeat.adapters._utils import AdapterFilter, split_message
from cordbeat.config import AdapterConfig, STTConfig, TTSConfig
from cordbeat.core.gateway import RetryableConnection

if TYPE_CHECKING:
    from cordbeat.ai.stt import STTBackend
    from cordbeat.ai.tts import TTSBackend

logger = logging.getLogger(__name__)

ADAPTER_ID = "telegram"
_MAX_IMAGES_PER_MESSAGE = 4
_IMAGE_SIZE_LIMIT_BYTES = 10 * 1024 * 1024
_TYPING_REFRESH_SECONDS = 4
_TELEGRAM_MESSAGE_LIMIT = 4096
_TELEGRAM_CAPTION_LIMIT = 1024
_PENDING_SKILL_CONFIRM_MAX = 1_000
_CORE_BOT_COMMANDS = (
    ("approve", "Approve a pending CordBeat proposal"),
    ("reject", "Reject a pending CordBeat proposal"),
    ("proposals", "List pending CordBeat proposals"),
    ("link", "Generate a cross-platform link token"),
    ("link_confirm", "Confirm a cross-platform link token"),
    ("unlink", "Unlink a platform from your account"),
    ("name", "Update CordBeat's displayed character name"),
    ("quiet", "Set heartbeat quiet hours"),
    ("prefer", "Set preferred heartbeat platform"),
    ("draw", "Run Draw DSL and return an image"),
)
_TELEGRAM_TO_CORE_COMMANDS = {
    "/link_confirm": "/link-confirm",
}


def _normalize_telegram_command_text(text: str) -> str:
    parts = text.split(maxsplit=1)
    if not parts:
        return ""
    command = parts[0].split("@", 1)[0]
    command = _TELEGRAM_TO_CORE_COMMANDS.get(command, command)
    rest = parts[1] if len(parts) > 1 else ""
    return f"{command} {rest}".strip()


class TelegramAdapter(RetryableConnection):
    """Telegram bot that forwards messages to CordBeat Core via WebSocket."""

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
        self._auth_token = config.auth_token
        self._token: str = config.options.get("token", "")
        self._app: Any = None
        self._ws: Any = None
        self._running = False
        self._max_backoff = config.reconnect_max_backoff
        # Map platform_user_id → chat_id for reply routing
        self._chat_map: dict[str, int] = {}
        self._filter = AdapterFilter.from_options(config.options)
        # Typing indicator tasks: chat_id → asyncio.Task
        self._typing_tasks: dict[int, asyncio.Task[None]] = {}
        # Track which users last communicated via voice (for TTS routing)
        self._voice_users: set[str] = set()
        self._pending_skill_confirm_owners: OrderedDict[str, str] = OrderedDict()

        self._stt: STTBackend | None = None
        self._tts: TTSBackend | None = None

        if stt_config is not None and stt_config.enabled:
            from cordbeat.ai.stt import create_stt_backend

            self._stt = create_stt_backend(stt_config)

        if tts_config is not None and tts_config.enabled:
            from cordbeat.ai.tts import create_tts_backend

            self._tts = create_tts_backend(tts_config)

    async def start(self) -> None:
        try:
            from telegram import BotCommand, Update
            from telegram.ext import (
                ApplicationBuilder,
                CommandHandler,
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

            # ── E-4 response filtering ────────────────────────────────
            is_private = getattr(msg.chat, "type", "private") == "private"
            raw_text = msg.text or msg.caption or ""
            bot_username: str = context.bot.username or ""
            # For Telegram: "mentioned" = @botname present in group text
            is_mentioned = bool(
                bot_username and f"@{bot_username}".lower() in raw_text.lower()
            )
            if not await self._filter.should_respond_async(
                user_id=user_id,
                channel_id=str(chat_id),
                is_dm=is_private,
                is_mentioned=is_mentioned,
                text=raw_text,
                extra_keywords=[bot_username] if bot_username else None,
            ):
                return
            # ─────────────────────────────────────────────────────────

            self._chat_map[user_id] = chat_id
            display_name = user.full_name or user.username or ""

            # Download photo if present (pick highest resolution).
            images: list[str] = []
            if msg.photo:
                try:
                    photo = msg.photo[-1]
                    if (photo.file_size or 0) <= _IMAGE_SIZE_LIMIT_BYTES:
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
                size_ok = (doc.file_size or 0) <= _IMAGE_SIZE_LIMIT_BYTES
                if mime and size_ok:
                    try:
                        file = await context.bot.get_file(doc.file_id)
                        raw = await file.download_as_bytearray()
                        images.append(base64.b64encode(bytes(raw)).decode("ascii"))
                    except Exception:
                        logger.warning("Failed to download Telegram document image")

            reply_context: dict[str, Any] | None = None
            reply = getattr(msg, "reply_to_message", None)
            if reply is not None:
                reply_image_start = len(images)
                if len(images) < _MAX_IMAGES_PER_MESSAGE and reply.photo:
                    try:
                        photo = reply.photo[-1]
                        if (photo.file_size or 0) <= _IMAGE_SIZE_LIMIT_BYTES:
                            file = await context.bot.get_file(photo.file_id)
                            raw = await file.download_as_bytearray()
                            images.append(base64.b64encode(bytes(raw)).decode("ascii"))
                    except Exception:
                        logger.warning("Failed to download replied Telegram photo")
                if (
                    len(images) < _MAX_IMAGES_PER_MESSAGE
                    and reply.document
                    and (reply.document.mime_type or "").startswith("image/")
                    and (reply.document.file_size or 0) <= _IMAGE_SIZE_LIMIT_BYTES
                ):
                    try:
                        file = await context.bot.get_file(reply.document.file_id)
                        raw = await file.download_as_bytearray()
                        images.append(base64.b64encode(bytes(raw)).decode("ascii"))
                    except Exception:
                        logger.warning("Failed to download replied Telegram document")

                reply_author = getattr(reply, "from_user", None)
                author_name = (
                    getattr(reply_author, "full_name", None)
                    or getattr(reply_author, "username", None)
                    or ""
                )
                reply_context = {
                    "author": author_name,
                    "content": reply.text or reply.caption or "",
                    "message_id": str(reply.message_id),
                    "is_bot": bool(getattr(reply_author, "is_bot", False)),
                    "image_count": len(images) - reply_image_start,
                }

            # Handle voice messages via STT when the backend is configured.
            text = msg.text or msg.caption or ""
            if msg.voice and self._stt is not None:
                try:
                    vfile = await context.bot.get_file(msg.voice.file_id)
                    audio_bytes = bytes(await vfile.download_as_bytearray())
                    transcribed = await self._stt.transcribe(audio_bytes)
                    if transcribed:
                        text = transcribed
                        self._voice_users.add(user_id)
                    else:
                        logger.warning(
                            "STT returned empty transcription for user %s", user_id
                        )
                except Exception:
                    logger.warning("Failed to process Telegram voice message")
            elif msg.voice:
                logger.debug(
                    "Received voice message from %s but STT is not configured", user_id
                )
            else:
                # Text/photo message: revert to text replies for this user.
                self._voice_users.discard(user_id)

            await self._forward_to_core(
                user_id,
                text,
                display_name=display_name,
                chat_id=chat_id,
                images=images,
                is_voice=user_id in self._voice_users,
                reply_context=reply_context,
            )

        async def handle_core_command(update: Update, context: Any) -> None:
            msg = update.message
            user = update.effective_user
            if msg is None or user is None:
                return

            normalized = _normalize_telegram_command_text(msg.text or "")
            if not normalized:
                return

            user_id = str(user.id)
            chat_id = msg.chat_id
            self._chat_map[user_id] = chat_id
            await self._forward_to_core(
                user_id,
                normalized,
                display_name=user.full_name or user.username or "",
                chat_id=chat_id,
            )

        from telegram.ext import CallbackQueryHandler  # noqa: PLC0415

        self._app.add_handler(
            CommandHandler(
                [name for name, _ in _CORE_BOT_COMMANDS],
                handle_core_command,
            )
        )
        self._app.add_handler(
            MessageHandler(
                (filters.TEXT | filters.PHOTO | filters.Document.IMAGE | filters.VOICE)
                & ~filters.COMMAND,
                handle_message,
            )
        )

        async def handle_skill_confirm_callback(
            update: Update,
            context: Any,
        ) -> None:
            """Process ✅/🔁/❌ inline button presses for skill confirmation."""
            query = update.callback_query
            if query is None:
                return
            data = query.data or ""
            # data format: "skill_confirm:<action>:<proposal_id>"
            if not data.startswith("skill_confirm:"):
                await query.answer()
                return

            parts = data.split(":", 2)
            if len(parts) != 3:
                await query.answer()
                return

            action, proposal_id = parts[1], parts[2]
            cmd_map = {
                "approve": f"/approve {proposal_id}",
                "reject": f"/reject {proposal_id}",
            }
            cmd = cmd_map.get(action)
            if cmd is None:
                await query.answer("Unknown action")
                return

            sent = False
            user = update.effective_user
            platform_user_id = str(user.id) if user is not None else ""
            owner_id = self._pending_skill_confirm_owners.get(proposal_id)
            if owner_id is not None and owner_id != platform_user_id:
                await query.answer("This approval isn't yours", show_alert=True)
                return

            await query.answer()
            if self._ws is not None:
                try:
                    await self._ws.send(
                        json.dumps(
                            {
                                "type": "message",
                                "adapter_id": ADAPTER_ID,
                                "platform_user_id": platform_user_id,
                                "content": cmd,
                                "timestamp": datetime.now(tz=UTC).isoformat(),
                            }
                        )
                    )
                    sent = True
                except Exception:
                    logger.warning("Failed to forward skill confirm to Core")

            if owner_id is None and sent:
                verb = "approve" if action == "approve" else "deny"
                label = f"📨 Sent {verb} request — see the reply."
            else:
                label_map = {
                    "approve": "✅ Approved once",
                    "reject": "❌ Denied",
                }
                label = label_map[action]
            if not sent:
                label = "⚠️ Failed to send approval to CordBeat Core."
            else:
                self._pending_skill_confirm_owners.pop(proposal_id, None)
            try:
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text(label)
            except Exception:
                pass

        self._app.add_handler(
            CallbackQueryHandler(
                handle_skill_confirm_callback,
                pattern=r"^skill_confirm:",
            )
        )

        # Connect to Core in background
        asyncio.create_task(self._connect_to_core())

        try:
            await self._app.bot.set_my_commands(
                [
                    BotCommand(name, description)
                    for name, description in _CORE_BOT_COMMANDS
                ]
            )
        except Exception:
            logger.warning("Failed to register Telegram bot commands", exc_info=True)

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

    async def _dispatch_skill_confirm(
        self, platform_user_id: str, data: dict[str, Any]
    ) -> None:
        """Send Telegram inline keyboard for skill confirmation."""
        if not self._app:
            await super()._dispatch_skill_confirm(platform_user_id, data)
            return

        meta = data.get("metadata") or {}
        proposal_id: str = str(meta.get("proposal_id", ""))
        skill_name: str = str(meta.get("skill_name", "unknown"))
        skill_params: dict[str, Any] = meta.get("skill_params") or {}

        chat_id = self._chat_map.get(platform_user_id)
        if chat_id is None:
            try:
                chat_id = int(platform_user_id)
            except ValueError:
                await super()._dispatch_skill_confirm(platform_user_id, data)
                return

        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup  # noqa: I001,PLC0415

            params_text = ""
            if skill_params:
                params_text = "\n".join(
                    f"  • {k}: {v}" for k, v in skill_params.items()
                )
                params_text = f"\n\nParameters:\n{params_text}"

            text = (
                "🔧 Skill Execution Required\n"
                f"Skill: {skill_name}{params_text}\n\n"
                f"Allow {skill_name} to run?"
            )

            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ Allow Once",
                            callback_data=f"skill_confirm:approve:{proposal_id}",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "❌ Deny",
                            callback_data=f"skill_confirm:reject:{proposal_id}",
                        ),
                    ],
                ]
            )

            self._stop_typing(chat_id)
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=keyboard,
            )
            if proposal_id:
                self._pending_skill_confirm_owners[proposal_id] = platform_user_id
                self._pending_skill_confirm_owners.move_to_end(proposal_id)
                if len(self._pending_skill_confirm_owners) > _PENDING_SKILL_CONFIRM_MAX:
                    self._pending_skill_confirm_owners.popitem(last=False)
        except ImportError:
            await super()._dispatch_skill_confirm(platform_user_id, data)
        except Exception:
            logger.exception("Failed to send Telegram skill confirm keyboard")
            await super()._dispatch_skill_confirm(platform_user_id, data)

    async def _dispatch_core_message(
        self,
        platform_user_id: str,
        content: str,
        images: list[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._tts is not None and platform_user_id in self._voice_users:
            await self._send_voice_to_telegram(platform_user_id, content)
        elif images:
            await self._send_images_to_telegram(platform_user_id, content, images)
        else:
            await self._send_to_telegram(platform_user_id, content)

    async def _send_voice_to_telegram(
        self,
        platform_user_id: str,
        content: str,
    ) -> None:
        """Synthesize *content* via TTS and send as voice/audio to Telegram."""
        if not self._app or not self._tts:
            return

        chat_id = self._chat_map.get(platform_user_id)
        if chat_id is None:
            try:
                chat_id = int(platform_user_id)
            except ValueError:
                await self._send_to_telegram(platform_user_id, content)
                return

        try:
            audio_bytes = await self._tts.synthesize(content)
            if not audio_bytes:
                await self._send_to_telegram(platform_user_id, content)
                return

            audio_io = io.BytesIO(audio_bytes)
            audio_io.name = (
                "response.ogg"
                if self._tts.content_type == "audio/ogg"
                else "response.mp3"
            )

            if self._tts.content_type == "audio/ogg":
                await self._app.bot.send_voice(chat_id=chat_id, voice=audio_io)
            else:
                await self._app.bot.send_audio(chat_id=chat_id, audio=audio_io)
        except Exception:
            logger.exception(
                "Failed to send TTS voice to Telegram chat %s; falling back to text",
                chat_id,
            )
            await self._send_to_telegram(platform_user_id, content)

    async def _forward_to_core(
        self,
        user_id: str,
        text: str,
        *,
        display_name: str = "",
        chat_id: int = 0,
        images: list[str] | None = None,
        is_voice: bool = False,
        reply_context: dict[str, Any] | None = None,
    ) -> None:
        payload = json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": user_id,
                "content": text,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "images": images or [],
                "is_voice": is_voice,
                "metadata": {
                    "chat_id": str(chat_id),
                    "display_name": display_name,
                    **({"reply_context": reply_context} if reply_context else {}),
                },
            }
        )
        # Buffered for resend if Core is unreachable (bounded outbox).
        sent = await self._send_to_core(payload)
        if sent and chat_id:
            # Show typing indicator while core processes the message
            self._start_typing(chat_id)

    def _start_typing(self, chat_id: int) -> None:
        """Start a background typing indicator loop for the given chat."""
        existing = self._typing_tasks.pop(chat_id, None)
        if existing is not None:
            existing.cancel()
        self._typing_tasks[chat_id] = asyncio.create_task(
            self._keep_typing_telegram(chat_id)
        )

    def _stop_typing(self, chat_id: int) -> None:
        """Cancel the typing indicator for the given chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task is not None:
            task.cancel()

    async def _keep_typing_telegram(self, chat_id: int) -> None:
        """Send Telegram typing action periodically until cancelled.

        Telegram typing actions expire after ~5 seconds.
        """
        try:
            while True:
                try:
                    from telegram import ChatAction  # type: ignore[attr-defined]

                    await self._app.bot.send_chat_action(
                        chat_id=chat_id, action=ChatAction.TYPING
                    )
                except Exception:
                    pass  # best-effort; don't crash if typing fails
                await asyncio.sleep(_TYPING_REFRESH_SECONDS)
        except asyncio.CancelledError:
            pass

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

        self._stop_typing(chat_id)
        try:
            for chunk in split_message(content, _TELEGRAM_MESSAGE_LIMIT):
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                )
        except Exception:
            logger.exception(
                "Failed to send message to Telegram chat %s",
                chat_id,
            )

    async def _send_images_to_telegram(
        self,
        platform_user_id: str,
        caption: str,
        images: list[str],
    ) -> None:
        """Send base64-encoded images to Telegram, falling back to text on error."""
        if not self._app or not platform_user_id:
            return

        chat_id = self._chat_map.get(platform_user_id)
        if chat_id is None:
            try:
                chat_id = int(platform_user_id)
            except ValueError:
                await self._send_to_telegram(platform_user_id, caption)
                return

        import base64  # noqa: PLC0415

        sent_any = False
        send_caption_as_text = bool(caption) and len(caption) > _TELEGRAM_CAPTION_LIMIT
        for idx, b64 in enumerate(images):
            try:
                raw = base64.b64decode(b64)
                photo_io = io.BytesIO(raw)
                photo_io.name = f"draw_{idx + 1}.png"
                img_caption = (
                    caption
                    if idx == 0 and not send_caption_as_text
                    else None
                )
                await self._app.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_io,
                    caption=img_caption,
                )
                sent_any = True
            except Exception:
                logger.exception(
                    "Failed to send image %d to Telegram chat %s", idx, chat_id
                )

        if not sent_any:
            await self._send_to_telegram(platform_user_id, caption)
        elif send_caption_as_text:
            await self._send_to_telegram(platform_user_id, caption)
