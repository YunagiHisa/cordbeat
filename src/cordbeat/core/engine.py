"""Core engine — processes incoming messages and generates responses."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from cordbeat.agent.react_types import ToolCallResult, ToolTrace
from cordbeat.agent.soul import Soul
from cordbeat.ai.backend import AIBackend, voice_context_scope
from cordbeat.ai.extraction import MemoryExtractor
from cordbeat.ai.prompt import (
    build_context,
    build_react_continuation_prompt,
    build_soul_system_prompt,
    sanitize,
    sanitize_tool_artifacts,
)
from cordbeat.ai.reasoning import sanitize_reasoning_artifacts
from cordbeat.config import MemoryConfig, ReActConfig
from cordbeat.memory.core import MemoryStore
from cordbeat.models import (
    GatewayMessage,
    MessageType,
    ProposalStatus,
    ProposalType,
    SafetyLevel,
    SoulCaller,
    UserSummary,
)
from cordbeat.skills.registry import SkillRegistry

from .gateway import GatewayServer

logger = logging.getLogger(__name__)

# Pattern for inline draw-intent tags the AI may emit in chat responses.
# Examples: [DRAW: a red circle], [A DRAW: a red circle]
_DRAW_TAG_RE = re.compile(
    r"\[(?:A\s+)?DRAW:\s*(.+?)\]",
    re.DOTALL | re.IGNORECASE,
)

_DRAW_SAFE_OPCODES = frozenset(
    {
        "SIZE",
        "CANVAS",
        "CIRCLE",
        "RECT",
        "ELLIPSE",
        "LINE",
        "POLYGON",
        "TEXT",
        "STAR",
        "SPIRAL",
        "ARC",
        "BEZIER",
        "GRADIENT",
        "DOTS",
        "OUTPUT",
    }
)
_DRAW_CONTENT_OPCODES = _DRAW_SAFE_OPCODES - {"SIZE", "CANVAS", "OUTPUT"}
_DRAW_MAX_AUTO_LINES = 80
_DRAW_MAX_SIMPLE_DESCRIPTION_CHARS = 360
_DRAW_MAX_SIMPLE_DESCRIPTION_WORDS = 60
_DRAW_MAX_AUTO_ATTEMPTS = 3
_DRAW_COMPLEX_PROMPT_TERMS = frozenset(
    {
        "8k",
        "anime",
        "camera",
        "cinematic",
        "complex",
        "crystal cave",
        "detailed",
        "dramatic",
        "ethereal",
        "gracefully",
        "high quality",
        "illustration prompt",
        "image-generation prompt",
        "lighting",
        "majestic",
        "masterpiece",
        "photorealistic",
        "realistic",
        "render",
        "translucent",
        "ultra",
    }
)

# Pattern for inline skill-invocation tags.
# Example: [SKILL: web_search | query=latest AI news]
_SKILL_TAG_RE = re.compile(
    r"\[SKILL:\s*([^\|\]\n]+?)(?:\s*\|\s*([^\]\n]*))?\]",
    re.IGNORECASE,
)


def _is_timeout_exception(exc: BaseException) -> bool:
    """Return True for common HTTP timeout exception wrappers."""

    timeout_names = {
        "ConnectTimeout",
        "PoolTimeout",
        "ReadTimeout",
        "TimeoutException",
        "WriteTimeout",
    }
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, TimeoutError):
            return True
        if current.__class__.__name__ in timeout_names:
            return True
        current = current.__cause__ or current.__context__
    return False


def _draw_line_has_minimum_args(opcode: str, line: str) -> bool:
    """Return True when a DSL line has enough tokens to be plausibly usable."""

    token_count = len(line.split())
    minimums = {
        "SIZE": 3,
        "CANVAS": 2,
        "CIRCLE": 5,
        "RECT": 6,
        "ELLIPSE": 6,
        "LINE": 6,
        "POLYGON": 7,
        "TEXT": 5,
        "STAR": 7,
        "SPIRAL": 6,
        "ARC": 7,
        "BEZIER": 10,
        "GRADIENT": 7,
        "DOTS": 7,
        "OUTPUT": 1,
    }
    return token_count >= minimums.get(opcode, 1)


def _normalize_draw_dsl(raw_dsl: str) -> str:
    """Keep safe Draw DSL lines and ensure the final command emits an image."""

    normalized: list[str] = []
    has_size = False
    has_canvas = False
    has_content = False

    for raw_line in raw_dsl.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        parts = line.split(maxsplit=1)
        opcode = parts[0].upper().rstrip(":")
        if opcode not in _DRAW_SAFE_OPCODES:
            continue
        normalized_line = f"{opcode} {parts[1]}".strip() if len(parts) > 1 else opcode
        if not _draw_line_has_minimum_args(opcode, normalized_line):
            continue
        if opcode == "OUTPUT":
            continue
        if opcode == "SIZE":
            if has_size:
                continue
            has_size = True
        elif opcode in {"CANVAS", "GRADIENT"}:
            has_canvas = True
        if opcode in _DRAW_CONTENT_OPCODES:
            has_content = True
        normalized.append(normalized_line)
        if len(normalized) >= _DRAW_MAX_AUTO_LINES:
            break

    if not has_content:
        return ""

    if not has_size:
        normalized.insert(0, "SIZE 800 600")
    if not has_canvas:
        canvas_index = (
            1 if normalized and normalized[0].upper().startswith("SIZE") else 0
        )
        normalized.insert(canvas_index, "CANVAS #f8fafc")
    normalized.append("OUTPUT")
    return "\n".join(normalized)


def _is_simple_draw_description(description: str) -> bool:
    """Return True for requests that fit the tiny Draw DSL renderer."""

    compact = " ".join(description.split())
    if not compact:
        return False
    if len(compact) > _DRAW_MAX_SIMPLE_DESCRIPTION_CHARS:
        return False
    words = re.findall(r"[A-Za-z0-9#%-]+", compact)
    if len(words) > _DRAW_MAX_SIMPLE_DESCRIPTION_WORDS:
        return False
    lower = compact.lower()
    if any(term in lower for term in _DRAW_COMPLEX_PROMPT_TERMS):
        return False
    return True


class CoreEngine:
    """Processes messages from the global queue and generates AI responses."""

    def __init__(
        self,
        ai: AIBackend,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        gateway: GatewayServer,
        memory_config: MemoryConfig | None = None,
        react_config: ReActConfig | None = None,
        vision_enabled: bool = False,
        timezone_name: str = "UTC",
    ) -> None:
        self._ai = ai
        self._soul = soul
        self._memory = memory
        self._skills = skills
        self._gateway = gateway
        self._memory_config = memory_config or MemoryConfig()
        self._react_config = react_config or ReActConfig()
        self._vision_enabled = vision_enabled
        self._timezone_name = timezone_name
        self._extractor = MemoryExtractor(ai, soul, memory, self._memory_config)
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._post_process_lock = asyncio.Lock()

    async def handle_message(self, message: GatewayMessage) -> None:
        """Handle a single incoming message from the queue."""
        if message.type == MessageType.LINK_REQUEST:
            await self._handle_link_request(message)
            return

        if message.type == MessageType.LINK_CONFIRM:
            await self._handle_link_confirm(message)
            return

        if message.type != MessageType.MESSAGE:
            return

        try:
            await self._process_chat_message(message)
        except Exception:
            logger.exception(
                "Unexpected error processing message from %s", message.adapter_id
            )
            error_reply = GatewayMessage(
                type=MessageType.ERROR,
                adapter_id=message.adapter_id,
                platform_user_id=message.platform_user_id,
                content="An internal error occurred. Please try again.",
                metadata=self._reply_metadata(message),
            )
            try:
                await self._gateway.send_to_adapter(message.adapter_id, error_reply)
            except Exception:
                logger.exception("Failed to send error reply to %s", message.adapter_id)

    async def _process_chat_message(self, message: GatewayMessage) -> None:
        """Core chat handling — command routing, AI generation, memory storage."""
        # ── Command routing ───────────────────────────────────────────
        text = message.content.strip()
        if text.startswith("/"):
            handled = await self._handle_command(message, text)
            if handled:
                return

        # Phase 1: Resolve user
        user_id, user = await self._resolve_user(message)

        # Voice-context scope: any LLM call inside this block consults
        # ``ai.options.voice_enable_thinking`` instead of
        # ``ai.options.enable_thinking`` so STT-originated messages get
        # a faster (non-thinking) response when configured.
        with voice_context_scope(message.is_voice):
            # Phase 2: Build prompt and generate initial response
            result = await self._generate_response(user_id, user, message)
            if result is None:
                return
            response, system_prompt, user_prompt = result

            # Phase 3: ReAct loop — execute skill tags and re-prompt
            response = await self._react_loop(
                response, message, system_prompt, user_prompt, user_id
            )

            if message.metadata.get("shared_voice"):
                clean_response = _DRAW_TAG_RE.sub("", response).strip()
                if not clean_response:
                    clean_response = (
                        "That action is unavailable in a shared voice channel. "
                        "Please ask me in text instead."
                    )
                draw_images: list[str] = []
            else:
                # Phase 4: Send reply immediately — do NOT wait for post-processing
                clean_response, draw_images = await self._maybe_draw(response)
        reply = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id=message.adapter_id,
            platform_user_id=message.platform_user_id,
            content=clean_response,
            images=draw_images,
            metadata=self._reply_metadata(message),
        )
        await self._gateway.send_to_adapter(message.adapter_id, reply)

        # Phase 5: Background post-processing (memory storage + emotion update).
        # Runs concurrently so the user already has the reply.
        task = asyncio.create_task(
            self._post_process_message(user_id, user, message, clean_response)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    @staticmethod
    def _reply_metadata(message: GatewayMessage) -> dict[str, Any]:
        """Preserve routing metadata while preventing channel replies leaking to DM."""

        metadata = dict(message.metadata or {})
        if metadata.get("is_dm") is False and "allow_dm_fallback" not in metadata:
            metadata["allow_dm_fallback"] = False
        return metadata

    async def drain(self) -> None:
        """Wait for all background post-processing tasks to complete.

        Primarily useful in tests to ensure side effects are visible before
        making assertions.
        """
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

    async def _resolve_user(self, message: GatewayMessage) -> tuple[str, UserSummary]:
        """Resolve or create the user and update their summary."""
        adapter_id = message.adapter_id
        platform_user_id = message.platform_user_id
        display_name = (
            str(message.metadata.get("display_name") or platform_user_id).strip()
            or platform_user_id
        )

        user_id = await self._memory.resolve_user(adapter_id, platform_user_id)
        if user_id is None:
            user_id = uuid.uuid4().hex
            user = await self._memory.get_or_create_user(user_id, display_name)
            await self._memory.link_platform(user_id, adapter_id, platform_user_id)
        else:
            user = await self._memory.get_or_create_user(user_id, display_name)

        if display_name and user.display_name != display_name:
            user.display_name = display_name
        user.last_talked_at = datetime.now(tz=UTC)
        user.last_platform = adapter_id
        await self._memory.update_user_summary(user)

        # Record the channel the user actually messaged us in, so that
        # Heartbeat can route proactive messages back to the same channel
        # instead of falling back to DM (see PR for context).
        channel_id = str(message.metadata.get("channel_id", "") or "")
        if channel_id and not message.metadata.get("via_vc"):
            guild_id = str(message.metadata.get("guild_id", "") or "")
            is_dm = bool(
                message.metadata.get(
                    "is_dm", not guild_id if "guild_id" in message.metadata else False
                )
            )
            try:
                await self._memory.record_last_seen_channel(
                    user_id, adapter_id, channel_id, is_dm
                )
            except Exception:  # pragma: no cover - persistence best-effort
                logger.exception(
                    "Failed to record last-seen channel for user=%s adapter=%s",
                    user_id,
                    adapter_id,
                )

        return user_id, user

    async def _generate_response(
        self,
        user_id: str,
        user: UserSummary,
        message: GatewayMessage,
    ) -> tuple[str, str, str] | None:
        """Build prompt, call AI, return (response, system_prompt, user_prompt).

        Returns None on failure.
        """
        soul_snap = self._soul.get_soul_snapshot()
        md = message.metadata or {}
        shared_voice = bool(md.get("shared_voice"))
        profile = None if shared_voice else await self._memory.get_core_profile(user_id)
        channel_id = str(md.get("channel_id") or "") or None
        is_dm_raw = md.get("is_dm")
        is_dm: bool | None = bool(is_dm_raw) if is_dm_raw is not None else None
        adapter_id = message.adapter_id or None
        history_limit = (
            self._memory_config.voice_conversation_history_limit
            if message.is_voice
            else self._memory_config.conversation_history_limit
        )
        history = (
            []
            if shared_voice
            else await self._memory.get_recent_messages(
                user_id,
                limit=history_limit,
                channel_id=channel_id,
                is_dm=is_dm,
                adapter_id=adapter_id,
            )
        )
        message_count = (
            None if shared_voice else await self._memory.count_messages(user_id)
        )

        system_prompt = build_soul_system_prompt(
            soul_snap,
            timezone_name=self._timezone_name,
            user_message_count=message_count,
        )
        if shared_voice:
            system_prompt += (
                "\n\nYou are participating in a shared voice channel with multiple"
                " people. The user message is a short room transcript with speaker"
                " labels. Reply to the group, not to a private individual. Never"
                " reveal or infer private personal memories. Keep the spoken reply"
                " concise and natural, usually one or two sentences. Safe skills"
                " explicitly enabled for shared voice may be used. Drawings and"
                " actions requiring confirmation are unavailable in shared voice"
                " channels; never emit [DRAW: ...] tags."
            )
        if not message.is_voice and self._skills.get("draw") is not None:
            system_prompt += (
                "\n\nYou can create simple vector-style drawings for the user"
                " through CordBeat's tiny Draw DSL renderer. This is not an"
                " image-generation model and cannot create complex"
                " illustrations, character art, photorealism, anime-style art,"
                " detailed fantasy scenes, camera/lighting/composition effects,"
                " or rich prompt-based images. Use it only when the user"
                " explicitly asks for a simple DSL-friendly sketch, diagram,"
                " icon, poster, or geometric drawing. If the user asks for a"
                " complex illustration, explain that Draw can only make simple"
                " vector-style DSL sketches and ask for a simpler target."
                " When appropriate, include exactly one short"
                " [DRAW: <simple shape/object description in English>] tag."
                " Example: [DRAW: a red circle on a white background]."
                " The tag will be converted to Draw DSL, rendered, and sent"
                " with your reply."
                " Do not call the draw skill directly with [SKILL: draw];"
                " use the [DRAW: ...] tag instead."
            )

        # Inject available skill catalog so the AI knows what tools it can call.
        skills_desc = ""
        if shared_voice:
            skills_desc = self._skills.get_skill_descriptions_for_prompt(
                exclude_names={"draw"},
                context="shared_voice",
            )
        elif not message.is_voice:
            skills_desc = self._skills.get_skill_descriptions_for_prompt(
                exclude_names={"draw"}
            )
        if skills_desc and skills_desc != "(no skills available)":
            system_prompt += (
                "\n\nYou have access to the following tools. To USE a tool you"
                " MUST include a [SKILL: <name> | <param>=<value>] tag in your"
                " reply — multiple tags per reply are allowed (they execute in"
                " order). Only safe tools run automatically; others are queued"
                " for approval."
                "\n\n**STRICT RULE**: If you state in natural language that you"
                " will look something up, search, check, investigate, fetch,"
                " confirm, or perform any other action that needs a skill"
                ", you MUST include the corresponding [SKILL: ...] tag in"
                " the SAME reply. Do NOT promise an action without emitting the"
                " tag — that produces dishonest replies and frustrates users."
                " Drawing is separate: never use [SKILL: draw]; only use"
                " [DRAW: ...] when the Draw DSL guidance says it is appropriate."
                " Conversely, if you have no need for a tool, do not promise one."
                "\nExample: [SKILL: web_search | query=latest AI news]"
                "\nExample (multi): [SKILL: fetch_url | url=https://example.com]"
                " followed by your reply."
                f"\nAvailable tools:\n{skills_desc}"
            )

        # Phase 1: Direct keyword search (message.content → vector search)
        semantic_memories: list[dict[str, Any]] = []
        episodic_memories: list[dict[str, Any]] = []
        if not shared_voice:
            semantic_memories = await self._memory.search_semantic(
                user_id,
                message.content,
                n_results=self._memory_config.memory_search_results,
            )
            episodic_memories = await self._memory.search_episodic(
                user_id,
                message.content,
                n_results=self._memory_config.memory_search_results,
            )

        # Phase 2: Context inference recall (AI extracts keywords → search).
        # Voice replies skip this extra LLM round-trip by default to preserve
        # conversational latency; direct vector and history recall still run.
        recall_keywords: list[str] = []
        if not shared_voice and (
            not message.is_voice or self._memory_config.voice_recall_keywords_enabled
        ):
            recall_keywords = await self._extractor.extract_recall_keywords(
                message.content, history or None
            )
        else:
            logger.debug("Skipping LLM recall-keyword extraction for voice message")
        seen_ids = {m["id"] for m in semantic_memories + episodic_memories}
        for keyword in recall_keywords:
            for mem in await self._memory.search_semantic(
                user_id,
                keyword,
                n_results=self._memory_config.recall_keyword_search_results,
            ):
                if mem["id"] not in seen_ids:
                    semantic_memories.append(mem)
                    seen_ids.add(mem["id"])
            for mem in await self._memory.search_episodic(
                user_id,
                keyword,
                n_results=self._memory_config.recall_keyword_search_results,
            ):
                if mem["id"] not in seen_ids:
                    episodic_memories.append(mem)
                    seen_ids.add(mem["id"])

        # Phase 3: Emotion association recall (current emotion → tag search)
        current_emotion = soul_snap["emotion"]["primary"]
        if not shared_voice and current_emotion and current_emotion != "calm":
            for mem in await self._memory.search_by_emotion(
                user_id,
                current_emotion,
                message.content,
                n_results=self._memory_config.emotion_recall_search_results,
            ):
                if mem["id"] not in seen_ids:
                    episodic_memories.append(mem)
                    seen_ids.add(mem["id"])

        # Phase 4a: Chain recall (precomputed associative links)
        if not shared_voice:
            try:
                recalled_ids = list(seen_ids)
                chain_contents = await self._memory.get_chain_links(
                    user_id,
                    recalled_ids,
                    max_depth=self._memory_config.chain_recall_max_depth,
                )
                existing_contents = {
                    m["content"] for m in semantic_memories + episodic_memories
                }
                for chain_text in chain_contents:
                    if chain_text not in existing_contents:
                        episodic_memories.append(
                            {"id": f"chain_{hash(chain_text)}", "content": chain_text}
                        )
                        existing_contents.add(chain_text)
            except Exception:
                logger.debug("Chain recall failed for user %s", user_id)

        # Phase 4b: Precomputed temporal recall hints
        hints: list[str] = []
        if not shared_voice:
            try:
                raw_hints = await self._memory.get_recall_hints(user_id)
                hints = [h["content"] for h in raw_hints if h.get("content")]
            except Exception:
                logger.debug("Recall hints lookup failed for user %s", user_id)

        context = build_context(
            user_display_name=(
                "participants in a shared voice channel"
                if shared_voice
                else user.display_name
            ),
            profile=profile or None,
            semantic_memories=semantic_memories or None,
            episodic_memories=episodic_memories or None,
            recall_hints=hints or None,
            history=history or None,
            soul_name=soul_snap["name"],
            max_user_input_len=self._memory_config.max_user_input_len,
        )

        safe_content = sanitize(
            message.content, max_len=self._memory_config.max_user_input_len
        )
        prompt = f"{context}\n\nUser says: {safe_content}"

        logger.debug(
            "[AI INPUT] system_prompt(%d chars):\n%s",
            len(system_prompt),
            system_prompt,
        )
        logger.debug(
            "[AI INPUT] full_prompt(%d chars):\n%s",
            len(prompt),
            prompt,
        )

        try:
            if message.images and not self._vision_enabled:
                logger.warning(
                    "Ignoring %d image(s) from %s because "
                    "ai_backend.vision_enabled is false",
                    len(message.images),
                    message.adapter_id,
                )
            if self._vision_enabled and message.images:
                try:
                    logger.debug(
                        "Generating vision response with %d image(s) from %s",
                        len(message.images),
                        message.adapter_id,
                    )
                    raw = await self._ai.generate_with_vision(
                        prompt=prompt,
                        images=message.images,
                        system=system_prompt,
                    )
                    cleaned = sanitize_reasoning_artifacts(raw)
                    return cleaned, system_prompt, prompt
                except Exception:
                    logger.warning(
                        "Vision generation failed for %d image(s), "
                        "falling back to text-only response",
                        len(message.images),
                        exc_info=True,
                    )
            raw = await self._generate_text_with_timeout_retry(prompt, system_prompt)
            cleaned = sanitize_reasoning_artifacts(raw)
            logger.debug(
                "[AI OUTPUT] raw(%d chars):\n%s",
                len(raw),
                raw,
            )
            logger.debug("[AI OUTPUT] cleaned: %s", cleaned)
            if not cleaned:
                logger.warning(
                    "[AI OUTPUT] empty response — model returned nothing useful"
                )
                error_reply = GatewayMessage(
                    type=MessageType.ERROR,
                    adapter_id=message.adapter_id,
                    platform_user_id=message.platform_user_id,
                    content="The AI could not generate a response. Please try again.",
                    metadata=self._reply_metadata(message),
                )
                await self._gateway.send_to_adapter(message.adapter_id, error_reply)
                return None
            return cleaned, system_prompt, prompt
        except Exception:
            logger.exception("AI generation failed")
            error_reply = GatewayMessage(
                type=MessageType.ERROR,
                adapter_id=message.adapter_id,
                platform_user_id=message.platform_user_id,
                content="AI generation failed. Please try again later.",
                metadata=self._reply_metadata(message),
            )
            await self._gateway.send_to_adapter(message.adapter_id, error_reply)
            return None

    async def _generate_text_with_timeout_retry(
        self,
        prompt: str,
        system_prompt: str,
    ) -> str:
        """Generate chat text, retrying timeouts once with no-think constraints."""

        try:
            return await self._ai.generate(prompt=prompt, system=system_prompt)
        except Exception as exc:
            if not _is_timeout_exception(exc):
                raise

        retry_system = (
            system_prompt
            + "\n/no_think\n"
            "The previous generation timed out. Reply concisely and directly. "
            "Do not use hidden reasoning."
        )
        logger.warning(
            "AI generation timed out; retrying once with no-think short output"
        )
        return await self._ai.generate(
            prompt=prompt,
            system=retry_system,
            temperature=0.5,
            max_tokens=1024,
        )

    async def _post_process_message(
        self,
        user_id: str,
        user: UserSummary,
        message: GatewayMessage,
        response: str,
    ) -> None:
        """Background task: persist conversation + run emotion/memory extraction."""
        try:
            md = message.metadata or {}
            if md.get("ephemeral"):
                logger.debug("Skipping persistence for ephemeral conversation")
                return
            channel_id = str(md.get("channel_id") or "")
            is_dm_raw = md.get("is_dm")
            is_dm = bool(is_dm_raw) if is_dm_raw is not None else True
            stored_user_content = sanitize_tool_artifacts(message.content)
            await self._memory.add_message(
                user_id,
                "user",
                stored_user_content,
                message.adapter_id,
                channel_id=channel_id,
                is_dm=is_dm,
            )
            stored_response = sanitize_tool_artifacts(
                sanitize_reasoning_artifacts(response)
            )
            await self._memory.add_message(
                user_id,
                "assistant",
                stored_response,
                message.adapter_id,
                channel_id=channel_id,
                is_dm=is_dm,
            )
            if (
                message.is_voice
                and not self._memory_config.voice_memory_extraction_enabled
            ):
                logger.debug(
                    "Stored voice conversation without LLM emotion/memory extraction"
                )
                return

            # Background tasks from rapid messages must not fan out into
            # concurrent calls against a single local LLM server.
            async with self._post_process_lock:
                await self._extractor.infer_and_update_emotion(
                    user_id, stored_user_content, stored_response
                )
                await self._extractor.extract_and_store_memories(
                    user_id, user.display_name, stored_user_content, stored_response
                )
        except Exception:
            logger.exception("Background post-processing failed for user %s", user_id)

    async def _react_loop(
        self,
        initial_response: str,
        message: GatewayMessage,
        system_prompt: str,
        user_prompt: str,
        user_id: str,
    ) -> str:
        """ReAct: multi-turn skill execution loop (D1-D16).

        Returns the final text response (all skill tags resolved or
        max iterations exhausted).
        """
        if not self._react_config.enabled:
            return _SKILL_TAG_RE.sub("", initial_response).strip()

        response = initial_response
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]
        trace = ToolTrace()
        shared_voice = bool(message.metadata.get("shared_voice"))

        for iteration in range(self._react_config.max_iterations):
            tags = list(_SKILL_TAG_RE.finditer(response))
            if not tags:
                break  # clean response, no more tools needed

            # D13: Extract pre-tag text and flush to user immediately
            first_tag_start = tags[0].start()
            pre_text = response[:first_tag_start].strip()
            if pre_text and not shared_voice:
                pre_msg = GatewayMessage(
                    type=MessageType.MESSAGE,
                    adapter_id=message.adapter_id,
                    platform_user_id=message.platform_user_id,
                    content=pre_text,
                    metadata=self._reply_metadata(message),
                )
                try:
                    await self._gateway.send_to_adapter(message.adapter_id, pre_msg)
                except Exception:
                    logger.warning(
                        "Failed to send pre-tool text to %s", message.adapter_id
                    )

            # D9/D15: Execute all tags in order
            results: list[ToolCallResult] = []
            stopped_early = False
            status_sent = False
            for m in tags:
                skill_name = m.group(1).strip()
                params_raw = (m.group(2) or "").strip()
                params: dict[str, Any] = {}
                for part in params_raw.split("|"):
                    part = part.strip()
                    if "=" in part:
                        k, _, v = part.partition("=")
                        params[k.strip()] = v.strip()

                skill = self._skills.get(skill_name)
                if skill is None or not skill.meta.enabled:
                    logger.debug("ReAct: unknown/disabled skill %r", skill_name)
                    results.append(
                        ToolCallResult(
                            skill_name=skill_name,
                            params=params,
                            output="Skill not found",
                            is_error=True,
                        )
                    )
                    continue

                if shared_voice and not skill.meta.shared_voice_enabled:
                    logger.info(
                        "ReAct: blocking skill %r disabled for shared voice",
                        skill_name,
                    )
                    results.append(
                        ToolCallResult(
                            skill_name=skill_name,
                            params=params,
                            output="Unavailable in shared voice",
                            is_error=True,
                        )
                    )
                    continue

                if skill.meta.safety_level != SafetyLevel.SAFE:
                    if shared_voice:
                        logger.info(
                            "ReAct: blocking non-safe skill %r in shared voice",
                            skill_name,
                        )
                        results.append(
                            ToolCallResult(
                                skill_name=skill_name,
                                params=params,
                                output="Unavailable in shared voice",
                                is_error=True,
                            )
                        )
                        continue
                    logger.info(
                        "ReAct: skill %r requires confirmation; requesting approval",
                        skill_name,
                    )
                    proposal_id = await self._request_skill_confirmation(
                        user_id=user_id,
                        message=message,
                        skill_name=skill_name,
                        skill_params=params,
                    )
                    results.append(
                        ToolCallResult(
                            skill_name=skill_name,
                            params=params,
                            output=f"Approval required: {proposal_id}",
                            is_error=True,
                        )
                    )
                    stopped_early = True
                    break

                if pre_text and not status_sent and not shared_voice:
                    status = GatewayMessage(
                        type=MessageType.ACK,
                        adapter_id=message.adapter_id,
                        platform_user_id=message.platform_user_id,
                        content="🔧 Running tool…",
                        metadata=self._reply_metadata(message),
                    )
                    try:
                        await self._gateway.send_to_adapter(message.adapter_id, status)
                    except Exception:
                        logger.warning(
                            "Failed to send tool-running status to %s",
                            message.adapter_id,
                        )
                    status_sent = True

                try:
                    result = await skill.execute(params, memory=self._memory)
                    output = str(result.get("output", result.get("result", ""))).strip()
                    results.append(
                        ToolCallResult(
                            skill_name=skill_name, params=params, output=output
                        )
                    )
                    logger.debug(
                        "ReAct iter %d: %r → %d chars",
                        iteration,
                        skill_name,
                        len(output),
                    )
                except Exception:
                    logger.warning("ReAct: skill %r failed", skill_name, exc_info=True)
                    results.append(
                        ToolCallResult(
                            skill_name=skill_name,
                            params=params,
                            output="Tool execution failed",
                            is_error=True,
                        )
                    )

            trace.calls.extend(results)
            trace.iterations = iteration + 1

            if stopped_early:
                return (
                    "🔧 This action requires approval. "
                    "Use the displayed confirmation or run "
                    "/approve <proposal_id> to proceed."
                )

            if not results:
                break

            # Build continuation prompt and get next AI response
            continuation = build_react_continuation_prompt(
                results,
                max_tool_output_chars=self._react_config.max_tool_output_chars,
            )
            messages.append({"role": "user", "content": continuation})

            try:
                raw = await self._ai.generate_chat(
                    messages,
                    max_tokens=self._react_config.max_tool_output_chars,
                )
            except Exception:
                logger.warning(
                    "ReAct: continuation generate_chat failed, using last response"
                )
                break

            cleaned = sanitize_reasoning_artifacts(raw)
            logger.debug(
                "ReAct iter %d response: %d chars", iteration + 1, len(cleaned)
            )
            if not cleaned:
                break
            response = cleaned
            messages.append({"role": "assistant", "content": response})

        logger.debug(
            "ReAct trace: %d iterations, %d tool calls",
            trace.iterations,
            len(trace.calls),
        )
        # C-1: Strip any leaked [SKILL: ...] tags before returning so users
        # never see raw tags. C-2: If max_iterations was exhausted and the
        # final response is empty after stripping (AI emitted only tags
        # with no natural language), fall back to a generic acknowledgement
        # so the user still gets a reply.
        cleaned_final = _SKILL_TAG_RE.sub("", response).strip()
        if not cleaned_final and trace.calls:
            cleaned_final = (
                "(The tool finished, but I could not generate a summary reply. "
                "Please ask again.)"
            )
        return cleaned_final

    async def _request_skill_confirmation(
        self,
        *,
        user_id: str,
        message: GatewayMessage,
        skill_name: str,
        skill_params: dict[str, Any],
    ) -> str:
        """Persist a skill-execution proposal and ask the adapter to confirm it."""
        metadata: dict[str, Any] = {
            "status": ProposalStatus.PENDING,
            "proposal_type": ProposalType.SKILL_EXECUTION,
            "skill_name": skill_name,
            "skill_params": skill_params,
            "adapter_id": message.adapter_id,
        }
        content = (
            f"Skill '{skill_name}' requires confirmation.\n"
            f"Parameters: {json.dumps(skill_params, ensure_ascii=False)}"
        )
        proposal_id = await self._memory.add_certain_record(
            user_id=user_id,
            content=content,
            record_type="proposal",
            metadata=metadata,
        )
        confirm = GatewayMessage(
            type=MessageType.SKILL_CONFIRM,
            adapter_id=message.adapter_id,
            platform_user_id=message.platform_user_id,
            content=f"🔧 Skill '{skill_name}' requires approval.",
            metadata={
                **self._reply_metadata(message),
                "proposal_id": proposal_id,
                "skill_name": skill_name,
                "skill_params": skill_params,
            },
        )
        await self._gateway.send_to_adapter(message.adapter_id, confirm)
        return proposal_id

    async def _maybe_draw(self, response: str) -> tuple[str, list[str]]:
        """Parse [DRAW: ...] tags from LLM response and execute the draw skill.

        Returns (clean_text, images). Only the first tag is processed.
        Falls back to (original_response, []) on any failure.
        """
        if self._skills.get("draw") is None:
            return response, []

        matches = _DRAW_TAG_RE.findall(response)
        if not matches:
            return response, []

        clean_text = _DRAW_TAG_RE.sub("", response).strip()
        description = matches[0].strip()

        if not _is_simple_draw_description(description):
            logger.warning(
                "Skipping auto-draw because description is too complex "
                "for Draw DSL: %r",
                description,
            )
            limitation = (
                "⚠️ I couldn't attach a drawing because Draw only supports "
                "simple vector-style DSL sketches, not complex image-generation "
                "prompts. Please ask for a simpler sketch, diagram, icon, or "
                "geometric drawing."
            )
            return limitation, []

        skill = self._skills.get("draw")
        if skill is None:
            return clean_text, []

        retry_reason: str | None = None
        for attempt in range(_DRAW_MAX_AUTO_ATTEMPTS):
            raw_dsl = await self._generate_draw_dsl(
                description,
                retry_reason=retry_reason,
            )
            dsl = _normalize_draw_dsl(raw_dsl)
            source = f"llm_attempt_{attempt + 1}"
            if not dsl:
                retry_reason = "previous attempt produced no valid Draw DSL commands"
                logger.warning(
                    "Auto-draw DSL generation produced no usable commands "
                    "(source=%s, description=%r, raw_len=%d)",
                    source,
                    description,
                    len(raw_dsl),
                )
                continue

            output, result = await self._execute_auto_draw(
                skill,
                dsl,
                description=description,
                source=source,
            )
            if output:
                review_reason = await self._review_auto_draw(
                    description=description,
                    dsl=dsl,
                    image_b64=output,
                    warnings=result.get("warnings") if isinstance(result, dict) else [],
                )
                if not review_reason:
                    return clean_text, [output]
                retry_reason = review_reason
                logger.info(
                    "Auto-draw review requested retry "
                    "(source=%s, description=%r, reason=%s)",
                    source,
                    description,
                    retry_reason,
                )
                continue
            retry_reason = "previous Draw DSL executed but produced no image"

        failure_note = "⚠️ Drawing failed, so I couldn't attach an image."
        return failure_note, []

    async def _execute_auto_draw(
        self,
        skill: Any,
        dsl: str,
        *,
        description: str,
        source: str,
    ) -> tuple[str, dict[str, Any]]:
        """Execute Draw DSL for auto-draw and return (base64 image, raw result)."""

        try:
            result = await skill.execute({"commands": dsl}, memory=self._memory)
        except Exception:
            logger.exception(
                "Auto-draw skill execution failed "
                "(source=%s, description=%r, dsl_len=%d)",
                source,
                description,
                len(dsl),
            )
            return "", {"error": "skill execution failed"}

        if not isinstance(result, dict):
            logger.warning(
                "Auto-draw skill returned non-dict result "
                "(source=%s, description=%r): %r",
                source,
                description,
                result,
            )
            return "", {"error": "non-dict skill result"}

        output = result.get("output")
        if isinstance(output, str) and output and "error" not in result:
            warnings = result.get("warnings") or []
            if warnings:
                logger.debug(
                    "Auto-draw completed with warnings (source=%s, description=%r): %s",
                    source,
                    description,
                    warnings,
                )
            return output, result

        logger.warning(
            "Auto-draw produced no image "
            "(source=%s, description=%r, error=%r, warnings=%r, dsl_len=%d)",
            source,
            description,
            result.get("error"),
            result.get("warnings"),
            len(dsl),
        )
        return "", result

    async def _review_auto_draw(
        self,
        *,
        description: str,
        dsl: str,
        image_b64: str,
        warnings: Any,
    ) -> str | None:
        """Return a retry reason when the rendered Draw image should be redone."""

        severe_warnings = [
            str(w)
            for w in warnings or []
            if any(
                marker in str(w).lower()
                for marker in (
                    "unknown command",
                    "requires",
                    "error",
                    "invalid",
                    "raised",
                )
            )
        ]
        if severe_warnings:
            return (
                "previous Draw DSL rendered with execution warnings: "
                f"{sanitize('; '.join(severe_warnings), strict=True, max_len=180)}"
            )

        if not self._vision_enabled:
            return None

        system = (
            "/no_think\n"
            "You review simple Draw DSL renderings. Return ONLY compact JSON: "
            '{"verdict":"pass"} or {"verdict":"retry","reason":"short reason"}. '
            "Retry only when the image is blank, broken, unreadable, or clearly "
            "does not match the requested simple drawing. Do not request style "
            "polish or complex image-generation details."
        )
        prompt = (
            "Requested drawing:\n"
            f"{sanitize(description, max_len=300)}\n\n"
            "Draw DSL used:\n"
            f"{sanitize(dsl, strict=True, max_len=1400)}\n\n"
            "Check whether the attached image is an acceptable simple "
            "vector-style rendering of the request."
        )
        try:
            raw = await self._ai.generate_with_vision(
                prompt=prompt,
                images=[image_b64],
                system=system,
                temperature=0.0,
                max_tokens=160,
            )
        except Exception:
            logger.warning(
                "Auto-draw vision review failed; accepting image",
                exc_info=True,
            )
            return None

        cleaned = sanitize_reasoning_artifacts(raw).strip()
        try:
            review = json.loads(cleaned)
        except json.JSONDecodeError:
            if "retry" not in cleaned.lower():
                return None
            return "vision review requested retry"

        verdict = str(review.get("verdict", "")).strip().lower()
        if verdict != "retry":
            return None
        reason = str(review.get("reason") or "vision review requested retry")
        return sanitize(reason, strict=True, max_len=180)

    async def _generate_draw_dsl(
        self,
        description: str,
        *,
        retry_reason: str | None = None,
    ) -> str:
        """Ask the AI to produce Draw DSL for the given plain-text description.

        Returns an empty string on failure so callers can skip gracefully.
        """
        safe_desc = sanitize(description, max_len=500)
        retry_line = ""
        if retry_reason:
            retry_line = (
                "\nPrevious attempt failed: "
                f"{sanitize(retry_reason, strict=True, max_len=160)}"
                "\nProduce a complete alternative Draw DSL for the same request."
            )
        system = (
            "/no_think\n"
            "You are a drawing DSL generator. "
            "Given a description, output ONLY valid Draw DSL"
            " commands — no prose, no markdown fences.\n"
            "Use at most 35 lines. Prefer a simple poster-like composition. "
            "Never use SAVE. Always end with OUTPUT as the final line.\n"
            "Available commands (one per line):\n"
            "  SIZE <width> <height>\n"
            "  CANVAS <color>\n"
            "  CIRCLE <cx> <cy> <radius> <color> [FILL]\n"
            "  RECT <x1> <y1> <x2> <y2> <color> [FILL]\n"
            "  ELLIPSE <x1> <y1> <x2> <y2> <color> [FILL]\n"
            "  LINE <x1> <y1> <x2> <y2> <color> [width]\n"
            "  POLYGON <x1> <y1> <x2> <y2> ... <color> [FILL]\n"
            '  TEXT <x> <y> "<text>" <color> [size]\n'
            "  STAR <cx> <cy> <outer_r> <inner_r> <points> <color> [FILL]\n"
            "  SPIRAL <cx> <cy> <turns> <spacing> <color> [width]\n"
            "  OUTPUT [PNG]\n"
            "Colors: named colors (white, red, blue, ...) or #RRGGBB."
            " Always end with OUTPUT."
        )
        prompt = f"Draw this: {safe_desc}{retry_line}"
        try:
            raw = await self._ai.generate(
                prompt=prompt,
                system=system,
                temperature=0.2,
                max_tokens=1200,
            )
            return raw.strip()
        except Exception:
            logger.exception("DSL generation failed for description=%r", safe_desc)
            return ""

    async def _handle_link_request(self, message: GatewayMessage) -> None:
        """Generate a link token and send it back to the requester."""
        token = await self._memory.store_link_token(
            requester_adapter_id=message.adapter_id,
            requester_platform_user_id=message.platform_user_id,
        )
        reply = GatewayMessage(
            type=MessageType.ACK,
            adapter_id=message.adapter_id,
            platform_user_id=message.platform_user_id,
            content=(
                f"Link token generated: {token}\n"
                "Send this token from your other platform "
                "using the link confirm command."
            ),
            metadata=self._reply_metadata(message),
        )
        await self._gateway.send_to_adapter(message.adapter_id, reply)
        logger.info(
            "Link token issued for %s on %s",
            message.platform_user_id,
            message.adapter_id,
        )
        await self._audit_link_event(
            message, "link_request", f"Token issued on {message.adapter_id}"
        )

    async def _handle_link_confirm(self, message: GatewayMessage) -> None:
        """Verify a link token and merge the requester's platform."""
        token = message.content.strip()
        result = await self._memory.verify_link_token(token)

        if result is None:
            reply = GatewayMessage(
                type=MessageType.ERROR,
                adapter_id=message.adapter_id,
                platform_user_id=message.platform_user_id,
                content="Invalid or expired link token.",
                metadata=self._reply_metadata(message),
            )
            await self._gateway.send_to_adapter(message.adapter_id, reply)
            return

        # Resolve the confirmer's user_id (must be an existing user)
        confirmer_user_id = await self._memory.resolve_user(
            message.adapter_id, message.platform_user_id
        )
        if confirmer_user_id is None:
            reply = GatewayMessage(
                type=MessageType.ERROR,
                adapter_id=message.adapter_id,
                platform_user_id=message.platform_user_id,
                content="You must have an existing account to confirm a link.",
                metadata=self._reply_metadata(message),
            )
            await self._gateway.send_to_adapter(message.adapter_id, reply)
            return

        # Link the requester's platform to the confirmer's user
        requester_adapter_id = result["requester_adapter_id"]
        requester_platform_user_id = result["requester_platform_user_id"]
        await self._memory.link_platform(
            confirmer_user_id,
            requester_adapter_id,
            requester_platform_user_id,
        )

        reply = GatewayMessage(
            type=MessageType.ACK,
            adapter_id=message.adapter_id,
            platform_user_id=message.platform_user_id,
            content=(
                f"Account linked! Platform {requester_adapter_id} "
                "is now connected to your account."
            ),
            metadata=self._reply_metadata(message),
        )
        await self._gateway.send_to_adapter(message.adapter_id, reply)
        logger.info(
            "Account linked: %s/%s -> user %s",
            requester_adapter_id,
            requester_platform_user_id,
            confirmer_user_id,
        )
        await self._audit_link_event(
            message,
            "link_confirm",
            f"Linked {requester_adapter_id}/{requester_platform_user_id} "
            f"to user {confirmer_user_id}",
        )

    # ── Slash Commands ────────────────────────────────────────────────

    async def _handle_command(self, message: GatewayMessage, text: str) -> bool:
        """Route slash commands. Returns True if the command was handled."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        handlers: dict[str, Callable[[], Coroutine[Any, Any, None]]] = {
            "/approve": lambda: self._cmd_approve(message, arg),
            "/reject": lambda: self._cmd_reject(message, arg),
            "/proposals": lambda: self._cmd_proposals(message),
            "/link": lambda: self._cmd_link(message),
            "/unlink": lambda: self._cmd_unlink(message, arg),
            "/name": lambda: self._cmd_name(message, arg),
            "/quiet": lambda: self._cmd_quiet(message, arg),
            "/prefer": lambda: self._cmd_prefer(message, arg),
            "/draw": lambda: self._cmd_draw(message, arg),
        }

        handler = handlers.get(cmd)
        if handler is None:
            return False
        await handler()
        return True

    async def _cmd_approve(self, message: GatewayMessage, proposal_id: str) -> None:
        """Approve a pending proposal."""
        if not proposal_id:
            await self._send_reply(message, "Usage: /approve <proposal_id>")
            return

        user_id = await self._memory.resolve_user(
            message.adapter_id, message.platform_user_id
        )
        if user_id is None:
            await self._send_reply(message, "User not found.")
            return

        proposal = await self._memory.get_proposal(proposal_id)
        if proposal is None:
            await self._send_reply(message, "Proposal not found.")
            return

        # Verify ownership: proposal must belong to this user
        if proposal["user_id"] != user_id:
            await self._send_reply(message, "Proposal not found.")
            return

        meta = json.loads(proposal.get("metadata") or "{}")
        status = meta.get("status", "")
        if status != ProposalStatus.PENDING:
            await self._send_reply(
                message,
                f"Cannot approve: proposal status is '{status}'.",
            )
            return

        await self._memory.update_proposal_status(proposal_id, ProposalStatus.APPROVED)
        await self._send_reply(
            message,
            f"✅ Proposal approved: {proposal['content'][:80]}",
        )
        logger.info("Proposal %s approved by user %s", proposal_id, user_id)

    async def _cmd_reject(self, message: GatewayMessage, proposal_id: str) -> None:
        """Reject a pending proposal."""
        if not proposal_id:
            await self._send_reply(message, "Usage: /reject <proposal_id>")
            return

        user_id = await self._memory.resolve_user(
            message.adapter_id, message.platform_user_id
        )
        if user_id is None:
            await self._send_reply(message, "User not found.")
            return

        proposal = await self._memory.get_proposal(proposal_id)
        if proposal is None:
            await self._send_reply(message, "Proposal not found.")
            return

        if proposal["user_id"] != user_id:
            await self._send_reply(message, "Proposal not found.")
            return

        meta = json.loads(proposal.get("metadata") or "{}")
        status = meta.get("status", "")
        if status != ProposalStatus.PENDING:
            await self._send_reply(
                message,
                f"Cannot reject: proposal status is '{status}'.",
            )
            return

        await self._memory.update_proposal_status(proposal_id, ProposalStatus.REJECTED)
        await self._send_reply(
            message,
            f"❌ Proposal rejected: {proposal['content'][:80]}",
        )
        logger.info("Proposal %s rejected by user %s", proposal_id, user_id)

    async def _cmd_proposals(self, message: GatewayMessage) -> None:
        """List pending proposals for the current user."""
        user_id = await self._memory.resolve_user(
            message.adapter_id, message.platform_user_id
        )
        if user_id is None:
            await self._send_reply(message, "User not found.")
            return

        proposals = await self._memory.get_pending_proposals(
            user_id=user_id, status=ProposalStatus.PENDING
        )
        if not proposals:
            await self._send_reply(message, "No pending proposals.")
            return

        lines = ["📋 Pending proposals:\n"]
        for p in proposals:
            content = p["content"][:60]
            lines.append(f"  • {p['id'][:8]}… — {content}")
        await self._send_reply(message, "\n".join(lines))

    async def _cmd_link(self, message: GatewayMessage) -> None:
        """Generate a link token for cross-platform account linking."""
        token = await self._memory.store_link_token(
            requester_adapter_id=message.adapter_id,
            requester_platform_user_id=message.platform_user_id,
        )
        await self._send_reply(
            message,
            f"Link token generated: {token}\n"
            "Send this token from your other platform "
            "using the link confirm command.",
        )
        await self._audit_link_event(
            message, "link_request", f"Token issued on {message.adapter_id}"
        )

    async def _cmd_unlink(self, message: GatewayMessage, platform: str) -> None:
        """Remove a platform link from the current user."""
        if not platform:
            await self._send_reply(message, "Usage: /unlink <platform>")
            return

        user_id = await self._memory.resolve_user(
            message.adapter_id, message.platform_user_id
        )
        if user_id is None:
            await self._send_reply(message, "User not found.")
            return

        # Prevent unlinking the current platform (last remaining link check)
        links = await self._memory.get_linked_platforms(user_id)
        if len(links) <= 1:
            await self._send_reply(
                message, "Cannot unlink: you must have at least one linked platform."
            )
            return

        target = platform.lower()
        if target == message.adapter_id:
            await self._send_reply(
                message,
                "Cannot unlink the platform you are currently using. "
                "Send /unlink from a different platform.",
            )
            return

        removed = await self._memory.unlink_platform(user_id, target)
        if not removed:
            await self._send_reply(
                message, f"Platform '{target}' is not linked to your account."
            )
            return

        await self._send_reply(
            message, f"✅ Platform '{target}' has been unlinked from your account."
        )
        await self._audit_link_event(
            message, "unlink", f"Unlinked {target} from user {user_id}"
        )

    async def _cmd_name(self, message: GatewayMessage, name: str) -> None:
        """Change the SOUL character name."""
        if not name:
            current = self._soul.name
            await self._send_reply(
                message, f"Current name: {current}\nUsage: /name <new_name>"
            )
            return

        self._soul.update_name(name, caller=SoulCaller.USER)
        await self._send_reply(message, f"✅ Name updated to: {name}")
        logger.info("SOUL name changed to '%s'", name)

    async def _cmd_quiet(self, message: GatewayMessage, arg: str) -> None:
        """Update HEARTBEAT quiet hours."""
        if not arg:
            start, end = self._soul.quiet_hours
            await self._send_reply(
                message,
                f"Current quiet hours: {start} - {end}\n"
                "Usage: /quiet <start> <end>  (e.g. /quiet 01:00 07:00)",
            )
            return

        parts = arg.split()
        if len(parts) != 2:
            await self._send_reply(
                message,
                "Usage: /quiet <start> <end>  (e.g. /quiet 01:00 07:00)",
            )
            return

        start, end = parts
        # Basic HH:MM validation
        if not re.fullmatch(r"\d{1,2}:\d{2}", start) or not re.fullmatch(
            r"\d{1,2}:\d{2}", end
        ):
            await self._send_reply(
                message, "Invalid time format. Use HH:MM (e.g. 01:00)."
            )
            return

        self._soul.update_quiet_hours(start, end, caller=SoulCaller.USER)
        await self._send_reply(message, f"✅ Quiet hours updated: {start} - {end}")
        logger.info("Quiet hours changed to %s - %s", start, end)

    async def _cmd_prefer(self, message: GatewayMessage, platform: str) -> None:
        """Set preferred reply platform for HEARTBEAT messages."""
        user_id = await self._memory.resolve_user(
            message.adapter_id, message.platform_user_id
        )
        if user_id is None:
            await self._send_reply(message, "User not found.")
            return

        if not platform:
            user = await self._memory.get_or_create_user(
                user_id, message.platform_user_id
            )
            current = user.preferred_platform or "(not set)"
            await self._send_reply(
                message,
                f"Preferred platform: {current}\n"
                "Usage: /prefer <platform>  (e.g. /prefer discord)\n"
                "Use /prefer clear to reset.",
            )
            return

        user = await self._memory.get_or_create_user(user_id, message.platform_user_id)
        if platform.lower() == "clear":
            user.preferred_platform = None
            await self._memory.update_user_summary(user)
            await self._send_reply(
                message,
                "✅ Preferred platform cleared. "
                "HEARTBEAT will reply on the last-used platform.",
            )
            return

        # Verify the platform is actually linked
        links = await self._memory.get_linked_platforms(user_id)
        linked_ids = {lnk["adapter_id"] for lnk in links}
        target = platform.lower()
        if target not in linked_ids:
            await self._send_reply(
                message,
                f"Platform '{target}' is not linked to your account. "
                f"Linked: {', '.join(sorted(linked_ids))}",
            )
            return

        user.preferred_platform = target
        await self._memory.update_user_summary(user)
        await self._send_reply(
            message,
            f"✅ Preferred platform set to: {target}",
        )
        logger.info(
            "User %s preferred platform set to '%s'",
            user_id,
            target,
        )

    async def _audit_link_event(
        self,
        message: GatewayMessage,
        action: str,
        detail: str,
    ) -> None:
        """Record a link/unlink operation in the audit log."""
        user_id = await self._memory.resolve_user(
            message.adapter_id, message.platform_user_id
        )
        await self._memory.add_certain_record(
            user_id=user_id or "__system__",
            content=detail,
            record_type="link_audit",
            metadata={
                "action": action,
                "adapter_id": message.adapter_id,
                "platform_user_id": message.platform_user_id,
            },
        )
        logger.info("Link audit: %s — %s", action, detail)

    async def _cmd_draw(self, message: GatewayMessage, commands_str: str) -> None:
        """Run the draw skill and return the generated image."""
        if not commands_str.strip():
            await self._send_reply(
                message,
                "Usage: /draw <DSL commands>\n"
                "Example: /draw SIZE 256 256"
                "\\nCANVAS white\\nCIRCLE 128 128 60 red\\nOUTPUT",
            )
            return

        skill = self._skills.get("draw")
        if skill is None:
            await self._send_reply(
                message,
                "Draw skill is not available. "
                "Make sure Pillow is installed: uv sync --extra draw",
            )
            return

        try:
            result = await skill.execute(
                {"commands": commands_str},
                memory=self._memory,
            )
        except Exception:
            logger.exception("Draw skill execution failed")
            await self._send_reply(
                message, "⚠️ Drawing failed. Check your DSL commands."
            )
            return

        if "error" in result:
            await self._send_reply(message, f"⚠️ Draw error: {result['error']}")
            return

        output_b64: str = result.get("output", "")
        if not output_b64:
            await self._send_reply(message, "⚠️ Draw skill returned no image.")
            return

        warnings: list[str] = result.get("warnings", [])
        caption = "🖼️ Here's your drawing!"
        if warnings:
            caption += "\n⚠️ " + "; ".join(warnings)

        await self._send_reply(message, caption, images=[output_b64])

    async def _send_reply(
        self,
        message: GatewayMessage,
        content: str,
        images: list[str] | None = None,
    ) -> None:
        """Send a reply message back to the user."""
        reply = GatewayMessage(
            type=MessageType.ACK,
            adapter_id=message.adapter_id,
            platform_user_id=message.platform_user_id,
            content=content,
            images=images or [],
            metadata=self._reply_metadata(message),
        )
        await self._gateway.send_to_adapter(message.adapter_id, reply)
