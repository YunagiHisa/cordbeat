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

from cordbeat.agent.soul import Soul
from cordbeat.ai.backend import AIBackend
from cordbeat.ai.compression import ConversationCompressor
from cordbeat.ai.extraction import MemoryExtractor
from cordbeat.ai.prompt import build_context, build_soul_system_prompt, sanitize
from cordbeat.config import MemoryConfig
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
# Example: [DRAW: a red circle on a white background]
_DRAW_TAG_RE = re.compile(r"\[DRAW:\s*(.+?)\]", re.DOTALL | re.IGNORECASE)

# Pattern for inline skill-invocation tags.
# Example: [SKILL: web_search | query=latest AI news]
_SKILL_TAG_RE = re.compile(
    r"\[SKILL:\s*([^\|\]\n]+?)(?:\s*\|\s*([^\]\n]*))?\]",
    re.IGNORECASE,
)


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
        vision_enabled: bool = False,
        timezone_name: str = "UTC",
    ) -> None:
        self._ai = ai
        self._soul = soul
        self._memory = memory
        self._skills = skills
        self._gateway = gateway
        self._memory_config = memory_config or MemoryConfig()
        self._vision_enabled = vision_enabled
        self._timezone_name = timezone_name
        self._extractor = MemoryExtractor(ai, soul, memory, self._memory_config)
        self._compressor = ConversationCompressor(ai)
        self._background_tasks: set[asyncio.Task[None]] = set()
        # Skills that the user has approved for this session (no re-confirmation).
        self._session_allowed_skills: set[str] = set()
        # Per-user stop flags: if user_id is present, abort in-flight processing.
        self._stop_flags: set[str] = set()

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

        # Check if a /stop was issued while we were queued
        if user_id in self._stop_flags:
            self._stop_flags.discard(user_id)
            return

        # Phase 2: Build prompt and generate response
        response = await self._generate_response(user_id, user, message)
        if response is None:
            return  # AI failure already handled

        # Phase 3: Dispatch any [SKILL: ...] tags the AI emitted
        response = await self._dispatch_skill_tags(response, message)

        # Phase 4: Send reply immediately — do NOT wait for post-processing
        clean_response, draw_images = await self._maybe_draw(response)
        reply = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id=message.adapter_id,
            platform_user_id=message.platform_user_id,
            content=clean_response,
            images=draw_images,
        )
        await self._gateway.send_to_adapter(message.adapter_id, reply)

        # Phase 5: Background post-processing (memory storage + emotion update).
        # Runs concurrently so the user already has the reply.
        task = asyncio.create_task(
            self._post_process_message(user_id, user, message, response)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

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

        user_id = await self._memory.resolve_user(adapter_id, platform_user_id)
        if user_id is None:
            user_id = uuid.uuid4().hex
        user = await self._memory.get_or_create_user(user_id, platform_user_id)
        # Always upsert the link so heartbeat can reverse-resolve even if the
        # user record was created via another code path.
        await self._memory.link_platform(user_id, adapter_id, platform_user_id)

        user.last_talked_at = datetime.now(tz=UTC)
        user.last_platform = adapter_id
        await self._memory.update_user_summary(user)
        return user_id, user

    async def _generate_response(
        self,
        user_id: str,
        user: UserSummary,
        message: GatewayMessage,
    ) -> str | None:
        """Build prompt, call AI, return response or None on failure."""
        soul_snap = self._soul.get_soul_snapshot()
        profile = await self._memory.get_core_profile(user_id)
        history = await self._memory.get_recent_messages(
            user_id, limit=self._memory_config.conversation_history_limit
        )

        # Compress in-memory if history is too large
        if self._memory_config.context_compression_enabled and history:
            history = await self._compressor.compress_in_memory(
                history,
                chars_threshold=self._memory_config.context_compression_chars_threshold,
                soul_name=soul_snap["name"],
            )

        system_prompt = build_soul_system_prompt(
            soul_snap, timezone_name=self._timezone_name
        )
        if self._skills.get("draw") is not None:
            system_prompt += (
                "\n\nYou can create images for the user. When drawing would enhance"
                " your response, include exactly one"
                " [DRAW: <description in English>] tag in your message."
                " Example: [DRAW: a red circle on a white background]."
                " The image will be rendered automatically and sent with your reply."
            )

        # Inject available skill catalog so the AI knows what tools it can call.
        skills_desc = self._skills.get_skill_descriptions_for_prompt()
        if skills_desc and skills_desc != "(no skills available)":
            system_prompt += (
                "\n\nYou have access to the following tools. When using a tool,"
                " include exactly one [SKILL: <name> | <param>=<value>] tag in your"
                " reply. Only safe tools run automatically; others are queued for"
                " approval. Only use a tool when it genuinely helps the user."
                "\nExample: [SKILL: web_search | query=latest AI news]"
                f"\nAvailable tools:\n{skills_desc}"
            )

        # Phase 1: Direct keyword search (message.content → vector search)
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

        # Phase 2: Context inference recall (AI extracts keywords → search)
        recall_keywords = await self._extractor.extract_recall_keywords(
            message.content, history or None
        )
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
        if current_emotion and current_emotion != "calm":
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
        try:
            raw_hints = await self._memory.get_recall_hints(user_id)
            hints = [h["content"] for h in raw_hints if h.get("content")]
        except Exception:
            logger.debug("Recall hints lookup failed for user %s", user_id)

        context = build_context(
            user_display_name=user.display_name,
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
            if self._vision_enabled and message.images:
                try:
                    raw = await self._ai.generate_with_vision(
                        prompt=prompt,
                        images=message.images,
                        system=system_prompt,
                    )
                    return re.sub(
                        r"<think>.*?</think>", "", raw, flags=re.DOTALL
                    ).strip()
                except Exception:
                    logger.warning(
                        "Vision generation failed, falling back to text-only response"
                    )
            raw = await self._ai.generate(prompt=prompt, system=system_prompt)
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
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
                    content="（AIが応答を生成できませんでした。もう一度お試しください）",
                )
                await self._gateway.send_to_adapter(message.adapter_id, error_reply)
                return None
            return cleaned
        except Exception:
            logger.exception("AI generation failed")
            error_reply = GatewayMessage(
                type=MessageType.ERROR,
                adapter_id=message.adapter_id,
                platform_user_id=message.platform_user_id,
                content="AI generation failed. Please try again later.",
            )
            await self._gateway.send_to_adapter(message.adapter_id, error_reply)
            return None

    async def _post_process_message(
        self,
        user_id: str,
        user: UserSummary,
        message: GatewayMessage,
        response: str,
    ) -> None:
        """Background task: persist conversation + run emotion/memory extraction."""
        try:
            await self._memory.add_message(
                user_id, "user", message.content, message.adapter_id
            )
            await self._memory.add_message(
                user_id, "assistant", response, message.adapter_id
            )
            await self._extractor.infer_and_update_emotion(
                user_id, message.content, response
            )
            await self._extractor.extract_and_store_memories(
                user_id, user.display_name, message.content, response
            )
        except Exception:
            logger.exception("Background post-processing failed for user %s", user_id)

    async def _dispatch_skill_tags(self, response: str, message: GatewayMessage) -> str:
        """Execute the first [SKILL: name | param=value] tag in the AI response.

        ``safe`` skills run immediately.  ``requires_confirmation`` skills are
        queued for user approval unless the skill is in the session allow-list.
        The tag is replaced with the skill's output inline.
        """
        match = _SKILL_TAG_RE.search(response)
        if match is None:
            return response

        skill_name = match.group(1).strip()
        params_raw = (match.group(2) or "").strip()

        skill = self._skills.get(skill_name)
        if skill is None or not skill.meta.enabled:
            logger.debug("SKILL tag references unknown/disabled skill: %r", skill_name)
            return _SKILL_TAG_RE.sub("", response, count=1).strip()

        # Parse "key=value | key2=value2 ..." pairs early so we can include them
        # in the confirmation request if needed.
        params: dict[str, Any] = {}
        for part in params_raw.split("|"):
            part = part.strip()
            if "=" in part:
                k, _, v = part.partition("=")
                params[k.strip()] = v.strip()

        if skill.meta.safety_level != SafetyLevel.SAFE:
            if skill_name not in self._session_allowed_skills:
                logger.info(
                    "SKILL tag %r requires confirmation; queuing proposal",
                    skill_name,
                )
                await self._request_skill_confirmation(message, skill_name, params)
                return _SKILL_TAG_RE.sub("", response, count=1).strip()
            logger.info(
                "SKILL tag %r is in session allow-list; executing", skill_name
            )

        try:
            result = await skill.execute(params, memory=self._memory)
            # Prefer explicit output/result key; fall back to compact JSON of
            # the whole result dict (preserves information, no whitespace waste).
            output = result.get("output") or result.get("result") or ""
            if not output:
                output = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
            output = str(output).strip()
            if output:
                inline = f"\n[{skill_name}: {output[:2000]}]"
                response = _SKILL_TAG_RE.sub(inline, response, count=1)
            else:
                response = _SKILL_TAG_RE.sub("", response, count=1)
        except Exception:
            logger.warning("Auto-skill execution failed: %s", skill_name, exc_info=True)
            response = _SKILL_TAG_RE.sub("", response, count=1)

        return response.strip()

    async def _request_skill_confirmation(
        self,
        message: GatewayMessage,
        skill_name: str,
        params: dict[str, Any],
    ) -> None:
        """Store a skill execution proposal and notify the adapter with a confirm UI."""
        user_id = await self._memory.resolve_user(
            message.adapter_id, message.platform_user_id
        )
        if user_id is None:
            logger.debug(
                "Cannot request skill confirmation: user not found for %s",
                message.platform_user_id,
            )
            return

        proposal_content = (
            f"Skill '{skill_name}' requires confirmation.\n"
            f"Parameters: {json.dumps(params)}"
        )
        metadata: dict[str, Any] = {
            "status": ProposalStatus.PENDING,
            "proposal_type": ProposalType.SKILL_EXECUTION,
            "skill_name": skill_name,
            "skill_params": params,
            "adapter_id": message.adapter_id,
        }
        proposal_id = await self._memory.add_certain_record(
            user_id=user_id,
            content=proposal_content,
            record_type="proposal",
            metadata=metadata,
        )

        notification = GatewayMessage(
            type=MessageType.SKILL_CONFIRM,
            adapter_id=message.adapter_id,
            platform_user_id=message.platform_user_id,
            content=(
                f"🔧 I want to run **{skill_name}**.\n"
                f"Parameters: {json.dumps(params)}\n\n"
                f"(proposal ID: {proposal_id})"
            ),
            metadata={
                "proposal_id": proposal_id,
                "skill_name": skill_name,
                "skill_params": params,
            },
        )
        await self._gateway.send_to_adapter(message.adapter_id, notification)
        logger.info(
            "Skill confirmation requested for '%s' (proposal=%s)",
            skill_name,
            proposal_id,
        )

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

        try:
            dsl = await self._generate_draw_dsl(description)
            if not dsl:
                return clean_text, []
            skill = self._skills.get("draw")
            if skill is None:
                return clean_text, []
            result = await skill.execute({"commands": dsl}, memory=self._memory)
            output = result.get("output", "")
            if output and "error" not in result:
                return clean_text, [output]
        except Exception:
            logger.exception("Auto-draw pipeline failed (description=%r)", description)

        return clean_text, []

    async def _generate_draw_dsl(self, description: str) -> str:
        """Ask the AI to produce Draw DSL for the given plain-text description.

        Returns an empty string on failure so callers can skip gracefully.
        """
        safe_desc = sanitize(description, max_len=500)
        system = (
            "You are a drawing DSL generator. "
            "Given a description, output ONLY valid Draw DSL"
            " commands — no prose, no markdown fences.\n"
            "Available commands (one per line):\n"
            "  SIZE <width> <height>\n"
            "  CANVAS <color>\n"
            "  CIRCLE <cx> <cy> <radius> <color>\n"
            "  RECT <x1> <y1> <x2> <y2> <color>\n"
            "  ELLIPSE <x1> <y1> <x2> <y2> <color>\n"
            "  LINE <x1> <y1> <x2> <y2> <color>\n"
            "  POLYGON <color> <x1> <y1> <x2> <y2> ...\n"
            "  TEXT <x> <y> <text> <color> [size]\n"
            "  STAR <cx> <cy> <outer_r> <inner_r> <points> <color>\n"
            "  SPIRAL <cx> <cy> <turns> <spacing> <color>\n"
            "  OUTPUT\n"
            "Colors: named colors (white, red, blue, ...) or #RRGGBB."
            " Always end with OUTPUT."
        )
        prompt = f"Draw this: {safe_desc}"
        try:
            raw = await self._ai.generate(prompt=prompt, system=system)
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
            "/approve_session": lambda: self._cmd_approve_session(message, arg),
            "/reject": lambda: self._cmd_reject(message, arg),
            "/proposals": lambda: self._cmd_proposals(message),
            "/link": lambda: self._cmd_link(message),
            "/unlink": lambda: self._cmd_unlink(message, arg),
            "/name": lambda: self._cmd_name(message, arg),
            "/quiet": lambda: self._cmd_quiet(message, arg),
            "/prefer": lambda: self._cmd_prefer(message, arg),
            "/draw": lambda: self._cmd_draw(message, arg),
            "/reset": lambda: self._cmd_reset(message),
            "/new": lambda: self._cmd_reset(message),
            "/stop": lambda: self._cmd_stop(message),
            "/skills": lambda: self._cmd_skills(message),
            "/help": lambda: self._cmd_help(message),
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

    async def _cmd_approve_session(
        self, message: GatewayMessage, proposal_id: str
    ) -> None:
        """Approve a pending skill proposal and allow it for this session."""
        if not proposal_id:
            await self._send_reply(
                message, "Usage: /approve_session <proposal_id>"
            )
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
                f"Cannot approve: proposal status is '{status}'.",
            )
            return

        await self._memory.update_proposal_status(proposal_id, ProposalStatus.APPROVED)

        skill_name = meta.get("skill_name", "")
        if skill_name:
            self._session_allowed_skills.add(skill_name)
            logger.info(
                "Skill '%s' added to session allow-list by user %s", skill_name, user_id
            )

        await self._send_reply(
            message,
            f"✅ Approved for this session: **{skill_name}**"
            f" (proposal {proposal_id[:8]}…)\n"
            "Future invocations of this skill will run automatically until restart.",
        )
        logger.info("Proposal %s approved (session) by user %s", proposal_id, user_id)

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

    async def _cmd_reset(self, message: GatewayMessage) -> None:
        """Clear conversation history for the current user."""
        user_id = await self._memory.resolve_user(
            message.adapter_id, message.platform_user_id
        )
        if user_id is None:
            await self._send_reply(message, "✅ Conversation cleared.")
            return
        deleted = await self._memory.clear_conversation_history(user_id)
        logger.info(
            "Conversation history cleared for user %s (%d rows)", user_id, deleted
        )
        await self._send_reply(
            message, "✅ Conversation history cleared. Starting fresh!"
        )

    async def _cmd_stop(self, message: GatewayMessage) -> None:
        """Request cancellation of any in-flight processing for this user."""
        user_id = await self._memory.resolve_user(
            message.adapter_id, message.platform_user_id
        )
        key = user_id or message.platform_user_id
        self._stop_flags.add(key)
        await self._send_reply(
            message, "⏹ Stop requested. Current processing will abort."
        )
        logger.info("Stop flag set for user key=%s", key)

    async def _cmd_skills(self, message: GatewayMessage) -> None:
        """List all available skills."""
        skills = self._skills.available_skills
        if not skills:
            await self._send_reply(message, "No skills available.")
            return
        lines = ["🛠️ Available skills:\n"]
        for name, skill in sorted(skills.items()):
            desc = getattr(skill, "description", "") or ""
            safety = getattr(skill, "safety_level", "")
            safety_icons: dict[str, str] = {
                "safe": "✅",
                "requires_confirmation": "⚠️",
                "dangerous": "🔴",
            }
            safety_icon = safety_icons.get(str(safety), "❓")
            lines.append(f"  {safety_icon} **{name}** — {desc}")
        await self._send_reply(message, "\n".join(lines))

    async def _cmd_help(self, message: GatewayMessage) -> None:
        """List all available slash commands."""
        help_text = """\
📖 Available commands:

**Conversation**
  /reset, /new        — Clear conversation history
  /stop               — Cancel in-flight processing

**Skills**
  /skills             — List available skills
  /draw <DSL>         — Draw an image using the draw skill

**Proposals & approvals**
  /proposals          — List pending proposals
  /approve <id>       — Approve a proposal
  /approve_session <id> — Approve for this session
  /reject <id>        — Reject a proposal

**Settings**
  /name <name>        — Change the AI's name
  /quiet <start> <end>— Set quiet hours (HH:MM)
  /prefer <platform>  — Set preferred reply platform

**Account linking**
  /link               — Generate a cross-platform link token
  /unlink <platform>  — Remove a platform link

  /help               — Show this help
"""
        await self._send_reply(message, help_text)

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
        )
        await self._gateway.send_to_adapter(message.adapter_id, reply)
