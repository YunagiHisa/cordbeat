"""Core engine — processes incoming messages and generates responses."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from cordbeat.ai_backend import AIBackend
from cordbeat.config import MemoryConfig
from cordbeat.extraction import MemoryExtractor
from cordbeat.gateway import GatewayServer
from cordbeat.memory import MemoryStore
from cordbeat.models import (
    GatewayMessage,
    MessageType,
    ProposalStatus,
    UserSummary,
)
from cordbeat.prompt import build_context, build_soul_system_prompt, sanitize
from cordbeat.skills import SkillRegistry
from cordbeat.soul import Soul

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._ai = ai
        self._soul = soul
        self._memory = memory
        self._skills = skills
        self._gateway = gateway
        self._memory_config = memory_config or MemoryConfig()
        self._extractor = MemoryExtractor(ai, soul, memory, self._memory_config)

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

        # ── Command routing ───────────────────────────────────────────
        text = message.content.strip()
        if text.startswith("/"):
            handled = await self._handle_command(message, text)
            if handled:
                return

        # Phase 1: Resolve user
        user_id, user = await self._resolve_user(message)

        # Phase 2: Build prompt and generate response
        response = await self._generate_response(user_id, user, message)
        if response is None:
            return  # AI failure already handled

        # Phase 3: Store conversation and extract memories
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

        # Phase 4: Send reply
        reply = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id=message.adapter_id,
            platform_user_id=message.platform_user_id,
            content=response,
        )
        await self._gateway.send_to_adapter(message.adapter_id, reply)

    async def _resolve_user(self, message: GatewayMessage) -> tuple[str, UserSummary]:
        """Resolve or create the user and update their summary."""
        adapter_id = message.adapter_id
        platform_user_id = message.platform_user_id

        user_id = await self._memory.resolve_user(adapter_id, platform_user_id)
        if user_id is None:
            user_id = f"cb_{adapter_id}_{platform_user_id}"
            user = await self._memory.get_or_create_user(user_id, platform_user_id)
            await self._memory.link_platform(user_id, adapter_id, platform_user_id)
        else:
            user = await self._memory.get_or_create_user(user_id, platform_user_id)

        user.last_talked_at = datetime.now()
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

        system_prompt = build_soul_system_prompt(soul_snap)

        # Phase 1: Direct keyword search (message.content → ChromaDB)
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

        # Phase 4a: Chain recall (芋づる想起 — precomputed links)
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

        try:
            return await self._ai.generate(prompt=prompt, system=system_prompt)
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

        if cmd == "/approve":
            await self._cmd_approve(message, arg)
            return True
        if cmd == "/reject":
            await self._cmd_reject(message, arg)
            return True
        if cmd == "/proposals":
            await self._cmd_proposals(message)
            return True
        if cmd == "/link":
            await self._cmd_link(message)
            return True
        if cmd == "/unlink":
            await self._cmd_unlink(message, arg)
            return True
        if cmd == "/name":
            await self._cmd_name(message, arg)
            return True
        if cmd == "/quiet":
            await self._cmd_quiet(message, arg)
            return True
        if cmd == "/prefer":
            await self._cmd_prefer(message, arg)
            return True

        # Unknown slash command — fall through to normal processing
        return False

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

        self._soul.update_name(name)
        await self._send_reply(
            message, f"✅ Name updated to: {name}"
        )
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

        self._soul.update_quiet_hours(start, end)
        await self._send_reply(
            message, f"✅ Quiet hours updated: {start} - {end}"
        )
        logger.info("Quiet hours changed to %s - %s", start, end)

    async def _cmd_prefer(
        self, message: GatewayMessage, platform: str
    ) -> None:
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

        user = await self._memory.get_or_create_user(
            user_id, message.platform_user_id
        )
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

    async def _send_reply(self, message: GatewayMessage, content: str) -> None:
        """Send a reply message back to the user."""
        reply = GatewayMessage(
            type=MessageType.ACK,
            adapter_id=message.adapter_id,
            platform_user_id=message.platform_user_id,
            content=content,
        )
        await self._gateway.send_to_adapter(message.adapter_id, reply)
