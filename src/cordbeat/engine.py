"""Core engine — processes incoming messages and generates responses."""

from __future__ import annotations

import logging
from datetime import datetime

from cordbeat.ai_backend import AIBackend
from cordbeat.config import MemoryConfig
from cordbeat.extraction import MemoryExtractor
from cordbeat.gateway import GatewayServer
from cordbeat.memory import MemoryStore
from cordbeat.models import (
    GatewayMessage,
    MessageType,
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

        context = build_context(
            user_display_name=user.display_name,
            profile=profile or None,
            semantic_memories=semantic_memories or None,
            episodic_memories=episodic_memories or None,
            history=history or None,
            soul_name=soul_snap["name"],
        )

        safe_content = sanitize(message.content)
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
