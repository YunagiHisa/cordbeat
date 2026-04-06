"""Core engine — processes incoming messages and generates responses."""

from __future__ import annotations

import logging
from datetime import datetime

from cordbeat.ai_backend import AIBackend
from cordbeat.gateway import GatewayServer
from cordbeat.memory import MemoryStore
from cordbeat.models import GatewayMessage, MessageType
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
    ) -> None:
        self._ai = ai
        self._soul = soul
        self._memory = memory
        self._skills = skills
        self._gateway = gateway

    async def handle_message(self, message: GatewayMessage) -> None:
        """Handle a single incoming message from the queue."""
        if message.type not in (MessageType.MESSAGE, MessageType.LINK_REQUEST):
            return

        adapter_id = message.adapter_id
        platform_user_id = message.platform_user_id

        # Resolve or create user
        user_id = await self._memory.resolve_user(
            adapter_id,
            platform_user_id,
        )
        if user_id is None:
            user_id = f"cb_{adapter_id}_{platform_user_id}"
            user = await self._memory.get_or_create_user(
                user_id,
                platform_user_id,
            )
            await self._memory.link_platform(
                user_id,
                adapter_id,
                platform_user_id,
            )
        else:
            user = await self._memory.get_or_create_user(
                user_id,
                platform_user_id,
            )

        # Update user summary
        user.last_talked_at = datetime.now()
        user.last_platform = adapter_id
        await self._memory.update_user_summary(user)

        # Build prompt
        soul_snap = self._soul.get_soul_snapshot()
        profile = await self._memory.get_core_profile(user_id)
        history = await self._memory.get_recent_messages(user_id, limit=20)

        system_prompt = (
            f"You are {soul_snap['name']}. "
            f"Personality: {', '.join(soul_snap['traits'])}. "
            f"Current emotion: {soul_snap['emotion']['primary']} "
            f"(intensity: {soul_snap['emotion']['intensity']}). "
            f"\nImmutable rules:\n"
            + "\n".join(f"- {r}" for r in soul_snap["immutable_rules"])
            + "\n\nRespond naturally to the user's message. "
            "Keep your response concise."
        )

        context_parts = [f"User: {user.display_name}"]
        if profile:
            context_parts.append(
                f"Known info: {', '.join(f'{k}={v}' for k, v in profile.items())}"
            )

        # Append conversation history
        if history:
            context_parts.append("\nConversation history:")
            for msg in history:
                prefix = "User" if msg["role"] == "user" else soul_snap["name"]
                context_parts.append(f"  {prefix}: {msg['content']}")

        context = "\n".join(context_parts)

        prompt = f"{context}\n\nUser says: {message.content}"

        # Generate response
        try:
            response = await self._ai.generate(
                prompt=prompt,
                system=system_prompt,
            )
        except Exception:
            logger.exception("AI generation failed")
            error_reply = GatewayMessage(
                type=MessageType.ERROR,
                adapter_id=adapter_id,
                platform_user_id=platform_user_id,
                content="AI generation failed. Please try again later.",
            )
            await self._gateway.send_to_adapter(adapter_id, error_reply)
            return

        # Store conversation in memory
        await self._memory.add_message(
            user_id,
            "user",
            message.content,
            adapter_id,
        )
        await self._memory.add_message(
            user_id,
            "assistant",
            response,
            adapter_id,
        )

        # Send response back
        reply = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id=adapter_id,
            platform_user_id=platform_user_id,
            content=response,
        )
        await self._gateway.send_to_adapter(adapter_id, reply)
