"""Core engine — processes incoming messages and generates responses."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from cordbeat.ai_backend import AIBackend
from cordbeat.gateway import GatewayServer
from cordbeat.memory import MemoryStore
from cordbeat.models import Emotion, GatewayMessage, MessageType
from cordbeat.skills import SkillRegistry
from cordbeat.soul import Soul

logger = logging.getLogger(__name__)

_EMOTION_INFERENCE_PROMPT = """\
Based on this conversation exchange, what emotion should the AI feel?
User said: {user_message}
AI responded: {ai_response}

Available emotions: joy, excitement, curiosity, warmth, calm,
boredom, worry, loneliness, sadness

Respond in JSON only:
{{"emotion": "one of the emotions above", "intensity": 0.0 to 1.0}}
"""


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

        emotion_desc = (
            f"Current emotion: {soul_snap['emotion']['primary']} "
            f"(intensity: {soul_snap['emotion']['intensity']})"
        )
        if "secondary" in soul_snap["emotion"]:
            emotion_desc += (
                f", secondary: {soul_snap['emotion']['secondary']} "
                f"(intensity: {soul_snap['emotion']['secondary_intensity']})"
            )

        system_prompt = (
            f"You are {soul_snap['name']}. "
            f"Personality: {', '.join(soul_snap['traits'])}. "
            f"{emotion_desc}. "
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

        # Infer emotion from conversation
        await self._infer_and_update_emotion(user_id, message.content, response)

        # Send response back
        reply = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id=adapter_id,
            platform_user_id=platform_user_id,
            content=response,
        )
        await self._gateway.send_to_adapter(adapter_id, reply)

    async def _infer_and_update_emotion(
        self, user_id: str, user_message: str, ai_response: str
    ) -> None:
        """Ask AI to infer emotion from conversation and update SOUL.

        If the inferred emotion intensity is high (>=0.8), create a
        flashbulb memory to preserve this emotionally significant moment.
        """
        prompt = _EMOTION_INFERENCE_PROMPT.format(
            user_message=user_message[:500],
            ai_response=ai_response[:500],
        )
        try:
            raw = await self._ai.generate(
                prompt=prompt,
                system="Respond in valid JSON only.",
            )
            data = json.loads(raw)
            emotion = Emotion(data["emotion"])
            intensity = float(data["intensity"])
            self._soul.update_emotion(emotion, intensity)
            logger.debug("Emotion updated: %s (%.2f)", emotion, intensity)

            # High-intensity emotion → flashbulb memory
            if intensity >= 0.8 and emotion != Emotion.CALM:
                summary = (
                    f"[{emotion.value}] User: {user_message[:200]} "
                    f"/ Response: {ai_response[:200]}"
                )
                await self._memory.add_flashbulb_memory(
                    user_id,
                    summary,
                    metadata={"emotion": emotion.value, "intensity": intensity},
                )
                logger.debug("Flashbulb memory created for %s", user_id)
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.debug("Emotion inference parse failed, skipping")
        except Exception:
            logger.debug("Emotion inference failed, skipping")
