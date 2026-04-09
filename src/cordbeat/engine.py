"""Core engine — processes incoming messages and generates responses."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime

from cordbeat.ai_backend import AIBackend
from cordbeat.gateway import GatewayServer
from cordbeat.memory import MemoryStore
from cordbeat.models import (
    Emotion,
    GatewayMessage,
    MemoryEntry,
    MemoryLayer,
    MessageType,
)
from cordbeat.skills import SkillRegistry
from cordbeat.soul import Soul

logger = logging.getLogger(__name__)

# Strip control characters that could manipulate prompt structure
_SANITIZE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MAX_USER_INPUT_LEN = 2000


def _sanitize_user_input(text: str) -> str:
    """Remove control characters and truncate user input for safe prompt use."""
    return _SANITIZE_RE.sub("", text)[:_MAX_USER_INPUT_LEN]


_EMOTION_INFERENCE_PROMPT = """\
Based on this conversation exchange, what emotion should the AI feel?
User said: {user_message}
AI responded: {ai_response}

Available emotions: joy, excitement, curiosity, warmth, calm,
boredom, worry, loneliness, sadness

Respond in JSON only:
{{"emotion": "one of the emotions above", "intensity": 0.0 to 1.0}}
"""

_MEMORY_EXTRACTION_PROMPT = """\
Analyze this conversation exchange and extract memory-worthy information.
User ({user_name}) said: {user_message}
AI responded: {ai_response}

Extract the following in JSON:
{{
  "topic": "brief topic label (3-5 words, or empty string if trivial)",
  "emotional_tone": "one word describing user's tone (e.g. happy, curious, \
frustrated, neutral)",
  "facts": ["list of new facts/preferences about the user, or empty list"],
  "episode_summary": "one-sentence summary if this is a memorable moment, \
or empty string"
}}

Only include facts that are clearly stated or strongly implied.
Do NOT fabricate or assume information.
Respond in valid JSON only.
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

        context_parts = [f"User: {_sanitize_user_input(user.display_name)}"]
        if profile:
            sanitized = ", ".join(
                f"{k}={_sanitize_user_input(str(v))}" for k, v in profile.items()
            )
            context_parts.append(f"Known info: {sanitized}")

        # Retrieve relevant semantic memories (preferences, facts)
        semantic_memories = await self._memory.search_semantic(
            user_id, message.content, n_results=3
        )
        if semantic_memories:
            context_parts.append("\nKnown preferences/facts:")
            for mem in semantic_memories:
                context_parts.append(f"  - {mem['content']}")

        # Retrieve relevant episodic memories (past moments)
        episodic_memories = await self._memory.search_episodic(
            user_id, message.content, n_results=3
        )
        if episodic_memories:
            context_parts.append("\nRelated past moments:")
            for mem in episodic_memories:
                context_parts.append(f"  - {mem['content']}")

        # Append conversation history
        if history:
            context_parts.append("\nConversation history:")
            for msg in history:
                prefix = "User" if msg["role"] == "user" else soul_snap["name"]
                sanitized = _sanitize_user_input(msg["content"])
                context_parts.append(f"  {prefix}: {sanitized}")

        context = "\n".join(context_parts)

        safe_content = _sanitize_user_input(message.content)
        prompt = f"{context}\n\nUser says: {safe_content}"

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

        # Extract and store memories (topic, facts, episodes)
        await self._extract_and_store_memories(
            user_id, user.display_name, message.content, response
        )

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

    async def _extract_and_store_memories(
        self,
        user_id: str,
        user_name: str,
        user_message: str,
        ai_response: str,
    ) -> None:
        """Extract topic, facts, and episodes from conversation via AI."""
        prompt = _MEMORY_EXTRACTION_PROMPT.format(
            user_name=user_name[:50],
            user_message=user_message[:500],
            ai_response=ai_response[:500],
        )
        try:
            raw = await self._ai.generate(
                prompt=prompt,
                system="Respond in valid JSON only.",
                temperature=0.2,
            )
            data = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            logger.debug("Memory extraction failed, skipping")
            return

        # Update user summary with topic and tone
        topic = data.get("topic", "")
        tone = data.get("emotional_tone", "")
        if topic or tone:
            user = await self._memory.get_or_create_user(user_id, user_name)
            if topic:
                user.last_topic = str(topic)[:100]
            if tone:
                user.emotional_tone = str(tone)[:50]
            await self._memory.update_user_summary(user)

        # Store semantic facts (preferences, knowledge)
        facts = data.get("facts", [])
        if isinstance(facts, list):
            for fact in facts[:5]:  # Cap at 5 facts per message
                if not isinstance(fact, str) or len(fact.strip()) < 3:
                    continue
                entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    layer=MemoryLayer.SEMANTIC,
                    content=fact.strip(),
                )
                await self._memory.add_semantic_memory(entry)
                logger.debug("Semantic memory stored for %s: %s", user_id, fact)

        # Store episodic summary if notable
        episode = data.get("episode_summary", "")
        if isinstance(episode, str) and len(episode.strip()) > 10:
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                user_id=user_id,
                layer=MemoryLayer.EPISODIC,
                content=episode.strip(),
            )
            await self._memory.add_episodic_memory(entry)
            logger.debug("Episodic memory stored for %s", user_id)
