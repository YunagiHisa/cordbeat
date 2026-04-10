"""Memory extraction — AI-driven fact/episode extraction from conversations."""

from __future__ import annotations

import json
import logging
import uuid

from cordbeat.ai_backend import AIBackend
from cordbeat.config import MemoryConfig
from cordbeat.memory import MemoryStore
from cordbeat.models import Emotion, MemoryEntry, MemoryLayer
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


class MemoryExtractor:
    """AI-driven extraction of emotions, facts, and episodes."""

    def __init__(
        self,
        ai: AIBackend,
        soul: Soul,
        memory: MemoryStore,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        self._ai = ai
        self._soul = soul
        self._memory = memory
        self._memory_config = memory_config or MemoryConfig()

    async def infer_and_update_emotion(
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
        except (OSError, RuntimeError, TypeError):
            logger.debug("Emotion inference failed, skipping")

    async def extract_and_store_memories(
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
        except (
            json.JSONDecodeError,
            KeyError,
            ValueError,
            OSError,
            RuntimeError,
            TypeError,
        ):
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
            for fact in facts[: self._memory_config.facts_per_message_limit]:
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
