"""Memory extraction — AI-driven fact/episode extraction from conversations."""

from __future__ import annotations

import json
import logging
import uuid

from cordbeat.agent.soul import Soul
from cordbeat.config import MemoryConfig
from cordbeat.memory.core import MemoryStore
from cordbeat.models import Emotion, MemoryEntry, MemoryLayer, SoulCaller

from .backend import AIBackend, internal_context_scope
from .reasoning import parse_json_object

logger = logging.getLogger(__name__)

# One stored fact/episode is a single extracted sentence; anything longer is
# runaway LLM output that would bloat embeddings and the DB (topic/tone are
# already capped at write time).
_MAX_MEMORY_CONTENT_CHARS = 500

_RECALL_KEYWORD_PROMPT = """\
Based on the recent conversation context below, extract exactly 3 short \
recall keywords that would help retrieve relevant memories about this user.

Focus on topics, names, events, or interests the user has mentioned recently.
Each keyword should be 1-4 words.

Conversation context:
{context}

Current message: {current_message}

Respond in JSON only:
{{"keywords": ["keyword1", "keyword2", "keyword3"]}}
"""

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
  "episode_summary": "one-sentence summary of a notable user-lived event, \
decision, preference, or meaningful shared moment; otherwise empty string"
}}

Only include facts that are clearly stated or strongly implied.
Do not store the AI's wording, promises, tool usage, failed actions, or response \
strategy as an episode. An ordinary request followed by an ordinary answer is \
not a memorable episode.
Do not store AI claims about hidden progress, files, sandbox work, tool \
results, or completed tasks unless the user independently confirmed the real \
artifact.
Do NOT fabricate or assume information.
Respond in valid JSON only.
"""

_SERVER_SHARED_NOTE_PROMPT = """\
Classify this single message from an explicitly shared public server channel.

Create a shared note only when the message explicitly states one of:
- decision: a clear decision or settled change
- schedule: a concrete planned date or time
- announcement: an explicit server-relevant announcement

Do not create notes for questions, requests, opinions, jokes, casual chat,
personal details, emotions, temporary activities, guesses, or implications.
The evidence must be an exact contiguous excerpt from the message. The summary
must contain no detail that is absent from that evidence.

Message:
{user_message}

Respond in JSON only:
{{
  "kind": "decision | schedule | announcement | none",
  "summary": "one grounded sentence, or empty",
  "evidence": "exact excerpt, or empty"
}}
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

    async def extract_recall_keywords(
        self,
        current_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[str]:
        """Extract recall keywords from conversation context via AI.

        Returns up to 3 short keywords for memory search.
        This is Phase2 of the recall model — context inference recall.
        """
        context_parts: list[str] = []
        if history:
            for msg in history[-6:]:
                prefix = "User" if msg["role"] == "user" else "AI"
                context_parts.append(f"{prefix}: {msg['content'][:200]}")
        context = "\n".join(context_parts) if context_parts else "(no prior context)"

        prompt = _RECALL_KEYWORD_PROMPT.format(
            context=context,
            current_message=current_message[:500],
        )
        try:
            raw = await self._ai.generate(
                prompt=prompt,
                system="/no_think\nRespond in valid JSON only.",
                temperature=self._memory_config.extraction_temperature,
            )
            data = parse_json_object(raw)
            keywords = data.get("keywords", [])
            if isinstance(keywords, list):
                return [
                    str(k).strip()[:50]
                    for k in keywords[:3]
                    if isinstance(k, str) and len(k.strip()) >= 2
                ]
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.debug("Recall keyword extraction parse failed, skipping")
        except Exception:
            logger.debug("Recall keyword extraction failed, skipping")
            return []
        return []

    async def extract_server_shared_note(
        self, user_message: str
    ) -> dict[str, str] | None:
        """Extract one source-grounded server note from an allowed message."""

        message = user_message.strip()[:2000]
        if not message:
            return None
        try:
            with internal_context_scope():
                raw = await self._ai.generate(
                    prompt=_SERVER_SHARED_NOTE_PROMPT.format(user_message=message),
                    system="/no_think\nRespond in valid JSON only.",
                    temperature=self._memory_config.extraction_temperature,
                )
            data = parse_json_object(raw)
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.debug("Server shared-note extraction parse failed, skipping")
            return None
        except Exception:
            logger.debug("Server shared-note extraction failed, skipping")
            return None

        kind = str(data.get("kind") or "").strip().lower()
        summary = str(data.get("summary") or "").strip()[:300]
        evidence = str(data.get("evidence") or "").strip()[:500]
        if kind not in {"decision", "schedule", "announcement"}:
            return None
        if not summary or not evidence or evidence not in message:
            logger.debug("Server shared-note evidence validation failed, skipping")
            return None
        return {"kind": kind, "summary": summary, "evidence": evidence}

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
            with internal_context_scope():
                raw = await self._ai.generate(
                    prompt=prompt,
                    system="Respond in valid JSON only.",
                )
            data = parse_json_object(raw)
            emotion = Emotion(data["emotion"])
            intensity = float(data["intensity"])
            self._soul.update_emotion(emotion, intensity, caller=SoulCaller.AI)
            logger.debug("Emotion updated: %s (%.2f)", emotion, intensity)

            # High-intensity emotion → flashbulb memory
            if (
                intensity >= self._memory_config.flashbulb_intensity_threshold
                and emotion != Emotion.CALM
            ):
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
            logger.warning("Emotion inference parse failed, skipping")
        except Exception as exc:
            logger.warning("Emotion inference failed, skipping: %s", exc)

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
            with internal_context_scope():
                raw = await self._ai.generate(
                    prompt=prompt,
                    system="Respond in valid JSON only.",
                    temperature=self._memory_config.extraction_temperature,
                )
            data = parse_json_object(raw)
        except Exception as exc:
            # Memories silently not being stored breaks relationship growth;
            # make extraction failures visible to operators.
            logger.warning("Memory extraction failed, skipping: %s", exc)
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
                    content=fact.strip()[:_MAX_MEMORY_CONTENT_CHARS],
                    metadata={"emotional_tone": tone} if tone else {},
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
                content=episode.strip()[:_MAX_MEMORY_CONTENT_CHARS],
                metadata={"emotional_tone": tone} if tone else {},
            )
            await self._memory.add_episodic_memory(entry)
            logger.debug("Episodic memory stored for %s", user_id)
