"""Sleep-phase memory consolidation for the HEARTBEAT loop."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from cordbeat.ai_backend import AIBackend
from cordbeat.config import MemoryConfig
from cordbeat.memory import MemoryStore
from cordbeat.models import MemoryEntry, MemoryLayer, UserSummary
from cordbeat.soul import Soul

logger = logging.getLogger(__name__)

_DIARY_SYSTEM_PROMPT = """\
You are {name}, reviewing today's conversations to write a diary entry.
Write a brief, reflective diary entry summarizing what happened today.
Note key topics, emotional moments, and anything worth remembering.
Keep it concise (3-5 sentences). Write in first person.
Respond with ONLY the diary text, no JSON.
"""

_PROMOTION_SYSTEM_PROMPT = """\
You are reviewing today's episodic memories to extract general facts.
For each episode, decide if it contains a general fact or preference about
the user that should be remembered long-term (e.g., "likes Python",
"works as a developer", "enjoys hiking").

Episodes:
{episodes}

Respond in JSON only:
{{"facts": ["general fact 1", "general fact 2"]}}

Only include genuinely general facts. Do NOT include specific events,
timestamps, or one-time occurrences. Empty list if nothing is generalizable.
"""

# Temporal recall lookback windows: (days_ago, human_label)
TEMPORAL_WINDOWS: list[tuple[int, str]] = [
    (1, "yesterday"),
    (7, "1 week ago"),
    (30, "1 month ago"),
    (365, "1 year ago"),
]


class SleepPhase:
    """Memory consolidation during quiet hours."""

    def __init__(
        self,
        memory: MemoryStore,
        ai: AIBackend,
        soul: Soul,
        memory_config: MemoryConfig,
    ) -> None:
        self._memory = memory
        self._ai = ai
        self._soul = soul
        self._memory_config = memory_config

    async def run(self) -> None:
        """Run full sleep-phase memory consolidation.

        1. For each user, review today's conversations and generate a diary.
        2. Promote notable episodic memories to semantic facts.
        3. Precompute Phase4 temporal recall hints (multiple windows).
        4. Apply forgetting curve and archive decayed memories.
        5. Trim old conversation messages and recall hints.
        """
        logger.info("Sleep phase started — consolidating memories")

        users = await self._memory.get_all_user_summaries()
        soul_snap = self._soul.get_soul_snapshot()

        # 1. Generate diary entries per user
        for user in users:
            await self._write_diary(user, soul_snap)

        # 2. Promote notable episodic memories to semantic facts
        for user in users:
            await self._promote_episodic_memories(user.user_id)

        # 3. Phase4: Temporal recall (check multiple lookback windows)
        for user in users:
            await self._precompute_temporal_recall(user)

        # 3b. Phase4: Chain recall (precompute associative memory links)
        for user in users:
            await self._precompute_chain_links(user.user_id)

        # 4. (Memory decay is now lazy — performed inside search_*; no
        #    nightly batch needed. See PR #81 / 設計書 項目 #4.)

        # 5. Trim old conversation messages and recall hints
        for user in users:
            try:
                deleted = await self._memory.trim_old_messages(user.user_id)
                if deleted > 0:
                    logger.info(
                        "Trimmed %d old messages for user %s",
                        deleted,
                        user.user_id,
                    )
            except Exception:
                logger.exception("Failed to trim messages for user %s", user.user_id)

        try:
            await self._memory.clear_old_recall_hints(
                keep_days=self._memory_config.recall_hints_retention_days,
            )
        except Exception:
            logger.exception("Failed to clear old recall hints")

        try:
            await self._memory.clear_old_chain_links(
                keep_days=self._memory_config.chain_links_retention_days,
            )
        except Exception:
            logger.exception("Failed to clear old chain links")

        # 6. Expire stale pending proposals (older than 7 days)
        try:
            expired = await self._memory.expire_old_proposals(
                max_age_days=self._memory_config.proposal_expiry_days,
            )
            if expired > 0:
                logger.info("Expired %d stale proposals", expired)
        except Exception:
            logger.exception("Failed to expire old proposals")

        logger.info("Sleep phase completed")

    async def _write_diary(
        self,
        user: UserSummary,
        soul_snap: dict[str, Any],
    ) -> None:
        try:
            messages = await self._memory.get_todays_messages(user.user_id)
            if not messages:
                return

            conversation = "\n".join(
                f"{'User' if m['role'] == 'user' else soul_snap['name']}: "
                f"{m['content']}"
                for m in messages
            )
            system = _DIARY_SYSTEM_PROMPT.format(name=soul_snap["name"])
            prompt = f"Today's conversation with {user.display_name}:\n\n{conversation}"

            diary_text = await self._ai.generate(
                prompt=prompt,
                system=system,
                temperature=self._memory_config.diary_temperature,
                max_tokens=self._memory_config.diary_max_tokens,
            )

            await self._memory.add_certain_record(
                user_id=user.user_id,
                content=diary_text.strip(),
                record_type="diary",
                metadata={"date": datetime.now(tz=UTC).strftime("%Y-%m-%d")},
            )
            logger.info("Diary written for user %s", user.user_id)
        except Exception:
            logger.exception("Failed to write diary for user %s", user.user_id)

    async def _promote_episodic_memories(self, user_id: str) -> None:
        """AI reviews today's episodic memories and promotes general facts."""
        try:
            episodic = await self._memory.search_episodic(
                user_id,
                "today",
                n_results=self._memory_config.consolidation_episode_results,
            )
            if not episodic:
                return

            episodes_text = "\n".join(f"- {m['content']}" for m in episodic)
            system = _PROMOTION_SYSTEM_PROMPT.format(episodes=episodes_text)

            raw = await self._ai.generate(
                prompt=("Extract generalizable facts from the episodes above."),
                system=system,
                temperature=self._memory_config.consolidation_temperature,
            )
            data = json.loads(raw)
            facts = data.get("facts", [])
            if not isinstance(facts, list):
                return

            for fact in facts[: self._memory_config.consolidation_facts_limit]:
                if not isinstance(fact, str) or len(fact.strip()) < 3:
                    continue
                entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    layer=MemoryLayer.SEMANTIC,
                    content=fact.strip(),
                    metadata={"source": "sleep_promotion"},
                )
                await self._memory.add_semantic_memory(entry)
                logger.debug(
                    "Promoted episodic → semantic for %s: %s",
                    user_id,
                    fact,
                )
        except Exception:
            logger.exception("Episodic promotion failed for user %s", user_id)

    async def _precompute_temporal_recall(self, user: UserSummary) -> None:
        """Precompute temporal recall hints for multiple lookback windows."""
        for days_ago, label in TEMPORAL_WINDOWS:
            try:
                target_date = (
                    datetime.now(tz=UTC) - timedelta(days=days_ago)
                ).strftime("%Y-%m-%d")
                messages = await self._memory.get_messages_on_date(
                    user.user_id, target_date
                )
                if not messages:
                    continue

                topics: list[str] = []
                for msg in messages:
                    if msg["role"] == "user" and len(msg["content"]) > 10:
                        topics.append(msg["content"][:100])
                if not topics:
                    continue

                summary = "; ".join(topics[:3])
                content = (
                    f"{label} ({target_date}), "
                    f"{user.display_name} talked about: {summary}"
                )
                await self._memory.store_recall_hint(
                    user_id=user.user_id,
                    hint_type="temporal",
                    content=content,
                    metadata={
                        "original_date": target_date,
                        "days_ago": days_ago,
                    },
                )
                logger.debug(
                    "Temporal recall hint stored for %s (%s): %s",
                    user.user_id,
                    label,
                    content[:80],
                )
            except Exception:
                logger.exception(
                    "Temporal recall (%s) failed for user %s",
                    label,
                    user.user_id,
                )

    async def _precompute_chain_links(self, user_id: str) -> None:
        """Precompute chain-recall links between related memories."""
        try:
            recent_episodes = await self._memory.search_episodic(
                user_id,
                "today",
                n_results=self._memory_config.chain_link_episode_results,
            )
            if not recent_episodes:
                return

            for episode in recent_episodes:
                ep_id = episode.get("id", "")
                ep_content = episode.get("content", "")
                if not ep_content:
                    continue

                related = await self._memory.search_semantic(
                    user_id,
                    ep_content,
                    n_results=(self._memory_config.chain_link_related_results),
                )
                for mem in related:
                    mem_id = mem.get("id", "")
                    mem_content = mem.get("content", "")
                    if mem_id == ep_id or not mem_content:
                        continue
                    await self._memory.store_chain_link(
                        user_id=user_id,
                        source_memory_id=ep_id,
                        linked_content=mem_content,
                        linked_memory_id=mem_id,
                        distance=mem.get("distance"),
                    )

                related_ep = await self._memory.search_episodic(
                    user_id,
                    ep_content,
                    n_results=(self._memory_config.chain_link_related_results),
                )
                for mem in related_ep:
                    mem_id = mem.get("id", "")
                    mem_content = mem.get("content", "")
                    if mem_id == ep_id or not mem_content:
                        continue
                    await self._memory.store_chain_link(
                        user_id=user_id,
                        source_memory_id=ep_id,
                        linked_content=mem_content,
                        linked_memory_id=mem_id,
                        distance=mem.get("distance"),
                    )

            logger.debug("Chain links computed for user %s", user_id)
        except Exception:
            logger.exception("Chain link precomputation failed for user %s", user_id)
