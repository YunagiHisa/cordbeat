"""Public memory subsystem facade.

The internals are split across ``_memory_*`` sibling modules; this module
re-exports :class:`MemoryStore` as the single public entry point.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import sqlite_vec

from cordbeat._memory_common import SCHEMA, VECTOR_SCHEMA
from cordbeat._memory_conversation import ConversationStore
from cordbeat._memory_records import RecordStore
from cordbeat._memory_users import UserStore
from cordbeat._memory_vector import VectorMemory
from cordbeat.config import MemoryConfig
from cordbeat.exceptions import MemorySubsystemError
from cordbeat.models import MemoryEntry, UserSummary

logger = logging.getLogger(__name__)


class MemoryStore:
    """Facade over the 4-layer memory system.

    Delegates to internal stores:
    - :class:`UserStore` — user CRUD and platform links
    - :class:`RecordStore` — core profiles and certain records
    - :class:`ConversationStore` — conversation history
    - :class:`VectorMemory` — semantic/episodic vectors via sqlite-vec
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._db_path = Path(config.sqlite_path)
        self.__conn: aiosqlite.Connection | None = None
        self.__users: UserStore | None = None
        self.__records: RecordStore | None = None
        self.__conversations: ConversationStore | None = None
        self.__vectors: VectorMemory | None = None

    # ── Type-narrowed accessors (raise if initialize() was not called) ──

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self.__conn is None:
            raise MemorySubsystemError(
                "MemoryStore not initialized: call initialize() first"
            )
        return self.__conn

    @property
    def _users(self) -> UserStore:
        if self.__users is None:
            raise MemorySubsystemError(
                "MemoryStore not initialized: call initialize() first"
            )
        return self.__users

    @property
    def _records(self) -> RecordStore:
        if self.__records is None:
            raise MemorySubsystemError(
                "MemoryStore not initialized: call initialize() first"
            )
        return self.__records

    @property
    def _conversations(self) -> ConversationStore:
        if self.__conversations is None:
            raise MemorySubsystemError(
                "MemoryStore not initialized: call initialize() first"
            )
        return self.__conversations

    @property
    def _vectors(self) -> VectorMemory:
        if self.__vectors is None:
            raise MemorySubsystemError(
                "MemoryStore not initialized: call initialize() first"
            )
        return self.__vectors

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(self._db_path))
        conn.row_factory = aiosqlite.Row

        # Load the sqlite-vec extension so the vec0 virtual tables in
        # VECTOR_SCHEMA can be created/used on this connection.
        await conn.enable_load_extension(True)
        await conn.load_extension(sqlite_vec.loadable_path())
        await conn.enable_load_extension(False)

        await conn.executescript(SCHEMA)
        await conn.executescript(VECTOR_SCHEMA)
        await conn.commit()
        self.__conn = conn

        self.__users = UserStore(conn)
        self.__records = RecordStore(conn)
        self.__conversations = ConversationStore(conn)
        self.__vectors = VectorMemory(conn)
        logger.info("Memory store initialized")

    async def close(self) -> None:
        if self.__conn:
            await self.__conn.close()
            self.__conn = None

    # ── User management (delegates to UserStore) ─────────────────

    async def get_or_create_user(
        self,
        user_id: str,
        display_name: str,
    ) -> UserSummary:
        return await self._users.get_or_create_user(user_id, display_name)

    async def update_user_summary(self, summary: UserSummary) -> None:
        await self._users.update_user_summary(summary)

    async def get_all_user_summaries(self) -> list[UserSummary]:
        return await self._users.get_all_user_summaries()

    async def link_platform(
        self,
        user_id: str,
        adapter_id: str,
        platform_user_id: str,
    ) -> None:
        await self._users.link_platform(user_id, adapter_id, platform_user_id)

    async def resolve_user(
        self,
        adapter_id: str,
        platform_user_id: str,
    ) -> str | None:
        return await self._users.resolve_user(adapter_id, platform_user_id)

    async def unlink_platform(
        self,
        user_id: str,
        adapter_id: str,
    ) -> bool:
        return await self._users.unlink_platform(user_id, adapter_id)

    async def get_linked_platforms(
        self,
        user_id: str,
    ) -> list[dict[str, str]]:
        return await self._users.get_linked_platforms(user_id)

    async def resolve_platform_user(
        self,
        user_id: str,
        adapter_id: str,
    ) -> str | None:
        return await self._users.resolve_platform_user(user_id, adapter_id)

    # ── Link tokens (delegates to RecordStore) ──────────────────

    async def store_link_token(
        self,
        requester_adapter_id: str,
        requester_platform_user_id: str,
        token_expiry_minutes: int | None = None,
    ) -> str:
        if token_expiry_minutes is None:
            token_expiry_minutes = self._config.token_expiry_minutes
        return await self._records.store_link_token(
            requester_adapter_id,
            requester_platform_user_id,
            token_expiry_minutes,
        )

    async def verify_link_token(
        self,
        token: str,
    ) -> dict[str, str] | None:
        return await self._records.verify_link_token(token)

    # ── Core profiles + certain records (delegates to RecordStore) ──

    async def set_core_profile(
        self,
        user_id: str,
        key: str,
        value: str,
    ) -> None:
        await self._records.set_core_profile(user_id, key, value)

    async def get_core_profile(self, user_id: str) -> dict[str, str]:
        return await self._records.get_core_profile(user_id)

    async def add_certain_record(
        self,
        user_id: str,
        content: str,
        record_type: str = "log",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self._records.add_certain_record(
            user_id, content, record_type, metadata
        )

    async def get_certain_records(
        self,
        user_id: str,
        record_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await self._records.get_certain_records(user_id, record_type, limit)

    async def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        return await self._records.get_proposal(proposal_id)

    async def update_proposal_status(
        self,
        proposal_id: str,
        status: str,
    ) -> bool:
        return await self._records.update_proposal_status(proposal_id, status)

    async def get_pending_proposals(
        self,
        user_id: str | None = None,
        status: str = "pending",
    ) -> list[dict[str, Any]]:
        return await self._records.get_pending_proposals(user_id, status)

    async def expire_old_proposals(self, max_age_days: int = 7) -> int:
        return await self._records.expire_old_proposals(max_age_days)

    # ── Conversation history (delegates to ConversationStore) ────

    async def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        adapter_id: str = "",
    ) -> None:
        await self._conversations.add_message(user_id, role, content, adapter_id)

    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        return await self._conversations.get_recent_messages(user_id, limit)

    async def get_todays_messages(
        self,
        user_id: str,
    ) -> list[dict[str, str]]:
        return await self._conversations.get_todays_messages(user_id)

    async def get_messages_on_date(
        self,
        user_id: str,
        date_str: str,
    ) -> list[dict[str, str]]:
        return await self._conversations.get_messages_on_date(user_id, date_str)

    async def trim_old_messages(
        self,
        user_id: str,
        keep: int | None = None,
    ) -> int:
        if keep is None:
            keep = self._config.message_trim_keep
        return await self._conversations.trim_old_messages(user_id, keep)

    # ── Semantic / episodic memory (delegates to VectorMemory) ───

    async def add_semantic_memory(self, entry: MemoryEntry) -> str:
        return await self._vectors.add_semantic(entry)

    async def search_semantic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._vectors.search_semantic(user_id, query, n_results)

    async def add_episodic_memory(self, entry: MemoryEntry) -> str:
        return await self._vectors.add_episodic(entry)

    async def search_episodic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._vectors.search_episodic(user_id, query, n_results)

    async def search_by_emotion(
        self,
        user_id: str,
        emotion: str,
        query: str,
        n_results: int = 3,
    ) -> list[dict[str, Any]]:
        """Search episodic memories by emotional_tone metadata."""
        return await self._vectors.search_by_emotion(user_id, emotion, query, n_results)

    async def add_flashbulb_memory(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self._vectors.add_flashbulb(user_id, content, metadata)

    # ── Forgetting (Ebbinghaus decay) ─────────────────────────────

    def calculate_strength(
        self,
        base_strength: float,
        elapsed_days: float,
        emotion_weight: float = 0.0,
    ) -> float:
        """Apply Ebbinghaus-inspired forgetting curve."""
        effective_decay = self._config.decay_rate * (1.0 - emotion_weight * 0.5)
        return base_strength * (1.0 / (1.0 + effective_decay * elapsed_days))

    async def decay_and_archive_memories(self) -> dict[str, int]:
        return await self._vectors.decay_and_archive(self._config)

    # ── Recall hints (Phase4 precomputation) ──────────────────────

    async def store_recall_hint(
        self,
        user_id: str,
        hint_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a precomputed recall hint."""
        meta = {
            "hint_type": hint_type,
            "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
            **(metadata or {}),
        }
        return await self._records.add_certain_record(
            user_id=user_id,
            content=content,
            record_type="recall_hint",
            metadata=meta,
        )

    async def get_recall_hints(
        self,
        user_id: str,
        date_str: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get precomputed recall hints for a user, optionally by date."""
        all_hints = await self._records.get_certain_records(
            user_id,
            record_type="recall_hint",
            limit=self._config.recall_hints_limit,
        )
        if date_str is None:
            return all_hints
        result = []
        for hint in all_hints:
            meta = json.loads(hint.get("metadata") or "{}")
            if meta.get("date") == date_str:
                result.append(hint)
        return result

    async def clear_old_recall_hints(self, keep_days: int = 2) -> int:
        """Remove recall hints older than keep_days."""
        cutoff = (datetime.now(tz=UTC) - timedelta(days=keep_days)).isoformat()
        cursor = await self._conn.execute(
            "DELETE FROM certain_records "
            "WHERE record_type = 'recall_hint' AND created_at < ?",
            (cutoff,),
        )
        await self._conn.commit()
        return cursor.rowcount

    # ── Chain links (Phase4 chain recall) ──────────────────────────

    async def store_chain_link(
        self,
        user_id: str,
        source_memory_id: str,
        linked_content: str,
        linked_memory_id: str = "",
        distance: float | None = None,
    ) -> str:
        """Store a precomputed chain-recall link between memories."""
        meta: dict[str, Any] = {
            "source_memory_id": source_memory_id,
            "linked_memory_id": linked_memory_id,
            "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        }
        if distance is not None:
            meta["distance"] = distance
        return await self._records.add_certain_record(
            user_id=user_id,
            content=linked_content,
            record_type="chain_link",
            metadata=meta,
        )

    async def get_chain_links(
        self,
        user_id: str,
        source_memory_ids: list[str],
        *,
        max_depth: int = 1,
        max_results: int | None = None,
    ) -> list[str]:
        """Get linked memory contents for the given source memory IDs.

        With max_depth > 1, follows chain links transitively (multi-hop).
        Results are sorted by vector distance (closest first) when available.
        """
        if max_results is None:
            max_results = self._config.chain_link_max_results
        if not source_memory_ids:
            return []
        all_links = await self._records.get_certain_records(
            user_id,
            record_type="chain_link",
            limit=self._config.chain_link_query_limit,
        )

        scored: list[tuple[float, str]] = []
        seen_contents: set[str] = set()
        visited_sources: set[str] = set()
        current_sources = set(source_memory_ids)

        for depth in range(max_depth):
            next_sources: set[str] = set()
            for link in all_links:
                meta = json.loads(link.get("metadata") or "{}")
                if meta.get("source_memory_id") not in current_sources:
                    continue
                content = link.get("content", "")
                if not content or content in seen_contents:
                    continue

                base_distance = meta.get("distance", 1.0)
                penalty = self._config.chain_recall_depth_penalty
                adjusted = base_distance + depth * penalty
                scored.append((adjusted, content))
                seen_contents.add(content)

                linked_id = meta.get("linked_memory_id", "")
                if linked_id:
                    next_sources.add(linked_id)

            visited_sources |= current_sources
            current_sources = next_sources - visited_sources
            if not current_sources:
                break

        scored.sort(key=lambda x: x[0])
        return [content for _, content in scored[:max_results]]

    async def clear_old_chain_links(self, keep_days: int = 2) -> int:
        """Remove chain links older than keep_days."""
        cutoff = (datetime.now(tz=UTC) - timedelta(days=keep_days)).isoformat()
        cursor = await self._conn.execute(
            "DELETE FROM certain_records "
            "WHERE record_type = 'chain_link' AND created_at < ?",
            (cutoff,),
        )
        await self._conn.commit()
        return cursor.rowcount
