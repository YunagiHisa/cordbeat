"""Public memory subsystem facade.

The internals are split across ``_memory_*`` sibling modules; this module
re-exports :class:`MemoryStore` as the single public entry point.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any

import aiosqlite

from cordbeat.config import MemoryConfig
from cordbeat.exceptions import MemorySubsystemError
from cordbeat.models import MemoryEntry, UserSummary
from cordbeat.tools.metrics import MEMORY_QUERY_LATENCY, time_block

from .conversation import ConversationStore
from .migrations import apply_migrations
from .records import RecordStore
from .users import UserStore
from .vector import VectorMemory

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
        try:
            import sqlite_vec  # noqa: PLC0415
        except ImportError as exc:
            raise MemorySubsystemError(
                "sqlite-vec is not installed. "
                "Run: uv add sqlite-vec  (or pip install sqlite-vec)"
            ) from exc
        await conn.enable_load_extension(True)
        await conn.load_extension(sqlite_vec.loadable_path())
        await conn.enable_load_extension(False)

        await apply_migrations(conn)
        await self._log_foreign_key_check(conn)
        self.__conn = conn

        self.__users = UserStore(conn)
        self.__records = RecordStore(conn)
        self.__conversations = ConversationStore(conn)
        self.__vectors = VectorMemory(conn, self._config)
        logger.info("Memory store initialized")

    async def _log_foreign_key_check(self, conn: aiosqlite.Connection) -> None:
        try:
            cursor = await conn.execute("PRAGMA foreign_key_check")
            rows = list(await cursor.fetchall())
            await cursor.close()
        except Exception:
            logger.debug("FK check unavailable", exc_info=True)
            return

        if not rows:
            logger.debug("FK check: clean")
            return

        counts: dict[str, int] = {}
        for row in rows:
            table = str(row["table"])
            counts[table] = counts.get(table, 0) + 1
        summary = ", ".join(
            f"{table}={count}" for table, count in sorted(counts.items())
        )
        logger.warning(
            "FK check: %d orphan row(s): %s. Foreign keys are not enforced; "
            "see MIN8-1.",
            len(rows),
            summary,
        )
        for row in rows[:10]:
            logger.debug(
                "FK check detail: table=%s rowid=%s parent=%s fkid=%s",
                row["table"],
                row["rowid"],
                row["parent"],
                row["fkid"],
            )

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
        *,
        allow_repoint: bool = False,
    ) -> None:
        await self._users.link_platform(
            user_id,
            adapter_id,
            platform_user_id,
            allow_repoint=allow_repoint,
        )

    async def link_platform_if_absent(
        self,
        user_id: str,
        adapter_id: str,
        platform_user_id: str,
    ) -> bool:
        return await self._users.link_platform_if_absent(
            user_id,
            adapter_id,
            platform_user_id,
        )

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

    async def record_last_seen_channel(
        self,
        user_id: str,
        adapter_id: str,
        channel_id: str,
        is_dm: bool,
    ) -> None:
        await self._users.record_last_seen_channel(
            user_id, adapter_id, channel_id, is_dm
        )

    async def get_last_seen_channel(
        self,
        user_id: str,
        adapter_id: str,
    ) -> tuple[str, bool] | None:
        return await self._users.get_last_seen_channel(user_id, adapter_id)

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

    async def update_record_metadata(
        self,
        record_id: str,
        metadata: dict[str, Any],
    ) -> bool:
        return await self._records.update_record_metadata(record_id, metadata)

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
        channel_id: str = "",
        is_dm: bool = True,
        created_at: datetime | None = None,
        received_at: datetime | None = None,
    ) -> int:
        message_id = await self._conversations.add_message(
            user_id,
            role,
            content,
            adapter_id,
            channel_id,
            is_dm,
            created_at,
            received_at,
        )
        # Lifetime counter survives nightly history trimming so the
        # familiarity stage keeps growing with the relationship.
        await self._users.increment_total_messages(user_id)
        return message_id

    async def add_media_observation(
        self,
        message_id: int,
        *,
        summary: str,
        relation: str,
        mime_type: str = "",
        content_sha256: str = "",
        source_ref: str = "",
    ) -> None:
        await self._conversations.add_media_observation(
            message_id,
            summary=summary,
            relation=relation,
            mime_type=mime_type,
            content_sha256=content_sha256,
            source_ref=source_ref,
        )

    async def get_lifetime_message_count(self, user_id: str) -> int:
        """Total messages ever exchanged (not reduced by trimming)."""
        return await self._users.get_total_messages(user_id)

    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 20,
        channel_id: str | None = None,
        is_dm: bool | None = None,
        adapter_id: str | None = None,
    ) -> list[dict[str, str]]:
        return await self._conversations.get_recent_messages(
            user_id,
            limit,
            channel_id=channel_id,
            is_dm=is_dm,
            adapter_id=adapter_id,
        )

    async def get_recent_messages_with_media(
        self,
        user_id: str,
        limit: int = 20,
        channel_id: str | None = None,
        is_dm: bool | None = None,
        adapter_id: str | None = None,
    ) -> list[dict[str, object]]:
        return await self._conversations.get_recent_messages_with_media(
            user_id,
            limit,
            channel_id=channel_id,
            is_dm=is_dm,
            adapter_id=adapter_id,
        )

    async def get_todays_messages(
        self,
        user_id: str,
        timezone: str | tzinfo | None = UTC,
    ) -> list[dict[str, str]]:
        return await self._conversations.get_todays_messages(user_id, timezone)

    async def get_messages_between(
        self,
        user_id: str,
        start_iso: str,
        end_iso: str,
    ) -> list[dict[str, str]]:
        return await self._conversations.get_messages_between(
            user_id, start_iso, end_iso
        )

    async def get_messages_on_date(
        self,
        user_id: str,
        date_str: str,
        timezone: str | tzinfo | None = UTC,
    ) -> list[dict[str, str]]:
        return await self._conversations.get_messages_on_date(
            user_id, date_str, timezone
        )

    async def trim_old_messages(
        self,
        user_id: str,
        keep: int | None = None,
    ) -> int:
        if keep is None:
            keep = self._config.message_trim_keep
        return await self._conversations.trim_old_messages(user_id, keep)

    async def clear_conversation_history(self, user_id: str) -> int:
        """Delete all conversation messages for a user."""
        return await self._conversations.clear_conversation_history(user_id)

    async def count_messages(self, user_id: str) -> int:
        """Return total number of stored conversation messages for *user_id*."""
        return await self._conversations.count_messages(user_id)

    async def get_oldest_messages(
        self, user_id: str, limit: int
    ) -> list[dict[str, str]]:
        """Return the *limit* oldest conversation messages (ascending)."""
        return await self._conversations.get_oldest_messages(user_id, limit)

    async def delete_messages_with_ids(self, ids: list[str]) -> int:
        """Delete conversation messages by ID.  Returns deleted count."""
        return await self._conversations.delete_messages_with_ids(ids)

    # ── Semantic / episodic memory (delegates to VectorMemory) ───

    async def add_semantic_memory(self, entry: MemoryEntry) -> str:
        return await self._vectors.add_semantic(entry)

    async def search_semantic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        async with time_block(MEMORY_QUERY_LATENCY, {"kind": "semantic"}):
            return await self._vectors.search_semantic(user_id, query, n_results)

    async def add_episodic_memory(self, entry: MemoryEntry) -> str:
        return await self._vectors.add_episodic(entry)

    async def search_episodic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        async with time_block(MEMORY_QUERY_LATENCY, {"kind": "episodic"}):
            return await self._vectors.search_episodic(user_id, query, n_results)

    async def get_episodic_since(
        self,
        user_id: str,
        since_iso: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        async with time_block(MEMORY_QUERY_LATENCY, {"kind": "episodic_since"}):
            return await self._vectors.get_episodic_since(user_id, since_iso, limit)

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
        """Apply Ebbinghaus-inspired forgetting curve.

        ``elapsed_days`` is measured since the current base strength was
        written (``last_accessed_at`` in stored rows). Kept as a public helper
        for diagnostics / tests. Production code no longer needs to call this
        — strength is computed lazily inside :meth:`VectorMemory._search`.
        """
        effective_decay = self._config.decay_rate * (1.0 - emotion_weight * 0.5)
        return base_strength * (1.0 / (1.0 + effective_decay * elapsed_days))

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
            try:
                meta = json.loads(hint.get("metadata") or "{}")
            except json.JSONDecodeError:
                logger.debug(
                    "Skipping recall hint with invalid metadata",
                    exc_info=True,
                )
                continue
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
                try:
                    meta = json.loads(link.get("metadata") or "{}")
                except json.JSONDecodeError:
                    logger.debug(
                        "Skipping chain link with invalid metadata",
                        exc_info=True,
                    )
                    continue
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
