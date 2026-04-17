"""MEMORY system — 4-layer memory architecture with forgetting curves."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from cordbeat.config import MemoryConfig
from cordbeat.models import MemoryEntry, MemoryLayer, UserSummary

logger = logging.getLogger(__name__)


def _ensure_aware(dt: datetime) -> datetime:
    """Return *dt* with UTC timezone attached if it was naive."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


# ── SQLite schema ─────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    last_talked_at TEXT,
    last_platform TEXT,
    last_topic    TEXT DEFAULT '',
    emotional_tone TEXT DEFAULT '',
    attention_score REAL DEFAULT 0.5,
    preferred_platform TEXT
);

CREATE TABLE IF NOT EXISTS platform_links (
    user_id          TEXT NOT NULL,
    adapter_id       TEXT NOT NULL,
    platform_user_id TEXT NOT NULL,
    linked_at        TEXT NOT NULL,
    PRIMARY KEY (adapter_id, platform_user_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS core_profiles (
    user_id  TEXT NOT NULL,
    key      TEXT NOT NULL,
    value    TEXT NOT NULL,
    PRIMARY KEY (user_id, key),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS certain_records (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    content    TEXT NOT NULL,
    record_type TEXT DEFAULT 'log',
    created_at TEXT NOT NULL,
    metadata   TEXT DEFAULT '{}',
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    adapter_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_conv_user_time
    ON conversation_messages (user_id, created_at DESC);
"""


# ── Internal: User management ─────────────────────────────────────────


class _UserStore:
    """Handles user CRUD and platform link resolution."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def get_or_create_user(
        self,
        user_id: str,
        display_name: str,
    ) -> UserSummary:
        cursor = await self._db.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row:
            return UserSummary(
                user_id=row["user_id"],
                display_name=row["display_name"],
                last_talked_at=_ensure_aware(
                    datetime.fromisoformat(row["last_talked_at"])
                )
                if row["last_talked_at"]
                else None,
                last_platform=row["last_platform"],
                last_topic=row["last_topic"] or "",
                emotional_tone=row["emotional_tone"] or "",
                attention_score=row["attention_score"] or 0.5,
                preferred_platform=row["preferred_platform"],
            )
        await self._db.execute(
            "INSERT INTO users (user_id, display_name) VALUES (?, ?)",
            (user_id, display_name),
        )
        await self._db.commit()
        return UserSummary(user_id=user_id, display_name=display_name)

    async def update_user_summary(self, summary: UserSummary) -> None:
        await self._db.execute(
            "UPDATE users SET display_name=?, last_talked_at=?, "
            "last_platform=?, last_topic=?, emotional_tone=?, "
            "attention_score=?, preferred_platform=? WHERE user_id=?",
            (
                summary.display_name,
                summary.last_talked_at.isoformat() if summary.last_talked_at else None,
                summary.last_platform,
                summary.last_topic,
                summary.emotional_tone,
                summary.attention_score,
                summary.preferred_platform,
                summary.user_id,
            ),
        )
        await self._db.commit()

    async def get_all_user_summaries(self) -> list[UserSummary]:
        cursor = await self._db.execute("SELECT * FROM users")
        rows = await cursor.fetchall()
        return [
            UserSummary(
                user_id=row["user_id"],
                display_name=row["display_name"],
                last_talked_at=_ensure_aware(
                    datetime.fromisoformat(row["last_talked_at"])
                )
                if row["last_talked_at"]
                else None,
                last_platform=row["last_platform"],
                last_topic=row["last_topic"] or "",
                emotional_tone=row["emotional_tone"] or "",
                attention_score=row["attention_score"] or 0.5,
                preferred_platform=row["preferred_platform"],
            )
            for row in rows
        ]

    async def link_platform(
        self,
        user_id: str,
        adapter_id: str,
        platform_user_id: str,
    ) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO platform_links "
            "(user_id, adapter_id, platform_user_id, linked_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, adapter_id, platform_user_id, datetime.now(tz=UTC).isoformat()),
        )
        await self._db.commit()

    async def unlink_platform(
        self,
        user_id: str,
        adapter_id: str,
    ) -> bool:
        """Remove a platform link. Returns True if a row was deleted."""
        cursor = await self._db.execute(
            "DELETE FROM platform_links WHERE user_id = ? AND adapter_id = ?",
            (user_id, adapter_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def get_linked_platforms(
        self,
        user_id: str,
    ) -> list[dict[str, str]]:
        """Return all platform links for a user."""
        cursor = await self._db.execute(
            "SELECT adapter_id, platform_user_id FROM platform_links WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "adapter_id": row["adapter_id"],
                "platform_user_id": row["platform_user_id"],
            }
            for row in rows
        ]

    async def resolve_user(
        self,
        adapter_id: str,
        platform_user_id: str,
    ) -> str | None:
        cursor = await self._db.execute(
            "SELECT user_id FROM platform_links "
            "WHERE adapter_id = ? AND platform_user_id = ?",
            (adapter_id, platform_user_id),
        )
        row = await cursor.fetchone()
        return row["user_id"] if row else None

    async def resolve_platform_user(
        self,
        user_id: str,
        adapter_id: str,
    ) -> str | None:
        """Reverse-lookup: internal user_id + adapter → platform_user_id."""
        cursor = await self._db.execute(
            "SELECT platform_user_id FROM platform_links "
            "WHERE user_id = ? AND adapter_id = ?",
            (user_id, adapter_id),
        )
        row = await cursor.fetchone()
        return row["platform_user_id"] if row else None


# ── Internal: Vector memory (semantic + episodic) ─────────────────────


def _build_memory_metadata(entry: MemoryEntry) -> dict[str, Any]:
    """Build ChromaDB-compatible metadata dict from a MemoryEntry."""
    metadata: dict[str, Any] = {
        "user_id": entry.user_id,
        "trust_level": entry.trust_level.value,
        "strength": entry.strength,
        "emotion_weight": entry.emotion_weight,
        "created_at": entry.created_at.isoformat(),
        "last_accessed_at": entry.last_accessed_at.isoformat(),
    }
    for k, v in entry.metadata.items():
        if k == "flashbulb":
            metadata["flashbulb"] = str(v)
        elif isinstance(v, (dict, list)):
            metadata[k] = json.dumps(v)
        else:
            metadata[k] = str(v)
    return metadata


class _VectorMemory:
    """Wraps ChromaDB collections for semantic and episodic memory."""

    def __init__(self, semantic: Any, episodic: Any) -> None:
        self._semantic = semantic
        self._episodic = episodic

    async def add_semantic(self, entry: MemoryEntry) -> str:
        entry_id = entry.id or str(uuid.uuid4())
        metadata = _build_memory_metadata(entry)
        await asyncio.to_thread(
            self._semantic.add,
            ids=[entry_id],
            documents=[entry.content],
            metadatas=[metadata],
        )
        return entry_id

    async def search_semantic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._query_collection(self._semantic, user_id, query, n_results)

    async def add_episodic(self, entry: MemoryEntry) -> str:
        entry_id = entry.id or str(uuid.uuid4())
        metadata = _build_memory_metadata(entry)
        await asyncio.to_thread(
            self._episodic.add,
            ids=[entry_id],
            documents=[entry.content],
            metadatas=[metadata],
        )
        return entry_id

    async def search_episodic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._query_collection(self._episodic, user_id, query, n_results)

    async def search_by_emotion(
        self,
        user_id: str,
        emotion: str,
        query: str,
        n_results: int = 3,
    ) -> list[dict[str, Any]]:
        """Search episodic memories filtered by emotional_tone metadata.

        Queries by user_id via ChromaDB, then post-filters by emotion tag.
        """
        candidates = await self._query_collection(
            self._episodic, user_id, query, n_results * 3
        )
        return [
            r
            for r in candidates
            if r.get("metadata", {}).get("emotional_tone") == emotion
        ][:n_results]

    async def add_flashbulb(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            layer=MemoryLayer.EPISODIC,
            content=content,
            strength=1.0,
            emotion_weight=1.0,
            metadata={"flashbulb": True, **(metadata or {})},
        )
        return await self.add_episodic(entry)

    async def decay_and_archive(
        self,
        config: MemoryConfig,
    ) -> dict[str, int]:
        threshold = config.archive_threshold
        now = datetime.now(tz=UTC)
        stats = {"decayed": 0, "archived": 0}

        for collection in (self._semantic, self._episodic):
            all_data = await asyncio.to_thread(collection.get, include=["metadatas"])
            if not all_data or not all_data.get("ids"):
                continue

            ids_to_delete: list[str] = []
            for i, entry_id in enumerate(all_data["ids"]):
                meta = all_data["metadatas"][i]

                if str(meta.get("flashbulb", "False")).lower() == "true":
                    continue

                created_str = meta.get("created_at", "")
                if not created_str:
                    continue

                created_at = _ensure_aware(datetime.fromisoformat(created_str))
                elapsed_days = (now - created_at).total_seconds() / 86400.0
                base_strength = float(meta.get("strength", 1.0))
                emotion_weight = float(meta.get("emotion_weight", 0.0))

                effective_decay = config.decay_rate * (1.0 - emotion_weight * 0.5)
                new_strength = base_strength * (
                    1.0 / (1.0 + effective_decay * elapsed_days)
                )

                if new_strength < threshold:
                    ids_to_delete.append(entry_id)
                    stats["archived"] += 1
                else:
                    stats["decayed"] += 1

            if ids_to_delete:
                await asyncio.to_thread(collection.delete, ids=ids_to_delete)

        return stats

    @staticmethod
    async def _query_collection(
        collection: Any,
        user_id: str,
        query: str,
        n_results: int,
    ) -> list[dict[str, Any]]:
        results = await asyncio.to_thread(
            collection.query,
            query_texts=[query],
            n_results=n_results,
            where={"user_id": user_id},
        )
        entries: list[dict[str, Any]] = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                entries.append(
                    {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                        if results.get("distances")
                        else None,
                    }
                )
        return entries


# ── Internal: Conversation history ────────────────────────────────────


class _ConversationStore:
    """SQLite-backed conversation message storage."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        adapter_id: str = "",
    ) -> None:
        await self._db.execute(
            "INSERT INTO conversation_messages "
            "(user_id, role, content, adapter_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, role, content, adapter_id, datetime.now(tz=UTC).isoformat()),
        )
        await self._db.commit()

    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        cursor = await self._db.execute(
            "SELECT role, content FROM ("
            "  SELECT role, content, created_at "
            "  FROM conversation_messages "
            "  WHERE user_id = ? "
            "  ORDER BY created_at DESC LIMIT ?"
            ") sub ORDER BY created_at ASC",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def get_todays_messages(
        self,
        user_id: str,
    ) -> list[dict[str, str]]:
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        cursor = await self._db.execute(
            "SELECT role, content FROM conversation_messages "
            "WHERE user_id = ? AND created_at >= ? "
            "ORDER BY created_at ASC",
            (user_id, today),
        )
        rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def get_messages_on_date(
        self,
        user_id: str,
        date_str: str,
    ) -> list[dict[str, str]]:
        """Get all messages on a specific date (YYYY-MM-DD)."""
        next_day = (datetime.fromisoformat(date_str) + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        cursor = await self._db.execute(
            "SELECT role, content FROM conversation_messages "
            "WHERE user_id = ? AND created_at >= ? AND created_at < ? "
            "ORDER BY created_at ASC",
            (user_id, date_str, next_day),
        )
        rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def trim_old_messages(
        self,
        user_id: str,
        keep: int = 100,
    ) -> int:
        cursor = await self._db.execute(
            "DELETE FROM conversation_messages WHERE user_id = ? "
            "AND id NOT IN ("
            "  SELECT id FROM conversation_messages "
            "  WHERE user_id = ? ORDER BY created_at DESC LIMIT ?"
            ")",
            (user_id, user_id, keep),
        )
        await self._db.commit()
        return cursor.rowcount


# ── Internal: Certain records ─────────────────────────────────────────


class _RecordStore:
    """SQLite-backed immutable records (core profiles + certain records)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def set_core_profile(
        self,
        user_id: str,
        key: str,
        value: str,
    ) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO core_profiles (user_id, key, value) "
            "VALUES (?, ?, ?)",
            (user_id, key, value),
        )
        await self._db.commit()

    async def get_core_profile(self, user_id: str) -> dict[str, str]:
        cursor = await self._db.execute(
            "SELECT key, value FROM core_profiles WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}

    async def add_certain_record(
        self,
        user_id: str,
        content: str,
        record_type: str = "log",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        record_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO certain_records "
            "(id, user_id, content, record_type, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record_id,
                user_id,
                content,
                record_type,
                datetime.now(tz=UTC).isoformat(),
                json.dumps(metadata or {}),
            ),
        )
        await self._db.commit()
        return record_id

    async def get_certain_records(
        self,
        user_id: str,
        record_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if record_type:
            cursor = await self._db.execute(
                "SELECT * FROM certain_records "
                "WHERE user_id = ? AND record_type = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, record_type, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM certain_records "
                "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        """Retrieve a single proposal by ID."""
        cursor = await self._db.execute(
            "SELECT * FROM certain_records WHERE id = ? AND record_type = 'proposal'",
            (proposal_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    # Valid proposal status transitions
    _VALID_TRANSITIONS: dict[str, set[str]] = {
        "pending": {"approved", "rejected", "expired"},
        "approved": {"executed", "expired"},
    }

    async def update_proposal_status(
        self,
        proposal_id: str,
        status: str,
    ) -> bool:
        """Update the status field inside a proposal's metadata.

        Returns True if the proposal was found and updated.
        Raises ValueError on invalid status transitions.
        """
        row_cursor = await self._db.execute(
            "SELECT metadata FROM certain_records "
            "WHERE id = ? AND record_type = 'proposal'",
            (proposal_id,),
        )
        row = await row_cursor.fetchone()
        if row is None:
            return False

        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        current = meta.get("status", "pending")
        allowed = self._VALID_TRANSITIONS.get(current, set())
        if status not in allowed:
            msg = (
                f"Invalid proposal transition: '{current}' → '{status}' "
                f"(allowed: {allowed})"
            )
            raise ValueError(msg)

        meta["status"] = status
        await self._db.execute(
            "UPDATE certain_records SET metadata = ? WHERE id = ?",
            (json.dumps(meta), proposal_id),
        )
        await self._db.commit()
        return True

    async def get_pending_proposals(
        self,
        user_id: str | None = None,
        status: str = "pending",
    ) -> list[dict[str, Any]]:
        """Return all proposals matching the given status.

        Uses SQLite's ``json_extract()`` to filter at the database level
        instead of loading every proposal into Python.
        """
        if user_id:
            cursor = await self._db.execute(
                "SELECT * FROM certain_records "
                "WHERE record_type = 'proposal' AND user_id = ? "
                "AND json_extract(metadata, '$.status') = ? "
                "ORDER BY created_at ASC",
                (user_id, status),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM certain_records "
                "WHERE record_type = 'proposal' "
                "AND json_extract(metadata, '$.status') = ? "
                "ORDER BY created_at ASC",
                (status,),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def expire_old_proposals(self, max_age_days: int = 7) -> int:
        """Expire PENDING proposals older than max_age_days.

        Returns the number of proposals expired.
        """
        cutoff = (datetime.now(tz=UTC) - timedelta(days=max_age_days)).isoformat()
        cursor = await self._db.execute(
            "SELECT id, metadata FROM certain_records "
            "WHERE record_type = 'proposal' AND created_at < ?",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        count = 0
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            if meta.get("status") == "pending":
                meta["status"] = "expired"
                await self._db.execute(
                    "UPDATE certain_records SET metadata = ? WHERE id = ?",
                    (json.dumps(meta), row["id"]),
                )
                count += 1
        if count > 0:
            await self._db.commit()
        return count

    async def store_link_token(
        self,
        requester_adapter_id: str,
        requester_platform_user_id: str,
        token_expiry_minutes: int = 10,
    ) -> str:
        """Generate and store a link token for cross-platform account linking.

        Returns the generated token string.
        """
        token = secrets.token_urlsafe(16)
        expiry = (
            datetime.now(tz=UTC) + timedelta(minutes=token_expiry_minutes)
        ).isoformat()
        metadata = {
            "requester_adapter_id": requester_adapter_id,
            "requester_platform_user_id": requester_platform_user_id,
            "expires_at": expiry,
            "used": False,
        }
        record_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO certain_records "
            "(id, user_id, content, record_type, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record_id,
                "__system__",
                token,
                "link_token",
                datetime.now(tz=UTC).isoformat(),
                json.dumps(metadata),
            ),
        )
        await self._db.commit()
        return token

    async def verify_link_token(
        self,
        token: str,
    ) -> dict[str, str] | None:
        """Verify a link token and return requester info if valid.

        Returns a dict with ``requester_adapter_id`` and
        ``requester_platform_user_id`` if the token is valid and not
        expired. Returns ``None`` otherwise. Marks the token as used.
        """
        cursor = await self._db.execute(
            "SELECT * FROM certain_records "
            "WHERE record_type = 'link_token' AND content = ?",
            (token,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        meta = json.loads(row["metadata"]) if row["metadata"] else {}

        # Check if already used
        if meta.get("used"):
            return None

        # Check expiry
        expires_at = meta.get("expires_at", "")
        if expires_at and _ensure_aware(
            datetime.fromisoformat(expires_at)
        ) < datetime.now(tz=UTC):
            return None

        # Mark as used
        meta["used"] = True
        await self._db.execute(
            "UPDATE certain_records SET metadata = ? WHERE id = ?",
            (json.dumps(meta), row["id"]),
        )
        await self._db.commit()

        return {
            "requester_adapter_id": meta["requester_adapter_id"],
            "requester_platform_user_id": meta["requester_platform_user_id"],
        }


# ── Facade ────────────────────────────────────────────────────────────


class MemoryStore:
    """Facade over the 4-layer memory system.

    Delegates to internal stores:
    - ``_UserStore`` — user CRUD and platform links
    - ``_RecordStore`` — core profiles and certain records
    - ``_ConversationStore`` — conversation history
    - ``_VectorMemory`` — semantic/episodic ChromaDB collections
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._db_path = Path(config.sqlite_path)
        self._chroma_path = Path(config.chroma_path)
        self.__conn: aiosqlite.Connection | None = None
        self.__users: _UserStore | None = None
        self.__records: _RecordStore | None = None
        self.__conversations: _ConversationStore | None = None
        self.__vectors: _VectorMemory | None = None

    # ── Type-narrowed accessors (raise if initialize() was not called) ──

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self.__conn is None:
            raise RuntimeError("MemoryStore not initialized: call initialize() first")
        return self.__conn

    @property
    def _users(self) -> _UserStore:
        if self.__users is None:
            raise RuntimeError("MemoryStore not initialized: call initialize() first")
        return self.__users

    @property
    def _records(self) -> _RecordStore:
        if self.__records is None:
            raise RuntimeError("MemoryStore not initialized: call initialize() first")
        return self.__records

    @property
    def _conversations(self) -> _ConversationStore:
        if self.__conversations is None:
            raise RuntimeError("MemoryStore not initialized: call initialize() first")
        return self.__conversations

    @property
    def _vectors(self) -> _VectorMemory:
        if self.__vectors is None:
            raise RuntimeError("MemoryStore not initialized: call initialize() first")
        return self.__vectors

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(self._db_path))
        conn.row_factory = aiosqlite.Row
        await conn.executescript(_SCHEMA)
        await conn.commit()
        self.__conn = conn

        self.__users = _UserStore(conn)
        self.__records = _RecordStore(conn)
        self.__conversations = _ConversationStore(conn)

        self._chroma_path.mkdir(parents=True, exist_ok=True)
        import chromadb

        client = await asyncio.to_thread(
            chromadb.PersistentClient,
            path=str(self._chroma_path),
        )
        semantic = await asyncio.to_thread(
            client.get_or_create_collection,
            name="semantic_memory",
        )
        episodic = await asyncio.to_thread(
            client.get_or_create_collection,
            name="episodic_memory",
        )
        self.__vectors = _VectorMemory(semantic, episodic)
        logger.info("Memory store initialized")

    async def close(self) -> None:
        if self.__conn:
            await self.__conn.close()
            self.__conn = None

    # ── User management (delegates to _UserStore) ─────────────────

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

    # ── Link tokens (delegates to _RecordStore) ──────────────────

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

    # ── Core profiles + certain records (delegates to _RecordStore) ──

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

    # ── Conversation history (delegates to _ConversationStore) ────

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

    # ── Semantic / episodic memory (delegates to _VectorMemory) ───

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

    # ── Chain links (Phase4 芋づる想起) ───────────────────────────

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

                # Distance increases with depth to prefer direct links
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
