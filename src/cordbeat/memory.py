"""MEMORY system — 4-layer memory architecture with forgetting curves."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from cordbeat.config import MemoryConfig
from cordbeat.models import MemoryEntry, MemoryLayer, UserSummary

logger = logging.getLogger(__name__)

# ── SQLite schema ─────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    last_talked_at TEXT,
    last_platform TEXT,
    last_topic    TEXT DEFAULT '',
    emotional_tone TEXT DEFAULT '',
    attention_score REAL DEFAULT 0.5
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
                last_talked_at=datetime.fromisoformat(row["last_talked_at"])
                if row["last_talked_at"]
                else None,
                last_platform=row["last_platform"],
                last_topic=row["last_topic"] or "",
                emotional_tone=row["emotional_tone"] or "",
                attention_score=row["attention_score"] or 0.5,
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
            "attention_score=? WHERE user_id=?",
            (
                summary.display_name,
                summary.last_talked_at.isoformat() if summary.last_talked_at else None,
                summary.last_platform,
                summary.last_topic,
                summary.emotional_tone,
                summary.attention_score,
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
                last_talked_at=datetime.fromisoformat(row["last_talked_at"])
                if row["last_talked_at"]
                else None,
                last_platform=row["last_platform"],
                last_topic=row["last_topic"] or "",
                emotional_tone=row["emotional_tone"] or "",
                attention_score=row["attention_score"] or 0.5,
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
            (user_id, adapter_id, platform_user_id, datetime.now().isoformat()),
        )
        await self._db.commit()

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
        now = datetime.now()
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

                created_at = datetime.fromisoformat(created_str)
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
            (user_id, role, content, adapter_id, datetime.now().isoformat()),
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
        today = datetime.now().strftime("%Y-%m-%d")
        cursor = await self._db.execute(
            "SELECT role, content FROM conversation_messages "
            "WHERE user_id = ? AND created_at >= ? "
            "ORDER BY created_at ASC",
            (user_id, today),
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
                datetime.now().isoformat(),
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

    async def update_proposal_status(
        self,
        proposal_id: str,
        status: str,
    ) -> bool:
        """Update the status field inside a proposal's metadata.

        Returns True if the proposal was found and updated.
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
        """Return all proposals matching the given status."""
        if user_id:
            cursor = await self._db.execute(
                "SELECT * FROM certain_records "
                "WHERE record_type = 'proposal' AND user_id = ? "
                "ORDER BY created_at ASC",
                (user_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM certain_records "
                "WHERE record_type = 'proposal' "
                "ORDER BY created_at ASC",
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            record = dict(row)
            meta = json.loads(record.get("metadata") or "{}")
            if meta.get("status") == status:
                results.append(record)
        return results


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
        self._conn: aiosqlite.Connection | None = None
        self._users: _UserStore | None = None
        self._records: _RecordStore | None = None
        self._conversations: _ConversationStore | None = None
        self._vectors: _VectorMemory | None = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

        self._users = _UserStore(self._conn)
        self._records = _RecordStore(self._conn)
        self._conversations = _ConversationStore(self._conn)

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
        self._vectors = _VectorMemory(semantic, episodic)
        logger.info("Memory store initialized")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ── User management (delegates to _UserStore) ─────────────────

    async def get_or_create_user(
        self,
        user_id: str,
        display_name: str,
    ) -> UserSummary:
        return await self._users.get_or_create_user(user_id, display_name)  # type: ignore[union-attr]

    async def update_user_summary(self, summary: UserSummary) -> None:
        await self._users.update_user_summary(summary)  # type: ignore[union-attr]

    async def get_all_user_summaries(self) -> list[UserSummary]:
        return await self._users.get_all_user_summaries()  # type: ignore[union-attr]

    async def link_platform(
        self,
        user_id: str,
        adapter_id: str,
        platform_user_id: str,
    ) -> None:
        await self._users.link_platform(user_id, adapter_id, platform_user_id)  # type: ignore[union-attr]

    async def resolve_user(
        self,
        adapter_id: str,
        platform_user_id: str,
    ) -> str | None:
        return await self._users.resolve_user(adapter_id, platform_user_id)  # type: ignore[union-attr]

    async def resolve_platform_user(
        self,
        user_id: str,
        adapter_id: str,
    ) -> str | None:
        return await self._users.resolve_platform_user(user_id, adapter_id)  # type: ignore[union-attr]

    # ── Core profiles + certain records (delegates to _RecordStore) ──

    async def set_core_profile(
        self,
        user_id: str,
        key: str,
        value: str,
    ) -> None:
        await self._records.set_core_profile(user_id, key, value)  # type: ignore[union-attr]

    async def get_core_profile(self, user_id: str) -> dict[str, str]:
        return await self._records.get_core_profile(user_id)  # type: ignore[union-attr]

    async def add_certain_record(
        self,
        user_id: str,
        content: str,
        record_type: str = "log",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self._records.add_certain_record(  # type: ignore[union-attr]
            user_id, content, record_type, metadata
        )

    async def get_certain_records(
        self,
        user_id: str,
        record_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await self._records.get_certain_records(user_id, record_type, limit)  # type: ignore[union-attr]

    async def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        return await self._records.get_proposal(proposal_id)  # type: ignore[union-attr]

    async def update_proposal_status(
        self,
        proposal_id: str,
        status: str,
    ) -> bool:
        return await self._records.update_proposal_status(proposal_id, status)  # type: ignore[union-attr]

    async def get_pending_proposals(
        self,
        user_id: str | None = None,
        status: str = "pending",
    ) -> list[dict[str, Any]]:
        return await self._records.get_pending_proposals(user_id, status)  # type: ignore[union-attr]

    # ── Conversation history (delegates to _ConversationStore) ────

    async def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        adapter_id: str = "",
    ) -> None:
        await self._conversations.add_message(user_id, role, content, adapter_id)  # type: ignore[union-attr]

    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        return await self._conversations.get_recent_messages(user_id, limit)  # type: ignore[union-attr]

    async def get_todays_messages(
        self,
        user_id: str,
    ) -> list[dict[str, str]]:
        return await self._conversations.get_todays_messages(user_id)  # type: ignore[union-attr]

    async def trim_old_messages(
        self,
        user_id: str,
        keep: int = 100,
    ) -> int:
        return await self._conversations.trim_old_messages(user_id, keep)  # type: ignore[union-attr]

    # ── Semantic / episodic memory (delegates to _VectorMemory) ───

    async def add_semantic_memory(self, entry: MemoryEntry) -> str:
        return await self._vectors.add_semantic(entry)  # type: ignore[union-attr]

    async def search_semantic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._vectors.search_semantic(user_id, query, n_results)  # type: ignore[union-attr]

    async def add_episodic_memory(self, entry: MemoryEntry) -> str:
        return await self._vectors.add_episodic(entry)  # type: ignore[union-attr]

    async def search_episodic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._vectors.search_episodic(user_id, query, n_results)  # type: ignore[union-attr]

    async def add_flashbulb_memory(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self._vectors.add_flashbulb(user_id, content, metadata)  # type: ignore[union-attr]

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
        return await self._vectors.decay_and_archive(self._config)  # type: ignore[union-attr]
