"""MEMORY system — 4-layer memory architecture with forgetting curves."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from cordbeat.config import MemoryConfig
from cordbeat.models import MemoryEntry, UserSummary

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


class MemoryStore:
    """4-layer memory system with SQLite for structured data
    and ChromaDB for vector-based semantic/episodic memory."""

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._db_path = Path(config.sqlite_path)
        self._chroma_path = Path(config.chroma_path)
        self._conn: aiosqlite.Connection | None = None
        self._chroma_client: Any = None
        self._semantic_collection: Any = None
        self._episodic_collection: Any = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

        self._chroma_path.mkdir(parents=True, exist_ok=True)
        import chromadb  # lazy: heavy dependency, loaded only when needed

        self._chroma_client = chromadb.PersistentClient(
            path=str(self._chroma_path),
        )
        self._semantic_collection = self._chroma_client.get_or_create_collection(
            name="semantic_memory",
        )
        self._episodic_collection = self._chroma_client.get_or_create_collection(
            name="episodic_memory",
        )
        logger.info("Memory store initialized")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            msg = "MemoryStore not initialized"
            raise RuntimeError(msg)
        return self._conn

    # ── Layer 1: Core Profile (SQLite, no forgetting) ─────────────────

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

    # ── Layer 2: Semantic Memory (ChromaDB, with forgetting) ──────────

    async def add_semantic_memory(self, entry: MemoryEntry) -> str:
        entry_id = entry.id or str(uuid.uuid4())
        self._semantic_collection.add(
            ids=[entry_id],
            documents=[entry.content],
            metadatas=[
                {
                    "user_id": entry.user_id,
                    "trust_level": entry.trust_level.value,
                    "strength": entry.strength,
                    "emotion_weight": entry.emotion_weight,
                    "created_at": entry.created_at.isoformat(),
                    "last_accessed_at": entry.last_accessed_at.isoformat(),
                    **{
                        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                        for k, v in entry.metadata.items()
                    },
                }
            ],
        )
        return entry_id

    async def search_semantic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        results = self._semantic_collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"user_id": user_id},
        )
        entries = []
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

    # ── Layer 3: Episodic Memory (ChromaDB, with forgetting) ──────────

    async def add_episodic_memory(self, entry: MemoryEntry) -> str:
        entry_id = entry.id or str(uuid.uuid4())
        self._episodic_collection.add(
            ids=[entry_id],
            documents=[entry.content],
            metadatas=[
                {
                    "user_id": entry.user_id,
                    "trust_level": entry.trust_level.value,
                    "strength": entry.strength,
                    "emotion_weight": entry.emotion_weight,
                    "created_at": entry.created_at.isoformat(),
                    "last_accessed_at": entry.last_accessed_at.isoformat(),
                    **{
                        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                        for k, v in entry.metadata.items()
                    },
                }
            ],
        )
        return entry_id

    async def search_episodic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        results = self._episodic_collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"user_id": user_id},
        )
        entries = []
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

    # ── Layer 4: Certain Records (SQLite, no forgetting) ──────────────

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

    # ── Conversation History ──────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        adapter_id: str = "",
    ) -> None:
        """Store a conversation message (role: 'user' or 'assistant')."""
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
        """Return recent messages in chronological order."""
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

    async def trim_old_messages(
        self,
        user_id: str,
        keep: int = 100,
    ) -> int:
        """Delete oldest messages beyond the keep limit. Returns count deleted."""
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

    # ── User management ───────────────────────────────────────────────

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

    # ── Forgetting (Ebbinghaus decay) ─────────────────────────────────

    def calculate_strength(
        self,
        base_strength: float,
        elapsed_days: float,
        emotion_weight: float = 0.0,
    ) -> float:
        """Apply Ebbinghaus-inspired forgetting curve.

        strength = base_strength * (1 / (1 + decay_rate * elapsed_days))
        Emotional memories decay slower.
        """
        effective_decay = self._config.decay_rate * (1.0 - emotion_weight * 0.5)
        return base_strength * (1.0 / (1.0 + effective_decay * elapsed_days))
