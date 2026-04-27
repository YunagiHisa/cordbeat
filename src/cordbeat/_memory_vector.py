"""sqlite-vec backed semantic and episodic memory (private).

Replaces the earlier ChromaDB-based implementation. Vectors are stored
alongside relational data in the same SQLite database via the
``sqlite-vec`` loadable extension, using the ``vec0`` virtual-table
variant with a ``user_id`` partition key for efficient per-user search.

Embeddings are produced locally via ``sentence-transformers`` so the
agent remains local-first. The model is loaded lazily on first use.
The model name and embedding dimension can be overridden via
:class:`cordbeat.config.MemoryConfig` (``embedding_model`` /
``embedding_dim``); changing the dimension requires a fresh database
because the ``vec0`` virtual table is created with a fixed size at
schema-migration time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import aiosqlite
import sqlite_vec

from cordbeat._memory_common import ensure_aware
from cordbeat.config import MemoryConfig
from cordbeat.models import MemoryEntry, MemoryLayer

logger = logging.getLogger(__name__)

# Default values used by schema migrations and for callers that do not
# go through MemoryConfig (e.g., direct VectorMemory construction in
# tests with no override). These mirror the dataclass defaults in
# ``MemoryConfig.embedding_model`` / ``embedding_dim``.
EMBEDDING_DIM = 384
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_models: dict[str, Any] = {}
_models_lock = Lock()


def _load_model(name: str) -> Any:
    """Lazily load (and cache) a sentence-transformers model by name."""
    cached = _models.get(name)
    if cached is not None:
        return cached
    with _models_lock:
        cached = _models.get(name)
        if cached is None:
            from sentence_transformers import SentenceTransformer

            cached = SentenceTransformer(name)
            _models[name] = cached
    return cached


async def embed_text(text: str, model_name: str = EMBEDDING_MODEL_NAME) -> bytes:
    """Encode *text* into a serialized float32 embedding for sqlite-vec."""

    def _run() -> bytes:
        model = _load_model(model_name)
        vec = model.encode(text, normalize_embeddings=True).tolist()
        serialized: bytes = sqlite_vec.serialize_float32(vec)
        return serialized

    return await asyncio.to_thread(_run)


def _serialize_metadata(entry: MemoryEntry) -> str:
    """Serialize entry.metadata to JSON, preserving native types."""
    payload: dict[str, Any] = dict(entry.metadata)
    return json.dumps(payload, default=str)


def _row_to_result(row: aiosqlite.Row, distance: float | None) -> dict[str, Any]:
    """Shape a metadata row + distance into the public result dict."""
    metadata: dict[str, Any] = {
        "user_id": row["user_id"],
        "trust_level": row["trust_level"],
        "strength": row["strength"],
        "emotion_weight": row["emotion_weight"],
        "created_at": row["created_at"],
        "last_accessed_at": row["last_accessed_at"],
    }
    extra = json.loads(row["metadata_json"] or "{}")
    metadata.update(extra)
    return {
        "id": row["id"],
        "content": row["content"],
        "metadata": metadata,
        "distance": distance,
    }


class VectorMemory:
    """sqlite-vec backed semantic/episodic vector store."""

    def __init__(self, conn: aiosqlite.Connection, config: MemoryConfig) -> None:
        self._conn = conn
        self._config = config
        self._model_name = config.embedding_model
        if config.embedding_dim != EMBEDDING_DIM:
            logger.warning(
                "MemoryConfig.embedding_dim=%d differs from the vec0 schema "
                "size (%d). The custom value is ignored; recreate the DB "
                "if you need a different dimension.",
                config.embedding_dim,
                EMBEDDING_DIM,
            )

    async def _embed(self, text: str) -> bytes:
        return await embed_text(text, self._model_name)

    async def add_semantic(self, entry: MemoryEntry) -> str:
        return await self._insert(entry, layer="semantic")

    async def add_episodic(self, entry: MemoryEntry) -> str:
        return await self._insert(entry, layer="episodic")

    async def search_semantic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._search("semantic", user_id, query, n_results)

    async def search_episodic(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._search("episodic", user_id, query, n_results)

    async def search_by_emotion(
        self,
        user_id: str,
        emotion: str,
        query: str,
        n_results: int = 3,
    ) -> list[dict[str, Any]]:
        """Search episodic memories and post-filter by ``emotional_tone``."""
        candidates = await self._search("episodic", user_id, query, n_results * 3)
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

    def _compute_strength(
        self,
        base_strength: float,
        created_at: datetime,
        emotion_weight: float,
        now: datetime,
    ) -> float:
        """Apply Ebbinghaus-inspired forgetting curve at read time."""
        elapsed_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
        effective_decay = self._config.decay_rate * (1.0 - emotion_weight * 0.5)
        return base_strength * (1.0 / (1.0 + effective_decay * elapsed_days))

    async def _insert(self, entry: MemoryEntry, *, layer: str) -> str:
        entry_id = entry.id or str(uuid.uuid4())
        embedding = await self._embed(entry.content)
        vec_table = f"{layer}_vectors"
        meta_table = f"{layer}_memory"

        if self._config.dedup_distance_threshold > 0.0:
            existing = await self._find_near_duplicate(
                vec_table, meta_table, entry.user_id, embedding
            )
            if existing is not None:
                await self._merge_duplicate(meta_table, existing, entry)
                return str(existing["id"])

        async with self._conn.execute(
            f"INSERT INTO {vec_table}(user_id, embedding) VALUES (?, ?)",  # noqa: S608
            (entry.user_id, embedding),
        ) as cur:
            vec_rowid = cur.lastrowid

        await self._conn.execute(
            f"""INSERT INTO {meta_table}
                (id, vec_rowid, user_id, content, trust_level, strength,
                 emotion_weight, created_at, last_accessed_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",  # noqa: S608
            (
                entry_id,
                vec_rowid,
                entry.user_id,
                entry.content,
                entry.trust_level.value,
                entry.strength,
                entry.emotion_weight,
                entry.created_at.isoformat(),
                entry.last_accessed_at.isoformat(),
                _serialize_metadata(entry),
            ),
        )
        await self._conn.commit()
        return entry_id

    async def _find_near_duplicate(
        self,
        vec_table: str,
        meta_table: str,
        user_id: str,
        embedding: bytes,
    ) -> dict[str, Any] | None:
        """Return the nearest existing memory if within dedup threshold."""
        sql = f"""
            SELECT m.id, m.strength, m.emotion_weight, v.distance
              FROM {vec_table} AS v
              JOIN {meta_table} AS m ON m.vec_rowid = v.rowid
             WHERE v.user_id = ?
               AND v.embedding MATCH ?
               AND k = 1
             ORDER BY v.distance
        """  # noqa: S608
        async with self._conn.execute(sql, (user_id, embedding)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        if float(row["distance"]) > self._config.dedup_distance_threshold:
            return None
        return {
            "id": row["id"],
            "strength": float(row["strength"]),
            "emotion_weight": float(row["emotion_weight"]),
            "distance": float(row["distance"]),
        }

    async def _merge_duplicate(
        self,
        meta_table: str,
        existing: dict[str, Any],
        new_entry: MemoryEntry,
    ) -> None:
        """Reinforce an existing memory instead of inserting a duplicate.

        Strength is bumped (capped at 1.0) and ``last_accessed_at`` is
        refreshed to ``now``. Emotion weight takes the max so that a
        more intense duplicate raises the floor.
        """
        new_strength = min(1.0, existing["strength"] + 0.1)
        new_emotion = max(existing["emotion_weight"], new_entry.emotion_weight)
        await self._conn.execute(
            f"""UPDATE {meta_table}
                   SET strength = ?,
                       emotion_weight = ?,
                       last_accessed_at = ?
                 WHERE id = ?""",  # noqa: S608
            (
                new_strength,
                new_emotion,
                datetime.now(tz=UTC).isoformat(),
                existing["id"],
            ),
        )
        await self._conn.commit()
        logger.debug(
            "memory dedup: merged near-duplicate (id=%s, distance=%.4f)",
            existing["id"],
            existing["distance"],
        )

    async def _search(
        self,
        layer: str,
        user_id: str,
        query: str,
        n_results: int,
    ) -> list[dict[str, Any]]:
        """Search with lazy decay.

        Strength is computed on read from ``base_strength`` + ``created_at``
        + ``emotion_weight`` (Ebbinghaus). Non-flashbulb entries that fall
        below ``archive_threshold`` are physically deleted (eager delete on
        read, per design questionnaire Q4-1=A). Results are ranked by a
        composite score ``strength / (1 + distance)`` so that strong
        memories outrank weakly-relevant ones at the margin.
        """
        vec_table = f"{layer}_vectors"
        meta_table = f"{layer}_memory"
        query_embedding = await self._embed(query)
        threshold = self._config.archive_threshold
        now = datetime.now(tz=UTC)

        # Oversample so eager-deleted rows don't starve the result set.
        fetch_k = max(n_results * 3, n_results + 5)

        sql = f"""
            SELECT m.id, m.vec_rowid, m.user_id, m.content, m.trust_level,
                   m.strength, m.emotion_weight, m.created_at,
                   m.last_accessed_at, m.metadata_json, v.distance
              FROM {vec_table} AS v
              JOIN {meta_table} AS m ON m.vec_rowid = v.rowid
             WHERE v.user_id = ?
               AND v.embedding MATCH ?
               AND k = ?
             ORDER BY v.distance
        """
        async with self._conn.execute(sql, (user_id, query_embedding, fetch_k)) as cur:
            rows = await cur.fetchall()

        enriched: list[tuple[float, dict[str, Any]]] = []
        to_delete: list[tuple[str, int, str]] = []

        for row in rows:
            extras = json.loads(row["metadata_json"] or "{}")
            is_flashbulb = bool(extras.get("flashbulb", False))
            created_at = ensure_aware(datetime.fromisoformat(row["created_at"]))
            current_strength = self._compute_strength(
                base_strength=float(row["strength"]),
                created_at=created_at,
                emotion_weight=float(row["emotion_weight"]),
                now=now,
            )

            if not is_flashbulb and current_strength < threshold:
                to_delete.append((row["id"], row["vec_rowid"], row["user_id"]))
                continue

            result = _row_to_result(row, row["distance"])
            result["metadata"]["strength"] = current_strength
            distance = float(row["distance"])
            score = current_strength / (1.0 + distance)
            enriched.append((score, result))

        for entry_id, vec_rowid, uid in to_delete:
            await self._conn.execute(
                f"DELETE FROM {vec_table} WHERE user_id = ? AND rowid = ?",
                (uid, vec_rowid),
            )
            await self._conn.execute(
                f"DELETE FROM {meta_table} WHERE id = ?",
                (entry_id,),
            )
        if to_delete:
            await self._conn.commit()

        enriched.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in enriched[:n_results]]
