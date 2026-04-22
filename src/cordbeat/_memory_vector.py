"""ChromaDB-backed semantic and episodic memory (private)."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from cordbeat._memory_common import ensure_aware
from cordbeat.config import MemoryConfig
from cordbeat.models import MemoryEntry, MemoryLayer


def build_memory_metadata(entry: MemoryEntry) -> dict[str, Any]:
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


class VectorMemory:
    """Wraps ChromaDB collections for semantic and episodic memory."""

    def __init__(self, semantic: Any, episodic: Any) -> None:
        self._semantic = semantic
        self._episodic = episodic

    async def add_semantic(self, entry: MemoryEntry) -> str:
        entry_id = entry.id or str(uuid.uuid4())
        metadata = build_memory_metadata(entry)
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
        metadata = build_memory_metadata(entry)
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

                created_at = ensure_aware(datetime.fromisoformat(created_str))
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
