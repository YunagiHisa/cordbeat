"""End-to-end integration test against real sqlite-vec + sentence-transformers.

Most of the existing ``test_memory.py`` tests already exercise the real
backends via ``tmp_path``. This file complements them with a single
high-level scenario that walks the full memory lifecycle in one place,
so a regression in the embedding/search/decay pipeline surfaces as a
clear "integration broken" signal rather than a scattered unit failure.

Run only this file with::

    uv run pytest tests/integration/ -v

The ``integration`` marker is purely informational; pytest still
executes these tests by default.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cordbeat.config import MemoryConfig
from cordbeat.memory import MemoryStore
from cordbeat.models import MemoryEntry, MemoryLayer, TrustLevel

pytestmark = pytest.mark.integration


@pytest.fixture
async def store(tmp_path: Path) -> MemoryStore:
    config = MemoryConfig(sqlite_path=str(tmp_path / "integration.db"))
    s = MemoryStore(config)
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


def _entry(
    user_id: str,
    layer: MemoryLayer,
    content: str,
    *,
    strength: float = 1.0,
    emotion_weight: float = 0.0,
    created_at: datetime | None = None,
) -> MemoryEntry:
    now = datetime.now(UTC)
    return MemoryEntry(
        id="",  # filled in by VectorMemory
        user_id=user_id,
        layer=layer,
        content=content,
        trust_level=TrustLevel.NORMAL,
        strength=strength,
        emotion_weight=emotion_weight,
        created_at=created_at or now,
        last_accessed_at=created_at or now,
    )


class TestMemoryEndToEnd:
    """Full insert → search → user-isolation → decay → flashbulb flow."""

    async def test_full_lifecycle(self, store: MemoryStore) -> None:
        # 1. Two users with distinct semantic facts.
        await store.get_or_create_user("alice", "Alice")
        await store.get_or_create_user("bob", "Bob")

        await store.add_semantic_memory(
            _entry("alice", MemoryLayer.SEMANTIC, "Alice loves jazz piano music"),
        )
        await store.add_semantic_memory(
            _entry(
                "alice", MemoryLayer.SEMANTIC, "Alice prefers green tea over coffee"
            ),
        )
        await store.add_semantic_memory(
            _entry("bob", MemoryLayer.SEMANTIC, "Bob plays competitive chess"),
        )

        # 2. Semantic search returns relevant + isolated results.
        alice_hits = await store.search_semantic("alice", "music", n_results=5)
        assert any("jazz" in hit["content"].lower() for hit in alice_hits)
        assert all(hit["metadata"]["user_id"] == "alice" for hit in alice_hits)

        bob_hits = await store.search_semantic("bob", "music", n_results=5)
        assert all(hit["metadata"]["user_id"] == "bob" for hit in bob_hits)
        assert not any("jazz" in hit["content"].lower() for hit in bob_hits)

        # 3. Episodic memory with emotional tone metadata.
        episodic = _entry(
            "alice",
            MemoryLayer.EPISODIC,
            "We had a wonderful conversation about Coltrane",
            emotion_weight=0.6,
        )
        episodic.metadata["emotional_tone"] = "joy"
        await store.add_episodic_memory(episodic)

        joy_hits = await store.search_by_emotion(
            "alice",
            "joy",
            "Coltrane",
            n_results=3,
        )
        assert len(joy_hits) >= 1
        assert "coltrane" in joy_hits[0]["content"].lower()

        # 4. Flashbulb memory persists and is retrievable via episodic search.
        await store.add_flashbulb_memory(
            "alice",
            "The first time Alice opened up about her dreams",
            metadata={"emotional_tone": "intimate"},
        )
        flash_hits = await store.search_episodic("alice", "dreams", n_results=5)
        assert any("dreams" in hit["content"].lower() for hit in flash_hits)

    async def test_decay_weakens_old_unimportant_memory(
        self,
        store: MemoryStore,
    ) -> None:
        await store.get_or_create_user("carol", "Carol")

        # Old, low-strength, no emotion → should be weak on read.
        old = _entry(
            "carol",
            MemoryLayer.SEMANTIC,
            "Carol mentioned a forgettable trivia about pebbles",
            strength=0.3,
            emotion_weight=0.0,
            created_at=datetime.now(UTC) - timedelta(days=400),
        )
        await store.add_semantic_memory(old)

        # Recent, full-strength fact → should rank high.
        recent = _entry(
            "carol",
            MemoryLayer.SEMANTIC,
            "Carol just adopted a kitten named Pebbles",
            strength=1.0,
        )
        await store.add_semantic_memory(recent)

        hits = await store.search_semantic("carol", "Pebbles", n_results=5)
        assert len(hits) >= 1
        # Recent entry should out-rank the decayed one.
        top = hits[0]
        assert "kitten" in top["content"].lower()
