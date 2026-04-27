"""Tests for D-3 memory quality features (embedding config + dedup)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cordbeat.config import MemoryConfig
from cordbeat.memory import MemoryStore
from cordbeat.models import MemoryEntry, MemoryLayer, TrustLevel

pytestmark = pytest.mark.integration


def _entry(user_id: str, content: str) -> MemoryEntry:
    return MemoryEntry(
        id="",
        user_id=user_id,
        layer=MemoryLayer.SEMANTIC,
        content=content,
        trust_level=TrustLevel.NORMAL,
        strength=0.5,
        emotion_weight=0.0,
        created_at=datetime.now(tz=UTC),
        last_accessed_at=datetime.now(tz=UTC),
        metadata={},
    )


@pytest.fixture
async def store(tmp_path: Path) -> MemoryStore:
    config = MemoryConfig(
        sqlite_path=str(tmp_path / "d3.db"),
        dedup_distance_threshold=0.01,  # only true near-duplicates
    )
    s = MemoryStore(config)
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


async def test_dedup_merges_identical_content(store: MemoryStore) -> None:
    """Identical content should reinforce the existing memory, not duplicate."""
    first_id = await store.add_semantic_memory(_entry("u1", "I love sushi"))
    second_id = await store.add_semantic_memory(_entry("u1", "I love sushi"))
    assert first_id == second_id

    results = await store.search_semantic("u1", "sushi", n_results=5)
    contents = [r["content"] for r in results]
    assert contents.count("I love sushi") == 1
    # strength got bumped from 0.5 -> 0.6
    assert results[0]["metadata"]["strength"] >= 0.55


async def test_dedup_keeps_distinct_content(store: MemoryStore) -> None:
    await store.add_semantic_memory(_entry("u1", "I love sushi"))
    await store.add_semantic_memory(_entry("u1", "I hate broccoli"))
    results = await store.search_semantic("u1", "food", n_results=5)
    assert len(results) == 2


async def test_dedup_disabled_by_default(tmp_path: Path) -> None:
    """With threshold=0 (default), identical inserts produce two rows."""
    config = MemoryConfig(sqlite_path=str(tmp_path / "nodedup.db"))
    s = MemoryStore(config)
    await s.initialize()
    try:
        a = await s.add_semantic_memory(_entry("u1", "duplicate text"))
        b = await s.add_semantic_memory(_entry("u1", "duplicate text"))
        assert a != b
    finally:
        await s.close()


async def test_dim_mismatch_emits_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = MemoryConfig(
        sqlite_path=str(tmp_path / "warn.db"),
        embedding_dim=512,
    )
    s = MemoryStore(config)
    with caplog.at_level(logging.WARNING, logger="cordbeat._memory_vector"):
        await s.initialize()
    try:
        assert any("embedding_dim=512" in rec.message for rec in caplog.records)
    finally:
        await s.close()
