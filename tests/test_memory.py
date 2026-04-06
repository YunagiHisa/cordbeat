"""Tests for memory store."""

from __future__ import annotations

from pathlib import Path

import pytest

from cordbeat.config import MemoryConfig
from cordbeat.memory import MemoryStore


@pytest.fixture
async def memory(tmp_path: Path) -> MemoryStore:
    config = MemoryConfig(
        sqlite_path=str(tmp_path / "test.db"),
        chroma_path=str(tmp_path / "chroma"),
    )
    store = MemoryStore(config)
    await store.initialize()
    yield store
    await store.close()


class TestUserManagement:
    async def test_create_user(self, memory: MemoryStore) -> None:
        user = await memory.get_or_create_user("u1", "Test User")
        assert user.user_id == "u1"
        assert user.display_name == "Test User"

    async def test_get_existing_user(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        user = await memory.get_or_create_user("u1", "Ignored")
        assert user.display_name == "Test"

    async def test_update_summary(self, memory: MemoryStore) -> None:
        user = await memory.get_or_create_user("u1", "Test")
        user.last_topic = "AI discussion"
        user.attention_score = 0.9
        await memory.update_user_summary(user)
        users = await memory.get_all_user_summaries()
        assert users[0].last_topic == "AI discussion"

    async def test_platform_link(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        await memory.link_platform("u1", "discord", "123456")
        resolved = await memory.resolve_user("discord", "123456")
        assert resolved == "u1"

    async def test_resolve_unknown_user(self, memory: MemoryStore) -> None:
        resolved = await memory.resolve_user("discord", "unknown")
        assert resolved is None


class TestCoreProfile:
    async def test_set_and_get(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        await memory.set_core_profile("u1", "nickname", "tatsuki")
        profile = await memory.get_core_profile("u1")
        assert profile["nickname"] == "tatsuki"


class TestCertainRecords:
    async def test_add_and_get(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        await memory.add_certain_record("u1", "Important event", "diary")
        records = await memory.get_certain_records("u1", "diary")
        assert len(records) == 1
        assert records[0]["content"] == "Important event"


class TestForgettingCurve:
    def test_fresh_memory(self) -> None:
        config = MemoryConfig()
        store = MemoryStore(config)
        strength = store.calculate_strength(1.0, elapsed_days=0)
        assert strength == 1.0

    def test_decay_over_time(self) -> None:
        config = MemoryConfig(decay_rate=0.1)
        store = MemoryStore(config)
        strength = store.calculate_strength(1.0, elapsed_days=10)
        assert strength < 1.0
        assert strength > 0.0

    def test_emotional_memories_decay_slower(self) -> None:
        config = MemoryConfig(decay_rate=0.1)
        store = MemoryStore(config)
        normal = store.calculate_strength(1.0, elapsed_days=10, emotion_weight=0.0)
        emotional = store.calculate_strength(1.0, elapsed_days=10, emotion_weight=0.8)
        assert emotional > normal


class TestConversationHistory:
    async def test_add_and_get_messages(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        await memory.add_message("u1", "user", "Hello!")
        await memory.add_message("u1", "assistant", "Hi there!")
        msgs = await memory.get_recent_messages("u1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello!"
        assert msgs[1]["role"] == "assistant"

    async def test_messages_chronological_order(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        for i in range(5):
            await memory.add_message("u1", "user", f"msg{i}")
        msgs = await memory.get_recent_messages("u1")
        assert [m["content"] for m in msgs] == [
            "msg0",
            "msg1",
            "msg2",
            "msg3",
            "msg4",
        ]

    async def test_limit_returns_most_recent(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        for i in range(10):
            await memory.add_message("u1", "user", f"msg{i}")
        msgs = await memory.get_recent_messages("u1", limit=3)
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg7"
        assert msgs[2]["content"] == "msg9"

    async def test_messages_isolated_per_user(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.get_or_create_user("u2", "Bob")
        await memory.add_message("u1", "user", "Alice says hi")
        await memory.add_message("u2", "user", "Bob says hi")
        alice_msgs = await memory.get_recent_messages("u1")
        bob_msgs = await memory.get_recent_messages("u2")
        assert len(alice_msgs) == 1
        assert len(bob_msgs) == 1
        assert alice_msgs[0]["content"] == "Alice says hi"

    async def test_trim_old_messages(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        for i in range(10):
            await memory.add_message("u1", "user", f"msg{i}")
        deleted = await memory.trim_old_messages("u1", keep=3)
        assert deleted == 7
        msgs = await memory.get_recent_messages("u1")
        assert len(msgs) == 3
