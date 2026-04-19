"""Tests for memory store."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from cordbeat.config import MemoryConfig
from cordbeat.memory import MemoryStore
from cordbeat.models import MemoryEntry, MemoryLayer


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


class TestFlashbulbMemory:
    async def test_add_flashbulb_memory(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        entry_id = await memory.add_flashbulb_memory("u1", "The day we first talked")
        assert entry_id is not None
        # Should be searchable in episodic memory
        results = await memory.search_episodic("u1", "first talked")
        assert len(results) >= 1
        assert results[0]["metadata"]["flashbulb"] == "True"
        assert float(results[0]["metadata"]["emotion_weight"]) == 1.0

    async def test_flashbulb_resists_decay(self, memory: MemoryStore) -> None:
        """Flashbulb memories should not be archived during decay."""
        await memory.get_or_create_user("u1", "Test")
        # Add a normal episodic memory with very low strength
        normal = MemoryEntry(
            id="normal-1",
            user_id="u1",
            layer=MemoryLayer.EPISODIC,
            content="Ordinary conversation",
            strength=0.001,
            emotion_weight=0.0,
        )
        await memory.add_episodic_memory(normal)

        # Add a flashbulb memory
        await memory.add_flashbulb_memory("u1", "Emotionally significant moment")

        # Run decay — normal should be archived, flashbulb should survive
        stats = await memory.decay_and_archive_memories()
        assert stats["archived"] >= 1

        # Flashbulb should still be searchable
        results = await memory.search_episodic("u1", "significant moment")
        assert len(results) >= 1
        assert results[0]["metadata"]["flashbulb"] == "True"

    async def test_flashbulb_with_custom_metadata(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        await memory.add_flashbulb_memory(
            "u1",
            "A joyful moment",
            metadata={"emotion": "joy", "intensity": 0.9},
        )
        results = await memory.search_episodic("u1", "joyful moment")
        assert len(results) >= 1
        assert results[0]["metadata"]["emotion"] == "joy"


class TestResolvePlatformUser:
    async def test_resolve_linked_user(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
        result = await memory.resolve_platform_user("u1", "discord")
        assert result == "discord_123"

    async def test_resolve_unknown_user(self, memory: MemoryStore) -> None:
        result = await memory.resolve_platform_user("no_user", "discord")
        assert result is None

    async def test_resolve_wrong_adapter(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
        result = await memory.resolve_platform_user("u1", "telegram")
        assert result is None


class TestProposalOperations:
    async def test_get_proposal_by_id(self, memory: MemoryStore) -> None:
        pid = await memory.add_certain_record(
            "u1",
            "Test proposal",
            record_type="proposal",
            metadata={"status": "pending"},
        )
        result = await memory.get_proposal(pid)
        assert result is not None
        assert result["id"] == pid
        assert result["content"] == "Test proposal"

    async def test_get_proposal_not_found(self, memory: MemoryStore) -> None:
        result = await memory.get_proposal("nonexistent-id")
        assert result is None

    async def test_get_proposal_ignores_non_proposals(
        self,
        memory: MemoryStore,
    ) -> None:
        """get_proposal only returns records with record_type=proposal."""
        rid = await memory.add_certain_record(
            "u1",
            "A log entry",
            record_type="log",
        )
        result = await memory.get_proposal(rid)
        assert result is None

    async def test_update_proposal_status(self, memory: MemoryStore) -> None:
        pid = await memory.add_certain_record(
            "u1",
            "Pending proposal",
            record_type="proposal",
            metadata={"status": "pending"},
        )
        updated = await memory.update_proposal_status(pid, "approved")
        assert updated is True

        result = await memory.get_proposal(pid)
        assert result is not None
        import json

        meta = json.loads(result["metadata"])
        assert meta["status"] == "approved"

    async def test_update_proposal_status_not_found(
        self,
        memory: MemoryStore,
    ) -> None:
        updated = await memory.update_proposal_status("bad-id", "approved")
        assert updated is False

    async def test_get_pending_proposals(self, memory: MemoryStore) -> None:
        await memory.add_certain_record(
            "u1",
            "Pending 1",
            record_type="proposal",
            metadata={"status": "pending"},
        )
        await memory.add_certain_record(
            "u1",
            "Approved",
            record_type="proposal",
            metadata={"status": "approved"},
        )
        await memory.add_certain_record(
            "u2",
            "Pending 2",
            record_type="proposal",
            metadata={"status": "pending"},
        )

        all_pending = await memory.get_pending_proposals()
        assert len(all_pending) == 2

        u1_pending = await memory.get_pending_proposals(user_id="u1")
        assert len(u1_pending) == 1
        assert u1_pending[0]["content"] == "Pending 1"


class TestLinkTokenOperations:
    """Tests for link token storage and verification."""

    async def test_store_and_verify_token(self, memory: MemoryStore) -> None:
        token = await memory.store_link_token("telegram", "tg_user_1")
        assert isinstance(token, str)
        assert len(token) > 0

        result = await memory.verify_link_token(token)
        assert result is not None
        assert result["requester_adapter_id"] == "telegram"
        assert result["requester_platform_user_id"] == "tg_user_1"

    async def test_verify_invalid_token(self, memory: MemoryStore) -> None:
        result = await memory.verify_link_token("nonexistent_token")
        assert result is None

    async def test_token_single_use(self, memory: MemoryStore) -> None:
        """Tokens can only be verified once."""
        token = await memory.store_link_token("telegram", "tg_user_1")

        first = await memory.verify_link_token(token)
        assert first is not None

        second = await memory.verify_link_token(token)
        assert second is None

    async def test_expired_token(self, memory: MemoryStore) -> None:
        """Expired tokens cannot be verified."""
        # Create token with 0-minute expiry (immediately expired)
        token = await memory.store_link_token(
            "telegram",
            "tg_user_1",
            token_expiry_minutes=0,
        )
        result = await memory.verify_link_token(token)
        assert result is None


class TestSearchByEmotion:
    """Tests for emotion-based memory search (Phase3 recall)."""

    async def test_finds_matching_emotion(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        entry = MemoryEntry(
            id="ep_joy",
            user_id="u1",
            layer=MemoryLayer.EPISODIC,
            content="Celebrated a milestone together",
            metadata={"emotional_tone": "joy"},
        )
        await memory.add_episodic_memory(entry)
        results = await memory.search_by_emotion("u1", "joy", "celebration")
        assert len(results) >= 1
        assert "milestone" in results[0]["content"]

    async def test_filters_non_matching_emotion(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        entry = MemoryEntry(
            id="ep_sad",
            user_id="u1",
            layer=MemoryLayer.EPISODIC,
            content="Had a tough day at work",
            metadata={"emotional_tone": "sadness"},
        )
        await memory.add_episodic_memory(entry)
        results = await memory.search_by_emotion("u1", "joy", "tough day")
        assert len(results) == 0

    async def test_isolates_by_user(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.get_or_create_user("u2", "Bob")
        entry = MemoryEntry(
            id="ep_u2",
            user_id="u2",
            layer=MemoryLayer.EPISODIC,
            content="Bob's happy memory",
            metadata={"emotional_tone": "joy"},
        )
        await memory.add_episodic_memory(entry)
        results = await memory.search_by_emotion("u1", "joy", "happy")
        assert len(results) == 0

    async def test_no_emotion_tag_not_returned(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Test")
        entry = MemoryEntry(
            id="ep_notag",
            user_id="u1",
            layer=MemoryLayer.EPISODIC,
            content="Memory without emotion tag",
        )
        await memory.add_episodic_memory(entry)
        results = await memory.search_by_emotion("u1", "joy", "memory")
        assert len(results) == 0


class TestGetMessagesOnDate:
    """Tests for date-based message retrieval."""

    async def test_returns_messages_on_date(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_message("u1", "user", "Hello", "discord")
        await memory.add_message("u1", "assistant", "Hi!", "discord")

        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        messages = await memory.get_messages_on_date("u1", today)
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi!"

    async def test_empty_for_other_date(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_message("u1", "user", "Hello", "discord")

        messages = await memory.get_messages_on_date("u1", "2020-01-01")
        assert messages == []

    async def test_isolates_by_user(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.get_or_create_user("u2", "Bob")
        await memory.add_message("u2", "user", "Bob's message", "discord")

        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        messages = await memory.get_messages_on_date("u1", today)
        assert messages == []


class TestRecallHints:
    """Tests for Phase4 recall hint operations."""

    async def test_store_and_retrieve_hint(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_recall_hint(
            user_id="u1",
            hint_type="temporal",
            content="7 days ago talked about OSS",
            metadata={"original_date": "2025-01-01"},
        )
        hints = await memory.get_recall_hints("u1")
        assert len(hints) == 1
        assert "OSS" in hints[0]["content"]

    async def test_filter_by_date(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_recall_hint(
            user_id="u1",
            hint_type="temporal",
            content="Hint for today",
        )
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        hints = await memory.get_recall_hints("u1", date_str=today)
        assert len(hints) == 1

        hints_other = await memory.get_recall_hints("u1", date_str="2020-01-01")
        assert len(hints_other) == 0

    async def test_clear_old_hints(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_recall_hint(
            user_id="u1",
            hint_type="temporal",
            content="Old hint",
        )
        # Insert with artificially old created_at
        await memory._conn.execute(
            "UPDATE certain_records SET created_at = '2020-01-01T00:00:00' "
            "WHERE record_type = 'recall_hint'"
        )
        await memory._conn.commit()

        deleted = await memory.clear_old_recall_hints(keep_days=2)
        assert deleted == 1

        remaining = await memory.get_recall_hints("u1")
        assert len(remaining) == 0

    async def test_isolates_by_user(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.get_or_create_user("u2", "Bob")
        await memory.store_recall_hint(
            user_id="u2",
            hint_type="temporal",
            content="Bob's hint",
        )
        hints = await memory.get_recall_hints("u1")
        assert len(hints) == 0


class TestChainLinks:
    """Tests for Phase4 chain-recall link operations."""

    async def test_store_and_retrieve_chain_link(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_A",
            linked_content="Related fact about Python",
            linked_memory_id="mem_B",
        )
        results = await memory.get_chain_links("u1", ["mem_A"])
        assert len(results) == 1
        assert "Python" in results[0]

    async def test_no_match_returns_empty(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_A",
            linked_content="Some content",
        )
        results = await memory.get_chain_links("u1", ["mem_X"])
        assert results == []

    async def test_empty_source_ids_returns_empty(self, memory: MemoryStore) -> None:
        results = await memory.get_chain_links("u1", [])
        assert results == []

    async def test_deduplicates_linked_content(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_A",
            linked_content="Same content",
        )
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_A",
            linked_content="Same content",
        )
        results = await memory.get_chain_links("u1", ["mem_A"])
        assert len(results) == 1

    async def test_clear_old_chain_links(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_A",
            linked_content="Old link",
        )
        await memory._conn.execute(
            "UPDATE certain_records SET created_at = '2020-01-01T00:00:00' "
            "WHERE record_type = 'chain_link'"
        )
        await memory._conn.commit()

        deleted = await memory.clear_old_chain_links(keep_days=2)
        assert deleted == 1
        results = await memory.get_chain_links("u1", ["mem_A"])
        assert results == []

    async def test_isolates_by_user(self, memory: MemoryStore) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.get_or_create_user("u2", "Bob")
        await memory.store_chain_link(
            user_id="u2",
            source_memory_id="mem_A",
            linked_content="Bob's linked memory",
        )
        results = await memory.get_chain_links("u1", ["mem_A"])
        assert results == []

    async def test_multi_hop_traversal(self, memory: MemoryStore) -> None:
        """Chain links follow multi-hop: A → B → C."""
        await memory.get_or_create_user("u1", "Alice")
        # A links to B
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_A",
            linked_content="Memory B content",
            linked_memory_id="mem_B",
        )
        # B links to C
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_B",
            linked_content="Memory C content",
            linked_memory_id="mem_C",
        )
        # Single hop: only B
        results_1 = await memory.get_chain_links("u1", ["mem_A"], max_depth=1)
        assert len(results_1) == 1
        assert "Memory B" in results_1[0]

        # Multi-hop: B and C
        results_2 = await memory.get_chain_links("u1", ["mem_A"], max_depth=2)
        assert len(results_2) == 2
        contents = set(results_2)
        assert "Memory B content" in contents
        assert "Memory C content" in contents

    async def test_multi_hop_cycle_prevention(self, memory: MemoryStore) -> None:
        """Circular chain links don't cause infinite loops."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_A",
            linked_content="Memory B",
            linked_memory_id="mem_B",
        )
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_B",
            linked_content="Memory A again",
            linked_memory_id="mem_A",
        )
        results = await memory.get_chain_links("u1", ["mem_A"], max_depth=3)
        # Should not loop — mem_A already visited
        assert len(results) <= 2

    async def test_distance_sorting(self, memory: MemoryStore) -> None:
        """Chain links are sorted by distance (closer first)."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_A",
            linked_content="Far memory",
            linked_memory_id="mem_far",
            distance=0.9,
        )
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_A",
            linked_content="Close memory",
            linked_memory_id="mem_close",
            distance=0.1,
        )
        results = await memory.get_chain_links("u1", ["mem_A"])
        assert results[0] == "Close memory"
        assert results[1] == "Far memory"

    async def test_max_results_limit(self, memory: MemoryStore) -> None:
        """Results are capped at max_results."""
        await memory.get_or_create_user("u1", "Alice")
        for i in range(5):
            await memory.store_chain_link(
                user_id="u1",
                source_memory_id="mem_A",
                linked_content=f"Link {i}",
                linked_memory_id=f"mem_{i}",
                distance=float(i) * 0.1,
            )
        results = await memory.get_chain_links("u1", ["mem_A"], max_results=3)
        assert len(results) == 3

    async def test_hop_distance_penalty(self, memory: MemoryStore) -> None:
        """Second-hop links are ranked lower than first-hop links."""
        await memory.get_or_create_user("u1", "Alice")
        # Hop 1: A → B (close)
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_A",
            linked_content="Direct link",
            linked_memory_id="mem_B",
            distance=0.5,
        )
        # Hop 2: B → C (close to B but further by depth penalty)
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_B",
            linked_content="Indirect link",
            linked_memory_id="mem_C",
            distance=0.1,
        )
        results = await memory.get_chain_links("u1", ["mem_A"], max_depth=2)
        # Direct link (0.5) should come before indirect (0.1 + 0.5 penalty = 0.6)
        assert results[0] == "Direct link"
        assert results[1] == "Indirect link"


class TestProposalStatusTransitions:
    """Tests for proposal status validation and expiry."""

    async def test_valid_transition_pending_to_approved(
        self, memory: MemoryStore
    ) -> None:
        pid = await memory.add_certain_record(
            "u1", "Test", record_type="proposal", metadata={"status": "pending"}
        )
        result = await memory.update_proposal_status(pid, "approved")
        assert result is True

    async def test_valid_transition_pending_to_rejected(
        self, memory: MemoryStore
    ) -> None:
        pid = await memory.add_certain_record(
            "u1", "Test", record_type="proposal", metadata={"status": "pending"}
        )
        result = await memory.update_proposal_status(pid, "rejected")
        assert result is True

    async def test_valid_transition_approved_to_executed(
        self, memory: MemoryStore
    ) -> None:
        pid = await memory.add_certain_record(
            "u1", "Test", record_type="proposal", metadata={"status": "pending"}
        )
        await memory.update_proposal_status(pid, "approved")
        result = await memory.update_proposal_status(pid, "executed")
        assert result is True

    async def test_invalid_transition_rejected_to_approved(
        self, memory: MemoryStore
    ) -> None:
        pid = await memory.add_certain_record(
            "u1", "Test", record_type="proposal", metadata={"status": "pending"}
        )
        await memory.update_proposal_status(pid, "rejected")
        with pytest.raises(ValueError, match="Invalid proposal transition"):
            await memory.update_proposal_status(pid, "approved")

    async def test_invalid_transition_executed_to_pending(
        self, memory: MemoryStore
    ) -> None:
        pid = await memory.add_certain_record(
            "u1", "Test", record_type="proposal", metadata={"status": "pending"}
        )
        await memory.update_proposal_status(pid, "approved")
        await memory.update_proposal_status(pid, "executed")
        with pytest.raises(ValueError, match="Invalid proposal transition"):
            await memory.update_proposal_status(pid, "pending")

    async def test_concurrent_approve_only_one_wins(
        self, memory: MemoryStore
    ) -> None:
        """Two concurrent approvals on the same pending proposal must race
        atomically: exactly one succeeds, the other raises
        ``Invalid proposal transition`` (because the first one already
        moved the status to 'approved', which has no 'approved' target).
        """
        import asyncio

        pid = await memory.add_certain_record(
            "u1", "Test", record_type="proposal", metadata={"status": "pending"}
        )

        results = await asyncio.gather(
            memory.update_proposal_status(pid, "approved"),
            memory.update_proposal_status(pid, "approved"),
            return_exceptions=True,
        )

        successes = [r for r in results if r is True]
        errors = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 1, f"Exactly one must win, got {results!r}"
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)

    async def test_update_nonexistent_proposal_returns_false(
        self, memory: MemoryStore
    ) -> None:
        result = await memory.update_proposal_status("no-such-id", "approved")
        assert result is False

    async def test_expire_old_proposals(self, memory: MemoryStore) -> None:
        """Pending proposals older than max_age_days are expired."""
        pid = await memory.add_certain_record(
            "u1",
            "Old proposal",
            record_type="proposal",
            metadata={"status": "pending"},
        )
        # Backdate the record
        await memory._conn.execute(
            "UPDATE certain_records SET created_at = '2020-01-01T00:00:00' "
            "WHERE id = ?",
            (pid,),
        )
        await memory._conn.commit()

        expired = await memory.expire_old_proposals(max_age_days=7)
        assert expired == 1

        proposal = await memory.get_proposal(pid)
        import json

        meta = json.loads(proposal["metadata"])
        assert meta["status"] == "expired"

    async def test_expire_does_not_touch_approved(self, memory: MemoryStore) -> None:
        """Only pending proposals are expired, not approved ones."""
        pid = await memory.add_certain_record(
            "u1",
            "Approved",
            record_type="proposal",
            metadata={"status": "pending"},
        )
        await memory.update_proposal_status(pid, "approved")
        await memory._conn.execute(
            "UPDATE certain_records SET created_at = '2020-01-01T00:00:00' "
            "WHERE id = ?",
            (pid,),
        )
        await memory._conn.commit()

        expired = await memory.expire_old_proposals(max_age_days=7)
        assert expired == 0

    async def test_expire_recent_proposals_untouched(self, memory: MemoryStore) -> None:
        """Recent pending proposals are not expired."""
        await memory.add_certain_record(
            "u1",
            "Fresh",
            record_type="proposal",
            metadata={"status": "pending"},
        )
        expired = await memory.expire_old_proposals(max_age_days=7)
        assert expired == 0
