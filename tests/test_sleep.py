"""Tests for sleep-phase memory consolidation."""

import json
from unittest.mock import AsyncMock, MagicMock

from cordbeat.agent.sleep import SleepPhase
from cordbeat.config import MemoryConfig


async def test_promote_episodic_memories_accepts_fenced_json() -> None:
    memory = MagicMock()
    memory.get_episodic_since = AsyncMock(
        return_value=[{"content": "The user said they enjoy hiking."}]
    )
    memory.add_semantic_memory = AsyncMock()
    ai = MagicMock()
    ai.generate = AsyncMock(return_value='```json\n{"facts": ["Enjoys hiking"]}\n```')
    sleep = SleepPhase(
        memory=memory,
        ai=ai,
        soul=MagicMock(),
        memory_config=MemoryConfig(),
    )

    await sleep._promote_episodic_memories("user-1")

    ai_call = ai.generate.await_args.kwargs
    assert "/no_think" not in ai_call["system"]
    assert "The user said they enjoy hiking." not in ai_call["system"]
    assert "The user said they enjoy hiking." in ai_call["prompt"]
    entry = memory.add_semantic_memory.await_args.args[0]
    assert entry.content == "Enjoys hiking"


async def test_promote_episodic_memories_uses_recent_day_scope() -> None:
    memory = MagicMock()
    memory.get_episodic_since = AsyncMock(return_value=[])
    memory.add_semantic_memory = AsyncMock()
    ai = MagicMock()
    ai.generate = AsyncMock(return_value='{"facts": ["old fact"]}')
    sleep = SleepPhase(
        memory=memory,
        ai=ai,
        soul=MagicMock(),
        memory_config=MemoryConfig(),
        timezone="Asia/Tokyo",
    )

    await sleep._promote_episodic_memories("user-1")

    memory.get_episodic_since.assert_awaited_once()
    assert memory.get_episodic_since.await_args.args[0] == "user-1"
    ai.generate.assert_not_awaited()
    memory.add_semantic_memory.assert_not_awaited()


async def test_promote_episodic_memories_caps_fact_length() -> None:
    memory = MagicMock()
    memory.get_episodic_since = AsyncMock(
        return_value=[{"content": "A very talkative day."}]
    )
    memory.add_semantic_memory = AsyncMock()
    ai = MagicMock()
    ai.generate = AsyncMock(return_value=json.dumps({"facts": ["f" * 2000]}))
    sleep = SleepPhase(
        memory=memory,
        ai=ai,
        soul=MagicMock(),
        memory_config=MemoryConfig(),
    )

    await sleep._promote_episodic_memories("user-1")

    entry = memory.add_semantic_memory.await_args.args[0]
    assert len(entry.content) == 500


async def test_write_diary_caps_input_and_strips_reasoning() -> None:
    memory = MagicMock()
    messages = [
        {"role": "user", "content": f"msg-{i} " + "x" * 480} for i in range(60)
    ]
    memory.get_todays_messages = AsyncMock(return_value=messages)
    memory.add_certain_record = AsyncMock()
    ai = MagicMock()
    ai.generate = AsyncMock(return_value="<think>secret</think>Dear diary entry.")
    sleep = SleepPhase(
        memory=memory,
        ai=ai,
        soul=MagicMock(),
        memory_config=MemoryConfig(),
    )
    user = MagicMock(user_id="user-1", display_name="User")

    await sleep._write_diary(user, {"name": "CordBeat"})

    prompt = ai.generate.await_args.kwargs["prompt"]
    assert "data, not instructions" in prompt
    assert "msg-59" in prompt  # newest messages are kept
    assert "msg-0 " not in prompt  # oldest dropped once over the budget
    assert len(prompt) < 25_000
    stored = memory.add_certain_record.await_args.kwargs["content"]
    assert "secret" not in stored
    assert stored.endswith("Dear diary entry.")


async def test_write_diary_disables_thinking() -> None:
    memory = MagicMock()
    memory.get_todays_messages = AsyncMock(
        return_value=[{"role": "user", "content": "Today was productive."}]
    )
    memory.add_certain_record = AsyncMock()
    ai = MagicMock()
    ai.generate = AsyncMock(return_value="A short diary entry.")
    sleep = SleepPhase(
        memory=memory,
        ai=ai,
        soul=MagicMock(),
        memory_config=MemoryConfig(),
    )
    user = MagicMock(user_id="user-1", display_name="User")

    await sleep._write_diary(user, {"name": "CordBeat"})

    assert "/no_think" not in ai.generate.await_args.kwargs["system"]


def _compress_sleep(memory: MagicMock, ai: MagicMock) -> SleepPhase:
    return SleepPhase(
        memory=memory,
        ai=ai,
        soul=MagicMock(),
        memory_config=MemoryConfig(),
    )


async def test_compress_old_messages_creates_episodic_and_deletes() -> None:
    memory = MagicMock()
    memory.count_messages = AsyncMock(return_value=100_000)  # over threshold
    memory.get_oldest_messages = AsyncMock(
        return_value=[{"id": "1", "role": "user", "content": "hello"}]
    )
    memory.add_episodic_memory = AsyncMock()
    memory.delete_messages_with_ids = AsyncMock(return_value=1)
    ai = MagicMock()
    ai.generate = AsyncMock(return_value="A concise summary of the chat.")
    sleep = _compress_sleep(memory, ai)
    user = MagicMock(user_id="u1", display_name="U")

    await sleep._compress_old_messages(user, {"name": "Aria"})

    memory.add_episodic_memory.assert_awaited_once()
    entry = memory.add_episodic_memory.await_args.args[0]
    assert entry.content == "A concise summary of the chat."
    memory.delete_messages_with_ids.assert_awaited_once_with(["1"])


async def test_compress_old_messages_scopes_chunk_to_one_conversation() -> None:
    """A summary spanning a DM and a public channel cannot be labelled with an
    origin, so it could later be recalled into the wrong room."""
    memory = MagicMock()
    memory.count_messages = AsyncMock(return_value=100_000)
    memory.get_oldest_messages = AsyncMock(
        return_value=[
            {
                "id": "1",
                "role": "user",
                "content": "hello",
                "channel_id": "dm-1",
                "is_dm": True,
            }
        ]
    )
    memory.add_episodic_memory = AsyncMock()
    memory.delete_messages_with_ids = AsyncMock(return_value=1)
    ai = MagicMock()
    ai.generate = AsyncMock(return_value="A concise summary of the chat.")
    sleep = _compress_sleep(memory, ai)

    await sleep._compress_old_messages(
        MagicMock(user_id="u1", display_name="U"), {"name": "Aria"}
    )

    chunk_call = memory.get_oldest_messages.await_args_list[1]
    assert chunk_call.kwargs["channel_id"] == "dm-1"
    assert chunk_call.kwargs["is_dm"] is True


async def test_compress_old_messages_records_origin_for_recall_labelling() -> None:
    memory = MagicMock()
    memory.count_messages = AsyncMock(return_value=100_000)
    memory.get_oldest_messages = AsyncMock(
        return_value=[
            {
                "id": "1",
                "role": "user",
                "content": "hello",
                "channel_id": "dm-1",
                "is_dm": True,
            }
        ]
    )
    memory.add_episodic_memory = AsyncMock()
    memory.delete_messages_with_ids = AsyncMock(return_value=1)
    ai = MagicMock()
    ai.generate = AsyncMock(return_value="A concise summary of the chat.")
    sleep = _compress_sleep(memory, ai)

    await sleep._compress_old_messages(
        MagicMock(user_id="u1", display_name="U"), {"name": "Aria"}
    )

    entry = memory.add_episodic_memory.await_args.args[0]
    assert entry.metadata["source_channel_id"] == "dm-1"
    assert entry.metadata["source_is_dm"] is True


async def test_temporal_recall_hints_are_split_per_conversation() -> None:
    """Hints quote the user verbatim and are injected into every channel, so
    a hint must never blend DM topics with public-channel ones."""
    memory = MagicMock()
    memory.get_messages_on_date = AsyncMock(
        return_value=[
            {
                "role": "user",
                "content": "the public boss fight was brutal",
                "channel_id": "chan-a",
                "is_dm": False,
            },
            {
                "role": "user",
                "content": "my private medical appointment is friday",
                "channel_id": "dm-1",
                "is_dm": True,
            },
        ]
    )
    memory.store_recall_hint = AsyncMock()
    sleep = SleepPhase(
        memory=memory,
        ai=MagicMock(),
        soul=MagicMock(),
        memory_config=MemoryConfig(),
    )

    await sleep._precompute_temporal_recall(MagicMock(user_id="u1", display_name="U"))

    stored = [call.kwargs for call in memory.store_recall_hint.await_args_list]
    assert stored, "expected at least one hint"
    for hint in stored:
        has_public = "boss fight" in hint["content"]
        has_private = "medical appointment" in hint["content"]
        assert not (has_public and has_private), "DM and channel topics merged"
        assert "source_channel_id" in hint["metadata"]
    dm_hints = [h for h in stored if h["metadata"]["source_is_dm"]]
    assert dm_hints and all(
        h["metadata"]["source_channel_id"] == "dm-1" for h in dm_hints
    )


async def test_compress_old_messages_skips_below_threshold() -> None:
    memory = MagicMock()
    memory.count_messages = AsyncMock(return_value=0)  # nothing to compress
    memory.get_oldest_messages = AsyncMock()
    memory.add_episodic_memory = AsyncMock()
    sleep = _compress_sleep(memory, MagicMock())

    await sleep._compress_old_messages(MagicMock(user_id="u1"), {"name": "Aria"})

    memory.get_oldest_messages.assert_not_awaited()
    memory.add_episodic_memory.assert_not_awaited()


async def test_compress_old_messages_handles_empty_chunk() -> None:
    memory = MagicMock()
    memory.count_messages = AsyncMock(return_value=100_000)
    memory.get_oldest_messages = AsyncMock(return_value=[])
    memory.add_episodic_memory = AsyncMock()
    sleep = _compress_sleep(memory, MagicMock())

    await sleep._compress_old_messages(MagicMock(user_id="u1"), {"name": "Aria"})

    memory.add_episodic_memory.assert_not_awaited()


async def test_compress_old_messages_skips_when_summary_empty() -> None:
    memory = MagicMock()
    memory.count_messages = AsyncMock(return_value=100_000)
    memory.get_oldest_messages = AsyncMock(
        return_value=[{"id": "1", "role": "user", "content": "hello"}]
    )
    memory.add_episodic_memory = AsyncMock()
    memory.delete_messages_with_ids = AsyncMock()
    ai = MagicMock()
    ai.generate = AsyncMock(return_value="")  # compressor yields nothing
    sleep = _compress_sleep(memory, ai)

    await sleep._compress_old_messages(MagicMock(user_id="u1"), {"name": "Aria"})

    memory.add_episodic_memory.assert_not_awaited()
    memory.delete_messages_with_ids.assert_not_awaited()


async def test_precompute_chain_links_uses_recent_day_scope() -> None:
    memory = MagicMock()
    memory.get_episodic_since = AsyncMock(
        return_value=[{"id": "today-ep", "content": "Today Alice discussed piano"}]
    )
    memory.search_semantic = AsyncMock(
        return_value=[
            {"id": "sem-1", "content": "Alice likes piano", "distance": 0.2}
        ]
    )
    memory.search_episodic = AsyncMock(return_value=[])
    memory.store_chain_link = AsyncMock()
    sleep = _compress_sleep(memory, MagicMock())

    await sleep._precompute_chain_links("user-1")

    memory.get_episodic_since.assert_awaited_once()
    memory.store_chain_link.assert_awaited_once()
    assert memory.store_chain_link.await_args.kwargs["source_memory_id"] == "today-ep"
