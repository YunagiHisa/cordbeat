"""Tests for sleep-phase memory consolidation."""

from unittest.mock import AsyncMock, MagicMock

from cordbeat.agent.sleep import SleepPhase
from cordbeat.config import MemoryConfig


async def test_promote_episodic_memories_accepts_fenced_json() -> None:
    memory = MagicMock()
    memory.search_episodic = AsyncMock(
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
    assert ai_call["system"].startswith("/no_think\n")
    entry = memory.add_semantic_memory.await_args.args[0]
    assert entry.content == "Enjoys hiking"


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

    assert ai.generate.await_args.kwargs["system"].startswith("/no_think\n")


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
