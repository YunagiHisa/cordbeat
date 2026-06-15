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
    ai.generate = AsyncMock(
        return_value='```json\n{"facts": ["Enjoys hiking"]}\n```'
    )
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
