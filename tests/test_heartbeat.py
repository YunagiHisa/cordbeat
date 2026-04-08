"""Tests for HEARTBEAT loop."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cordbeat.config import HeartbeatConfig, MemoryConfig
from cordbeat.gateway import MessageQueue
from cordbeat.heartbeat import HeartbeatLoop, _in_quiet_hours, _parse_time
from cordbeat.memory import MemoryStore
from cordbeat.models import (
    HeartbeatAction,
    HeartbeatDecision,
    MessageType,
    UserSummary,
)
from cordbeat.skills import SkillRegistry
from cordbeat.soul import Soul

# ── Helper time parsing ───────────────────────────────────────────────


class TestParseTime:
    def test_normal(self) -> None:
        assert _parse_time("01:30") == time(1, 30)

    def test_midnight(self) -> None:
        assert _parse_time("00:00") == time(0, 0)

    def test_end_of_day(self) -> None:
        assert _parse_time("23:59") == time(23, 59)


# ── Quiet hours ───────────────────────────────────────────────────────


class TestInQuietHours:
    def test_inside_range(self) -> None:
        """03:00 is between 01:00-07:00."""
        with patch("cordbeat.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(3, 0))
            assert _in_quiet_hours("01:00", "07:00") is True

    def test_outside_range(self) -> None:
        """12:00 is outside 01:00-07:00."""
        with patch("cordbeat.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(12, 0))
            assert _in_quiet_hours("01:00", "07:00") is False

    def test_midnight_wrap_inside(self) -> None:
        """23:30 is inside 22:00-06:00 (wraps midnight)."""
        with patch("cordbeat.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(23, 30))
            assert _in_quiet_hours("22:00", "06:00") is True

    def test_midnight_wrap_inside_early(self) -> None:
        """02:00 is inside 22:00-06:00 (wraps midnight)."""
        with patch("cordbeat.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(2, 0))
            assert _in_quiet_hours("22:00", "06:00") is True

    def test_midnight_wrap_outside(self) -> None:
        """12:00 is outside 22:00-06:00 (wraps midnight)."""
        with patch("cordbeat.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(12, 0))
            assert _in_quiet_hours("22:00", "06:00") is False

    def test_at_boundary_start(self) -> None:
        """01:00 (boundary) is inside 01:00-07:00."""
        with patch("cordbeat.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(1, 0))
            assert _in_quiet_hours("01:00", "07:00") is True

    def test_at_boundary_end(self) -> None:
        """07:00 (boundary) is inside 01:00-07:00."""
        with patch("cordbeat.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(7, 0))
            assert _in_quiet_hours("01:00", "07:00") is True


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def heartbeat_config() -> HeartbeatConfig:
    return HeartbeatConfig(
        default_interval_minutes=60,
        min_interval_minutes=5,
        max_interval_minutes=1440,
    )


@pytest.fixture
def soul(tmp_path: Path) -> Soul:
    return Soul(tmp_path / "soul")


@pytest.fixture
async def memory(tmp_path: Path) -> MemoryStore:
    config = MemoryConfig(
        sqlite_path=str(tmp_path / "test.db"),
        chroma_path=str(tmp_path / "chroma"),
    )
    store = MemoryStore(config)
    await store.initialize()
    yield store  # type: ignore[misc]
    await store.close()


@pytest.fixture
def mock_ai() -> AsyncMock:
    ai = AsyncMock()
    ai.generate = AsyncMock(return_value="test")
    ai.generate_json = AsyncMock(
        return_value={
            "action": "none",
            "content": "",
            "next_heartbeat_minutes": 60,
        }
    )
    return ai


@pytest.fixture
def mock_gateway() -> AsyncMock:
    gw = AsyncMock()
    gw.send_to_adapter = AsyncMock()
    return gw


@pytest.fixture
def queue() -> MessageQueue:
    return MessageQueue()


@pytest.fixture
def skills(tmp_path: Path) -> SkillRegistry:
    return SkillRegistry(tmp_path / "skills")


@pytest.fixture
def heartbeat(
    heartbeat_config: HeartbeatConfig,
    mock_ai: AsyncMock,
    soul: Soul,
    memory: MemoryStore,
    skills: SkillRegistry,
    mock_gateway: AsyncMock,
    queue: MessageQueue,
) -> HeartbeatLoop:
    return HeartbeatLoop(
        config=heartbeat_config,
        ai=mock_ai,
        soul=soul,
        memory=memory,
        skills=skills,
        gateway=mock_gateway,
        queue=queue,
    )


# ── Build global context ─────────────────────────────────────────────


class TestBuildGlobalContext:
    def test_includes_user_summaries(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        users = [
            UserSummary(
                user_id="u1",
                display_name="Alice",
                last_topic="AI research",
                emotional_tone="neutral",
                attention_score=0.5,
            ),
        ]
        ctx = heartbeat._build_global_context(users)
        assert "Alice" in ctx
        assert "AI research" in ctx
        assert "Total users: 1" in ctx

    def test_sanitizes_user_data(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Prompt injection characters are stripped."""
        users = [
            UserSummary(
                user_id="u1",
                display_name="Bad\nUser#Name",
                last_topic="inject\r\nme",
                emotional_tone="evil\x00",
            ),
        ]
        ctx = heartbeat._build_global_context(users)
        assert "\n" not in ctx.split("- Bad")[1].split(",")[0]
        assert "#" not in ctx.split("- Bad")[1].split(",")[0]
        assert "BadUserName" in ctx


# ── Execute decision ──────────────────────────────────────────────────


class TestExecuteDecision:
    async def test_action_none_does_nothing(
        self,
        heartbeat: HeartbeatLoop,
        mock_gateway: AsyncMock,
    ) -> None:
        decision = HeartbeatDecision(action=HeartbeatAction.NONE)
        await heartbeat._execute_decision(decision)
        mock_gateway.send_to_adapter.assert_not_called()

    async def test_action_message_sends(
        self,
        heartbeat: HeartbeatLoop,
        mock_gateway: AsyncMock,
    ) -> None:
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="Hello!",
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)
        mock_gateway.send_to_adapter.assert_called_once()
        call_args = mock_gateway.send_to_adapter.call_args
        assert call_args[0][0] == "discord"
        assert call_args[0][1].type == MessageType.HEARTBEAT_MESSAGE

    async def test_action_message_missing_target(
        self,
        heartbeat: HeartbeatLoop,
        mock_gateway: AsyncMock,
    ) -> None:
        """Missing target_user_id or target_adapter_id → warning, no send."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="Hello!",
        )
        await heartbeat._execute_decision(decision)
        mock_gateway.send_to_adapter.assert_not_called()

    async def test_action_skill_unknown(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Unknown skill name → warning, no crash."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="nonexistent",
        )
        await heartbeat._execute_decision(decision)

    async def test_action_skill_no_name(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Missing skill_name → warning, no crash."""
        decision = HeartbeatDecision(action=HeartbeatAction.SKILL)
        await heartbeat._execute_decision(decision)

    async def test_action_propose_improvement(
        self,
        heartbeat: HeartbeatLoop,
        mock_gateway: AsyncMock,
    ) -> None:
        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_IMPROVEMENT,
            content="Improve memory decay",
        )
        await heartbeat._execute_decision(decision)
        mock_gateway.send_to_adapter.assert_not_called()


# ── Tick ──────────────────────────────────────────────────────────────


class TestTick:
    async def test_tick_no_users_returns_default(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """No users → return default interval."""
        result = await heartbeat._tick()
        assert result == 60

    async def test_tick_quiet_hours_skips(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """During quiet hours → runs sleep once, then skips."""
        with patch("cordbeat.heartbeat._in_quiet_hours", return_value=True):
            result = await heartbeat._tick()
        assert result == 60
        assert heartbeat._sleep_done_today is True

    async def test_tick_calls_ai(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """With users, AI should be called for decision."""
        await memory.get_or_create_user("u1", "user1")
        with patch("cordbeat.heartbeat._in_quiet_hours", return_value=False):
            await heartbeat._tick()
        mock_ai.generate_json.assert_called_once()


# ── Sleep Phase ───────────────────────────────────────────────────────


class TestSleepPhase:
    async def test_sleep_phase_generates_diary(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Sleep phase generates a diary entry from today's messages."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_message("u1", "user", "Hello!", "discord")
        await memory.add_message("u1", "assistant", "Hi Alice!", "discord")

        mock_ai.generate = AsyncMock(return_value="Today I chatted with Alice.")

        await heartbeat._run_sleep_phase()

        records = await memory.get_certain_records("u1", record_type="diary")
        assert len(records) == 1
        assert "Alice" in records[0]["content"]

    async def test_sleep_phase_skips_users_without_messages(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Users with no messages today don't get a diary entry."""
        await memory.get_or_create_user("u1", "Silent")

        await heartbeat._run_sleep_phase()

        records = await memory.get_certain_records("u1", record_type="diary")
        assert len(records) == 0
        mock_ai.generate.assert_not_called()

    async def test_sleep_phase_runs_decay(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Sleep phase calls decay_and_archive_memories."""
        memory.decay_and_archive_memories = AsyncMock(
            return_value={"decayed": 5, "archived": 2}
        )
        await heartbeat._run_sleep_phase()
        memory.decay_and_archive_memories.assert_called_once()

    async def test_sleep_phase_trims_messages(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Sleep phase trims old messages for each user."""
        await memory.get_or_create_user("u1", "Alice")
        memory.trim_old_messages = AsyncMock(return_value=10)

        await heartbeat._run_sleep_phase()
        memory.trim_old_messages.assert_called_once_with("u1")

    async def test_sleep_phase_only_runs_once_per_quiet_period(
        self,
        heartbeat: HeartbeatLoop,
        mock_ai: AsyncMock,
    ) -> None:
        """Sleep phase runs only once during quiet hours."""
        with patch("cordbeat.heartbeat._in_quiet_hours", return_value=True):
            await heartbeat._tick()
            await heartbeat._tick()

        # _run_sleep_phase should only be invoked once
        assert heartbeat._sleep_done_today is True

    async def test_sleep_flag_resets_outside_quiet_hours(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """_sleep_done_today resets when leaving quiet hours."""
        heartbeat._sleep_done_today = True
        await memory.get_or_create_user("u1", "Alice")

        with patch("cordbeat.heartbeat._in_quiet_hours", return_value=False):
            await heartbeat._tick()

        assert heartbeat._sleep_done_today is False


# ── Memory consolidation ─────────────────────────────────────────────


class TestMemoryConsolidation:
    async def test_get_todays_messages(
        self,
        memory: MemoryStore,
    ) -> None:
        """get_todays_messages returns only today's messages."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_message("u1", "user", "Hello", "discord")
        await memory.add_message("u1", "assistant", "Hi!", "discord")

        messages = await memory.get_todays_messages("u1")
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi!"

    async def test_get_todays_messages_empty(
        self,
        memory: MemoryStore,
    ) -> None:
        """No messages → empty list."""
        await memory.get_or_create_user("u1", "Alice")
        messages = await memory.get_todays_messages("u1")
        assert messages == []

    async def test_decay_and_archive_empty(
        self,
        memory: MemoryStore,
    ) -> None:
        """No memories → zero counts."""
        stats = await memory.decay_and_archive_memories()
        assert stats["decayed"] == 0
        assert stats["archived"] == 0
