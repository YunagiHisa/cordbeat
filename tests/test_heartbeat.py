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
    SafetyLevel,
    SkillMeta,
    UserSummary,
)
from cordbeat.skills import Skill, SkillRegistry
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
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
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
        msg = call_args[0][1]
        assert msg.type == MessageType.HEARTBEAT_MESSAGE
        assert msg.platform_user_id == "discord_123"

    async def test_action_message_unresolvable_user(
        self,
        heartbeat: HeartbeatLoop,
        mock_gateway: AsyncMock,
    ) -> None:
        """No platform link → warning, no send."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="Hello!",
            target_user_id="u_unknown",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)
        mock_gateway.send_to_adapter.assert_not_called()

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

    async def test_dangerous_skill_blocked(
        self,
        heartbeat: HeartbeatLoop,
        skills: SkillRegistry,
    ) -> None:
        """Dangerous skills should be blocked from autonomous execution."""
        # Create a mock skill with DANGEROUS safety level
        from types import SimpleNamespace

        module = SimpleNamespace(execute=lambda **kw: {"result": "should not run"})
        meta = SkillMeta(
            name="danger_skill",
            description="Dangerous",
            usage="",
            safety_level=SafetyLevel.DANGEROUS,
            enabled=True,
        )
        skills._skills["danger_skill"] = Skill(meta=meta, module=module)

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="danger_skill",
        )
        # Should not raise, should just log and skip
        await heartbeat._execute_decision(decision)

    async def test_requires_confirmation_skill_blocked(
        self,
        heartbeat: HeartbeatLoop,
        skills: SkillRegistry,
    ) -> None:
        """Skills requiring confirmation should be skipped autonomously."""
        from types import SimpleNamespace

        module = SimpleNamespace(execute=lambda **kw: {"result": "should not run"})
        meta = SkillMeta(
            name="confirm_skill",
            description="Needs confirmation",
            usage="",
            safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
            enabled=True,
        )
        skills._skills["confirm_skill"] = Skill(meta=meta, module=module)

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="confirm_skill",
        )
        await heartbeat._execute_decision(decision)

    async def test_disabled_skill_blocked(
        self,
        heartbeat: HeartbeatLoop,
        skills: SkillRegistry,
    ) -> None:
        """Disabled skills should be skipped."""
        from types import SimpleNamespace

        module = SimpleNamespace(execute=lambda **kw: {"result": "should not run"})
        meta = SkillMeta(
            name="disabled_skill",
            description="Disabled",
            usage="",
            enabled=False,
        )
        skills._skills["disabled_skill"] = Skill(meta=meta, module=module)

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="disabled_skill",
        )
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

    async def test_tick_uses_configured_timezone(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Timezone from config is passed to _in_quiet_hours."""
        heartbeat._config.timezone = "Asia/Tokyo"
        with patch(
            "cordbeat.heartbeat._in_quiet_hours",
            return_value=True,
        ) as mock_quiet:
            await heartbeat._tick()
        _, kwargs = mock_quiet.call_args
        import zoneinfo

        assert kwargs["tz"] == zoneinfo.ZoneInfo("Asia/Tokyo")

    async def test_tick_invalid_timezone_falls_back_to_utc(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Invalid timezone string falls back to UTC."""
        heartbeat._config.timezone = "Invalid/Zone"
        with patch(
            "cordbeat.heartbeat._in_quiet_hours",
            return_value=True,
        ) as mock_quiet:
            await heartbeat._tick()
        _, kwargs = mock_quiet.call_args
        from datetime import UTC

        assert kwargs["tz"] is UTC


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


# ── Lifecycle ─────────────────────────────────────────────────────────


class TestLifecycle:
    async def test_start_and_stop(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """start() creates a task, stop() cancels it."""
        await heartbeat.start()
        assert heartbeat._task is not None
        assert heartbeat._running is True

        await heartbeat.stop()
        assert heartbeat._running is False

    async def test_stop_without_start(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """stop() before start() should not crash."""
        await heartbeat.stop()


# ── Global context: elapsed time ──────────────────────────────────────


class TestBuildGlobalContextElapsed:
    def test_user_with_last_talked_at_today(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        from datetime import datetime

        users = [
            UserSummary(
                user_id="u1",
                display_name="Alice",
                last_topic="",
                emotional_tone="",
                attention_score=0.5,
                last_talked_at=datetime.now(),
            ),
        ]
        ctx = heartbeat._build_global_context(users)
        assert "(today)" in ctx

    def test_user_with_last_talked_at_days_ago(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        from datetime import datetime, timedelta

        users = [
            UserSummary(
                user_id="u1",
                display_name="Bob",
                last_topic="",
                emotional_tone="",
                attention_score=0.5,
                last_talked_at=datetime.now() - timedelta(days=3),
            ),
        ]
        ctx = heartbeat._build_global_context(users)
        assert "(3d ago)" in ctx


# ── Skill execution in heartbeat ─────────────────────────────────────


class TestSkillExecution:
    async def test_safe_skill_executes(
        self,
        heartbeat: HeartbeatLoop,
        skills: SkillRegistry,
    ) -> None:
        """Safe enabled skill runs successfully."""
        from types import SimpleNamespace

        module = SimpleNamespace(execute=lambda **kw: {"result": "done"})
        meta = SkillMeta(
            name="safe_skill",
            description="Safe",
            usage="",
            safety_level=SafetyLevel.SAFE,
            enabled=True,
        )
        skills._skills["safe_skill"] = Skill(meta=meta, module=module)

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="safe_skill",
            skill_params={},
        )
        # Should not raise
        await heartbeat._execute_decision(decision)

    async def test_skill_execution_failure_handled(
        self,
        heartbeat: HeartbeatLoop,
        skills: SkillRegistry,
    ) -> None:
        """Skill that raises is caught and logged."""
        from types import SimpleNamespace

        def bad_execute(**kw: object) -> None:
            msg = "skill error"
            raise RuntimeError(msg)

        module = SimpleNamespace(execute=bad_execute)
        meta = SkillMeta(
            name="bad_skill",
            description="Broken",
            usage="",
            safety_level=SafetyLevel.SAFE,
            enabled=True,
        )
        skills._skills["bad_skill"] = Skill(meta=meta, module=module)

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="bad_skill",
            skill_params={},
        )
        # Should not raise
        await heartbeat._execute_decision(decision)


# ── Sleep phase error handling ────────────────────────────────────────


class TestSleepPhaseErrors:
    async def test_diary_error_does_not_stop_sleep(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """If diary generation fails for one user, sleep continues."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_message("u1", "user", "hi", "discord")

        mock_ai.generate = AsyncMock(side_effect=RuntimeError("AI down"))
        # Should not raise — error is logged and skipped
        await heartbeat._run_sleep_phase()

    async def test_decay_error_does_not_stop_sleep(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """If decay_and_archive fails, sleep continues."""
        memory.decay_and_archive_memories = AsyncMock(
            side_effect=RuntimeError("DB error")
        )
        # Should not raise
        await heartbeat._run_sleep_phase()

    async def test_trim_error_does_not_stop_sleep(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """If trim_old_messages fails, sleep continues."""
        await memory.get_or_create_user("u1", "Alice")
        memory.trim_old_messages = AsyncMock(side_effect=RuntimeError("DB error"))
        # Should not raise
        await heartbeat._run_sleep_phase()
