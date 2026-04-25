"""Tests for HEARTBEAT loop."""

from __future__ import annotations

import json
from datetime import UTC, datetime, time, timedelta
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
    ProposalStatus,
    ProposalType,
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
        skills._skills["danger_skill"] = Skill(meta=meta, _test_callable=module.execute)

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
        skills._skills["confirm_skill"] = Skill(
            meta=meta, _test_callable=module.execute
        )

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
        skills._skills["disabled_skill"] = Skill(
            meta=meta, _test_callable=module.execute
        )

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="disabled_skill",
        )
        await heartbeat._execute_decision(decision)

    async def test_action_propose_improvement(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Proposal is stored in memory."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_IMPROVEMENT,
            content="Improve memory decay",
        )
        await heartbeat._execute_decision(decision)
        # Stored under __system__ since no target user
        records = await memory.get_certain_records("__system__", record_type="proposal")
        assert len(records) == 1
        assert "Improve memory decay" in records[0]["content"]


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
        """With users, AI should be called for triage (Layer 1)."""
        await memory.get_or_create_user("u1", "user1")
        # Layer 1 returns no users to act on → only triage call
        mock_ai.generate_json = AsyncMock(
            return_value={
                "users": [],
                "next_heartbeat_minutes": 30,
            }
        )
        with patch("cordbeat.heartbeat._in_quiet_hours", return_value=False):
            result = await heartbeat._tick()
        mock_ai.generate_json.assert_called_once()
        assert result == 30

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

        await heartbeat._sleep.run()

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

        await heartbeat._sleep.run()

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
        await heartbeat._sleep.run()
        memory.decay_and_archive_memories.assert_called_once()

    async def test_sleep_phase_trims_messages(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Sleep phase trims old messages for each user."""
        await memory.get_or_create_user("u1", "Alice")
        memory.trim_old_messages = AsyncMock(return_value=10)

        await heartbeat._sleep.run()
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
        users = [
            UserSummary(
                user_id="u1",
                display_name="Alice",
                last_topic="",
                emotional_tone="",
                attention_score=0.5,
                last_talked_at=datetime.now(tz=UTC),
            ),
        ]
        ctx = heartbeat._build_global_context(users)
        assert "(today)" in ctx

    def test_user_with_last_talked_at_days_ago(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        users = [
            UserSummary(
                user_id="u1",
                display_name="Bob",
                last_topic="",
                emotional_tone="",
                attention_score=0.5,
                last_talked_at=datetime.now(tz=UTC) - timedelta(days=3),
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
        skills._skills["safe_skill"] = Skill(meta=meta, _test_callable=module.execute)

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
        skills._skills["bad_skill"] = Skill(meta=meta, _test_callable=module.execute)

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
        await heartbeat._sleep.run()

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
        await heartbeat._sleep.run()

    async def test_trim_error_does_not_stop_sleep(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """If trim_old_messages fails, sleep continues."""
        await memory.get_or_create_user("u1", "Alice")
        memory.trim_old_messages = AsyncMock(side_effect=RuntimeError("DB error"))
        # Should not raise
        await heartbeat._sleep.run()


# ── Sleep phase: memory promotion ─────────────────────────────────────


class TestMemoryPromotion:
    async def test_promotes_episodic_to_semantic(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Sleep phase promotes notable episodic memories to semantic."""
        from cordbeat.models import MemoryEntry, MemoryLayer

        await memory.get_or_create_user("u1", "Alice")
        entry = MemoryEntry(
            id="ep1",
            user_id="u1",
            layer=MemoryLayer.EPISODIC,
            content="Alice mentioned that she loves hiking in the mountains",
        )
        await memory.add_episodic_memory(entry)

        # AI returns promoted facts for diary + promotion calls
        async def _generate(**kwargs: object) -> str:
            system = kwargs.get("system", "")
            if isinstance(system, str) and "generalizable" in system.lower():
                return '{"facts": ["Alice enjoys hiking in the mountains"]}'
            return "Diary: quiet day."

        mock_ai.generate = AsyncMock(side_effect=_generate)

        await heartbeat._sleep.run()

        # Check that a semantic memory was created
        results = await memory.search_semantic("u1", "hiking")
        promoted = [r for r in results if "hiking" in r["content"]]
        assert len(promoted) >= 1

    async def test_promotion_caps_at_five_facts(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Promotion should only store up to 5 facts."""
        from cordbeat.models import MemoryEntry, MemoryLayer

        await memory.get_or_create_user("u1", "Alice")
        entry = MemoryEntry(
            id="ep2",
            user_id="u1",
            layer=MemoryLayer.EPISODIC,
            content="Many topics discussed today",
        )
        await memory.add_episodic_memory(entry)

        async def _generate(**kwargs: object) -> str:
            system = kwargs.get("system", "")
            if isinstance(system, str) and "generalizable" in system.lower():
                return json.dumps({"facts": [f"Fact {i}" for i in range(10)]})
            return "Diary"

        mock_ai.generate = AsyncMock(side_effect=_generate)

        await heartbeat._sleep.run()

        results = await memory.search_semantic("u1", "Fact")
        fact_results = [r for r in results if r["content"].startswith("Fact")]
        assert len(fact_results) <= 5

    async def test_promotion_failure_does_not_crash(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """AI returning invalid JSON for promotion should not crash."""
        from cordbeat.models import MemoryEntry, MemoryLayer

        await memory.get_or_create_user("u1", "Alice")
        entry = MemoryEntry(
            id="ep3",
            user_id="u1",
            layer=MemoryLayer.EPISODIC,
            content="Some episode",
        )
        await memory.add_episodic_memory(entry)

        async def _generate(**kwargs: object) -> str:
            system = kwargs.get("system", "")
            if isinstance(system, str) and "generalizable" in system.lower():
                return "not valid json"
            return "Diary"

        mock_ai.generate = AsyncMock(side_effect=_generate)
        # Should not raise
        await heartbeat._sleep.run()

    async def test_skips_user_without_episodic_memories(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Users with no episodic memories should not trigger promotion."""
        await memory.get_or_create_user("u1", "Silent")

        async def _generate(**kwargs: object) -> str:
            system = kwargs.get("system", "")
            if isinstance(system, str) and "generalizable" in system.lower():
                pytest.fail("Should not call promotion AI for user without episodes")
            return "Diary"

        mock_ai.generate = AsyncMock(side_effect=_generate)
        await heartbeat._sleep.run()


# ── Sleep phase: temporal recall precomputation ───────────────────────


class TestTemporalRecall:
    async def test_stores_temporal_recall_hint(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Sleep phase stores temporal recall hint for 7-day-old messages."""
        from datetime import datetime, timedelta

        await memory.get_or_create_user("u1", "Alice")

        # Insert a message dated 7 days ago
        seven_days_ago = (datetime.now(tz=UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
        await memory._conn.execute(
            "INSERT INTO conversation_messages "
            "(user_id, role, content, adapter_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("u1", "user", "Let's discuss OSS licensing", "discord", seven_days_ago),
        )
        await memory._conn.commit()

        mock_ai.generate = AsyncMock(return_value="Diary entry")

        await heartbeat._sleep.run()

        hints = await memory.get_recall_hints("u1")
        assert len(hints) >= 1
        assert any("OSS" in h["content"] for h in hints)

    async def test_multiple_lookback_windows(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Temporal recall should check 1/7/30/365 day windows."""
        from datetime import datetime, timedelta

        await memory.get_or_create_user("u1", "Alice")

        # Insert messages at 1-day and 30-day marks
        for days_ago, topic in [(1, "Yesterday topic"), (30, "Monthly topic")]:
            target_date = (datetime.now(tz=UTC) - timedelta(days=days_ago)).strftime(
                "%Y-%m-%d"
            )
            await memory._conn.execute(
                "INSERT INTO conversation_messages "
                "(user_id, role, content, adapter_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("u1", "user", topic, "discord", target_date),
            )
        await memory._conn.commit()

        mock_ai.generate = AsyncMock(return_value="Diary")

        await heartbeat._sleep.run()

        hints = await memory.get_recall_hints("u1")
        contents = [h["content"] for h in hints]
        assert any("Yesterday topic" in c for c in contents)
        assert any("Monthly topic" in c for c in contents)

    async def test_hint_label_matches_window(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Temporal recall hints should use human-readable labels."""
        from datetime import datetime, timedelta

        await memory.get_or_create_user("u1", "Alice")

        yesterday = (datetime.now(tz=UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
        await memory._conn.execute(
            "INSERT INTO conversation_messages "
            "(user_id, role, content, adapter_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("u1", "user", "Discussed Python typing", "discord", yesterday),
        )
        await memory._conn.commit()

        mock_ai.generate = AsyncMock(return_value="Diary")
        await heartbeat._sleep.run()

        hints = await memory.get_recall_hints("u1")
        assert any("yesterday" in h["content"] for h in hints)

    async def test_no_hint_when_no_old_messages(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """No temporal hint stored when there are no matching old messages."""
        await memory.get_or_create_user("u1", "Alice")

        mock_ai.generate = AsyncMock(return_value="Diary")

        await heartbeat._sleep.run()

        hints = await memory.get_recall_hints("u1")
        assert len(hints) == 0

    async def test_temporal_recall_error_does_not_crash(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Temporal recall failure should not crash sleep phase."""
        await memory.get_or_create_user("u1", "Alice")
        memory.get_messages_on_date = AsyncMock(side_effect=RuntimeError("DB error"))

        mock_ai.generate = AsyncMock(return_value="Diary")
        # Should not raise
        await heartbeat._sleep.run()

    async def test_clears_old_recall_hints(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Sleep phase clears old recall hints."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_recall_hint(
            user_id="u1",
            hint_type="temporal",
            content="Old hint",
        )
        # Make it old
        await memory._conn.execute(
            "UPDATE certain_records SET created_at = '2020-01-01T00:00:00' "
            "WHERE record_type = 'recall_hint'"
        )
        await memory._conn.commit()

        mock_ai.generate = AsyncMock(return_value="Diary")
        await heartbeat._sleep.run()

        hints = await memory.get_recall_hints("u1")
        assert len(hints) == 0


# ── Sleep phase: chain recall precomputation ──────────────────────────


class TestChainRecallPrecomputation:
    async def test_stores_chain_links_for_related_memories(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Sleep phase should store chain links between related memories."""
        from cordbeat.models import MemoryEntry, MemoryLayer

        await memory.get_or_create_user("u1", "Alice")

        # Store related episodic and semantic memories
        ep_entry = MemoryEntry(
            id="ep_python",
            user_id="u1",
            layer=MemoryLayer.EPISODIC,
            content="Alice discussed Python 3.12 new features",
        )
        sem_entry = MemoryEntry(
            id="sem_python",
            user_id="u1",
            layer=MemoryLayer.SEMANTIC,
            content="Alice prefers Python over JavaScript",
        )
        await memory.add_episodic_memory(ep_entry)
        await memory.add_semantic_memory(sem_entry)

        mock_ai.generate = AsyncMock(return_value="Diary")

        await heartbeat._sleep.run()

        # Chain links should have been created
        links = await memory.get_chain_links("u1", ["ep_python"])
        assert len(links) >= 1
        assert any("Python" in link for link in links)

    async def test_no_chain_links_without_memories(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """No chain links if user has no episodic memories."""
        await memory.get_or_create_user("u1", "Alice")
        mock_ai.generate = AsyncMock(return_value="Diary")

        await heartbeat._sleep.run()

        # Should not crash, and no chain_link records
        records = await memory.get_certain_records("u1", record_type="chain_link")
        assert len(records) == 0

    async def test_chain_precomputation_error_does_not_crash(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Chain link failure should not crash sleep phase."""
        await memory.get_or_create_user("u1", "Alice")
        memory.search_episodic = AsyncMock(side_effect=RuntimeError("Vector DB error"))
        mock_ai.generate = AsyncMock(return_value="Diary")

        # Should not raise
        await heartbeat._sleep.run()

    async def test_clears_old_chain_links(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Sleep phase clears old chain links."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_old",
            linked_content="Old chain link",
        )
        await memory._conn.execute(
            "UPDATE certain_records SET created_at = '2020-01-01T00:00:00' "
            "WHERE record_type = 'chain_link'"
        )
        await memory._conn.commit()

        mock_ai.generate = AsyncMock(return_value="Diary")
        await heartbeat._sleep.run()

        links = await memory.get_chain_links("u1", ["mem_old"])
        assert len(links) == 0


# ── Two-layer architecture ────────────────────────────────────────────


class TestLayer1Triage:
    async def test_triage_returns_selected_users(
        self,
        heartbeat: HeartbeatLoop,
        mock_ai: AsyncMock,
    ) -> None:
        """Layer 1 should return list of users needing attention."""
        mock_ai.generate_json = AsyncMock(
            return_value={
                "users": [{"user_id": "u1", "reason": "lonely"}],
                "next_heartbeat_minutes": 30,
            }
        )
        users = [
            UserSummary(user_id="u1", display_name="Alice"),
            UserSummary(user_id="u2", display_name="Bob"),
        ]
        result = await heartbeat._layer1_triage(users)
        assert len(result["users"]) == 1
        assert result["users"][0]["user_id"] == "u1"
        assert result["next_heartbeat_minutes"] == 30

    async def test_triage_empty_result(
        self,
        heartbeat: HeartbeatLoop,
        mock_ai: AsyncMock,
    ) -> None:
        """Layer 1 returning empty means nobody needs attention."""
        mock_ai.generate_json = AsyncMock(
            return_value={"users": [], "next_heartbeat_minutes": 60}
        )
        users = [UserSummary(user_id="u1", display_name="Alice")]
        result = await heartbeat._layer1_triage(users)
        assert result["users"] == []

    async def test_triage_fallback_on_validation_failure(
        self,
        heartbeat: HeartbeatLoop,
        mock_ai: AsyncMock,
    ) -> None:
        """Invalid AI output → fallback with empty users list."""
        import json

        mock_ai.generate_json = AsyncMock(
            side_effect=json.JSONDecodeError("bad", "", 0)
        )
        users = [UserSummary(user_id="u1", display_name="Alice")]
        result = await heartbeat._layer1_triage(users)
        assert result["users"] == []


class TestLayer2Evaluate:
    async def test_evaluate_returns_decision(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Layer 2 should return a HeartbeatDecision for the user."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_message("u1", "user", "Hello", "discord")
        mock_ai.generate_json = AsyncMock(
            return_value={
                "action": "message",
                "content": "How are you?",
                "target_user_id": "u1",
                "target_adapter_id": "discord",
                "next_heartbeat_minutes": 30,
            }
        )
        user = UserSummary(
            user_id="u1",
            display_name="Alice",
            last_platform="discord",
        )
        decision = await heartbeat._layer2_evaluate(user, "lonely user")
        assert decision.action == HeartbeatAction.MESSAGE
        assert decision.content == "How are you?"
        assert decision.target_user_id == "u1"

    async def test_evaluate_fallback_on_validation_failure(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Invalid AI output → fallback to action=none."""
        import json

        await memory.get_or_create_user("u1", "Alice")
        mock_ai.generate_json = AsyncMock(
            side_effect=json.JSONDecodeError("bad", "", 0)
        )
        user = UserSummary(user_id="u1", display_name="Alice")
        decision = await heartbeat._layer2_evaluate(user, "check in")
        assert decision.action == HeartbeatAction.NONE

    async def test_evaluate_defaults_target_from_user(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """If AI doesn't specify target, defaults from user summary."""
        await memory.get_or_create_user("u1", "Alice")
        mock_ai.generate_json = AsyncMock(
            return_value={
                "action": "none",
                "content": "",
                "next_heartbeat_minutes": 60,
            }
        )
        user = UserSummary(
            user_id="u1",
            display_name="Alice",
            last_platform="telegram",
        )
        decision = await heartbeat._layer2_evaluate(user, "routine check")
        assert decision.target_user_id == "u1"
        assert decision.target_adapter_id == "telegram"


class TestTwoLayerIntegration:
    async def test_tick_triage_then_evaluate(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        """Full tick: triage selects user → evaluate sends message."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
        await memory.add_message("u1", "user", "Hello", "discord")

        call_count = 0

        async def mock_generate_json(prompt: str, **kwargs: object) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Layer 1: triage
                return {
                    "users": [{"user_id": "u1", "reason": "hasn't talked recently"}],
                    "next_heartbeat_minutes": 45,
                }
            # Layer 2: decision
            return {
                "action": "message",
                "content": "Hey Alice!",
                "target_user_id": "u1",
                "target_adapter_id": "discord",
                "next_heartbeat_minutes": 30,
            }

        mock_ai.generate_json = AsyncMock(side_effect=mock_generate_json)

        with patch("cordbeat.heartbeat._in_quiet_hours", return_value=False):
            result = await heartbeat._tick()

        # 2 AI calls: triage + evaluate
        assert mock_ai.generate_json.call_count == 2
        # Message was sent
        mock_gateway.send_to_adapter.assert_called_once()
        # Min of triage(45) and decision(30)
        assert result == 30

    async def test_tick_triage_selects_no_one(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        """Triage returns empty → no Layer 2 calls, no messages."""
        await memory.get_or_create_user("u1", "Alice")
        mock_ai.generate_json = AsyncMock(
            return_value={"users": [], "next_heartbeat_minutes": 60}
        )

        with patch("cordbeat.heartbeat._in_quiet_hours", return_value=False):
            result = await heartbeat._tick()

        assert mock_ai.generate_json.call_count == 1
        mock_gateway.send_to_adapter.assert_not_called()
        assert result == 60

    async def test_tick_unknown_user_in_triage_skipped(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """If triage returns unknown user_id, it's skipped in Layer 2."""
        await memory.get_or_create_user("u1", "Alice")
        mock_ai.generate_json = AsyncMock(
            return_value={
                "users": [{"user_id": "nonexistent", "reason": "???"}],
                "next_heartbeat_minutes": 60,
            }
        )

        with patch("cordbeat.heartbeat._in_quiet_hours", return_value=False):
            result = await heartbeat._tick()

        # Only triage call, no Layer 2
        assert mock_ai.generate_json.call_count == 1
        assert result == 60


# ── Proposal workflow ─────────────────────────────────────────────────


class TestProposalWorkflow:
    async def test_proposal_stored_with_target_user(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Proposal with target user is stored and notification sent."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_IMPROVEMENT,
            content="Add weather skill",
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        # Proposal stored under user
        records = await memory.get_certain_records("u1", record_type="proposal")
        assert len(records) == 1
        assert "Add weather skill" in records[0]["content"]

        # Notification sent via gateway
        mock_gateway.send_to_adapter.assert_called_once()
        call_args = mock_gateway.send_to_adapter.call_args
        msg = call_args[0][1]
        assert "suggestion" in msg.content
        assert "Add weather skill" in msg.content

    async def test_proposal_stored_without_target(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Proposal without target user is stored under __system__."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_IMPROVEMENT,
            content="Optimize sleep phase",
        )
        await heartbeat._execute_decision(decision)

        # Stored under __system__
        records = await memory.get_certain_records("__system__", record_type="proposal")
        assert len(records) == 1

        # No notification sent (no target)
        mock_gateway.send_to_adapter.assert_not_called()

    async def test_proposal_unresolvable_user_still_stored(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Proposal with unresolvable user is stored but no notification."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_IMPROVEMENT,
            content="Improve diary quality",
            target_user_id="unknown_user",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        records = await memory.get_certain_records(
            "unknown_user", record_type="proposal"
        )
        assert len(records) == 1

        # No notification (can't resolve platform user)
        mock_gateway.send_to_adapter.assert_not_called()

    async def test_proposal_metadata_has_status(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Proposal metadata contains pending status."""
        import json

        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_IMPROVEMENT,
            content="Add new skill",
            target_user_id="u_sys",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        records = await memory.get_certain_records("u_sys", record_type="proposal")
        assert len(records) == 1
        meta = json.loads(records[0]["metadata"])
        assert meta["status"] == "pending"
        assert meta["adapter_id"] == "discord"


class TestSkillProposal:
    """Tests for requires_confirmation skill → proposal flow."""

    async def test_requires_confirmation_creates_proposal(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Skill with requires_confirmation stores a proposal instead of executing."""
        from types import SimpleNamespace

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        execute_fn = MagicMock(return_value={"ok": True})
        module = SimpleNamespace(execute=execute_fn)
        skill = Skill(
            meta=SkillMeta(
                name="deploy",
                description="Deploy app",
                usage="deploy",
                safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
            ),
            _test_callable=module.execute,
        )
        skills._skills["deploy"] = skill

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="deploy",
            skill_params={"target": "production"},
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_skill(decision)

        # Skill was NOT executed
        execute_fn.assert_not_called()

        # Proposal was stored
        records = await memory.get_certain_records("u1", record_type="proposal")
        assert len(records) == 1
        meta = json.loads(records[0]["metadata"])
        assert meta["proposal_type"] == ProposalType.SKILL_EXECUTION
        assert meta["skill_name"] == "deploy"
        assert meta["status"] == ProposalStatus.PENDING

    async def test_skill_proposal_notifies_user(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Skill proposal sends notification with proposal ID."""
        from types import SimpleNamespace

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        module = SimpleNamespace(execute=lambda **kw: {"ok": True})
        skill = Skill(
            meta=SkillMeta(
                name="cleanup",
                description="Clean up data",
                usage="cleanup",
                safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
            ),
            _test_callable=module.execute,
        )
        skills._skills["cleanup"] = skill

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="cleanup",
            skill_params={},
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_skill(decision)

        mock_gateway.send_to_adapter.assert_called_once()
        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert "cleanup" in msg.content
        assert "proposal ID:" in msg.content


class TestApprovedProposalExecution:
    """Tests for executing approved proposals on heartbeat tick."""

    async def test_approved_skill_proposal_executed(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        """An approved skill proposal gets executed and marked as executed."""
        from types import SimpleNamespace

        execute_fn = MagicMock(return_value={"ok": True})
        module = SimpleNamespace(execute=execute_fn)
        skill = Skill(
            meta=SkillMeta(
                name="test_skill",
                description="Test",
                usage="test",
                safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
            ),
            _test_callable=module.execute,
        )
        skills._skills["test_skill"] = skill

        # Store a proposal and approve it
        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Run test_skill",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "test_skill",
                "skill_params": {"key": "value"},
            },
        )

        await heartbeat._proposals.execute_approved()

        # Skill was executed with correct params
        execute_fn.assert_called_once_with(key="value")

        # Proposal marked as executed
        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.EXECUTED

    async def test_approved_missing_skill_marked_expired(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Approved proposal for missing skill is marked expired."""
        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Run unknown_skill",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "nonexistent",
                "skill_params": {},
            },
        )

        await heartbeat._proposals.execute_approved()

        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.EXPIRED

    async def test_pending_proposals_not_executed(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        """Pending (not approved) proposals are NOT executed."""
        from types import SimpleNamespace

        execute_fn = MagicMock(return_value={"ok": True})
        module = SimpleNamespace(execute=execute_fn)
        skill = Skill(
            meta=SkillMeta(
                name="test_skill",
                description="Test",
                usage="test",
            ),
            _test_callable=module.execute,
        )
        skills._skills["test_skill"] = skill

        await memory.add_certain_record(
            user_id="u1",
            content="Run test_skill",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.PENDING,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "test_skill",
                "skill_params": {},
            },
        )

        await heartbeat._proposals.execute_approved()

        # Skill was NOT executed
        execute_fn.assert_not_called()

    async def test_general_proposal_marked_executed(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """General approved proposals are acknowledged and marked executed."""
        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="General improvement idea",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.GENERAL,
            },
        )

        await heartbeat._proposals.execute_approved()

        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.EXECUTED

    async def test_skill_execution_failure_marks_expired(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        """Failed skill execution marks proposal as expired."""
        from types import SimpleNamespace

        def boom(**kw: object) -> None:
            raise RuntimeError("boom")

        module = SimpleNamespace(execute=boom)
        skill = Skill(
            meta=SkillMeta(
                name="failing_skill",
                description="Fails",
                usage="fail",
            ),
            _test_callable=module.execute,
        )
        skills._skills["failing_skill"] = skill

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Run failing_skill",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "failing_skill",
                "skill_params": {},
            },
        )

        await heartbeat._proposals.execute_approved()

        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.EXPIRED


class TestTraitChangeProposal:
    """Tests for propose_trait_change -> approval -> application flow."""

    async def test_trait_proposal_stored(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        soul: Soul,
    ) -> None:
        """propose_trait_change action stores a TRAIT_CHANGE proposal."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_TRAIT_CHANGE,
            content="I want to become more playful",
            trait_add=["playful"],
            trait_remove=[],
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._proposals.store_trait_proposal(decision)

        records = await memory.get_certain_records("u1", record_type="proposal")
        assert len(records) == 1
        meta = json.loads(records[0]["metadata"])
        assert meta["proposal_type"] == ProposalType.TRAIT_CHANGE
        assert meta["status"] == ProposalStatus.PENDING
        assert meta["trait_add"] == ["playful"]
        assert meta["trait_remove"] == []
        assert "playful" in meta["trait_preview"]

    async def test_trait_proposal_does_not_apply(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        soul: Soul,
    ) -> None:
        """Storing a trait proposal does NOT modify soul traits."""
        original_traits = soul.traits[:]
        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_TRAIT_CHANGE,
            content="Add boldness",
            trait_add=["bold"],
            trait_remove=[],
        )
        await heartbeat._proposals.store_trait_proposal(decision)

        assert soul.traits == original_traits
        assert "bold" not in soul.traits

    async def test_trait_proposal_notifies_user(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Trait proposal sends notification with preview."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_TRAIT_CHANGE,
            content="Become more playful",
            trait_add=["playful"],
            trait_remove=[],
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._proposals.store_trait_proposal(decision)

        mock_gateway.send_to_adapter.assert_called_once()
        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert "proposal ID:" in msg.content
        assert "playful" in msg.content

    async def test_execute_decision_dispatches_trait_change(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """_execute_decision routes propose_trait_change correctly."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_TRAIT_CHANGE,
            content="Add empathetic trait",
            trait_add=["empathetic"],
            trait_remove=[],
        )
        await heartbeat._execute_decision(decision)

        records = await memory.get_certain_records("__system__", record_type="proposal")
        assert len(records) == 1
        meta = json.loads(records[0]["metadata"])
        assert meta["proposal_type"] == ProposalType.TRAIT_CHANGE

    async def test_trait_proposal_with_remove(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        soul: Soul,
    ) -> None:
        """Trait proposal can include removals and preview reflects them."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_TRAIT_CHANGE,
            content="Replace curious with adventurous",
            trait_add=["adventurous"],
            trait_remove=["curious"],
        )
        await heartbeat._proposals.store_trait_proposal(decision)

        records = await memory.get_certain_records("__system__", record_type="proposal")
        assert len(records) == 1
        meta = json.loads(records[0]["metadata"])
        assert "adventurous" in meta["trait_preview"]
        assert "curious" not in meta["trait_preview"]


class TestApprovedTraitExecution:
    """Tests for executing approved trait change proposals."""

    async def test_approved_trait_change_applied(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        soul: Soul,
    ) -> None:
        """Approved trait change proposal applies traits to soul."""
        assert "playful" not in soul.traits

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Add playful trait",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.TRAIT_CHANGE,
                "trait_add": ["playful"],
                "trait_remove": [],
            },
        )

        await heartbeat._proposals.execute_approved()

        # Trait was applied
        assert "playful" in soul.traits

        # Proposal marked as executed
        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.EXECUTED

    async def test_approved_trait_remove(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        soul: Soul,
    ) -> None:
        """Approved trait change can remove traits."""
        assert "curious" in soul.traits

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Remove curious",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.TRAIT_CHANGE,
                "trait_add": [],
                "trait_remove": ["curious"],
            },
        )

        await heartbeat._proposals.execute_approved()

        assert "curious" not in soul.traits

        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.EXECUTED

    async def test_pending_trait_change_not_applied(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        soul: Soul,
    ) -> None:
        """Pending trait change proposals are NOT applied."""
        original = soul.traits[:]

        await memory.add_certain_record(
            user_id="u1",
            content="Add bold",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.PENDING,
                "proposal_type": ProposalType.TRAIT_CHANGE,
                "trait_add": ["bold"],
                "trait_remove": [],
            },
        )

        await heartbeat._proposals.execute_approved()

        assert soul.traits == original
        assert "bold" not in soul.traits


class TestProposalExecutionNotification:
    """Tests for user notification after proposal execution."""

    async def test_skill_success_notifies_user(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Successful skill execution sends success notification."""
        from types import SimpleNamespace

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        execute_fn = MagicMock(return_value={"result": "data fetched"})
        module = SimpleNamespace(execute=execute_fn)
        skill = Skill(
            meta=SkillMeta(
                name="test_skill",
                description="Test",
                usage="test",
            ),
            _test_callable=module.execute,
        )
        skills._skills["test_skill"] = skill

        await memory.add_certain_record(
            user_id="u1",
            content="Run test_skill",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "test_skill",
                "skill_params": {},
                "adapter_id": "discord",
            },
        )

        await heartbeat._proposals.execute_approved()

        mock_gateway.send_to_adapter.assert_called_once()
        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert "✅" in msg.content
        assert "test_skill" in msg.content

    async def test_skill_failure_notifies_user(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Failed skill execution sends failure notification."""
        from types import SimpleNamespace

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        def boom(**kw: object) -> None:
            raise RuntimeError("boom")

        module = SimpleNamespace(execute=boom)
        skill = Skill(
            meta=SkillMeta(
                name="bad_skill",
                description="Fails",
                usage="fail",
            ),
            _test_callable=module.execute,
        )
        skills._skills["bad_skill"] = skill

        await memory.add_certain_record(
            user_id="u1",
            content="Run bad_skill",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "bad_skill",
                "skill_params": {},
                "adapter_id": "discord",
            },
        )

        await heartbeat._proposals.execute_approved()

        mock_gateway.send_to_adapter.assert_called_once()
        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert "❌" in msg.content
        assert "bad_skill" in msg.content

    async def test_missing_skill_notifies_user(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Missing skill sends warning notification."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        await memory.add_certain_record(
            user_id="u1",
            content="Run nonexistent",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "nonexistent",
                "skill_params": {},
                "adapter_id": "discord",
            },
        )

        await heartbeat._proposals.execute_approved()

        mock_gateway.send_to_adapter.assert_called_once()
        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert "⚠️" in msg.content
        assert "nonexistent" in msg.content

    async def test_trait_change_notifies_user(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        soul: Soul,
        mock_gateway: AsyncMock,
    ) -> None:
        """Successful trait change sends notification with details."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        await memory.add_certain_record(
            user_id="u1",
            content="Add playful trait",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.TRAIT_CHANGE,
                "trait_add": ["playful"],
                "trait_remove": [],
                "adapter_id": "discord",
            },
        )

        await heartbeat._proposals.execute_approved()

        mock_gateway.send_to_adapter.assert_called_once()
        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert "✅" in msg.content
        assert "playful" in msg.content

    async def test_general_proposal_notifies_user(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """General proposal execution sends acknowledgment."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        await memory.add_certain_record(
            user_id="u1",
            content="General idea",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.GENERAL,
                "adapter_id": "discord",
            },
        )

        await heartbeat._proposals.execute_approved()

        mock_gateway.send_to_adapter.assert_called_once()
        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert "✅" in msg.content

    async def test_no_adapter_id_skips_notification(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Proposals without adapter_id skip notification gracefully."""
        await memory.add_certain_record(
            user_id="u1",
            content="No adapter proposal",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.GENERAL,
            },
        )

        await heartbeat._proposals.execute_approved()

        # Proposal still executed but no notification
        mock_gateway.send_to_adapter.assert_not_called()


class TestSkillCreationProposal:
    """Tests for AI-generated skill proposal feature."""

    async def test_store_skill_creation_proposal(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Skill creation proposal is stored with proposed_skill metadata."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_SKILL,
            content="Create a translation skill",
            proposed_skill={
                "name": "translate",
                "description": "Translate text",
                "usage": "When translation is needed",
                "parameters": [
                    {
                        "name": "text",
                        "type": "string",
                        "required": True,
                        "description": "Text to translate",
                    }
                ],
                "code": (
                    "async def execute(*, text, **_kw):\n    return {'result': text}\n"
                ),
            },
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        records = await memory.get_certain_records("u1", record_type="proposal")
        assert len(records) == 1
        meta = json.loads(records[0]["metadata"])
        assert meta["proposal_type"] == ProposalType.SKILL_PROPOSAL
        assert meta["proposed_skill"]["name"] == "translate"

    async def test_skill_creation_notification(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """User is notified about skill creation proposal."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_SKILL,
            content="Create translate skill",
            proposed_skill={
                "name": "translate",
                "description": "Translate text",
                "parameters": [],
                "code": "def execute(**kw):\n    return {}\n",
            },
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._proposals.store_skill_creation_proposal(decision)

        mock_gateway.send_to_adapter.assert_called_once()
        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert "translate" in msg.content
        assert "proposal ID:" in msg.content

    async def test_install_proposed_skill(
        self,
        heartbeat: HeartbeatLoop,
        tmp_path: Path,
    ) -> None:
        """Approved skill proposal writes files and reloads registry."""
        skills_dir = heartbeat._skills.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)

        proposed = {
            "name": "greet",
            "description": "Greet the user",
            "usage": "When greeting is needed",
            "parameters": [
                {
                    "name": "name",
                    "type": "string",
                    "required": True,
                    "description": "User name",
                }
            ],
            "code": (
                "def execute(*, name, **_kw):\n"
                "    return {'greeting': f'Hello {name}'}\n"
            ),
        }
        await heartbeat._proposals.install_proposed_skill(proposed)

        # Files written
        skill_dir = skills_dir / "greet"
        assert (skill_dir / "skill.yaml").exists()
        assert (skill_dir / "main.py").exists()

        # Skill loaded in registry
        skill = heartbeat._skills.get("greet")
        assert skill is not None
        assert skill.meta.safety_level == SafetyLevel.REQUIRES_CONFIRMATION

    async def test_install_rejects_invalid_name(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Invalid skill names are rejected."""
        proposed = {
            "name": "Bad-Name!",
            "description": "test",
            "code": "def execute(**kw):\n    pass\n",
        }
        with pytest.raises(ValueError, match="Invalid skill name"):
            await heartbeat._proposals.install_proposed_skill(proposed)

    async def test_install_rejects_existing_skill(
        self,
        heartbeat: HeartbeatLoop,
        tmp_path: Path,
    ) -> None:
        """Cannot overwrite an existing skill."""
        skills_dir = heartbeat._skills.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)

        # Install first
        proposed = {
            "name": "myskill",
            "description": "test",
            "parameters": [],
            "code": "def execute(**kw):\n    return {}\n",
        }
        await heartbeat._proposals.install_proposed_skill(proposed)

        # Try again
        with pytest.raises(ValueError, match="already exists"):
            await heartbeat._proposals.install_proposed_skill(proposed)

    async def test_install_rejects_dangerous_code(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Code importing disallowed modules is rejected by AST validator."""
        skills_dir = heartbeat._skills.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)

        proposed = {
            "name": "evil",
            "description": "test",
            "parameters": [],
            "code": "import subprocess\ndef execute(**kw):\n    pass\n",
        }
        with pytest.raises(ValueError, match="subprocess"):
            await heartbeat._proposals.install_proposed_skill(proposed)

    async def test_install_rejects_invalid_syntax(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Code with syntax errors is rejected."""
        skills_dir = heartbeat._skills.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)

        proposed = {
            "name": "broken",
            "description": "test",
            "parameters": [],
            "code": "def execute(**kw)\n    pass\n",  # missing colon
        }
        with pytest.raises(ValueError, match="syntax error"):
            await heartbeat._proposals.install_proposed_skill(proposed)

    async def test_install_rejects_empty_code(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Empty code is rejected."""
        skills_dir = heartbeat._skills.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)

        proposed = {
            "name": "empty",
            "description": "test",
            "parameters": [],
            "code": "",
        }
        with pytest.raises(ValueError, match="empty"):
            await heartbeat._proposals.install_proposed_skill(proposed)

    async def test_install_sanitizes_usage_yaml_injection(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Malicious `usage` content cannot inject top-level YAML keys.

        Regression test: before the fix, newlines in `usage` broke out of the
        YAML literal block, letting an attacker override `safety`, `sandbox`,
        etc. with arbitrary values.
        """
        import yaml

        skills_dir = heartbeat._skills.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)

        malicious_usage = (
            "Example usage\n"
            "safety:\n"
            "  level: safe\n"
            "  sandbox: true\n"
            "  network: true\n"
            "  filesystem: true\n"
        )
        proposed = {
            "name": "injected",
            "description": "test",
            "usage": malicious_usage,
            "parameters": [],
            "code": "def execute(**kw):\n    return {}\n",
        }
        await heartbeat._proposals.install_proposed_skill(proposed)

        yaml_path = skills_dir / "injected" / "skill.yaml"
        data = yaml.safe_load(yaml_path.read_text())

        assert data["safety"]["level"] == "requires_confirmation"
        assert data["safety"]["sandbox"] is False
        assert data["safety"]["network"] is False
        assert data["safety"]["filesystem"] is False
        assert "safety:" in data["usage"]

        skill = heartbeat._skills.get("injected")
        assert skill is not None
        assert skill.meta.safety_level == SafetyLevel.REQUIRES_CONFIRMATION

    async def test_execute_approved_skill_proposal(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Approved skill proposal is installed via _execute_approved_proposals."""
        skills_dir = heartbeat._skills.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        await memory.add_certain_record(
            user_id="u1",
            content="New skill: hello",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_PROPOSAL,
                "adapter_id": "discord",
                "proposed_skill": {
                    "name": "hello",
                    "description": "Say hello",
                    "usage": "Greeting",
                    "parameters": [],
                    "code": ("def execute(**kw):\n    return {'msg': 'hello'}\n"),
                },
            },
        )

        await heartbeat._proposals.execute_approved()

        # Skill installed
        skill = heartbeat._skills.get("hello")
        assert skill is not None

        # Notification sent
        mock_gateway.send_to_adapter.assert_called_once()
        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert "✅" in msg.content
        assert "hello" in msg.content

    async def test_execute_failed_skill_proposal(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Failed skill proposal is marked expired."""
        skills_dir = heartbeat._skills.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        await memory.add_certain_record(
            user_id="u1",
            content="Bad skill",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_PROPOSAL,
                "adapter_id": "discord",
                "proposed_skill": {
                    "name": "Bad!Name",
                    "description": "Invalid",
                    "code": "def execute(**kw): pass",
                },
            },
        )

        await heartbeat._proposals.execute_approved()

        mock_gateway.send_to_adapter.assert_called_once()
        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert "❌" in msg.content

    async def test_forced_safety_level(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """AI-generated skills always get requires_confirmation level."""
        skills_dir = heartbeat._skills.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)

        proposed = {
            "name": "safeskill",
            "description": "test",
            "parameters": [],
            "code": "def execute(**kw):\n    return {}\n",
        }
        await heartbeat._proposals.install_proposed_skill(proposed)

        yaml_content = (skills_dir / "safeskill" / "skill.yaml").read_text(
            encoding="utf-8"
        )
        assert "requires_confirmation" in yaml_content
        # AI cannot set dangerous level
        assert "dangerous" not in yaml_content
