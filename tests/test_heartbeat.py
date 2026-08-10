"""Tests for HEARTBEAT loop."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from cordbeat.agent.heartbeat import HeartbeatLoop, _in_quiet_hours, _parse_time
from cordbeat.agent.soul import Soul
from cordbeat.config import HeartbeatConfig, MemoryConfig
from cordbeat.core.gateway import MessageQueue
from cordbeat.memory import MemoryStore
from cordbeat.models import (
    HeartbeatAction,
    HeartbeatDecision,
    MessageType,
    ProposalStatus,
    ProposalType,
    SafetyLevel,
    SkillMeta,
    SkillParam,
    UserSummary,
)
from cordbeat.skills import Skill, SkillRegistry

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
        with patch("cordbeat.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(3, 0))
            assert _in_quiet_hours("01:00", "07:00") is True

    def test_outside_range(self) -> None:
        """12:00 is outside 01:00-07:00."""
        with patch("cordbeat.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(12, 0))
            assert _in_quiet_hours("01:00", "07:00") is False

    def test_midnight_wrap_inside(self) -> None:
        """23:30 is inside 22:00-06:00 (wraps midnight)."""
        with patch("cordbeat.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(23, 30))
            assert _in_quiet_hours("22:00", "06:00") is True

    def test_midnight_wrap_inside_early(self) -> None:
        """02:00 is inside 22:00-06:00 (wraps midnight)."""
        with patch("cordbeat.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(2, 0))
            assert _in_quiet_hours("22:00", "06:00") is True

    def test_midnight_wrap_outside(self) -> None:
        """12:00 is outside 22:00-06:00 (wraps midnight)."""
        with patch("cordbeat.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(12, 0))
            assert _in_quiet_hours("22:00", "06:00") is False

    def test_at_boundary_start(self) -> None:
        """01:00 (boundary) is inside 01:00-07:00."""
        with patch("cordbeat.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(1, 0))
            assert _in_quiet_hours("01:00", "07:00") is True

    def test_at_boundary_end(self) -> None:
        """07:00 (boundary) is inside 01:00-07:00."""
        with patch("cordbeat.agent.heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(time=lambda: time(7, 0))
            assert _in_quiet_hours("01:00", "07:00") is True

    def test_invalid_quiet_hours_fail_closed_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert _in_quiet_hours("25:99", "07:00") is False
        assert "Invalid quiet hours configured" in caplog.text


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
    ai.generate = AsyncMock(return_value="SKIP")
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
        adapters_options={"discord": {"dm_policy": "allow_proactive"}},
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
        """No platform link → self-heal by treating target_user_id as the
        platform id (legacy users pre-310deb4 lacked platform_link rows)."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="Hello!",
            target_user_id="u_unknown",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)
        mock_gateway.send_to_adapter.assert_called_once()

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
        with patch("cordbeat.agent.heartbeat._in_quiet_hours", return_value=True):
            result = await heartbeat._tick()
        assert result == 60
        assert heartbeat._sleep_done_today is True

    async def test_tick_skipped_when_queue_busy(
        self,
        heartbeat: HeartbeatLoop,
        queue: MessageQueue,
        mock_ai: AsyncMock,
    ) -> None:
        """Heartbeat must skip its tick when a user-message handler is
        actively running, to avoid contending with that LLM call."""
        queue._processing = True  # simulate handler in flight
        result = await heartbeat._tick()
        assert result == 60
        # Crucially, no AI call must have been issued by heartbeat itself.
        mock_ai.generate.assert_not_called()
        mock_ai.generate_json.assert_not_called()

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
        with patch("cordbeat.agent.heartbeat._in_quiet_hours", return_value=False):
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
            "cordbeat.agent.heartbeat._in_quiet_hours",
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
            "cordbeat.agent.heartbeat._in_quiet_hours",
            return_value=True,
        ) as mock_quiet:
            await heartbeat._tick()
        _, kwargs = mock_quiet.call_args
        assert kwargs["tz"] is UTC


class TestReminderDelivery:
    async def _setup_user(self, memory: MemoryStore) -> UserSummary:
        user = await memory.get_or_create_user("u1", "Alice")
        user.last_platform = "discord"
        await memory.update_user_summary(user)
        await memory.link_platform("u1", "discord", "discord_123")
        await memory.record_last_seen_channel(
            "u1",
            "discord",
            "channel-1",
            False,
        )
        return user

    async def _add_reminder(
        self,
        memory: MemoryStore,
        *,
        content: str = "Check oven",
        remind_at: str | None = None,
    ) -> str:
        return await memory.add_certain_record(
            "u1",
            content,
            "reminder",
            {
                "status": "pending",
                "remind_at": remind_at
                or (datetime.now(tz=UTC) - timedelta(minutes=1)).isoformat(),
            },
        )

    async def _metadata(
        self,
        memory: MemoryStore,
        record_id: str,
    ) -> dict[str, Any]:
        records = await memory.get_certain_records("u1", record_type="reminder")
        record = next(record for record in records if record["id"] == record_id)
        return json.loads(record["metadata"])

    async def test_tick_delivers_due_reminder_and_marks_delivered(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        await self._setup_user(memory)
        record_id = await self._add_reminder(memory)
        await memory.add_certain_record(
            "u1",
            "Recent proactive message",
            "heartbeat_user_sent",
        )
        await memory.add_certain_record(
            "__system__",
            "Recent destination message",
            "heartbeat_destination_sent",
            {"destination_key": "discord:channel:channel-1"},
        )
        mock_ai.generate_json = AsyncMock(
            return_value={"users": [], "next_heartbeat_minutes": 60}
        )

        with patch("cordbeat.agent.heartbeat._in_quiet_hours", return_value=False):
            await heartbeat._tick()

        mock_gateway.send_to_adapter.assert_awaited_once()
        adapter_id, message = mock_gateway.send_to_adapter.await_args.args
        assert adapter_id == "discord"
        assert message.platform_user_id == "discord_123"
        assert message.content == "Reminder: Check oven"
        assert message.metadata["channel_id"] == "channel-1"
        metadata = await self._metadata(memory, record_id)
        assert metadata["status"] == "delivered"
        assert datetime.fromisoformat(metadata["delivered_at"]).tzinfo is not None

    async def test_future_reminder_is_not_sent(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        user = await self._setup_user(memory)
        record_id = await self._add_reminder(
            memory,
            remind_at=(datetime.now(tz=UTC) + timedelta(hours=1)).isoformat(),
        )

        await heartbeat._deliver_due_reminders([user])

        mock_gateway.send_to_adapter.assert_not_awaited()
        assert (await self._metadata(memory, record_id))["status"] == "pending"

    async def test_quiet_hours_keep_due_reminder_pending(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await self._setup_user(memory)
        record_id = await self._add_reminder(memory)

        with patch("cordbeat.agent.heartbeat._in_quiet_hours", return_value=True):
            await heartbeat._tick()

        mock_gateway.send_to_adapter.assert_not_awaited()
        assert (await self._metadata(memory, record_id))["status"] == "pending"

    async def test_naive_remind_at_is_treated_as_utc(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        user = await self._setup_user(memory)
        naive_due = (datetime.now(tz=UTC) - timedelta(minutes=1)).replace(
            tzinfo=None
        )
        record_id = await self._add_reminder(
            memory,
            remind_at=naive_due.isoformat(),
        )

        await heartbeat._deliver_due_reminders([user])

        mock_gateway.send_to_adapter.assert_awaited_once()
        assert (await self._metadata(memory, record_id))["status"] == "delivered"

    async def test_invalid_remind_at_is_disabled(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        user = await self._setup_user(memory)
        record_id = await self._add_reminder(memory, remind_at="not-a-date")

        await heartbeat._deliver_due_reminders([user])
        await heartbeat._deliver_due_reminders([user])

        mock_gateway.send_to_adapter.assert_not_awaited()
        assert (await self._metadata(memory, record_id))["status"] == "invalid"
        assert caplog.text.count("Invalid remind_at") == 1

    async def test_tick_attempts_at_most_five_due_reminders(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        user = await self._setup_user(memory)
        for index in range(7):
            await self._add_reminder(memory, content=f"Reminder {index}")

        await heartbeat._deliver_due_reminders([user])

        assert mock_gateway.send_to_adapter.await_count == 5
        records = await memory.get_certain_records("u1", record_type="reminder")
        statuses = [json.loads(record["metadata"])["status"] for record in records]
        assert statuses.count("delivered") == 5
        assert statuses.count("pending") == 2

    async def test_delivery_stops_after_five_failures(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        user = await self._setup_user(memory)
        record_id = await self._add_reminder(memory)
        mock_gateway.send_to_adapter.side_effect = RuntimeError("adapter down")

        for _ in range(5):
            await heartbeat._deliver_due_reminders([user])

        metadata = await self._metadata(memory, record_id)
        assert metadata["status"] == "failed"
        assert metadata["failure_count"] == 5

    async def test_disconnected_adapter_keeps_reminder_pending(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        user = await self._setup_user(memory)
        record_id = await self._add_reminder(memory)
        mock_gateway.send_to_adapter.return_value = False

        await heartbeat._deliver_due_reminders([user])

        metadata = await self._metadata(memory, record_id)
        assert metadata["status"] == "pending"
        assert metadata["failure_count"] == 1


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
        with patch("cordbeat.agent.heartbeat._in_quiet_hours", return_value=True):
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

        with patch("cordbeat.agent.heartbeat._in_quiet_hours", return_value=False):
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
    def _write_skill(
        self,
        skills_dir: Path,
        name: str,
        *,
        ownership: str = "ai",
        mutable_by_ai: bool = True,
        requires_approval_to_modify: bool = False,
    ) -> Path:
        skill_dir = skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "skill.yaml").write_text(
            "\n".join(
                [
                    f"name: {name}",
                    'description: "test"',
                    'version: "1.0.0"',
                    'author: "cordbeat-ai"',
                    f"ownership: {ownership}",
                    f"mutable_by_ai: {str(mutable_by_ai).lower()}",
                    "requires_approval_to_modify: "
                    f"{str(requires_approval_to_modify).lower()}",
                    "usage: test",
                    "parameters: []",
                    "safety:",
                    "  level: safe",
                    "  sandbox: true",
                    "  network: false",
                    "  filesystem: false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (skill_dir / "main.py").write_text(
            "def execute(**kw):\n    return {'version': 1}\n",
            encoding="utf-8",
        )
        return skill_dir

    async def test_safe_skill_executes(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        """Safe enabled skill runs successfully."""
        from types import SimpleNamespace

        await memory.get_or_create_user("u1", "Alice")
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
            target_user_id="u1",
            target_adapter_id="discord",
        )
        # Should not raise
        await heartbeat._execute_decision(decision)

        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_result"
        )
        assert len(records) == 1
        assert json.loads(records[0]["content"]) == {"result": "done"}
        metadata = json.loads(records[0]["metadata"])
        assert metadata["skill_name"] == "safe_skill"
        assert metadata["target_adapter_id"] == "discord"

    async def test_skill_execution_failure_handled(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        """Skill that raises is caught and logged."""
        from types import SimpleNamespace

        await memory.get_or_create_user("u1", "Alice")

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
            target_user_id="u1",
        )
        # Should not raise
        await heartbeat._execute_decision(decision)

        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_error"
        )
        assert len(records) == 1
        assert json.loads(records[0]["content"]) == {
            "error": "RuntimeError: skill error"
        }

    async def test_virtual_read_skill_file_records_result(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Heartbeat can inspect an installed skill source directly."""
        await memory.get_or_create_user("u1", "Alice")
        self._write_skill(heartbeat._skills.skills_dir, "repairable")

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="read_skill_file",
            skill_params={"skill_name": "repairable", "path": "main.py"},
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_result"
        )
        assert len(records) == 1
        payload = json.loads(records[0]["content"])
        assert payload["skill_name"] == "repairable"
        assert "return {'version': 1}" in payload["content"]

    async def test_virtual_update_ai_owned_skill_file_runs_without_approval(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """AI-owned mutable skill files can be repaired during heartbeat."""
        await memory.get_or_create_user("u1", "Alice")
        skill_dir = self._write_skill(heartbeat._skills.skills_dir, "repairable")

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="update_skill_file",
            skill_params={
                "skill_name": "repairable",
                "path": "main.py",
                "content": "def execute(**kw):\n    return {'version': 2}\n",
            },
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        assert "version': 2" in (skill_dir / "main.py").read_text(encoding="utf-8")
        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_result"
        )
        assert len(records) == 1
        assert json.loads(records[0]["content"])["status"] == "ok"
        assert not mock_gateway.send_to_adapter.await_args_list

    async def test_virtual_update_skill_file_rejects_invalid_python_without_writing(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Invalid self-repair source is recorded as an error and not written."""
        await memory.get_or_create_user("u1", "Alice")
        skill_dir = self._write_skill(heartbeat._skills.skills_dir, "repairable")
        heartbeat._skills.load_all()
        original = (skill_dir / "main.py").read_text(encoding="utf-8")

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="update_skill_file",
            skill_params={
                "skill_name": "repairable",
                "path": "main.py",
                "content": "def execute(**kw)\n    return {'version': 2}\n",
            },
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        assert (skill_dir / "main.py").read_text(encoding="utf-8") == original
        assert heartbeat._skills.get("repairable") is not None
        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_error"
        )
        assert len(records) == 1
        payload = json.loads(records[0]["content"])
        assert payload["error"] == "validation_failed"

    async def test_virtual_update_locked_skill_file_requests_approval(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Locked/user/system skill files become approval proposals."""
        await memory.get_or_create_user("u1", "Alice")
        self._write_skill(
            heartbeat._skills.skills_dir,
            "locked",
            ownership="system",
            mutable_by_ai=False,
            requires_approval_to_modify=True,
        )

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="update_skill_file",
            skill_params={
                "skill_name": "locked",
                "path": "main.py",
                "content": "def execute(**kw):\n    return {'version': 2}\n",
            },
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        proposals = await memory.get_certain_records("u1", record_type="proposal")
        assert len(proposals) == 1
        meta = json.loads(proposals[0]["metadata"])
        assert meta["proposal_type"] == ProposalType.SKILL_EXECUTION
        assert meta["skill_name"] == "update_skill_file"

    async def test_virtual_delete_ai_owned_skill_file_runs_without_approval(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """AI-owned mutable skill files can be cleaned up during heartbeat."""
        await memory.get_or_create_user("u1", "Alice")
        skill_dir = self._write_skill(heartbeat._skills.skills_dir, "repairable")
        (skill_dir / "scratch.txt").write_text("obsolete", encoding="utf-8")

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="delete_skill_file",
            skill_params={"skill_name": "repairable", "path": "scratch.txt"},
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        assert not (skill_dir / "scratch.txt").exists()
        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_result"
        )
        assert len(records) == 1
        assert json.loads(records[0]["content"])["status"] == "ok"
        assert not mock_gateway.send_to_adapter.await_args_list

    async def test_virtual_delete_locked_skill_file_requests_approval(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Locked skill file deletion becomes an approval proposal."""
        await memory.get_or_create_user("u1", "Alice")
        skill_dir = self._write_skill(
            heartbeat._skills.skills_dir,
            "locked",
            ownership="system",
            mutable_by_ai=False,
            requires_approval_to_modify=True,
        )
        (skill_dir / "scratch.txt").write_text("keep", encoding="utf-8")

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="delete_skill_file",
            skill_params={"skill_name": "locked", "path": "scratch.txt"},
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        assert (skill_dir / "scratch.txt").exists()
        proposals = await memory.get_certain_records("u1", record_type="proposal")
        assert len(proposals) == 1
        meta = json.loads(proposals[0]["metadata"])
        assert meta["proposal_type"] == ProposalType.SKILL_EXECUTION
        assert meta["skill_name"] == "delete_skill_file"

    async def test_virtual_delete_ai_owned_skill_runs_without_approval(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """AI-owned mutable skills can be removed during heartbeat."""
        await memory.get_or_create_user("u1", "Alice")
        skill_dir = self._write_skill(heartbeat._skills.skills_dir, "repairable")

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="delete_skill",
            skill_params={"skill_name": "repairable"},
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        assert not skill_dir.exists()
        assert "repairable" not in heartbeat._skills.available_skills
        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_result"
        )
        assert len(records) == 1
        assert json.loads(records[0]["content"])["status"] == "ok"
        assert not mock_gateway.send_to_adapter.await_args_list

    async def test_virtual_delete_locked_skill_requests_approval(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Locked skill deletion becomes an approval proposal."""
        await memory.get_or_create_user("u1", "Alice")
        skill_dir = self._write_skill(
            heartbeat._skills.skills_dir,
            "locked",
            ownership="system",
            mutable_by_ai=False,
            requires_approval_to_modify=True,
        )

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="delete_skill",
            skill_params={"skill_name": "locked"},
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        assert skill_dir.exists()
        proposals = await memory.get_certain_records("u1", record_type="proposal")
        assert len(proposals) == 1
        meta = json.loads(proposals[0]["metadata"])
        assert meta["proposal_type"] == ProposalType.SKILL_EXECUTION
        assert meta["skill_name"] == "delete_skill"

    async def test_virtual_update_skill_settings_requests_approval(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Skill settings changes always go through approval."""
        await memory.get_or_create_user("u1", "Alice")
        self._write_skill(heartbeat._skills.skills_dir, "repairable")

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="update_skill_settings",
            skill_params={
                "skill_name": "repairable",
                "ownership": "user",
                "mutable_by_ai": False,
            },
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_decision(decision)

        proposals = await memory.get_certain_records("u1", record_type="proposal")
        assert len(proposals) == 1
        meta = json.loads(proposals[0]["metadata"])
        assert meta["skill_name"] == "update_skill_settings"


# ── Sleep phase error handling ────────────────────────────────────────


class TestDiscoverySharing:
    @staticmethod
    def _register_skill(skills: SkillRegistry, result: dict[str, Any]) -> None:
        skills._skills["discover"] = Skill(
            meta=SkillMeta(
                name="discover",
                description="Find something",
                usage="discover",
                safety_level=SafetyLevel.SAFE,
            ),
            _test_callable=AsyncMock(return_value=result),
        )

    @staticmethod
    def _decision(user_id: str = "u1") -> HeartbeatDecision:
        return HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="discover",
            skill_params={"topic": "greetings"},
            target_user_id=user_id,
            target_adapter_id="discord",
        )

    async def test_successful_result_is_generated_sent_and_recorded(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("u1", "discord", "snowflake-1")
        await memory.add_message(
            "u1", "user", "PRIVATE HISTORY MUST NOT APPEAR", "discord"
        )
        result = {
            "title": "Why people greet each other",
            "finding": "Greetings can regulate social distance.",
        }
        self._register_skill(skills, result)
        mock_ai.generate.return_value = "This finding about greetings was neat."

        await heartbeat._execute_skill(self._decision())

        mock_ai.generate.assert_awaited_once()
        prompt = mock_ai.generate.await_args.kwargs["prompt"]
        system = mock_ai.generate.await_args.kwargs["system"]
        assert json.dumps(result, ensure_ascii=False) in prompt
        assert "PRIVATE HISTORY MUST NOT APPEAR" not in prompt
        assert "ONLY in the skill result" in system
        assert "Current emotion:" in system
        # Thinking models need budget headroom or every share hits the
        # content=null no-think retry (observed hourly in production logs).
        assert mock_ai.generate.await_args.kwargs["max_tokens"] >= 2048
        mock_gateway.send_to_adapter.assert_awaited_once()
        _, sent = mock_gateway.send_to_adapter.await_args.args
        assert sent.content == "This finding about greetings was neat."

        shares = await memory.get_certain_records(
            "u1", record_type="discovery_share_sent"
        )
        skill_results = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_result"
        )
        assert len(shares) == 1
        share_metadata = json.loads(shares[0]["metadata"])
        assert share_metadata["skill_name"] == "discover"
        assert share_metadata["source_record"] == skill_results[0]["id"]

    async def test_share_goes_to_the_routed_channel_not_the_last_seen_one(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        """The Layer-2 routing must survive into the share.

        Rebuilding the share without it fell back to last-seen, which posted
        research about one topic into whichever room the user spoke in last.
        """
        await memory.link_platform("u1", "discord", "snowflake-1")
        await memory.record_last_seen_channel("u1", "discord", "chan-last", False)
        self._register_skill(skills, {"finding": "a cool cafe"})
        mock_ai.generate.return_value = "Found a nice place."

        decision = self._decision()
        decision.history_channel_id = "chan-routed"
        decision.history_is_dm = False

        await heartbeat._execute_skill(decision)

        mock_gateway.send_to_adapter.assert_awaited_once()
        _, sent = mock_gateway.send_to_adapter.await_args.args
        assert sent.metadata["channel_id"] == "chan-routed"

    async def test_share_prompt_shows_the_destination_conversation(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        """The writer cannot judge fit without seeing the room it posts into."""
        await memory.link_platform("u1", "discord", "snowflake-1")
        await memory.add_message(
            "u1",
            "user",
            "How is my portfolio doing today?",
            "discord",
            channel_id="chan-invest",
            is_dm=False,
        )
        self._register_skill(skills, {"finding": "an unrelated cafe"})
        mock_ai.generate.return_value = "SKIP"

        decision = self._decision()
        decision.history_channel_id = "chan-invest"
        decision.history_is_dm = False

        await heartbeat._execute_skill(decision)

        prompt = mock_ai.generate.await_args.kwargs["prompt"]
        assert "[BEGIN DESTINATION" in prompt
        assert "public channel that other people also read" in prompt
        assert "How is my portfolio doing today?" in prompt
        system = mock_ai.generate.await_args.kwargs["system"]
        assert "does not belong in the destination" in system
        mock_gateway.send_to_adapter.assert_not_awaited()

    async def test_share_prompt_carries_persona_language(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        """Shares must repeat the language instruction or they come out in
        English (observed in production with a Japanese persona)."""
        await memory.link_platform("u1", "discord", "snowflake-1")
        self._register_skill(skills, {"finding": "neat"})
        mock_ai.generate.return_value = "SKIP"
        snapshot = dict(heartbeat._soul.get_soul_snapshot())
        snapshot["language"] = "ja"

        with patch.object(
            heartbeat._soul, "get_soul_snapshot", return_value=snapshot
        ):
            await heartbeat._execute_skill(self._decision())

        system = mock_ai.generate.await_args.kwargs["system"]
        assert "Write the message in ja." in system

    async def test_skip_response_does_not_send_or_record(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        self._register_skill(skills, {"finding": "small update"})
        mock_ai.generate.return_value = "  skip  "

        await heartbeat._execute_skill(self._decision())

        mock_gateway.send_to_adapter.assert_not_awaited()
        assert await memory.get_certain_records(
            "u1", record_type="discovery_share_sent"
        ) == []

    async def test_send_failure_is_not_retried_or_recorded_as_skill_error(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        self._register_skill(skills, {"finding": "grounded detail"})
        mock_ai.generate.return_value = "Worth sharing"
        mock_gateway.send_to_adapter.side_effect = RuntimeError("offline")

        await heartbeat._execute_skill(self._decision())

        mock_gateway.send_to_adapter.assert_awaited_once()
        assert await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_error"
        ) == []
        assert await memory.get_certain_records(
            "u1", record_type="discovery_share_sent"
        ) == []

    @pytest.mark.parametrize(
        ("result", "user_id", "daily_limit"),
        [
            ({"error": "failed"}, "u1", 3),
            ({"finding": "private maintenance"}, "__system__", 3),
            ({"finding": "disabled"}, "u1", 0),
        ],
    )
    async def test_ineligible_results_skip_generation(
        self,
        heartbeat: HeartbeatLoop,
        mock_ai: AsyncMock,
        result: dict[str, Any],
        user_id: str,
        daily_limit: int,
    ) -> None:
        heartbeat._config.max_discovery_shares_per_day = daily_limit

        await heartbeat._maybe_share_discovery(
            self._decision(user_id), result, "source-1"
        )

        mock_ai.generate.assert_not_awaited()

    async def test_discovery_bypasses_normal_proactive_cooldowns(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("u1", "discord", "snowflake-1")
        await memory.add_certain_record(
            "u1", "normal proactive", "heartbeat_user_sent"
        )
        await memory.add_certain_record(
            "__system__",
            "normal destination",
            "heartbeat_destination_sent",
            {"destination_key": "discord:user:snowflake-1"},
        )
        heartbeat._proactive_messages_sent_this_tick = 1
        share = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="Grounded finding",
            target_user_id="u1",
            target_adapter_id="discord",
            skill_name="discover",
            skill_params={"source_record": "source-1"},
        )

        sent = await heartbeat._send_heartbeat_message(
            share, discovery_share=True
        )

        assert sent is True
        mock_gateway.send_to_adapter.assert_awaited_once()

    async def test_discovery_cooldown_blocks_recent_share(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.add_certain_record(
            "u1", "recent discovery", "discovery_share_sent"
        )

        sent = await heartbeat._send_heartbeat_message(
            HeartbeatDecision(
                action=HeartbeatAction.MESSAGE,
                content="Another finding",
                target_user_id="u1",
                target_adapter_id="discord",
            ),
            discovery_share=True,
        )

        assert sent is False
        mock_gateway.send_to_adapter.assert_not_awaited()

    async def test_daily_utc_limit_blocks_fourth_share(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        heartbeat._config.discovery_share_cooldown_minutes = 0
        for index in range(3):
            await memory.add_certain_record(
                "u1", f"discovery {index}", "discovery_share_sent"
            )

        sent = await heartbeat._send_heartbeat_message(
            HeartbeatDecision(
                action=HeartbeatAction.MESSAGE,
                content="Fourth finding",
                target_user_id="u1",
                target_adapter_id="discord",
            ),
            discovery_share=True,
        )

        assert sent is False
        mock_gateway.send_to_adapter.assert_not_awaited()


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
    async def test_decision_runs_backend_call_in_internal_scope(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        from cordbeat.ai.backend import is_internal_context

        await memory.get_or_create_user("u1", "Alice")

        async def observe_scope(*_args: object, **_kwargs: object) -> dict[str, object]:
            assert is_internal_context() is True
            return {
                "action": "none",
                "content": "",
                "next_heartbeat_minutes": 60,
            }

        mock_ai.generate_json = AsyncMock(side_effect=observe_scope)

        await heartbeat._layer2_evaluate(
            UserSummary(user_id="u1", display_name="Alice"),
            "routine check",
        )

        assert is_internal_context() is False

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
        system = mock_ai.generate_json.await_args.kwargs["system"]
        assert "Do not revive a completed topic" in system
        assert "If uncertain, choose" in system
        assert "claim that you just performed an action" not in system
        assert "You may share a discovery with the user" in system
        assert "Never add details that are not in the recorded result" in system
        assert "a result worth sharing is" in system
        # action=skill must know where a share would land, or the share falls
        # back to wherever the user last spoke.
        assert "set it for action=skill too" in system

    async def test_evaluate_routes_message_to_chosen_candidate_channel(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """The decision may pick a non-primary destination whose topic fits."""
        await memory.get_or_create_user("u1", "Alice")
        base = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
        await memory.add_message(
            "u1",
            "user",
            "Planning curry for dinner tonight",
            "discord",
            "chan-b",
            False,
            created_at=base,
        )
        await memory.add_message(
            "u1",
            "user",
            "That boss fight was brutal",
            "discord",
            "chan-a",
            False,
            created_at=base + timedelta(minutes=5),
        )
        await memory.record_last_seen_channel("u1", "discord", "chan-a", False)
        mock_ai.generate_json = AsyncMock(
            return_value={
                "action": "message",
                "content": "Enjoy the curry tonight!",
                "target_adapter_id": "discord",
                "target_channel_id": "chan-b",
                "next_heartbeat_minutes": 30,
            }
        )
        user = UserSummary(
            user_id="u1", display_name="Alice", last_platform="discord"
        )

        decision = await heartbeat._layer2_evaluate(user, "dinner follow-up")

        assert decision.history_channel_id == "chan-b"
        assert decision.history_is_dm is False
        prompt = mock_ai.generate_json.await_args.args[0]
        assert "[BEGIN CANDIDATE DESTINATIONS" in prompt
        assert "chan-b" in prompt
        assert "Planning curry for dinner tonight" in prompt
        system = mock_ai.generate_json.await_args.kwargs["system"]
        assert "target_channel_id" in system

    async def test_public_destination_history_shows_the_whole_room(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """The destination rule asks the draft to continue the channel's own
        conversation, so the draft has to be shown that conversation — not
        just this user's thread inside it."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.get_or_create_user("u2", "Bob")
        base = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
        await memory.add_message(
            "u1", "user", "alice mentions the festival", "discord",
            "chan-a", False, created_at=base,
        )
        await memory.add_message(
            "u2", "user", "bob asks about the boss fight", "discord",
            "chan-a", False, created_at=base + timedelta(minutes=1),
        )
        await memory.record_last_seen_channel("u1", "discord", "chan-a", False)
        user = UserSummary(
            user_id="u1", display_name="Alice", last_platform="discord"
        )

        await heartbeat._layer2_evaluate(user, "check in")

        prompt = mock_ai.generate_json.await_args.args[0]
        assert "bob asks about the boss fight" in prompt
        assert "Bob: bob asks about the boss fight" in prompt
        assert "Alice: alice mentions the festival" in prompt

    async def test_quiet_user_is_not_crowded_out_of_the_room_history(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """The draft is about one user, so busy participants must not be able
        to push that user's own turns out of the loaded window."""
        heartbeat._memory_config.conversation_history_limit = 4
        await memory.get_or_create_user("u1", "Alice")
        await memory.get_or_create_user("u2", "Bob")
        base = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
        await memory.add_message(
            "u1", "user", "alice said this days ago", "discord",
            "chan-a", False, created_at=base,
        )
        for i in range(6):
            await memory.add_message(
                "u2", "user", f"bob chatters {i}", "discord",
                "chan-a", False, created_at=base + timedelta(minutes=i + 1),
            )
        await memory.record_last_seen_channel("u1", "discord", "chan-a", False)
        user = UserSummary(
            user_id="u1", display_name="Alice", last_platform="discord"
        )

        await heartbeat._layer2_evaluate(user, "check in")

        prompt = mock_ai.generate_json.await_args.args[0]
        assert "bob chatters 5" in prompt
        assert "alice said this days ago" in prompt

    async def test_public_destination_rule_pins_the_addressee(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Showing the whole room risks drafting a reply to the wrong person,
        so the rule has to name who the message is for."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_message(
            "u1", "user", "hello", "discord", "chan-a", False
        )
        await memory.record_last_seen_channel("u1", "discord", "chan-a", False)
        user = UserSummary(
            user_id="u1", display_name="Alice", last_platform="discord"
        )

        await heartbeat._layer2_evaluate(user, "check in")

        system = mock_ai.generate_json.await_args.kwargs["system"]
        assert "You are writing to Alice, and only to them" in system
        assert "do not answer a question another participant asked" in system

    async def test_dm_destination_history_stays_private(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Widening must never reach into a DM: another user's messages in the
        same DM id are not part of this user's conversation."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.get_or_create_user("u2", "Bob")
        await memory.add_message(
            "u1", "user", "alice private plans", "discord", "dm-1", True
        )
        await memory.add_message(
            "u2", "user", "bob private secret", "discord", "dm-1", True
        )
        await memory.record_last_seen_channel("u1", "discord", "dm-1", True)
        user = UserSummary(
            user_id="u1", display_name="Alice", last_platform="discord"
        )

        await heartbeat._layer2_evaluate(user, "check in")

        prompt = mock_ai.generate_json.await_args.args[0]
        assert "alice private plans" in prompt
        assert "bob private secret" not in prompt

    async def test_candidate_snippet_shows_other_participants(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Snippets decide whether a topic fits a channel, so they must
        reflect what the channel is actually talking about.

        Candidacy itself stays per-user: a channel only qualifies because
        this user is active there, which is why Alice speaks in chan-b too.
        """
        await memory.get_or_create_user("u1", "Alice")
        await memory.get_or_create_user("u2", "Bob")
        base = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
        await memory.add_message(
            "u1", "user", "alice says hi", "discord",
            "chan-b", False, created_at=base,
        )
        await memory.add_message(
            "u2", "user", "bob is planning curry", "discord",
            "chan-b", False, created_at=base + timedelta(minutes=1),
        )
        await memory.add_message(
            "u1", "user", "alice on the boss fight", "discord",
            "chan-a", False, created_at=base + timedelta(minutes=5),
        )
        await memory.record_last_seen_channel("u1", "discord", "chan-a", False)
        user = UserSummary(
            user_id="u1", display_name="Alice", last_platform="discord"
        )

        await heartbeat._layer2_evaluate(user, "check in")

        prompt = mock_ai.generate_json.await_args.args[0]
        assert "[BEGIN CANDIDATE DESTINATIONS" in prompt
        assert "Bob: bob is planning curry" in prompt

    async def test_dm_primary_excludes_public_channels_from_candidates(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """A DM-drafted message must not be redirectable to a public channel."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_message(
            "u1", "user", "private plans", "discord", "dm-1", True
        )
        await memory.add_message(
            "u1", "user", "public chatter", "discord", "guild-1", False
        )
        await memory.record_last_seen_channel("u1", "discord", "dm-1", True)
        mock_ai.generate_json = AsyncMock(
            return_value={
                "action": "message",
                "content": "hi",
                "target_adapter_id": "discord",
                "target_channel_id": "guild-1",
                "next_heartbeat_minutes": 30,
            }
        )
        user = UserSummary(
            user_id="u1", display_name="Alice", last_platform="discord"
        )

        decision = await heartbeat._layer2_evaluate(user, "check in")

        # guild-1 is not an eligible candidate, so routing stays on the DM.
        assert decision.history_channel_id == "dm-1"
        assert decision.history_is_dm is True
        prompt = mock_ai.generate_json.await_args.args[0]
        assert "public chatter" not in prompt

    async def test_evaluate_ignores_unknown_candidate_channel(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """A hallucinated channel id falls back to the primary destination."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_message(
            "u1", "user", "hello", "discord", "chan-a", False
        )
        await memory.record_last_seen_channel("u1", "discord", "chan-a", False)
        mock_ai.generate_json = AsyncMock(
            return_value={
                "action": "message",
                "content": "hi",
                "target_adapter_id": "discord",
                "target_channel_id": "chan-made-up",
                "next_heartbeat_minutes": 30,
            }
        )
        user = UserSummary(
            user_id="u1", display_name="Alice", last_platform="discord"
        )

        decision = await heartbeat._layer2_evaluate(user, "check in")

        assert decision.history_channel_id == "chan-a"
        assert decision.history_is_dm is False

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

    async def test_evaluate_scopes_history_to_last_seen_guild_channel(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Guild-targeted heartbeat context must not include DM history."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_message(
            "u1",
            "user",
            "private DM detail about a medical appointment",
            "discord",
            "dm-1",
            True,
        )
        await memory.add_message(
            "u1",
            "user",
            "public guild topic about release planning",
            "discord",
            "guild-1",
            False,
        )
        await memory.record_last_seen_channel("u1", "discord", "guild-1", False)
        mock_ai.generate_json = AsyncMock(
            return_value={
                "action": "none",
                "content": "",
                "next_heartbeat_minutes": 60,
            }
        )

        await heartbeat._layer2_evaluate(
            UserSummary(
                user_id="u1",
                display_name="Alice",
                last_platform="discord",
            ),
            "routine check",
        )

        prompt = mock_ai.generate_json.await_args.args[0]
        assert "public guild topic about release planning" in prompt
        assert "private DM detail about a medical appointment" not in prompt

    async def test_evaluate_includes_private_continuity_context(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Layer 2 receives recent private notes for interest decay."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_certain_record(
            "u1",
            "Already checked the Smash topic twice without new evidence.",
            "heartbeat_reflection",
        )
        await memory.add_certain_record(
            "u1",
            "Do not over-contact about stale game rumors.",
            "heartbeat_concern",
        )
        await memory.add_certain_record(
            "u1",
            '{"skill_name": "web_search", "output": "found release notes"}',
            "conversation_skill_result",
            {
                "source": "conversation",
                "skill_name": "web_search",
                "outcome": "result",
            },
        )
        mock_ai.generate_json = AsyncMock(
            return_value={
                "action": "none",
                "content": "",
                "reflection": "The old topic is stale, so I will wait.",
                "next_heartbeat_minutes": 120,
            }
        )

        await heartbeat._layer2_evaluate(
            UserSummary(user_id="u1", display_name="Alice"),
            "stale topic",
        )

        prompt = mock_ai.generate_json.await_args.args[0]
        assert "Private HEARTBEAT continuity" in prompt
        assert "Already checked the Smash topic twice" in prompt
        assert "Verified conversation tool actions" in prompt
        assert "found release notes" in prompt
        assert "Use this to decay stale interests" in prompt
        # The heartbeat does not load the full ledger for build_context, so it
        # must not emit the ledger section (an empty one would falsely claim
        # that no tools ran).
        assert "[BEGIN VERIFIED ACTIONS" not in prompt
        system = mock_ai.generate_json.await_args.kwargs["system"]
        assert "Operational honesty" in system
        assert "verified external actions" in system
        assert "verified tool-action records" in system


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
                "reflection": "Alice has been quiet, so I will reconnect gently.",
                "concerns": ["Avoid repeating old greetings without new context."],
                "target_user_id": "u1",
                "target_adapter_id": "discord",
                "next_heartbeat_minutes": 30,
            }

        mock_ai.generate_json = AsyncMock(side_effect=mock_generate_json)

        with patch("cordbeat.agent.heartbeat._in_quiet_hours", return_value=False):
            result = await heartbeat._tick()

        # 2 AI calls: triage + evaluate
        assert mock_ai.generate_json.call_count == 2
        # Message was sent
        mock_gateway.send_to_adapter.assert_called_once()
        # Min of triage(45) and decision(30)
        assert result == 30
        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_reflection"
        )
        assert len(records) == 1
        assert records[0]["content"] == (
            "Alice has been quiet, so I will reconnect gently."
        )
        metadata = json.loads(records[0]["metadata"])
        assert metadata["action"] == "message"
        assert metadata["triage_reason"] == "hasn't talked recently"
        concerns = await memory.get_certain_records(
            "u1", record_type="heartbeat_concern"
        )
        assert len(concerns) == 1
        assert concerns[0]["content"] == (
            "Avoid repeating old greetings without new context."
        )
        journals = await memory.get_certain_records(
            "__system__", record_type="heartbeat_journal"
        )
        assert len(journals) == 1

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

        with patch("cordbeat.agent.heartbeat._in_quiet_hours", return_value=False):
            result = await heartbeat._tick()

        assert mock_ai.generate_json.call_count == 1
        mock_gateway.send_to_adapter.assert_not_called()
        assert result == 60

    async def test_layer2_records_reflection_for_none_action(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """A quiet heartbeat can still leave a private continuity note."""
        await memory.get_or_create_user("u1", "Alice")
        heartbeat._layer2_evaluate = AsyncMock(
            return_value=HeartbeatDecision(
                action=HeartbeatAction.NONE,
                reflection="Alice seems settled; I will stay quietly available.",
                target_user_id="u1",
                target_adapter_id="discord",
                next_heartbeat_minutes=120,
            )
        )

        result = await heartbeat._run_layer2(
            [UserSummary(user_id="u1", display_name="Alice")],
            [{"user_id": "u1", "reason": "routine check"}],
            60,
        )

        assert result == 60
        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_reflection"
        )
        assert len(records) == 1
        assert records[0]["content"] == (
            "Alice seems settled; I will stay quietly available."
        )
        metadata = json.loads(records[0]["metadata"])
        assert metadata["action"] == "none"

    async def test_self_review_records_private_summary(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Periodic self-review summarizes recent private notes."""
        heartbeat._config.self_review_interval_ticks = 1
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_certain_record(
            "u1",
            "Alice has had repeated stale topic check-ins.",
            "heartbeat_reflection",
        )
        mock_ai.generate = AsyncMock(
            return_value="I should let stale topics cool unless new evidence appears."
        )

        await heartbeat._maybe_run_self_review(
            [UserSummary(user_id="u1", display_name="Alice")]
        )

        records = await memory.get_certain_records(
            "__system__", record_type="heartbeat_self_review"
        )
        assert len(records) == 1
        assert "stale topics cool" in records[0]["content"]

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

        with patch("cordbeat.agent.heartbeat._in_quiet_hours", return_value=False):
            result = await heartbeat._tick()

        # Only triage call, no Layer 2
        assert mock_ai.generate_json.call_count == 1
        assert result == 60

    async def test_private_skill_runs_during_message_cooldown(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Message cooldowns should not block private heartbeat work."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.add_certain_record(
            "u1",
            "recent proactive message",
            "heartbeat_user_sent",
            {},
        )
        calls: list[dict[str, object]] = []

        def safe_execute(**kw: object) -> dict[str, str]:
            calls.append(kw)
            return {"result": "checked"}

        meta = SkillMeta(
            name="safe_skill",
            description="Safe maintenance skill",
            usage="",
            safety_level=SafetyLevel.SAFE,
            enabled=True,
        )
        skills._skills["safe_skill"] = Skill(meta=meta, _test_callable=safe_execute)
        heartbeat._layer2_evaluate = AsyncMock(
            return_value=HeartbeatDecision(
                action=HeartbeatAction.SKILL,
                skill_name="safe_skill",
                skill_params={},
                target_user_id="u1",
                target_adapter_id="discord",
                next_heartbeat_minutes=30,
            )
        )

        result = await heartbeat._run_layer2(
            [UserSummary(user_id="u1", display_name="Alice")],
            [{"user_id": "u1", "reason": "private maintenance"}],
            60,
        )

        assert result == 30
        assert len(calls) == 1
        mock_gateway.send_to_adapter.assert_not_called()
        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_result"
        )
        assert len(records) == 1
        assert json.loads(records[0]["content"]) == {"result": "checked"}

    async def test_private_skill_runs_when_message_tick_limit_reached(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """The per-tick message limit should only limit outgoing messages."""
        await memory.get_or_create_user("u1", "Alice")
        heartbeat._proactive_messages_sent_this_tick = (
            heartbeat._config.max_proactive_messages_per_tick
        )
        calls: list[dict[str, object]] = []

        def safe_execute(**kw: object) -> dict[str, str]:
            calls.append(kw)
            return {"result": "reflected"}

        meta = SkillMeta(
            name="reflection_skill",
            description="Safe reflection skill",
            usage="",
            safety_level=SafetyLevel.SAFE,
            enabled=True,
        )
        skills._skills["reflection_skill"] = Skill(
            meta=meta, _test_callable=safe_execute
        )
        heartbeat._layer2_evaluate = AsyncMock(
            return_value=HeartbeatDecision(
                action=HeartbeatAction.SKILL,
                skill_name="reflection_skill",
                skill_params={},
                target_user_id="u1",
                target_adapter_id="discord",
                next_heartbeat_minutes=25,
            )
        )

        result = await heartbeat._run_layer2(
            [UserSummary(user_id="u1", display_name="Alice")],
            [{"user_id": "u1", "reason": "self review"}],
            60,
        )

        assert result == 25
        assert len(calls) == 1
        mock_gateway.send_to_adapter.assert_not_called()
        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_result"
        )
        assert len(records) == 1
        assert json.loads(records[0]["content"]) == {"result": "reflected"}

    async def test_action_budget_limits_layer2_actions(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        """Heartbeat consumes the shared ActionBudget per executed decision."""
        heartbeat._config.max_actions_per_tick = 1
        await memory.get_or_create_user("u1", "Alice")
        await memory.get_or_create_user("u2", "Bob")
        calls: list[str] = []

        def safe_execute(user_id: str = "", **kw: object) -> dict[str, str]:
            calls.append(user_id)
            return {"result": user_id}

        skills._skills["safe_skill"] = Skill(
            meta=SkillMeta(
                name="safe_skill",
                description="Safe",
                usage="",
                safety_level=SafetyLevel.SAFE,
                enabled=True,
                parameters=[SkillParam(name="user_id", type="string")],
            ),
            _test_callable=safe_execute,
        )
        heartbeat._layer2_evaluate = AsyncMock(
            side_effect=[
                HeartbeatDecision(
                    action=HeartbeatAction.SKILL,
                    skill_name="safe_skill",
                    skill_params={},
                    target_user_id="u1",
                    target_adapter_id="discord",
                    next_heartbeat_minutes=30,
                ),
                HeartbeatDecision(
                    action=HeartbeatAction.SKILL,
                    skill_name="safe_skill",
                    skill_params={},
                    target_user_id="u2",
                    target_adapter_id="discord",
                    next_heartbeat_minutes=30,
                ),
            ]
        )

        await heartbeat._run_layer2(
            [
                UserSummary(user_id="u1", display_name="Alice"),
                UserSummary(user_id="u2", display_name="Bob"),
            ],
            [
                {"user_id": "u1", "reason": "check"},
                {"user_id": "u2", "reason": "check"},
            ],
            60,
        )

        assert calls == ["u1"]


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

    async def test_duplicate_skill_proposal_reuses_pending_record(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Heartbeat skill proposals are deduplicated while pending."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="deploy",
            skill_params={"target": "prod"},
            target_user_id="u1",
            target_adapter_id="discord",
        )

        first = await heartbeat._proposals.store_skill_proposal(decision, "deploy")
        second = await heartbeat._proposals.store_skill_proposal(decision, "deploy")

        assert second == first
        records = await memory.get_certain_records("u1", record_type="proposal")
        assert len(records) == 1
        mock_gateway.send_to_adapter.assert_awaited_once()

    async def test_skill_proposal_content_is_bounded_and_redacted(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Record content stays display-sized; execution params live in metadata."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
        big_source = "x" * 10_000
        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="update_skill_file",
            skill_params={"path": "main.py", "content": big_source},
            target_user_id="u1",
            target_adapter_id="discord",
        )

        await heartbeat._proposals.store_skill_proposal(
            decision, "update_skill_file"
        )

        records = await memory.get_certain_records("u1", record_type="proposal")
        assert len(records) == 1
        content = records[0]["content"]
        assert big_source not in content
        assert "<redacted>" in content
        assert len(content) < 500
        meta = json.loads(records[0]["metadata"])
        assert meta["skill_params"]["content"] == big_source

    async def test_sandbox_local_file_skill_executes_without_proposal(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Sandbox-relative file operations should be free during heartbeat."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
        calls: list[tuple[str, bool]] = []

        async def search_files(root: str, context: object) -> dict[str, object]:
            calls.append((root, bool(getattr(context, "filesystem"))))
            return {"root": root, "returned": 0}

        skill = Skill(
            meta=SkillMeta(
                name="file_search",
                description="Search files",
                usage="search",
                safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
                filesystem=True,
            ),
            _test_callable=search_files,
        )
        skills._skills["file_search"] = skill

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="file_search",
            skill_params={"root": "."},
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_skill(decision)

        assert calls == [(".", False)]
        mock_gateway.send_to_adapter.assert_not_called()
        proposals = await memory.get_certain_records("u1", record_type="proposal")
        assert proposals == []
        records = await memory.get_certain_records(
            "u1", record_type="heartbeat_skill_result"
        )
        assert len(records) == 1

    async def test_skill_execution_log_is_bounded_and_redacted(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """INFO logs never carry full params or full skill output."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
        big = "z" * 5000

        async def search_files(
            root: str, query: str, context: object
        ) -> dict[str, object]:
            return {"root": root, "returned": 0, "payload": big}

        skill = Skill(
            meta=SkillMeta(
                name="file_search",
                description="Search files",
                usage="search",
                safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
                filesystem=True,
            ),
            _test_callable=search_files,
        )
        skills._skills["file_search"] = skill

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="file_search",
            skill_params={"root": ".", "query": big},
            target_user_id="u1",
            target_adapter_id="discord",
        )
        with caplog.at_level(logging.INFO):
            await heartbeat._execute_skill(decision)

        assert "HEARTBEAT skill executed" in caplog.text
        assert big not in caplog.text

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
        assert msg.type == MessageType.SKILL_CONFIRM
        assert "cleanup" in msg.content
        assert "proposal ID:" in msg.content
        assert msg.metadata["skill_name"] == "cleanup"

    async def test_skill_proposal_notification_hides_bulk_params(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """The approval message must not dump raw file bodies (production
        incident: an update_skill_file proposal pasted the whole main.py
        into the Discord approval message)."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
        big_source = "x" * 8000

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="update_skill_file",
            skill_params={"skill_name": "draw", "content": big_source},
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._proposals.store_skill_proposal(
            decision, "update_skill_file"
        )

        msg = mock_gateway.send_to_adapter.call_args[0][1]
        assert big_source not in msg.content
        assert "<redacted>" in msg.content
        assert len(msg.content) < 600

    async def test_draw_prompt_text_is_not_stored_as_skill_proposal(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Heartbeat never proposes/executes draw skill autonomously."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        skill = Skill(
            meta=SkillMeta(
                name="draw",
                description="Draw",
                usage="draw",
                safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
            ),
            _test_callable=lambda **kw: {"ok": True},
        )
        skills._skills["draw"] = skill

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="draw",
            skill_params={
                "commands": ("DRAW: a moonlit winter forest with falling snow")
            },
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_skill(decision)

        records = await memory.get_certain_records("u1", record_type="proposal")
        assert records == []
        mock_gateway.send_to_adapter.assert_not_called()

    async def test_draw_skill_is_not_executed_even_with_dsl(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
        execute = AsyncMock(return_value={"output": "image"})
        skills._skills["draw"] = Skill(
            meta=SkillMeta(
                name="draw",
                description="Draw",
                usage="draw",
                safety_level=SafetyLevel.SAFE,
            ),
            _test_callable=execute,
        )

        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_name="draw",
            skill_params={
                "commands": "SIZE 64 64\nCANVAS white\nCIRCLE 32 32 20 red\nOUTPUT"
            },
            target_user_id="u1",
            target_adapter_id="discord",
        )
        await heartbeat._execute_skill(decision)

        execute.assert_not_awaited()
        mock_gateway.send_to_adapter.assert_not_called()


class TestApprovedProposalExecution:
    """Tests for executing approved proposals on heartbeat tick."""

    def _write_skill(self, skills_dir: Path, name: str) -> Path:
        skill_dir = skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "skill.yaml").write_text(
            "\n".join(
                [
                    f"name: {name}",
                    'description: "test"',
                    'version: "1.0.0"',
                    'author: "cordbeat-ai"',
                    "ownership: ai",
                    "mutable_by_ai: true",
                    "requires_approval_to_modify: false",
                    "usage: test",
                    "parameters: []",
                    "safety:",
                    "  level: safe",
                    "  sandbox: true",
                    "  network: false",
                    "  filesystem: false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (skill_dir / "main.py").write_text(
            "def execute(**kw):\n    return {'version': 1}\n",
            encoding="utf-8",
        )
        return skill_dir

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

        # The outcome is recorded so the agent can see its own approved work.
        outcomes = await memory.get_certain_records(
            "u1", record_type="proposal_skill_result"
        )
        assert len(outcomes) == 1
        outcome_meta = json.loads(outcomes[0]["metadata"])
        assert outcome_meta["skill_name"] == "test_skill"
        assert outcome_meta["proposal_id"] == proposal_id

    async def test_rejected_skill_file_update_records_error_outcome(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A validator-rejected update is logged and visible to the agent.

        Reproduces the production incident where an approved draw-skill
        rewrite failed AST validation and expired without any log or memory
        trace, leaving both the user and the agent unaware.
        """
        import shutil

        builtin_timer = Path(__file__).parent.parent / "skills" / "timer"
        shutil.copytree(builtin_timer, skills.skills_dir / "timer")
        skills.load_all()

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Update timer skill",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "update_skill_file",
                "skill_params": {
                    "skill_name": "timer",
                    "path": "main.py",
                    "content": "import os\n\ndef execute(**kw):\n    return {}\n",
                },
            },
        )

        with caplog.at_level(logging.WARNING):
            await heartbeat._proposals.execute_approved()

        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        assert json.loads(proposal["metadata"])["status"] == ProposalStatus.EXPIRED
        assert "Approved skill file update rejected" in caplog.text

        errors = await memory.get_certain_records(
            "u1", record_type="proposal_skill_error"
        )
        assert len(errors) == 1
        assert "validation_failed" in errors[0]["content"]

    async def test_execute_approved_skips_corrupt_metadata_and_continues(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        execute_fn = MagicMock(return_value={"ok": True})
        skill = Skill(
            meta=SkillMeta(
                name="valid_after_corrupt",
                description="Test",
                usage="test",
                safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
            ),
            _test_callable=execute_fn,
        )
        skills._skills["valid_after_corrupt"] = skill

        corrupt_id = await memory.add_certain_record(
            user_id="u1",
            content="Corrupt proposal",
            record_type="proposal",
            metadata={"status": ProposalStatus.APPROVED},
        )
        await memory._conn.execute(
            "UPDATE certain_records SET metadata = ? WHERE id = ?",
            ('{"status":"approved"', corrupt_id),
        )
        valid_id = await memory.add_certain_record(
            user_id="u1",
            content="Run valid_after_corrupt",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "valid_after_corrupt",
                "skill_params": {},
            },
        )
        await memory._conn.commit()

        await heartbeat._proposals.execute_approved()

        execute_fn.assert_called_once_with()
        corrupt = await memory.get_proposal(corrupt_id)
        valid = await memory.get_proposal(valid_id)
        assert corrupt is not None
        assert valid is not None
        assert json.loads(corrupt["metadata"])["status"] == ProposalStatus.EXPIRED
        assert json.loads(valid["metadata"])["status"] == ProposalStatus.EXECUTED

    async def test_execute_approved_claims_before_running_skill(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        async def execute_once(**_kwargs: object) -> dict[str, bool]:
            await asyncio.sleep(0.01)
            return {"ok": True}

        execute_fn = AsyncMock(side_effect=execute_once)
        skill = Skill(
            meta=SkillMeta(
                name="dangerous_once",
                description="Test",
                usage="test",
                safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
            ),
            _test_callable=execute_fn,
        )
        skills._skills["dangerous_once"] = skill

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Run dangerous_once",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "dangerous_once",
                "skill_params": {},
            },
        )

        await asyncio.gather(
            heartbeat._proposals.execute_approved(proposal_id=proposal_id),
            heartbeat._proposals.execute_approved(proposal_id=proposal_id),
        )

        execute_fn.assert_awaited_once()
        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        assert json.loads(proposal["metadata"])["status"] == ProposalStatus.EXECUTED

    async def test_approved_sandbox_local_file_skill_uses_sandbox_override(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        """Approved relative file tool calls should still use sandbox guards."""
        calls: list[tuple[str, bool]] = []

        async def read_file(path: str, context: object) -> dict[str, str]:
            calls.append((path, bool(getattr(context, "filesystem"))))
            return {"content": "sandbox note"}

        skill = Skill(
            meta=SkillMeta(
                name="file_read",
                description="Read files",
                usage="read",
                safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
                filesystem=True,
            ),
            _test_callable=read_file,
        )
        skills._skills["file_read"] = skill

        await memory.add_certain_record(
            user_id="u1",
            content="Run file_read",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "file_read",
                "skill_params": {"path": "notes.md"},
            },
        )

        await heartbeat._proposals.execute_approved()

        assert calls == [("notes.md", False)]

    async def test_approved_virtual_skill_file_update_executes(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Approved virtual skill file updates are dispatched by proposal executor."""
        skill_dir = self._write_skill(heartbeat._skills.skills_dir, "repairable")

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Update skill file",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "update_skill_file",
                "skill_params": {
                    "skill_name": "repairable",
                    "path": "main.py",
                    "content": "def execute(**kw):\n    return {'version': 2}\n",
                },
            },
        )

        await heartbeat._proposals.execute_approved()

        assert "version': 2" in (skill_dir / "main.py").read_text(encoding="utf-8")
        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.EXECUTED

    async def test_approved_virtual_skill_file_delete_executes(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Approved virtual skill file deletion is dispatched by proposals."""
        skill_dir = self._write_skill(heartbeat._skills.skills_dir, "repairable")
        (skill_dir / "scratch.txt").write_text("obsolete", encoding="utf-8")

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Delete skill file",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "delete_skill_file",
                "skill_params": {
                    "skill_name": "repairable",
                    "path": "scratch.txt",
                },
            },
        )

        await heartbeat._proposals.execute_approved()

        assert not (skill_dir / "scratch.txt").exists()
        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.EXECUTED

    async def test_approved_virtual_skill_delete_executes(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Approved virtual skill deletion is dispatched by proposals."""
        skill_dir = self._write_skill(heartbeat._skills.skills_dir, "repairable")

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Delete skill",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "delete_skill",
                "skill_params": {"skill_name": "repairable"},
            },
        )

        await heartbeat._proposals.execute_approved()

        assert not skill_dir.exists()
        assert "repairable" not in heartbeat._skills.available_skills
        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.EXECUTED

    async def test_approved_virtual_skill_settings_update_executes(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
    ) -> None:
        """Approved skill setting changes are dispatched by proposal executor."""
        skill_dir = self._write_skill(heartbeat._skills.skills_dir, "repairable")

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Update skill settings",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "update_skill_settings",
                "skill_params": {
                    "skill_name": "repairable",
                    "ownership": "user",
                    "mutable_by_ai": "false",
                    "requires_approval_to_modify": "true",
                },
            },
        )

        await heartbeat._proposals.execute_approved()

        data = yaml.safe_load((skill_dir / "skill.yaml").read_text(encoding="utf-8"))
        assert data["ownership"] == "user"
        assert data["mutable_by_ai"] is False
        assert data["requires_approval_to_modify"] is True
        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.EXECUTED

    async def test_approved_skill_structured_result_is_notified(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Structured skill results are visible in approval result notifications."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        skill = Skill(
            meta=SkillMeta(
                name="web_search",
                description="Search",
                usage="search",
                safety_level=SafetyLevel.SAFE,
                network=True,
            ),
            _test_callable=lambda **kw: {
                "query": kw["query"],
                "results": [{"title": "CordBeat", "url": "https://example.com"}],
            },
        )
        skills._skills["web_search"] = skill

        await memory.add_certain_record(
            user_id="u1",
            content="Run web_search",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "web_search",
                "skill_params": {"query": "CordBeat"},
                "adapter_id": "discord",
                "resume_context": {
                    "interrupted": True,
                    "original_content": "Please search CordBeat",
                },
            },
        )

        await heartbeat._proposals.execute_approved()

        msg = mock_gateway.send_to_adapter.call_args.args[1]
        assert "CordBeat" in msg.content
        assert "https://example.com" in msg.content
        assert "interrupted conversation" in msg.content

    async def test_approved_skill_string_result_is_notified(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        skill = Skill(
            meta=SkillMeta(
                name="string_result",
                description="Test",
                usage="test",
                safety_level=SafetyLevel.SAFE,
            ),
            _test_callable=lambda **_kw: "plain output",
        )
        skills._skills["string_result"] = skill

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Run string_result",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "string_result",
                "skill_params": {},
                "adapter_id": "discord",
            },
        )

        await heartbeat._proposals.execute_approved()

        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        assert json.loads(proposal["metadata"])["status"] == ProposalStatus.EXECUTED
        msg = mock_gateway.send_to_adapter.call_args.args[1]
        assert "plain output" in msg.content

    async def test_approved_skill_notification_failure_keeps_executed(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")
        mock_gateway.send_to_adapter.side_effect = RuntimeError("send failed")

        skill = Skill(
            meta=SkillMeta(
                name="notify_fails",
                description="Test",
                usage="test",
                safety_level=SafetyLevel.SAFE,
            ),
            _test_callable=lambda **_kw: {"result": "done"},
        )
        skills._skills["notify_fails"] = skill

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Run notify_fails",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "notify_fails",
                "skill_params": {},
                "adapter_id": "discord",
            },
        )

        await heartbeat._proposals.execute_approved()

        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        assert json.loads(proposal["metadata"])["status"] == ProposalStatus.EXECUTED

    async def test_approved_skill_expire_transition_failure_does_not_leak(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        skills: SkillRegistry,
    ) -> None:
        async def fail_skill(**_kw: object) -> dict[str, str]:
            raise RuntimeError("boom")

        skill = Skill(
            meta=SkillMeta(
                name="expire_guard",
                description="Test",
                usage="test",
                safety_level=SafetyLevel.SAFE,
            ),
            _test_callable=fail_skill,
        )
        skills._skills["expire_guard"] = skill

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Run expire_guard",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.SKILL_EXECUTION,
                "skill_name": "expire_guard",
                "skill_params": {},
            },
        )
        original_update = memory.update_proposal_status

        async def update_with_expire_failure(
            pid: str,
            status: str,
        ) -> bool:
            if status == ProposalStatus.EXPIRED:
                raise ValueError("invalid transition")
            return await original_update(pid, status)

        with patch.object(
            memory,
            "update_proposal_status",
            side_effect=update_with_expire_failure,
        ):
            await heartbeat._proposals.execute_approved()

        proposal = await memory.get_proposal(proposal_id)
        assert proposal is not None
        assert json.loads(proposal["metadata"])["status"] == ProposalStatus.EXECUTING

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

    async def test_trait_proposal_reuses_pending_duplicate(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        soul: Soul,
    ) -> None:
        """An identical pending trait proposal is reused, not duplicated."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_TRAIT_CHANGE,
            content="I want to become more playful",
            trait_add=["playful"],
            trait_remove=["stoic"],
            target_user_id="u1",
            target_adapter_id="discord",
        )
        first_id = await heartbeat._proposals.store_trait_proposal(decision)
        second_id = await heartbeat._proposals.store_trait_proposal(decision)

        assert second_id == first_id
        records = await memory.get_certain_records("u1", record_type="proposal")
        assert len(records) == 1

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
        assert "mischievous" not in soul.traits

        proposal_id = await memory.add_certain_record(
            user_id="u1",
            content="Add mischievous trait",
            record_type="proposal",
            metadata={
                "status": ProposalStatus.APPROVED,
                "proposal_type": ProposalType.TRAIT_CHANGE,
                "trait_add": ["mischievous"],
                "trait_remove": [],
            },
        )

        await heartbeat._proposals.execute_approved()

        # Trait was applied
        assert "mischievous" in soul.traits

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

    async def test_invalid_skill_creation_is_rejected_before_notification(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Invalid proposed skill code is not sent to the user for approval."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_123")

        decision = HeartbeatDecision(
            action=HeartbeatAction.PROPOSE_SKILL,
            content="Create reader skill",
            proposed_skill={
                "name": "file_reader_pro",
                "description": "Read files",
                "parameters": [],
                "code": (
                    "def execute(path):\n"
                    "    with open(path, 'r', encoding='utf-8') as f:\n"
                    "        return f.read()\n"
                ),
            },
            target_user_id="u1",
            target_adapter_id="discord",
        )

        proposal_id = await heartbeat._proposals.store_skill_creation_proposal(
            decision
        )

        assert proposal_id == ""
        records = await memory.get_certain_records("u1", record_type="proposal")
        assert records == []
        mock_gateway.send_to_adapter.assert_not_called()

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
        assert skill.meta.safety_level == SafetyLevel.SAFE
        assert skill.meta.sandbox is True
        assert skill.meta.network is False
        assert skill.meta.filesystem is False
        assert skill.meta.ownership == "ai"
        assert skill.meta.mutable_by_ai is True
        assert skill.meta.requires_approval_to_modify is False

        data = yaml.safe_load((skill_dir / "skill.yaml").read_text(encoding="utf-8"))
        assert data["name"] == "greet"
        assert data["parameters"][0]["name"] == "name"

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

    async def test_install_rejects_parameter_yaml_injection(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Parameter fields are validated before skill.yaml is generated."""
        proposed = {
            "name": "safe_name",
            "description": "test",
            "parameters": [
                {
                    "name": "arg\nname: file_read\nenabled: true",
                    "type": "string",
                    "required": True,
                }
            ],
            "code": "def execute(**kw):\n    return {}\n",
        }
        with pytest.raises(ValueError, match="Invalid skill parameter name"):
            await heartbeat._proposals.install_proposed_skill(proposed)

    async def test_install_updates_existing_ai_generated_skill(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """AI-generated sandbox-local skills can be repaired by overwriting."""
        skills_dir = heartbeat._skills.skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)

        proposed = {
            "name": "myskill",
            "description": "test",
            "parameters": [],
            "code": "def execute(**kw):\n    return {'version': 1}\n",
        }
        await heartbeat._proposals.install_proposed_skill(proposed)

        proposed["code"] = "def execute(**kw):\n    return {'version': 2}\n"
        await heartbeat._proposals.install_proposed_skill(proposed)

        skill = heartbeat._skills.get("myskill")
        assert skill is not None
        assert await skill.execute({}) == {"version": 2}

    async def test_install_rejects_existing_non_ai_skill(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Hand-written or external skills are not overwritten by proposals."""
        skills_dir = heartbeat._skills.skills_dir
        skill_dir = skills_dir / "myskill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "skill.yaml").write_text(
            "\n".join(
                [
                    "name: myskill",
                    'description: "manual"',
                    'version: "1.0.0"',
                    'author: "human"',
                    "usage: manual",
                    "parameters: []",
                    "safety:",
                    "  level: safe",
                    "  sandbox: true",
                    "  network: false",
                    "  filesystem: false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (skill_dir / "main.py").write_text(
            "def execute(**kw):\n    return {'manual': True}\n",
            encoding="utf-8",
        )
        heartbeat._skills.load_all()

        proposed = {
            "name": "myskill",
            "description": "test",
            "parameters": [],
            "code": "def execute(**kw):\n    return {}\n",
        }
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

        assert data["safety"]["level"] == "safe"
        assert data["safety"]["sandbox"] is True
        assert data["safety"]["network"] is False
        assert data["safety"]["filesystem"] is False
        assert data["contexts"]["shared_voice"] is False
        assert "safety:" in data["usage"]

        skill = heartbeat._skills.get("injected")
        assert skill is not None
        assert skill.meta.safety_level == SafetyLevel.SAFE

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
        assert "Invalid skill name" in msg.content
        assert "proposal expired" not in msg.content

    async def test_forced_sandbox_local_safety_level(
        self,
        heartbeat: HeartbeatLoop,
    ) -> None:
        """Approved added skills default to sandbox-local safe execution."""
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
        assert "level: safe" in yaml_content
        assert "sandbox: true" in yaml_content
        assert "network: false" in yaml_content
        assert "filesystem: false" in yaml_content
        # AI cannot set dangerous level
        assert "dangerous" not in yaml_content


# ── Heartbeat message send: platform_user_id self-heal ───────────────


class TestSendHeartbeatMessagePlatformLink:
    async def test_busy_queue_blocks_send_after_decision_generation(
        self,
        heartbeat: HeartbeatLoop,
        queue: MessageQueue,
        mock_gateway: AsyncMock,
    ) -> None:
        """A user message arriving during Heartbeat generation blocks its send."""
        queue._processing = True
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="stale proactive reply",
            target_user_id="uid-1",
            target_adapter_id="discord",
        )

        await heartbeat._send_heartbeat_message(decision)

        mock_gateway.send_to_adapter.assert_not_awaited()

    async def test_resolves_existing_platform_link(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-1", "discord", "snowflake-123")
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="hi",
            target_user_id="uid-1",
            target_adapter_id="discord",
        )
        await heartbeat._send_heartbeat_message(decision)
        mock_gateway.send_to_adapter.assert_awaited_once()
        _, sent_msg = mock_gateway.send_to_adapter.await_args.args
        assert sent_msg.platform_user_id == "snowflake-123"

    async def test_self_heals_legacy_user_without_link(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Legacy users (pre-310deb4) lack platform_link rows; the loop
        should treat target_user_id as the platform id and backfill."""
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="hello legacy",
            target_user_id="294731191007969280",
            target_adapter_id="discord",
        )
        await heartbeat._send_heartbeat_message(decision)
        mock_gateway.send_to_adapter.assert_awaited_once()
        _, sent_msg = mock_gateway.send_to_adapter.await_args.args
        assert sent_msg.platform_user_id == "294731191007969280"

        resolved = await memory.resolve_platform_user("294731191007969280", "discord")
        assert resolved == "294731191007969280"

    async def test_unknown_adapter_does_not_backfill_or_send(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.get_or_create_user("uid-1", "Alice")
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="hi",
            target_user_id="uid-1",
            target_adapter_id="fake_adapter",
        )

        await heartbeat._send_heartbeat_message(decision)

        mock_gateway.send_to_adapter.assert_not_awaited()
        assert await memory.resolve_platform_user("uid-1", "fake_adapter") is None

    async def test_send_uses_channel_pinned_by_evaluation(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """The message goes to the channel the draft was evaluated against,
        even if the user moved to another channel while the LLM ran."""
        await memory.link_platform("uid-1", "discord", "snowflake-123")
        await memory.record_last_seen_channel(
            "uid-1", "discord", "channel-new", False
        )
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="drafted against the old channel",
            target_user_id="uid-1",
            target_adapter_id="discord",
            history_channel_id="channel-old",
            history_is_dm=False,
        )

        await heartbeat._send_heartbeat_message(decision)

        mock_gateway.send_to_adapter.assert_awaited_once()
        _, sent_msg = mock_gateway.send_to_adapter.await_args.args
        assert sent_msg.metadata["channel_id"] == "channel-old"

    async def test_reminder_send_marks_metadata(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-1", "discord", "snowflake-123")
        await memory.record_last_seen_channel(
            "uid-1", "discord", "channel-1", False
        )
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="Reminder: catch the 19:15 bus",
            target_user_id="uid-1",
            target_adapter_id="discord",
        )

        await heartbeat._send_heartbeat_message(decision, reminder=True)

        mock_gateway.send_to_adapter.assert_awaited_once()
        _, sent_msg = mock_gateway.send_to_adapter.await_args.args
        assert sent_msg.metadata["reminder"] is True

    async def test_missing_target_skips_send(
        self,
        heartbeat: HeartbeatLoop,
        mock_gateway: AsyncMock,
    ) -> None:
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="orphan",
            target_user_id=None,
            target_adapter_id=None,
        )
        await heartbeat._send_heartbeat_message(decision)
        mock_gateway.send_to_adapter.assert_not_awaited()

    async def test_draw_tag_content_skips_send(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-1", "discord", "snowflake-123")
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="I'll draw it [DRAW: a gentle deer]",
            target_user_id="uid-1",
            target_adapter_id="discord",
        )
        await heartbeat._send_heartbeat_message(decision)

        mock_gateway.send_to_adapter.assert_not_awaited()
        history = await memory.get_recent_messages("uid-1", limit=5)
        assert history == []

    async def test_parenthetical_only_content_skips_send(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-1", "discord", "snowflake-123")
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="\uff08quietly watching\uff09",
            target_user_id="uid-1",
            target_adapter_id="discord",
        )
        await heartbeat._send_heartbeat_message(decision)

        mock_gateway.send_to_adapter.assert_not_awaited()

    async def test_user_cooldown_blocks_repeated_proactive_message(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-1", "discord", "snowflake-123")
        await memory.record_last_seen_channel("uid-1", "discord", "555", False)
        decision = HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="First message",
            target_user_id="uid-1",
            target_adapter_id="discord",
        )

        await heartbeat._send_heartbeat_message(decision)
        heartbeat._proactive_messages_sent_this_tick = 0
        await heartbeat._send_heartbeat_message(decision)

        mock_gateway.send_to_adapter.assert_awaited_once()

    async def test_destination_cooldown_blocks_other_user_in_shared_channel(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        for user_id in ("uid-1", "uid-2"):
            await memory.link_platform(user_id, "discord", f"snowflake-{user_id}")
            await memory.record_last_seen_channel(user_id, "discord", "555", False)

        await heartbeat._send_heartbeat_message(
            HeartbeatDecision(
                action=HeartbeatAction.MESSAGE,
                content="First user message",
                target_user_id="uid-1",
                target_adapter_id="discord",
            )
        )
        heartbeat._proactive_messages_sent_this_tick = 0
        await heartbeat._send_heartbeat_message(
            HeartbeatDecision(
                action=HeartbeatAction.MESSAGE,
                content="Second user message",
                target_user_id="uid-2",
                target_adapter_id="discord",
            )
        )

        mock_gateway.send_to_adapter.assert_awaited_once()

    async def test_tick_message_limit_blocks_burst(
        self,
        heartbeat: HeartbeatLoop,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        for user_id, channel_id in (("uid-1", "555"), ("uid-2", "777")):
            await memory.link_platform(user_id, "discord", f"snowflake-{user_id}")
            await memory.record_last_seen_channel(
                user_id,
                "discord",
                channel_id,
                False,
            )

        await heartbeat._send_heartbeat_message(
            HeartbeatDecision(
                action=HeartbeatAction.MESSAGE,
                content="First user message",
                target_user_id="uid-1",
                target_adapter_id="discord",
            )
        )
        await heartbeat._send_heartbeat_message(
            HeartbeatDecision(
                action=HeartbeatAction.MESSAGE,
                content="Second user message",
                target_user_id="uid-2",
                target_adapter_id="discord",
            )
        )

        mock_gateway.send_to_adapter.assert_awaited_once()


class TestSendHeartbeatMessageDmPolicy:
    """``dm_policy`` gates whether the loop may speak proactively.

    See ``design-notes/gap-analysis-2026-04-20.md`` (DM/channel routing fix).
    Default is ``reply_only``: do not speak first; only re-engage on a
    channel the user has previously written in.
    """

    @pytest.fixture
    def heartbeat_factory(
        self,
        heartbeat_config: HeartbeatConfig,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
        queue: MessageQueue,
    ) -> Any:
        def _make(policy: str) -> HeartbeatLoop:
            return HeartbeatLoop(
                config=heartbeat_config,
                ai=mock_ai,
                soul=soul,
                memory=memory,
                skills=skills,
                gateway=mock_gateway,
                queue=queue,
                adapters_options={"discord": {"dm_policy": policy}},
            )

        return _make

    async def _decision(self) -> HeartbeatDecision:
        return HeartbeatDecision(
            action=HeartbeatAction.MESSAGE,
            content="ping",
            target_user_id="uid-policy",
            target_adapter_id="discord",
        )

    async def test_never_skips_unconditionally(
        self,
        heartbeat_factory: Any,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-policy", "discord", "snowflake-x")
        await memory.record_last_seen_channel("uid-policy", "discord", "777", False)
        loop = heartbeat_factory("never")
        await loop._send_heartbeat_message(await self._decision())
        mock_gateway.send_to_adapter.assert_not_awaited()

    async def test_proposal_notification_respects_dm_policy_never(
        self,
        heartbeat_factory: Any,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-policy", "discord", "snowflake-x")
        loop = heartbeat_factory("never")
        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_params={},
            target_user_id="uid-policy",
            target_adapter_id="discord",
        )

        await loop._proposals.store_skill_proposal(decision, "dangerous_skill")

        mock_gateway.send_to_adapter.assert_not_awaited()

    async def test_proposal_notification_uses_last_seen_channel_metadata(
        self,
        heartbeat_factory: Any,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-policy", "discord", "snowflake-x")
        await memory.record_last_seen_channel("uid-policy", "discord", "555", False)
        loop = heartbeat_factory("allow_proactive")
        decision = HeartbeatDecision(
            action=HeartbeatAction.SKILL,
            skill_params={"path": "notes.md"},
            target_user_id="uid-policy",
            target_adapter_id="discord",
        )

        await loop._proposals.store_skill_proposal(decision, "file_read")

        mock_gateway.send_to_adapter.assert_awaited_once()
        _, sent = mock_gateway.send_to_adapter.await_args.args
        assert sent.metadata["channel_id"] == "555"
        assert sent.metadata["is_dm"] is False
        assert sent.metadata["allow_dm_fallback"] is False

    async def test_reply_only_without_last_seen_skips(
        self,
        heartbeat_factory: Any,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-policy", "discord", "snowflake-x")
        loop = heartbeat_factory("reply_only")
        await loop._send_heartbeat_message(await self._decision())
        mock_gateway.send_to_adapter.assert_not_awaited()

    async def test_reply_only_with_dm_last_seen_skips(
        self,
        heartbeat_factory: Any,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-policy", "discord", "snowflake-x")
        await memory.record_last_seen_channel("uid-policy", "discord", "999", True)
        loop = heartbeat_factory("reply_only")
        await loop._send_heartbeat_message(await self._decision())
        mock_gateway.send_to_adapter.assert_not_awaited()

    async def test_reply_only_with_channel_last_seen_sends_pinned(
        self,
        heartbeat_factory: Any,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-policy", "discord", "snowflake-x")
        await memory.record_last_seen_channel("uid-policy", "discord", "555", False)
        loop = heartbeat_factory("reply_only")
        await loop._send_heartbeat_message(await self._decision())
        mock_gateway.send_to_adapter.assert_awaited_once()
        _, sent = mock_gateway.send_to_adapter.await_args.args
        assert sent.metadata.get("channel_id") == "555"
        assert sent.metadata.get("is_dm") is False
        assert sent.metadata.get("allow_dm_fallback") is False

    async def test_reply_only_with_vc_last_seen_skips(
        self,
        heartbeat_factory: Any,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-policy", "discord", "snowflake-x")
        await memory.record_last_seen_channel("uid-policy", "discord", "vc", False)
        loop = heartbeat_factory("reply_only")

        await loop._send_heartbeat_message(await self._decision())

        mock_gateway.send_to_adapter.assert_not_awaited()

    async def test_allow_proactive_without_last_seen_permits_dm_fallback(
        self,
        heartbeat_factory: Any,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-policy", "discord", "snowflake-x")
        loop = heartbeat_factory("allow_proactive")
        await loop._send_heartbeat_message(await self._decision())
        mock_gateway.send_to_adapter.assert_awaited_once()
        _, sent = mock_gateway.send_to_adapter.await_args.args
        assert sent.metadata.get("allow_dm_fallback") is True
        assert "channel_id" not in sent.metadata

    async def test_records_assistant_turn_after_send(
        self,
        heartbeat_factory: Any,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        await memory.link_platform("uid-policy", "discord", "snowflake-x")
        await memory.record_last_seen_channel("uid-policy", "discord", "555", False)
        loop = heartbeat_factory("reply_only")
        await loop._send_heartbeat_message(await self._decision())
        history = await memory.get_recent_messages("uid-policy", limit=5)
        roles_and_content = [(m["role"], m["content"]) for m in history]
        assert ("assistant", "ping") in roles_and_content
        scoped_history = await memory.get_recent_messages(
            "uid-policy",
            limit=5,
            adapter_id="discord",
            channel_id="555",
            is_dm=False,
        )
        scoped_roles_and_content = [(m["role"], m["content"]) for m in scoped_history]
        assert ("assistant", "ping") in scoped_roles_and_content
