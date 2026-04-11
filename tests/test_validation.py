"""Tests for validation layer."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cordbeat.validation import (
    validate_heartbeat_decision,
    validate_heartbeat_triage,
    validate_skill_selection,
    validate_soul_update,
    validate_user_summary_update,
    validated_ai_json,
)


class TestHeartbeatValidation:
    def test_valid_none(self) -> None:
        r = validate_heartbeat_decision(
            {
                "action": "none",
                "next_heartbeat_minutes": 60,
            }
        )
        assert r.valid

    def test_valid_message(self) -> None:
        r = validate_heartbeat_decision(
            {
                "action": "message",
                "content": "hello",
                "next_heartbeat_minutes": 30,
            }
        )
        assert r.valid

    def test_invalid_action(self) -> None:
        r = validate_heartbeat_decision({"action": "destroy"})
        assert not r.valid

    def test_missing_content_for_message(self) -> None:
        r = validate_heartbeat_decision({"action": "message", "content": ""})
        assert not r.valid

    def test_invalid_interval(self) -> None:
        r = validate_heartbeat_decision(
            {
                "action": "none",
                "next_heartbeat_minutes": 99999,
            }
        )
        assert not r.valid

    def test_valid_propose_trait_change(self) -> None:
        r = validate_heartbeat_decision(
            {
                "action": "propose_trait_change",
                "content": "I want to become more playful",
                "trait_add": ["playful"],
                "next_heartbeat_minutes": 60,
            }
        )
        assert r.valid

    def test_propose_trait_change_requires_content(self) -> None:
        r = validate_heartbeat_decision(
            {
                "action": "propose_trait_change",
                "content": "",
            }
        )
        assert not r.valid


class TestUserSummaryValidation:
    def test_valid(self) -> None:
        r = validate_user_summary_update(
            {
                "attention_score": 0.8,
                "last_topic": "AI",
                "emotional_tone": "happy",
            }
        )
        assert r.valid

    def test_score_out_of_range(self) -> None:
        r = validate_user_summary_update({"attention_score": 1.5})
        assert not r.valid

    def test_topic_too_long(self) -> None:
        r = validate_user_summary_update({"last_topic": "x" * 100})
        assert not r.valid


class TestSoulUpdateValidation:
    def test_immutable_rules_blocked(self) -> None:
        r = validate_soul_update({"immutable_rules": ["new rule"]})
        assert not r.valid

    def test_valid_emotion_update(self) -> None:
        r = validate_soul_update(
            {
                "current_emotion": {"primary": "joy", "primary_intensity": 0.8},
            }
        )
        assert r.valid

    def test_invalid_intensity(self) -> None:
        r = validate_soul_update(
            {
                "current_emotion": {"primary_intensity": 2.0},
            }
        )
        assert not r.valid


class TestSkillSelectionValidation:
    def test_valid_skill(self) -> None:
        r = validate_skill_selection(
            {"skill_name": "web_search"},
            available_skills={"web_search", "timer"},
        )
        assert r.valid

    def test_unknown_skill(self) -> None:
        r = validate_skill_selection(
            {"skill_name": "nonexistent"},
            available_skills={"web_search", "timer"},
        )
        assert not r.valid

    def test_no_skill_name(self) -> None:
        r = validate_skill_selection(
            {"skill_name": ""},
            available_skills={"web_search"},
        )
        assert r.valid  # empty string is falsy, skipped


class TestTriageValidation:
    def test_valid_triage(self) -> None:
        r = validate_heartbeat_triage(
            {
                "users": [{"user_id": "u1", "reason": "lonely"}],
                "next_heartbeat_minutes": 30,
            }
        )
        assert r.valid

    def test_valid_empty_users(self) -> None:
        r = validate_heartbeat_triage({"users": [], "next_heartbeat_minutes": 60})
        assert r.valid

    def test_users_not_a_list(self) -> None:
        r = validate_heartbeat_triage(
            {"users": "invalid", "next_heartbeat_minutes": 60}
        )
        assert not r.valid
        assert any("must be a list" in e.message for e in r.errors)

    def test_user_entry_not_dict(self) -> None:
        r = validate_heartbeat_triage(
            {"users": ["not a dict"], "next_heartbeat_minutes": 60}
        )
        assert not r.valid
        assert any("must be a dict" in e.message for e in r.errors)

    def test_user_entry_missing_user_id(self) -> None:
        r = validate_heartbeat_triage(
            {"users": [{"reason": "lonely"}], "next_heartbeat_minutes": 60}
        )
        assert not r.valid
        assert any("user_id is required" in e.message for e in r.errors)

    def test_invalid_interval(self) -> None:
        r = validate_heartbeat_triage({"users": [], "next_heartbeat_minutes": 0})
        assert not r.valid

    def test_missing_users_key(self) -> None:
        r = validate_heartbeat_triage({"next_heartbeat_minutes": 60})
        assert not r.valid


class TestValidatedAiJson:
    async def test_valid_on_first_attempt(self) -> None:
        backend = AsyncMock()
        backend.generate_json = AsyncMock(
            return_value={"action": "none", "next_heartbeat_minutes": 60}
        )
        result = await validated_ai_json(
            backend,
            prompt="test",
            system="sys",
            validator=validate_heartbeat_decision,
        )
        assert result["action"] == "none"
        assert backend.generate_json.call_count == 1

    async def test_retry_on_invalid(self) -> None:
        backend = AsyncMock()
        backend.generate_json = AsyncMock(
            side_effect=[
                {"action": "destroy"},  # invalid
                {"action": "none", "next_heartbeat_minutes": 60},  # valid
            ]
        )
        result = await validated_ai_json(
            backend,
            prompt="test",
            system="sys",
            validator=validate_heartbeat_decision,
        )
        assert result["action"] == "none"
        assert backend.generate_json.call_count == 2

    async def test_retry_on_json_error(self) -> None:
        import json

        backend = AsyncMock()
        backend.generate_json = AsyncMock(
            side_effect=[
                json.JSONDecodeError("bad", "", 0),
                {"action": "none", "next_heartbeat_minutes": 60},
            ]
        )
        result = await validated_ai_json(
            backend,
            prompt="test",
            system="sys",
            validator=validate_heartbeat_decision,
        )
        assert result["action"] == "none"

    async def test_fallback_after_max_retries(self) -> None:
        backend = AsyncMock()
        backend.generate_json = AsyncMock(return_value={"action": "destroy"})
        fallback = {"action": "none", "next_heartbeat_minutes": 60}
        result = await validated_ai_json(
            backend,
            prompt="test",
            system="sys",
            validator=validate_heartbeat_decision,
            fallback=fallback,
        )
        assert result is fallback
        assert backend.generate_json.call_count == 3  # 1 + MAX_RETRIES

    async def test_raises_without_fallback(self) -> None:
        backend = AsyncMock()
        backend.generate_json = AsyncMock(return_value={"action": "destroy"})
        with pytest.raises(ValueError, match="Validation failed"):
            await validated_ai_json(
                backend,
                prompt="test",
                system="sys",
                validator=validate_heartbeat_decision,
            )
