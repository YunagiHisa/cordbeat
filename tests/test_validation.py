"""Tests for validation layer."""

from __future__ import annotations

from cordbeat.validation import (
    validate_heartbeat_decision,
    validate_soul_update,
    validate_user_summary_update,
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
