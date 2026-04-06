"""Tests for core data models."""

from __future__ import annotations

from cordbeat.models import (
    Emotion,
    EmotionState,
    HeartbeatAction,
    HeartbeatDecision,
    MemoryLayer,
    SafetyLevel,
    TrustLevel,
    UserSummary,
    ValidationError,
    ValidationResult,
)


class TestEmotionState:
    def test_default(self) -> None:
        state = EmotionState()
        assert state.primary == Emotion.CALM
        assert state.primary_intensity == 0.5

    def test_clamp_intensity(self) -> None:
        state = EmotionState(primary=Emotion.JOY, primary_intensity=1.5)
        assert state.primary_intensity == 1.0
        state2 = EmotionState(primary=Emotion.SADNESS, primary_intensity=-0.3)
        assert state2.primary_intensity == 0.0


class TestUserSummary:
    def test_default(self) -> None:
        user = UserSummary(user_id="u1", display_name="Test")
        assert user.attention_score == 0.5

    def test_clamp_attention(self) -> None:
        user = UserSummary(user_id="u1", display_name="T", attention_score=2.0)
        assert user.attention_score == 1.0


class TestHeartbeatDecision:
    def test_default_interval(self) -> None:
        d = HeartbeatDecision(action=HeartbeatAction.NONE)
        assert d.next_heartbeat_minutes == 60


class TestValidationResult:
    def test_valid(self) -> None:
        r = ValidationResult(valid=True)
        assert r.valid
        assert r.error_summary == ""

    def test_errors(self) -> None:
        r = ValidationResult(
            valid=False,
            errors=[ValidationError(field="x", message="bad")],
        )
        assert not r.valid
        assert "x" in r.error_summary


class TestEnums:
    def test_emotion_values(self) -> None:
        assert Emotion.JOY.value == "joy"
        assert Emotion.LONELINESS.value == "loneliness"

    def test_safety_level(self) -> None:
        assert SafetyLevel.DANGEROUS.value == "dangerous"

    def test_memory_layer(self) -> None:
        assert MemoryLayer.SEMANTIC.value == "semantic"

    def test_trust_level(self) -> None:
        assert TrustLevel.CERTAIN.value == 3
        assert TrustLevel.ALMOST_FORGOTTEN.value == 0
