"""Tests for SOUL system."""

from __future__ import annotations

from pathlib import Path

import pytest

from cordbeat.models import Emotion, SoulCaller, SoulPermissionError
from cordbeat.soul import _BASELINE_INTENSITY, _DECAY_RATE, Soul


class TestSoul:
    def test_default_creation(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        assert soul.name == "CordBeat"
        assert soul.pronoun == "I"
        assert soul.emotion.primary == Emotion.CALM
        assert len(soul.immutable_rules) > 0

    def test_update_emotion(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.9, caller=SoulCaller.AI)
        assert soul.emotion.primary == Emotion.JOY
        assert soul.emotion.primary_intensity == 0.9
        # Old CALM should become secondary (halved intensity)
        assert soul.emotion.secondary == Emotion.CALM
        assert soul.emotion.secondary_intensity == pytest.approx(0.25)

    def test_persistence(self, tmp_path: Path) -> None:
        soul_dir = tmp_path / "soul"
        soul = Soul(soul_dir)
        soul.update_name("テスト", caller=SoulCaller.USER)
        soul.update_emotion(Emotion.EXCITEMENT, 0.8, caller=SoulCaller.AI)

        # Reload
        soul2 = Soul(soul_dir)
        assert soul2.name == "テスト"
        assert soul2.emotion.primary == Emotion.EXCITEMENT

    def test_trait_proposal(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        proposal = soul.propose_trait_change(add=["talkative"])
        assert "talkative" in proposal["preview"]
        # Proposal should NOT change state
        assert "talkative" not in soul.traits

    def test_apply_trait_change(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.apply_trait_change(add=["talkative"], caller=SoulCaller.SYSTEM)
        assert "talkative" in soul.traits

    def test_soul_snapshot(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        snap = soul.get_soul_snapshot()
        assert "name" in snap
        assert "immutable_rules" in snap
        assert isinstance(snap["traits"], list)

    def test_immutable_rules_exist(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        rules = soul.immutable_rules
        assert any("harm" in r for r in rules)
        assert any("lie" in r for r in rules)


class TestEmotionTransition:
    """Test emotion transition logic when primary changes."""

    def test_primary_change_moves_old_to_secondary(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        # Start: CALM 0.5, secondary: CURIOSITY 0.3 (default)
        soul.update_emotion(Emotion.JOY, 0.8, caller=SoulCaller.AI)
        assert soul.emotion.primary == Emotion.JOY
        assert soul.emotion.primary_intensity == pytest.approx(0.8)
        # Old primary (CALM 0.5) → secondary at half intensity
        assert soul.emotion.secondary == Emotion.CALM
        assert soul.emotion.secondary_intensity == pytest.approx(0.25)

    def test_same_emotion_keeps_secondary(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        # Default secondary is CURIOSITY 0.3
        soul.update_emotion(Emotion.CALM, 0.7, caller=SoulCaller.AI)
        # Primary didn't change, secondary should remain
        assert soul.emotion.secondary == Emotion.CURIOSITY
        assert soul.emotion.secondary_intensity == pytest.approx(0.3)

    def test_explicit_secondary(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(
            Emotion.EXCITEMENT,
            0.9,
            secondary=Emotion.WARMTH,
            secondary_intensity=0.6,
            caller=SoulCaller.AI,
        )
        assert soul.emotion.primary == Emotion.EXCITEMENT
        assert soul.emotion.secondary == Emotion.WARMTH
        assert soul.emotion.secondary_intensity == pytest.approx(0.6)

    def test_chained_transitions(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.8, caller=SoulCaller.AI)
        soul.update_emotion(Emotion.SADNESS, 0.6, caller=SoulCaller.AI)
        # JOY should be the secondary now (halved)
        assert soul.emotion.primary == Emotion.SADNESS
        assert soul.emotion.secondary == Emotion.JOY
        assert soul.emotion.secondary_intensity == pytest.approx(0.4)

    def test_intensity_clamped(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 1.5, caller=SoulCaller.AI)
        assert soul.emotion.primary_intensity == 1.0
        soul.update_emotion(Emotion.SADNESS, -0.5, caller=SoulCaller.AI)
        assert soul.emotion.primary_intensity == 0.0


class TestEmotionDecay:
    """Test natural emotion decay toward CALM."""

    def test_decay_reduces_intensity(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.9, caller=SoulCaller.AI)
        soul.decay_emotion()
        assert soul.emotion.primary == Emotion.JOY
        assert soul.emotion.primary_intensity == pytest.approx(0.9 - _DECAY_RATE)

    def test_decay_reverts_to_calm(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        # Set intensity just above baseline + decay rate
        soul.update_emotion(
            Emotion.JOY, _BASELINE_INTENSITY + _DECAY_RATE, caller=SoulCaller.AI
        )
        soul.decay_emotion()
        # Should revert to CALM at baseline
        assert soul.emotion.primary == Emotion.CALM
        assert soul.emotion.primary_intensity == pytest.approx(_BASELINE_INTENSITY)

    def test_calm_does_not_decay(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        # Default is CALM
        original_intensity = soul.emotion.primary_intensity
        soul.decay_emotion()
        # Should apply secondary decay only; primary CALM shouldn't decay
        assert soul.emotion.primary == Emotion.CALM
        assert soul.emotion.primary_intensity == original_intensity

    def test_secondary_clears_below_threshold(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(
            Emotion.JOY,
            0.8,
            secondary=Emotion.WARMTH,
            secondary_intensity=0.08,
            caller=SoulCaller.AI,
        )
        soul.decay_emotion()
        # Secondary was below threshold after decay → cleared
        assert soul.emotion.secondary is None
        assert soul.emotion.secondary_intensity == 0.0

    def test_multiple_decay_cycles(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.EXCITEMENT, 0.9, caller=SoulCaller.AI)
        # Apply many decay cycles
        for _ in range(50):
            soul.decay_emotion()
        # Should eventually revert to CALM
        assert soul.emotion.primary == Emotion.CALM


class TestEmotionHistory:
    """Test emotion history tracking."""

    def test_history_recorded(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.8, caller=SoulCaller.AI)
        history = soul.emotion_history
        assert len(history) == 1
        assert history[0]["emotion"] == "joy"
        assert history[0]["intensity"] == 0.8
        assert "timestamp" in history[0]

    def test_history_accumulates(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.8, caller=SoulCaller.AI)
        soul.update_emotion(Emotion.SADNESS, 0.3, caller=SoulCaller.AI)
        soul.update_emotion(Emotion.CALM, 0.5, caller=SoulCaller.AI)
        assert len(soul.emotion_history) == 3

    def test_history_capped_at_50(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        for i in range(60):
            soul.update_emotion(Emotion.JOY, i / 100, caller=SoulCaller.AI)
        assert len(soul.emotion_history) == 50

    def test_history_persists(self, tmp_path: Path) -> None:
        soul_dir = tmp_path / "soul"
        soul = Soul(soul_dir)
        soul.update_emotion(Emotion.WARMTH, 0.7, caller=SoulCaller.AI)
        # Reload
        soul2 = Soul(soul_dir)
        assert len(soul2.emotion_history) == 1
        assert soul2.emotion_history[0]["emotion"] == "warmth"


class TestDefaultSecondaryEmotion:
    """Test that default soul includes secondary emotion per spec."""

    def test_default_has_secondary(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        assert soul.emotion.secondary == Emotion.CURIOSITY
        assert soul.emotion.secondary_intensity == pytest.approx(0.3)

    def test_snapshot_includes_secondary(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        snap = soul.get_soul_snapshot()
        assert "secondary" in snap["emotion"]
        assert snap["emotion"]["secondary"] == "curiosity"
        assert snap["emotion"]["secondary_intensity"] == pytest.approx(0.3)


class TestSoulNotes:
    """Tests for soul_notes.md loading and management."""

    def test_default_notes_created(self, tmp_path: Path) -> None:
        """Default soul_notes.md is created if it doesn't exist."""
        soul_dir = tmp_path / "soul"
        soul = Soul(soul_dir)
        notes_path = soul_dir / "soul_notes.md"
        assert notes_path.exists()
        assert "Soul Notes" in soul.notes

    def test_notes_loaded_from_file(self, tmp_path: Path) -> None:
        """Existing soul_notes.md is loaded correctly."""
        soul_dir = tmp_path / "soul"
        soul_dir.mkdir(parents=True)
        notes_path = soul_dir / "soul_notes.md"
        notes_path.write_text("Custom tone: casual and warm", encoding="utf-8")

        soul = Soul(soul_dir)
        assert soul.notes == "Custom tone: casual and warm"

    def test_update_notes(self, tmp_path: Path) -> None:
        """update_notes() saves content to soul_notes.md."""
        soul_dir = tmp_path / "soul"
        soul = Soul(soul_dir)
        soul.update_notes("Speak casually with lots of emoji", caller=SoulCaller.USER)

        assert soul.notes == "Speak casually with lots of emoji"
        # Verify persistence
        soul2 = Soul(soul_dir)
        assert soul2.notes == "Speak casually with lots of emoji"

    def test_snapshot_includes_notes(self, tmp_path: Path) -> None:
        """get_soul_snapshot() includes notes field."""
        soul_dir = tmp_path / "soul"
        soul = Soul(soul_dir)
        soul.update_notes("Be friendly and supportive", caller=SoulCaller.USER)

        snap = soul.get_soul_snapshot()
        assert "notes" in snap
        assert "friendly and supportive" in snap["notes"]

    def test_empty_notes(self, tmp_path: Path) -> None:
        """Empty notes do not cause errors."""
        soul_dir = tmp_path / "soul"
        soul = Soul(soul_dir)
        soul.update_notes("", caller=SoulCaller.USER)

        assert soul.notes == ""
        snap = soul.get_soul_snapshot()
        assert snap["notes"] == ""


class TestPermissionMatrix:
    """Tests for SOUL permission matrix enforcement."""

    def test_ai_can_update_emotion(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.8, caller=SoulCaller.AI)
        assert soul.emotion.primary == Emotion.JOY

    def test_system_can_update_emotion(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.8, caller=SoulCaller.SYSTEM)
        assert soul.emotion.primary == Emotion.JOY

    def test_user_cannot_update_emotion(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        with pytest.raises(SoulPermissionError):
            soul.update_emotion(Emotion.JOY, 0.8, caller=SoulCaller.USER)

    def test_system_can_apply_trait_change(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.apply_trait_change(add=["bold"], caller=SoulCaller.SYSTEM)
        assert "bold" in soul.traits

    def test_ai_cannot_apply_trait_change(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        with pytest.raises(SoulPermissionError):
            soul.apply_trait_change(add=["bold"], caller=SoulCaller.AI)

    def test_user_cannot_apply_trait_change(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        with pytest.raises(SoulPermissionError):
            soul.apply_trait_change(add=["bold"], caller=SoulCaller.USER)

    def test_user_can_update_name(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_name("Athena", caller=SoulCaller.USER)
        assert soul.name == "Athena"

    def test_ai_cannot_update_name(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        with pytest.raises(SoulPermissionError):
            soul.update_name("Hacked", caller=SoulCaller.AI)

    def test_system_cannot_update_name(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        with pytest.raises(SoulPermissionError):
            soul.update_name("Override", caller=SoulCaller.SYSTEM)

    def test_user_can_update_notes(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_notes("User notes", caller=SoulCaller.USER)
        assert soul.notes == "User notes"

    def test_system_can_update_notes(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_notes("Approved notes", caller=SoulCaller.SYSTEM)
        assert soul.notes == "Approved notes"

    def test_ai_cannot_update_notes(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        with pytest.raises(SoulPermissionError):
            soul.update_notes("Hacked notes", caller=SoulCaller.AI)

    def test_user_can_update_quiet_hours(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_quiet_hours("23:00", "06:00", caller=SoulCaller.USER)
        assert soul.quiet_hours == ("23:00", "06:00")

    def test_system_can_update_quiet_hours(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_quiet_hours("02:00", "08:00", caller=SoulCaller.SYSTEM)
        assert soul.quiet_hours == ("02:00", "08:00")

    def test_ai_cannot_update_quiet_hours(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        with pytest.raises(SoulPermissionError):
            soul.update_quiet_hours("00:00", "00:00", caller=SoulCaller.AI)

    def test_core_is_frozen(self, tmp_path: Path) -> None:
        """soul_core data cannot be modified at runtime."""
        soul = Soul(tmp_path / "soul")
        with pytest.raises(TypeError):
            soul._core["immutable_rules"] = []  # type: ignore[index]

    def test_permission_error_message(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        with pytest.raises(SoulPermissionError, match="ai.*not allowed.*name"):
            soul.update_name("Bad", caller=SoulCaller.AI)
