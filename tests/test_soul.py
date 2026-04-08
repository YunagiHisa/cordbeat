"""Tests for SOUL system."""

from __future__ import annotations

from pathlib import Path

import pytest

from cordbeat.models import Emotion
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
        soul.update_emotion(Emotion.JOY, 0.9)
        assert soul.emotion.primary == Emotion.JOY
        assert soul.emotion.primary_intensity == 0.9
        # Old CALM should become secondary (halved intensity)
        assert soul.emotion.secondary == Emotion.CALM
        assert soul.emotion.secondary_intensity == pytest.approx(0.25)

    def test_persistence(self, tmp_path: Path) -> None:
        soul_dir = tmp_path / "soul"
        soul = Soul(soul_dir)
        soul.update_name("テスト")
        soul.update_emotion(Emotion.EXCITEMENT, 0.8)

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
        soul.apply_trait_change(add=["talkative"])
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
        soul.update_emotion(Emotion.JOY, 0.8)
        assert soul.emotion.primary == Emotion.JOY
        assert soul.emotion.primary_intensity == pytest.approx(0.8)
        # Old primary (CALM 0.5) → secondary at half intensity
        assert soul.emotion.secondary == Emotion.CALM
        assert soul.emotion.secondary_intensity == pytest.approx(0.25)

    def test_same_emotion_keeps_secondary(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        # Default secondary is CURIOSITY 0.3
        soul.update_emotion(Emotion.CALM, 0.7)
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
        )
        assert soul.emotion.primary == Emotion.EXCITEMENT
        assert soul.emotion.secondary == Emotion.WARMTH
        assert soul.emotion.secondary_intensity == pytest.approx(0.6)

    def test_chained_transitions(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.8)
        soul.update_emotion(Emotion.SADNESS, 0.6)
        # JOY should be the secondary now (halved)
        assert soul.emotion.primary == Emotion.SADNESS
        assert soul.emotion.secondary == Emotion.JOY
        assert soul.emotion.secondary_intensity == pytest.approx(0.4)

    def test_intensity_clamped(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 1.5)
        assert soul.emotion.primary_intensity == 1.0
        soul.update_emotion(Emotion.SADNESS, -0.5)
        assert soul.emotion.primary_intensity == 0.0


class TestEmotionDecay:
    """Test natural emotion decay toward CALM."""

    def test_decay_reduces_intensity(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.9)
        soul.decay_emotion()
        assert soul.emotion.primary == Emotion.JOY
        assert soul.emotion.primary_intensity == pytest.approx(0.9 - _DECAY_RATE)

    def test_decay_reverts_to_calm(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        # Set intensity just above baseline + decay rate
        soul.update_emotion(Emotion.JOY, _BASELINE_INTENSITY + _DECAY_RATE)
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
        )
        soul.decay_emotion()
        # Secondary was below threshold after decay → cleared
        assert soul.emotion.secondary is None
        assert soul.emotion.secondary_intensity == 0.0

    def test_multiple_decay_cycles(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.EXCITEMENT, 0.9)
        # Apply many decay cycles
        for _ in range(50):
            soul.decay_emotion()
        # Should eventually revert to CALM
        assert soul.emotion.primary == Emotion.CALM


class TestEmotionHistory:
    """Test emotion history tracking."""

    def test_history_recorded(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.8)
        history = soul.emotion_history
        assert len(history) == 1
        assert history[0]["emotion"] == "joy"
        assert history[0]["intensity"] == 0.8
        assert "timestamp" in history[0]

    def test_history_accumulates(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        soul.update_emotion(Emotion.JOY, 0.8)
        soul.update_emotion(Emotion.SADNESS, 0.3)
        soul.update_emotion(Emotion.CALM, 0.5)
        assert len(soul.emotion_history) == 3

    def test_history_capped_at_50(self, tmp_path: Path) -> None:
        soul = Soul(tmp_path / "soul")
        for i in range(60):
            soul.update_emotion(Emotion.JOY, i / 100)
        assert len(soul.emotion_history) == 50

    def test_history_persists(self, tmp_path: Path) -> None:
        soul_dir = tmp_path / "soul"
        soul = Soul(soul_dir)
        soul.update_emotion(Emotion.WARMTH, 0.7)
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
