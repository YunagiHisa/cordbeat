"""Tests for SOUL system."""

from __future__ import annotations

from pathlib import Path

from cordbeat.models import Emotion
from cordbeat.soul import Soul


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
