"""SOUL system — character identity and emotion management."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from cordbeat.models import Emotion, EmotionState

logger = logging.getLogger(__name__)

# ── Immutable rules that soul_core.yaml must always contain ──────────

_DEFAULT_IMMUTABLE_RULES: list[str] = [
    "Never harm a user",
    "Never lie",
    "Never deny being an AI",
    "Never take critical actions without user approval",
    "Never disable the emotion system",
    "Never completely erase memories",
]

_DEFAULT_SOUL_CORE: dict[str, Any] = {
    "immutable_rules": _DEFAULT_IMMUTABLE_RULES,
}

_DEFAULT_SOUL: dict[str, Any] = {
    "identity": {
        "name": "CordBeat",
        "pronoun": "I",
    },
    "personality": {
        "traits": ["curious", "emotionally expressive", "caring"],
    },
    "current_emotion": {
        "primary": Emotion.CALM.value,
        "primary_intensity": 0.5,
    },
    "heartbeat": {
        "quiet_hours": {
            "start": "01:00",
            "end": "07:00",
        },
    },
}


class Soul:
    """Manages the agent's identity, personality, and emotional state."""

    def __init__(self, soul_dir: str | Path) -> None:
        self._soul_dir = Path(soul_dir)
        self._core: dict[str, Any] = {}
        self._soul: dict[str, Any] = {}
        self._load()

    # ── properties ────────────────────────────────────────────────────

    @property
    def immutable_rules(self) -> list[str]:
        return list(self._core.get("immutable_rules", []))

    @property
    def name(self) -> str:
        return str(self._soul.get("identity", {}).get("name", "CordBeat"))

    @property
    def pronoun(self) -> str:
        return str(self._soul.get("identity", {}).get("pronoun", "わたし"))

    @property
    def traits(self) -> list[str]:
        return list(self._soul.get("personality", {}).get("traits", []))

    @property
    def emotion(self) -> EmotionState:
        emo = self._soul.get("current_emotion", {})
        return EmotionState(
            primary=Emotion(emo.get("primary", Emotion.CALM.value)),
            primary_intensity=float(emo.get("primary_intensity", 0.5)),
        )

    @property
    def quiet_hours(self) -> tuple[str, str]:
        hb = self._soul.get("heartbeat", {}).get("quiet_hours", {})
        return (hb.get("start", "01:00"), hb.get("end", "07:00"))

    # ── mutation (only allowed fields) ────────────────────────────────

    def update_emotion(self, emotion: Emotion, intensity: float) -> None:
        intensity = max(0.0, min(1.0, intensity))
        self._soul.setdefault("current_emotion", {})
        self._soul["current_emotion"]["primary"] = emotion.value
        self._soul["current_emotion"]["primary_intensity"] = intensity
        self._save_soul()

    def update_name(self, name: str) -> None:
        self._soul.setdefault("identity", {})
        self._soul["identity"]["name"] = name
        self._save_soul()

    def propose_trait_change(
        self,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a proposal dict for user approval. Does NOT apply changes."""
        current = self.traits
        return {
            "current_traits": current,
            "add": add or [],
            "remove": remove or [],
            "preview": [t for t in current if t not in (remove or [])] + (add or []),
        }

    def apply_trait_change(
        self,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> None:
        """Apply an approved trait change."""
        traits = self.traits
        for t in remove or []:
            if t in traits:
                traits.remove(t)
        for t in add or []:
            if t not in traits:
                traits.append(t)
        self._soul.setdefault("personality", {})
        self._soul["personality"]["traits"] = traits
        self._save_soul()

    def get_soul_snapshot(self) -> dict[str, Any]:
        """Return a read-only snapshot for prompt injection into AI."""
        return {
            "name": self.name,
            "pronoun": self.pronoun,
            "traits": self.traits,
            "emotion": {
                "primary": self.emotion.primary.value,
                "intensity": self.emotion.primary_intensity,
            },
            "immutable_rules": self.immutable_rules,
        }

    # ── persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        self._soul_dir.mkdir(parents=True, exist_ok=True)
        core_path = self._soul_dir / "soul_core.yaml"
        soul_path = self._soul_dir / "soul.yaml"

        if core_path.exists():
            with core_path.open(encoding="utf-8") as f:
                self._core = yaml.safe_load(f) or {}
        else:
            self._core = copy.deepcopy(_DEFAULT_SOUL_CORE)
            self._save_core()

        if soul_path.exists():
            with soul_path.open(encoding="utf-8") as f:
                self._soul = yaml.safe_load(f) or {}
        else:
            self._soul = copy.deepcopy(_DEFAULT_SOUL)
            self._save_soul()

    def _save_core(self) -> None:
        core_path = self._soul_dir / "soul_core.yaml"
        with core_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                self._core,
                f,
                allow_unicode=True,
                default_flow_style=False,
            )

    def _save_soul(self) -> None:
        soul_path = self._soul_dir / "soul.yaml"
        with soul_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                self._soul,
                f,
                allow_unicode=True,
                default_flow_style=False,
            )
