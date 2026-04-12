"""SOUL system — character identity and emotion management."""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from cordbeat.models import Emotion, EmotionState, SoulCaller, SoulPermissionError

logger = logging.getLogger(__name__)

# ── Emotion decay constants ───────────────────────────────────────────

# Per-tick decay rate: intensity moves toward baseline by this fraction
_DECAY_RATE: float = 0.05
# Baseline intensity — emotions decay toward this, not zero
_BASELINE_INTENSITY: float = 0.3
# Below this threshold, secondary emotion is cleared
_SECONDARY_CLEAR_THRESHOLD: float = 0.1

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
        "secondary": Emotion.CURIOSITY.value,
        "secondary_intensity": 0.3,
    },
    "heartbeat": {
        "quiet_hours": {
            "start": "01:00",
            "end": "07:00",
        },
    },
}


_DEFAULT_SOUL_NOTES = """\
# Soul Notes

Free-form notes about this character's personality nuances.
Write anything here: speech patterns, favorite phrases, tone preferences, etc.
"""

# ── Permission matrix (from 設計書/SOUL.md) ──────────────────────────
# Maps each mutable field to the set of callers allowed to change it.
# Fields NOT listed (immutable_rules, emotion_states) cannot be changed
# by anyone at runtime.

_PERMISSIONS: dict[str, frozenset[SoulCaller]] = {
    "emotion": frozenset({SoulCaller.AI, SoulCaller.SYSTEM}),
    "traits": frozenset({SoulCaller.SYSTEM}),
    "name": frozenset({SoulCaller.USER}),
    "quiet_hours": frozenset({SoulCaller.SYSTEM, SoulCaller.USER}),
    "notes": frozenset({SoulCaller.SYSTEM, SoulCaller.USER}),
}


def _check_permission(field: str, caller: SoulCaller) -> None:
    """Raise SoulPermissionError if caller is not allowed to modify field."""
    allowed = _PERMISSIONS.get(field, frozenset())
    if caller not in allowed:
        msg = (
            f"{caller.value!r} is not allowed to modify {field!r}. "
            f"Allowed: {sorted(c.value for c in allowed)}"
        )
        raise SoulPermissionError(msg)


class Soul:
    """Manages the agent's identity, personality, and emotional state."""

    def __init__(self, soul_dir: str | Path) -> None:
        self._soul_dir = Path(soul_dir)
        self._core: MappingProxyType[str, Any] = MappingProxyType({})
        self._soul: dict[str, Any] = {}
        self._notes: str = ""
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
    def notes(self) -> str:
        return self._notes

    @property
    def emotion(self) -> EmotionState:
        emo = self._soul.get("current_emotion", {})
        secondary_raw = emo.get("secondary")
        secondary = Emotion(secondary_raw) if secondary_raw else None
        return EmotionState(
            primary=Emotion(emo.get("primary", Emotion.CALM.value)),
            primary_intensity=float(emo.get("primary_intensity", 0.5)),
            secondary=secondary,
            secondary_intensity=float(emo.get("secondary_intensity", 0.0)),
        )

    @property
    def quiet_hours(self) -> tuple[str, str]:
        hb = self._soul.get("heartbeat", {}).get("quiet_hours", {})
        return (hb.get("start", "01:00"), hb.get("end", "07:00"))

    # ── mutation (only allowed fields) ────────────────────────────────

    def update_emotion(
        self,
        emotion: Emotion,
        intensity: float,
        *,
        secondary: Emotion | None = None,
        secondary_intensity: float | None = None,
        caller: SoulCaller = SoulCaller.AI,
    ) -> None:
        """Update emotion with automatic transition logic.

        When the primary emotion changes, the old primary becomes the
        secondary (with its intensity halved) unless an explicit secondary
        is provided.
        """
        _check_permission("emotion", caller)
        intensity = max(0.0, min(1.0, intensity))
        emo = self._soul.setdefault("current_emotion", {})

        old_primary = emo.get("primary", Emotion.CALM.value)
        old_primary_intensity = float(emo.get("primary_intensity", 0.5))

        # Transition: old primary → secondary (unless overridden)
        if emotion.value != old_primary and secondary is None:
            secondary = Emotion(old_primary)
            secondary_intensity = old_primary_intensity * 0.5

        emo["primary"] = emotion.value
        emo["primary_intensity"] = intensity

        if secondary is not None:
            sec_int = max(0.0, min(1.0, secondary_intensity or 0.0))
            emo["secondary"] = secondary.value
            emo["secondary_intensity"] = sec_int
        # If secondary not provided and primary didn't change, leave it alone

        self._append_emotion_history(emotion, intensity)
        self._save_soul()

    def decay_emotion(self) -> None:
        """Apply natural decay toward CALM/baseline.

        Called by the heartbeat tick to simulate emotional cooldown.
        """
        emo = self._soul.get("current_emotion", {})
        changed = False

        # Decay primary intensity toward baseline
        primary_intensity = float(emo.get("primary_intensity", 0.5))
        if emo.get("primary", Emotion.CALM.value) != Emotion.CALM.value:
            new_intensity = primary_intensity - _DECAY_RATE
            if new_intensity <= _BASELINE_INTENSITY:
                # Emotion faded enough — revert to calm
                emo["primary"] = Emotion.CALM.value
                emo["primary_intensity"] = _BASELINE_INTENSITY
            else:
                emo["primary_intensity"] = new_intensity
            changed = True

        # Decay secondary
        sec_intensity = float(emo.get("secondary_intensity", 0.0))
        if sec_intensity > 0:
            new_sec = sec_intensity - _DECAY_RATE
            if new_sec < _SECONDARY_CLEAR_THRESHOLD:
                emo.pop("secondary", None)
                emo["secondary_intensity"] = 0.0
            else:
                emo["secondary_intensity"] = new_sec
            changed = True

        if changed:
            self._save_soul()

    @property
    def emotion_history(self) -> list[dict[str, Any]]:
        """Return recent emotion change log (newest first)."""
        return list(self._soul.get("emotion_history", []))

    def _append_emotion_history(self, emotion: Emotion, intensity: float) -> None:
        """Record an emotion change, keeping last 50 entries."""
        history: list[dict[str, Any]] = self._soul.setdefault("emotion_history", [])
        history.append(
            {
                "emotion": emotion.value,
                "intensity": round(intensity, 2),
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Keep last 50
        if len(history) > 50:
            self._soul["emotion_history"] = history[-50:]

    def update_name(self, name: str, *, caller: SoulCaller = SoulCaller.USER) -> None:
        _check_permission("name", caller)
        self._soul.setdefault("identity", {})
        self._soul["identity"]["name"] = name
        self._save_soul()

    def update_notes(
        self,
        notes: str,
        *,
        caller: SoulCaller = SoulCaller.USER,
    ) -> None:
        """Update soul_notes.md content."""
        _check_permission("notes", caller)
        self._notes = notes
        self._save_notes()

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
        *,
        caller: SoulCaller = SoulCaller.SYSTEM,
    ) -> None:
        """Apply an approved trait change."""
        _check_permission("traits", caller)
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

    def update_quiet_hours(
        self,
        start: str,
        end: str,
        *,
        caller: SoulCaller = SoulCaller.USER,
    ) -> None:
        """Update heartbeat quiet-hours window."""
        _check_permission("quiet_hours", caller)
        hb = self._soul.setdefault("heartbeat", {})
        hb["quiet_hours"] = {"start": start, "end": end}
        self._save_soul()

    def get_soul_snapshot(self) -> dict[str, Any]:
        """Return a read-only snapshot for prompt injection into AI."""
        emo = self.emotion
        emotion_data: dict[str, Any] = {
            "primary": emo.primary.value,
            "intensity": emo.primary_intensity,
        }
        if emo.secondary is not None:
            emotion_data["secondary"] = emo.secondary.value
            emotion_data["secondary_intensity"] = emo.secondary_intensity
        return {
            "name": self.name,
            "pronoun": self.pronoun,
            "traits": self.traits,
            "emotion": emotion_data,
            "immutable_rules": self.immutable_rules,
            "notes": self._notes,
        }

    # ── persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        self._soul_dir.mkdir(parents=True, exist_ok=True)
        core_path = self._soul_dir / "soul_core.yaml"
        soul_path = self._soul_dir / "soul.yaml"
        notes_path = self._soul_dir / "soul_notes.md"

        if core_path.exists():
            with core_path.open(encoding="utf-8") as f:
                raw_core = yaml.safe_load(f) or {}
        else:
            raw_core = copy.deepcopy(_DEFAULT_SOUL_CORE)
            with core_path.open("w", encoding="utf-8") as f:
                yaml.dump(raw_core, f, allow_unicode=True, default_flow_style=False)

        # Freeze core data — immutable at runtime
        self._core = MappingProxyType(raw_core)

        if soul_path.exists():
            with soul_path.open(encoding="utf-8") as f:
                self._soul = yaml.safe_load(f) or {}
        else:
            self._soul = copy.deepcopy(_DEFAULT_SOUL)
            self._save_soul()

        if notes_path.exists():
            self._notes = notes_path.read_text(encoding="utf-8")
        else:
            self._notes = _DEFAULT_SOUL_NOTES
            self._save_notes()

    def _save_soul(self) -> None:
        soul_path = self._soul_dir / "soul.yaml"
        with soul_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                self._soul,
                f,
                allow_unicode=True,
                default_flow_style=False,
            )

    def _save_notes(self) -> None:
        notes_path = self._soul_dir / "soul_notes.md"
        notes_path.write_text(self._notes, encoding="utf-8")
