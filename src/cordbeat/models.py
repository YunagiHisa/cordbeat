"""Core data models for CordBeat."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

# ── Emotions ──────────────────────────────────────────────────────────


class Emotion(StrEnum):
    JOY = "joy"
    EXCITEMENT = "excitement"
    CURIOSITY = "curiosity"
    WARMTH = "warmth"
    CALM = "calm"
    BOREDOM = "boredom"
    WORRY = "worry"
    LONELINESS = "loneliness"
    SADNESS = "sadness"


@dataclass
class EmotionState:
    primary: Emotion = Emotion.CALM
    primary_intensity: float = 0.5
    secondary: Emotion | None = None
    secondary_intensity: float = 0.0

    def __post_init__(self) -> None:
        self.primary_intensity = max(0.0, min(1.0, self.primary_intensity))
        self.secondary_intensity = max(0.0, min(1.0, self.secondary_intensity))


# ── Gateway messages ──────────────────────────────────────────────────


class MessageType(StrEnum):
    MESSAGE = "message"
    HEARTBEAT_MESSAGE = "heartbeat_message"
    ACK = "ack"
    ERROR = "error"
    LINK_REQUEST = "link_request"
    LINK_CONFIRM = "link_confirm"


@dataclass
class GatewayMessage:
    type: MessageType
    adapter_id: str
    platform_user_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── User ──────────────────────────────────────────────────────────────


@dataclass
class UserSummary:
    user_id: str
    display_name: str
    last_talked_at: datetime | None = None
    last_platform: str | None = None
    last_topic: str = ""
    emotional_tone: str = ""
    attention_score: float = 0.5

    def __post_init__(self) -> None:
        self.attention_score = max(0.0, min(1.0, self.attention_score))


@dataclass
class PlatformLink:
    user_id: str
    adapter_id: str
    platform_user_id: str
    linked_at: datetime = field(default_factory=datetime.now)


# ── HEARTBEAT ─────────────────────────────────────────────────────────


class HeartbeatAction(StrEnum):
    MESSAGE = "message"
    SKILL = "skill"
    PROPOSE_IMPROVEMENT = "propose_improvement"
    PROPOSE_TRAIT_CHANGE = "propose_trait_change"
    NONE = "none"


# ── Proposals ─────────────────────────────────────────────────────────


class ProposalType(StrEnum):
    SKILL_EXECUTION = "skill_execution"
    TRAIT_CHANGE = "trait_change"
    GENERAL = "general"


class ProposalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"


@dataclass
class HeartbeatDecision:
    action: HeartbeatAction
    content: str = ""
    skill_name: str | None = None
    skill_params: dict[str, Any] = field(default_factory=dict)
    trait_add: list[str] = field(default_factory=list)
    trait_remove: list[str] = field(default_factory=list)
    target_user_id: str | None = None
    target_adapter_id: str | None = None
    next_heartbeat_minutes: int = 60


# ── SKILL ─────────────────────────────────────────────────────────────


class SafetyLevel(StrEnum):
    SAFE = "safe"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    DANGEROUS = "dangerous"


@dataclass
class SkillParam:
    name: str
    type: str
    required: bool = True
    description: str = ""


@dataclass
class SkillMeta:
    name: str
    description: str
    usage: str
    parameters: list[SkillParam] = field(default_factory=list)
    safety_level: SafetyLevel = SafetyLevel.SAFE
    sandbox: bool = False
    network: bool = False
    filesystem: bool = False
    enabled: bool = True


@dataclass
class SkillContext:
    """Runtime permissions passed to skill execute() functions."""

    sandbox: bool = False
    network: bool = False
    filesystem: bool = False
    work_dir: Path | None = None
    memory: Any | None = None


# ── MEMORY ────────────────────────────────────────────────────────────


class MemoryLayer(StrEnum):
    CORE_PROFILE = "core_profile"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    CERTAIN_RECORD = "certain_record"


class TrustLevel(int, Enum):
    ALMOST_FORGOTTEN = 0
    VAGUE = 1
    NORMAL = 2
    CERTAIN = 3


@dataclass
class MemoryEntry:
    id: str
    user_id: str
    layer: MemoryLayer
    content: str
    trust_level: TrustLevel = TrustLevel.NORMAL
    strength: float = 1.0
    emotion_weight: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Validation ────────────────────────────────────────────────────────


@dataclass
class ValidationError:
    field: str
    message: str
    value: Any = None


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def error_summary(self) -> str:
        return "\n".join(f"- {e.field}: {e.message}" for e in self.errors)
