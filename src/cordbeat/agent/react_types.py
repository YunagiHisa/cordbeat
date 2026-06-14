"""ReAct loop type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MediaArtifact:
    """Ephemeral media returned by a tool or attached to the current turn."""

    image_b64: str
    relation: str
    mime_type: str = "image/jpeg"
    source_ref: str = ""


@dataclass
class SkillCall:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    skill_name: str
    params: dict[str, Any]
    output: str
    is_error: bool = False
    media: list[MediaArtifact] = field(default_factory=list)


@dataclass
class ToolTrace:
    calls: list[ToolCallResult] = field(default_factory=list)
    iterations: int = 0
