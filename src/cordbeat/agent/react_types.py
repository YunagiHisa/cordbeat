"""ReAct loop type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


@dataclass
class ToolTrace:
    calls: list[ToolCallResult] = field(default_factory=list)
    iterations: int = 0
