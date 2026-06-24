"""Shared action budgeting for user turns and heartbeat ticks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionBudget:
    """Small shared counter for bounded tool/action execution."""

    limit: int
    scope: str
    used: int = 0
    labels: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def consume(self, label: str) -> bool:
        """Consume one action slot, returning False when the budget is spent."""
        if self.exhausted:
            return False
        self.used += 1
        self.labels.append(label)
        return True
