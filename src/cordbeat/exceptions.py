"""CordBeat-specific exception hierarchy.

All CordBeat errors inherit from :class:`CordBeatError` so callers can
distinguish application-level failures from third-party or stdlib
exceptions. Boundary handlers (heartbeat loops, adapter message loops)
may still use broad ``except Exception:`` to avoid crashing, but internal
code should raise typed exceptions below.

Hierarchy
---------
- CordBeatError
    - SoulPermissionError
    - SkillError
        - SkillPermissionError
        - SkillSandboxError
        - SkillValidationError   (also subclasses ``ValueError``)
    - MemorySubsystemError
    - AIBackendError
    - OutputValidationError

The existing concrete classes continue to live in the modules where they
were first defined (``models``, ``skill_sandbox``, ``skill_validator``);
this module re-exports them so new callers have a single import site.
"""

from __future__ import annotations


class CordBeatError(Exception):
    """Base class for all CordBeat-specific errors."""


class SkillError(CordBeatError):
    """Base class for skill subsystem errors."""


class SkillExecutionError(SkillError):
    """Raised when a skill fails to execute for an infrastructural reason
    (missing ``execute()``, memory proxy failure, subprocess transport
    failure, etc.). Distinct from :class:`SkillValidationError`, which is
    raised for static problems in the skill source.
    """


class MemorySubsystemError(CordBeatError):
    """Raised when the memory subsystem fails irrecoverably.

    Named ``MemorySubsystemError`` rather than ``MemoryError`` to avoid
    shadowing the builtin ``MemoryError`` (raised on allocation failure).
    """


class AIBackendError(CordBeatError):
    """Raised when an AI backend call fails irrecoverably."""


class OutputValidationError(CordBeatError):
    """Raised when AI output validation fails after all retries/fallbacks.

    ``models.ValidationError`` remains a *data* class describing a single
    validation finding; this class is the *exception* equivalent.
    """


__all__ = [
    "CordBeatError",
    "SkillError",
    "SkillExecutionError",
    "MemorySubsystemError",
    "AIBackendError",
    "OutputValidationError",
]
