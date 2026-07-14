"""CordBeat — A local-first autonomous AI agent."""

from .exceptions import (
    AIBackendError,
    CordBeatError,
    MemorySubsystemError,
    OutputValidationError,
    SkillError,
    SkillExecutionError,
)

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "CordBeatError",
    "SkillError",
    "SkillExecutionError",
    "MemorySubsystemError",
    "AIBackendError",
    "OutputValidationError",
]
