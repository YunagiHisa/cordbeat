"""CordBeat — A local-first autonomous AI agent."""

from .exceptions import (
    AIBackendError,
    CordBeatError,
    MemorySubsystemError,
    OutputValidationError,
    SkillError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CordBeatError",
    "SkillError",
    "MemorySubsystemError",
    "AIBackendError",
    "OutputValidationError",
]
