"""Tests for the CordBeat exception hierarchy.

The gap analysis (2026-04-20) called out a need for a unified typed
exception hierarchy so callers can distinguish CordBeat-specific errors
from third-party / stdlib exceptions. This module pins the contract:

* Every CordBeat-specific error ultimately inherits from
  :class:`cordbeat.CordBeatError`.
* Skill-layer errors additionally inherit from
  :class:`cordbeat.SkillError`.
* :class:`cordbeat.exceptions` is the canonical import site; the
  classes remain re-exported from their originating modules for
  backward compatibility.
"""

from __future__ import annotations

import cordbeat
from cordbeat import exceptions
from cordbeat.models import SoulPermissionError
from cordbeat.skill_sandbox import SkillPermissionError, SkillSandboxError
from cordbeat.skill_validator import SkillValidationError


def test_base_is_exception() -> None:
    assert issubclass(cordbeat.CordBeatError, Exception)


def test_soul_permission_error_is_cordbeat_error() -> None:
    assert issubclass(SoulPermissionError, cordbeat.CordBeatError)


def test_skill_errors_are_skill_error() -> None:
    assert issubclass(SkillPermissionError, exceptions.SkillError)
    assert issubclass(SkillSandboxError, exceptions.SkillError)
    assert issubclass(SkillValidationError, exceptions.SkillError)


def test_skill_error_is_cordbeat_error() -> None:
    assert issubclass(exceptions.SkillError, cordbeat.CordBeatError)


def test_validation_error_keeps_value_error() -> None:
    # Code that used ``except ValueError:`` on skill validation must
    # continue to work after the hierarchy change.
    assert issubclass(SkillValidationError, ValueError)


def test_catch_all_cordbeat_errors() -> None:
    for exc in (
        SoulPermissionError("x"),
        SkillPermissionError("x"),
        SkillSandboxError("x"),
        SkillValidationError("x"),
        exceptions.MemorySubsystemError("x"),
        exceptions.AIBackendError("x"),
        exceptions.OutputValidationError("x"),
    ):
        try:
            raise exc
        except cordbeat.CordBeatError as caught:
            assert caught is exc
