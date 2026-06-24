"""Shared policy helpers for deciding how skills may run."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from cordbeat.models import SafetyLevel

SANDBOX_SHARED_WORK_DIR = "shared"

_SANDBOX_LOCAL_FILE_PARAMS: dict[str, str] = {
    "file_read": "path",
    "file_write": "path",
    "file_search": "root",
}


def is_sandbox_relative_path(value: Any) -> bool:
    """Return true when a file parameter stays within CordBeat's sandbox."""
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    if not text:
        return False
    windows = PureWindowsPath(text)
    posix = PurePosixPath(text)
    if (
        windows.is_absolute()
        or posix.is_absolute()
        or windows.drive
        or windows.root
        or posix.root
    ):
        return False
    parts = tuple(windows.parts) + tuple(posix.parts)
    if any(part == ".." or part.startswith("~") for part in parts):
        return False
    return True


def sandbox_overrides_for_skill(
    skill_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Return execution overrides for sandbox-local file tool calls."""
    path_param = _SANDBOX_LOCAL_FILE_PARAMS.get(skill_name)
    if path_param is None:
        return {}
    if is_sandbox_relative_path(params.get(path_param)):
        return {
            "filesystem": False,
            "work_dir": SANDBOX_SHARED_WORK_DIR,
        }
    return {}


def skill_requires_confirmation(skill: Any, params: dict[str, Any]) -> bool:
    """Return whether a skill call must be approved before execution."""
    meta = skill.meta
    if sandbox_overrides_for_skill(meta.name, params).get("filesystem") is False:
        return False
    return meta.safety_level != SafetyLevel.SAFE or bool(meta.filesystem)
