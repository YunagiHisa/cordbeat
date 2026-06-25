"""file_mkdir skill — create directories within sandbox."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _is_safe_relative(path: str) -> bool:
    windows = PureWindowsPath(path)
    posix = PurePosixPath(path)
    if windows.is_absolute() or posix.is_absolute() or windows.drive:
        return False
    parts = tuple(windows.parts) + tuple(posix.parts)
    return bool(path.strip()) and not any(
        part == ".." or part.startswith("~") for part in parts
    )


def execute(
    *,
    path: str,
    parents: bool = True,
    exist_ok: bool = True,
    context: Any = None,
) -> dict[str, Any]:
    """Create a sandbox-local directory."""
    if not _is_safe_relative(str(path)):
        return {"error": "path must be sandbox-relative"}

    target = Path(path)
    try:
        target.mkdir(parents=bool(parents), exist_ok=bool(exist_ok))
    except OSError as exc:
        return {"error": f"failed to create directory: {exc}", "path": str(target)}

    return {
        "path": str(target),
        "parents": bool(parents),
        "exist_ok": bool(exist_ok),
        "status": "ok",
    }
