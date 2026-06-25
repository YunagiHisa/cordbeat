"""file_delete skill — delete files or directories within sandbox."""

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


def _delete_tree(path: Path) -> None:
    children = sorted(
        path.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for child in children:
        if child.is_dir():
            child.rmdir()
        else:
            child.unlink()
    path.rmdir()


def execute(
    *,
    path: str,
    recursive: bool = False,
    context: Any = None,
) -> dict[str, Any]:
    """Delete a sandbox-local file or directory."""
    if not _is_safe_relative(str(path)):
        return {"error": "path must be sandbox-relative"}

    target = Path(path)
    if not target.exists():
        return {"error": f"not found: {path}", "path": str(target)}

    try:
        if target.is_dir():
            if recursive:
                _delete_tree(target)
            else:
                target.rmdir()
            kind = "directory"
        else:
            target.unlink()
            kind = "file"
    except OSError as exc:
        return {"error": f"failed to delete: {exc}", "path": str(target)}

    return {
        "path": str(target),
        "kind": kind,
        "recursive": bool(recursive),
        "status": "ok",
    }
