"""file_write skill — write content to a file within sandbox."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _is_safe_relative(path: str) -> bool:
    windows = PureWindowsPath(path)
    posix = PurePosixPath(path)
    if (
        windows.is_absolute()
        or posix.is_absolute()
        or windows.drive
        or windows.root
        or posix.root
    ):
        return False
    parts = tuple(windows.parts) + tuple(posix.parts)
    return bool(path.strip()) and not any(
        part == ".." or part.startswith("~") for part in parts
    )


def execute(
    *,
    path: str,
    content: str,
    mode: str = "write",
    context: Any = None,
) -> dict[str, Any]:
    """Write text content to a file.

    Supports 'write' (overwrite) and 'append' modes.
    """
    if mode not in ("write", "append"):
        return {"error": f"Invalid mode: {mode!r}. Use 'write' or 'append'."}

    if not _is_safe_relative(str(path)):
        return {"error": "path must be sandbox-relative"}

    target = Path(path)

    # Ensure parent directory exists
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "append":
            with target.open("a", encoding="utf-8") as f:
                f.write(content)
        else:
            target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return {"error": f"Failed to write file: {exc}"}

    return {
        "path": str(target),
        "mode": mode,
        "bytes_written": len(content.encode("utf-8")),
        "status": "ok",
    }
