"""file_read skill — read contents of a file within sandbox."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

_MAX_READ_BYTES = 1_000_000


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
    max_lines: int = 100,
    context: Any = None,
) -> dict[str, Any]:
    """Read a bounded text prefix without loading the whole file.

    ``total_lines`` is exact when ``truncated`` is false. For truncated files it
    reports only the number of lines inspected before a configured limit stopped
    the read.
    """
    if not _is_safe_relative(str(path)):
        return {"error": "path must be sandbox-relative"}

    target = Path(path)
    if not target.exists():
        return {"error": f"File not found: {path}"}

    line_limit = max(0, max_lines)
    lines: list[str] = []
    total_lines = 0
    total_bytes = 0
    truncated = False
    try:
        with target.open(encoding="utf-8") as file_handle:
            file_handle.read(4096)
            file_handle.seek(0)
            while True:
                line = file_handle.readline(_MAX_READ_BYTES + 1)
                if not line:
                    break
                total_lines += 1
                encoded = line.encode("utf-8")
                remaining = _MAX_READ_BYTES - total_bytes
                if len(encoded) > remaining:
                    prefix = encoded[:remaining].decode("utf-8", errors="ignore")
                    if len(lines) < line_limit and prefix:
                        lines.append(prefix.rstrip("\r\n"))
                    truncated = True
                    break
                total_bytes += len(encoded)
                if len(lines) >= line_limit:
                    truncated = True
                    break
                lines.append(line.rstrip("\r\n"))
    except UnicodeDecodeError:
        return {"error": f"Cannot read binary file: {path}"}

    content = "\n".join(lines)

    return {
        "path": str(target),
        "lines": len(lines),
        "total_lines": total_lines,
        "truncated": truncated,
        "content": content,
    }
