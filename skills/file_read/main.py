"""file_read skill — read contents of a file within sandbox."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def execute(
    *,
    path: str,
    max_lines: int = 100,
    context: Any = None,
) -> dict[str, Any]:
    """Read a file and return its contents (limited by max_lines)."""
    target = Path(path)
    if not target.exists():
        return {"error": f"File not found: {path}"}

    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return {"error": f"Cannot read binary file: {path}"}

    truncated = len(lines) > max_lines
    content = "\n".join(lines[:max_lines])

    return {
        "path": str(target),
        "lines": min(len(lines), max_lines),
        "total_lines": len(lines),
        "truncated": truncated,
        "content": content,
    }
