"""file_write skill — write content to a file within sandbox."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
