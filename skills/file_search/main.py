"""file_search skill - scan local files within an approved directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def _as_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _looks_textual(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES or not path.suffix


def _content_matches(path: Path, query: str, max_file_bytes: int) -> bool:
    if not query or not _looks_textual(path):
        return False
    try:
        with path.open("rb") as f:
            data = f.read(max_file_bytes + 1)
    except OSError:
        return False
    if b"\x00" in data:
        return False
    text = data[:max_file_bytes].decode("utf-8", errors="ignore").lower()
    return query.lower() in text


def execute(
    *,
    root: str,
    query: str = "",
    name_glob: str = "*",
    max_results: int = 50,
    max_file_bytes: int = 65536,
    context: Any = None,
) -> dict[str, Any]:
    """Search files below *root* by glob, filename, and optional text content."""
    base = Path(root).expanduser()
    if not base.exists():
        return {"error": f"Directory not found: {root}"}
    if not base.is_dir():
        return {"error": f"Not a directory: {root}"}

    limit = _as_int(max_results, 50, minimum=1, maximum=200)
    read_limit = _as_int(max_file_bytes, 65536, minimum=1024, maximum=1048576)
    glob_text = (name_glob or "*").strip() or "*"
    query_text = (query or "").strip()
    query_lower = query_text.lower()

    matches: list[dict[str, Any]] = []
    scanned = 0
    skipped = 0

    try:
        iterator = base.rglob("*")
        for path in iterator:
            try:
                if path.is_symlink() or not path.is_file():
                    continue
            except OSError:
                skipped += 1
                continue

            scanned += 1
            rel = path.relative_to(base)
            rel_text = rel.as_posix()
            name_hit = path.match(glob_text)
            query_hit = bool(query_lower and query_lower in rel_text.lower())
            content_hit = False

            if query_text and not query_hit:
                content_hit = _content_matches(path, query_text, read_limit)

            if name_hit and (not query_text or query_hit or content_hit):
                try:
                    size = path.stat().st_size
                except OSError:
                    size = None
                matches.append(
                    {
                        "path": str(path),
                        "relative_path": rel_text,
                        "size_bytes": size,
                        "match": "content" if content_hit else "name",
                    }
                )
                if len(matches) >= limit:
                    break
    except OSError as exc:
        return {"error": f"Failed to scan directory: {exc}"}

    return {
        "root": str(base),
        "query": query_text,
        "name_glob": glob_text,
        "scanned_files": scanned,
        "skipped_files": skipped,
        "returned": len(matches),
        "truncated": len(matches) >= limit,
        "matches": matches,
    }
