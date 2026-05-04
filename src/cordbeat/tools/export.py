"""Conversation export CLI.

Exports the conversation history stored in the CordBeat SQLite database
to JSON or Markdown for backup, debugging, or sharing. Entry point is
registered as ``cordbeat-export`` in ``pyproject.toml``.

Usage::

    cordbeat-export                      # list users with messages
    cordbeat-export --user-id <id>       # export to stdout (JSON)
    cordbeat-export --user-id <id> --format md --output chat.md
    cordbeat-export --user-id <id> --since 2025-01-01 --until 2025-02-01

The agent does **not** need to be stopped — the SQL queries here are
read-only and use ``sqlite3`` directly (no schema changes).
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

from cordbeat.config import Config, cordbeat_home, load_config

logger = logging.getLogger(__name__)

__all__ = ["export_main", "fetch_messages", "format_as_json", "format_as_markdown"]


def _resolve_db_path(config_path: str | None) -> Path:
    if config_path:
        cfg: Config = load_config(Path(config_path))
    else:
        cfg = load_config(cordbeat_home() / "config.yaml")
    return Path(cfg.memory.sqlite_path)


def _list_users(db_path: Path) -> list[tuple[str, int, str, str]]:
    """Return ``(user_id, message_count, first_at, last_at)`` per user."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT user_id, COUNT(*) AS n, "
            "       MIN(created_at) AS first_at, MAX(created_at) AS last_at "
            "FROM conversation_messages "
            "GROUP BY user_id ORDER BY n DESC"
        )
        return [(r["user_id"], r["n"], r["first_at"], r["last_at"]) for r in cursor]
    finally:
        conn.close()


def fetch_messages(
    db_path: Path,
    user_id: str,
    *,
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]]:
    """Return conversation messages for ``user_id`` in chronological order.

    ``since`` / ``until`` are ISO-8601 strings (``YYYY-MM-DD`` or full
    ISO timestamps). Both bounds are inclusive on the ``since`` side and
    exclusive on the ``until`` side, matching ``ConversationStore``.
    """
    query = (
        "SELECT role, content, adapter_id, created_at "
        "FROM conversation_messages WHERE user_id = ?"
    )
    params: list[Any] = [user_id]
    if since:
        query += " AND created_at >= ?"
        params.append(since)
    if until:
        query += " AND created_at < ?"
        params.append(until)
    query += " ORDER BY created_at ASC"

    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def format_as_json(
    user_id: str, messages: list[dict[str, Any]], *, db_path: Path | None = None
) -> str:
    payload = {
        "user_id": user_id,
        "message_count": len(messages),
        "source_db": str(db_path) if db_path else None,
        "messages": messages,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def format_as_markdown(user_id: str, messages: list[dict[str, Any]]) -> str:
    lines = [f"# Conversation export — `{user_id}`", ""]
    lines.append(f"**Messages:** {len(messages)}")
    if messages:
        lines.append(f"**From:** {messages[0]['created_at']}")
        lines.append(f"**To:** {messages[-1]['created_at']}")
    lines.append("")
    for msg in messages:
        role = msg["role"]
        ts = msg["created_at"]
        adapter = msg.get("adapter_id") or ""
        header = f"### {role} · {ts}"
        if adapter:
            header += f" · _{adapter}_"
        lines.append(header)
        lines.append("")
        # Quote-block the content so Markdown rendering preserves
        # multi-line user messages without spurious heading collisions.
        for content_line in msg["content"].splitlines() or [""]:
            lines.append(f"> {content_line}")
        lines.append("")
    return "\n".join(lines)


def export_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cordbeat-export",
        description=(
            "Export CordBeat conversation history to JSON or Markdown. "
            "Without --user-id, lists users with messages."
        ),
    )
    parser.add_argument("--user-id", help="User ID to export.")
    parser.add_argument(
        "--format",
        choices=["json", "md"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--output",
        help="Write to this file instead of stdout.",
    )
    parser.add_argument(
        "--since",
        help="Only include messages with created_at >= this ISO date/timestamp.",
    )
    parser.add_argument(
        "--until",
        help="Only include messages with created_at < this ISO date/timestamp.",
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml (default: ~/.cordbeat/config.yaml).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    db_path = _resolve_db_path(args.config)
    if not db_path.exists():
        logger.error("Database does not exist: %s", db_path)
        return 2

    if not args.user_id:
        users = _list_users(db_path)
        if not users:
            print("No conversations found.", file=sys.stderr)
            return 0
        print(
            "User ID                                  Msgs  First                Last"
        )
        for uid, n, first_at, last_at in users:
            print(f"{uid:<40} {n:>5}  {first_at[:19]}  {last_at[:19]}")
        return 0

    messages = fetch_messages(db_path, args.user_id, since=args.since, until=args.until)
    if args.format == "md":
        rendered = format_as_markdown(args.user_id, messages)
    else:
        rendered = format_as_json(args.user_id, messages, db_path=db_path)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        logger.info("Wrote %d messages to %s", len(messages), out_path)
    else:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")
    return 0
