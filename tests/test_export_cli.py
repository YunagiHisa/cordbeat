"""Tests for the conversation export CLI (D-9)."""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from textwrap import dedent

import pytest

from cordbeat import export_cli


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(
            dedent(
                """
                CREATE TABLE conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    adapter_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                INSERT INTO conversation_messages
                    (user_id, role, content, adapter_id, created_at)
                VALUES
                    ('alice', 'user', 'hello', 'cli', '2025-01-01T10:00:00+00:00'),
                    ('alice', 'assistant', 'hi there', 'cli',
                        '2025-01-01T10:00:01+00:00'),
                    ('alice', 'user', 'multi\nline\nmsg', 'cli',
                        '2025-01-02T09:00:00+00:00'),
                    ('bob', 'user', 'how are you', 'discord',
                        '2025-01-15T12:00:00+00:00'),
                    ('bob', 'assistant', 'great', 'discord',
                        '2025-01-15T12:00:05+00:00');
                """
            ).strip()
        )
        conn.commit()
    finally:
        conn.close()
    return db


class TestFetchMessages:
    def test_returns_chronological_order(self, populated_db: Path) -> None:
        msgs = export_cli.fetch_messages(populated_db, "alice")
        assert len(msgs) == 3
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert [m["content"] for m in msgs] == ["hello", "hi there", "multi\nline\nmsg"]

    def test_since_filter(self, populated_db: Path) -> None:
        msgs = export_cli.fetch_messages(populated_db, "alice", since="2025-01-02")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "multi\nline\nmsg"

    def test_until_filter_is_exclusive(self, populated_db: Path) -> None:
        msgs = export_cli.fetch_messages(populated_db, "alice", until="2025-01-02")
        assert len(msgs) == 2
        assert all(m["content"] != "multi\nline\nmsg" for m in msgs)

    def test_unknown_user_returns_empty(self, populated_db: Path) -> None:
        assert export_cli.fetch_messages(populated_db, "nobody") == []


class TestFormatAsJson:
    def test_includes_count_and_user(self, populated_db: Path) -> None:
        msgs = export_cli.fetch_messages(populated_db, "alice")
        rendered = export_cli.format_as_json("alice", msgs, db_path=populated_db)
        payload = json.loads(rendered)
        assert payload["user_id"] == "alice"
        assert payload["message_count"] == 3
        assert payload["messages"][0]["content"] == "hello"
        assert payload["source_db"] == str(populated_db)

    def test_unicode_preserved(self) -> None:
        rendered = export_cli.format_as_json(
            "u",
            [
                {
                    "role": "user",
                    "content": "こんにちは🌸",
                    "adapter_id": "",
                    "created_at": "2025-01-01T00:00:00+00:00",
                }
            ],
        )
        assert "こんにちは🌸" in rendered


class TestFormatAsMarkdown:
    def test_renders_headers_and_blockquotes(self, populated_db: Path) -> None:
        msgs = export_cli.fetch_messages(populated_db, "alice")
        rendered = export_cli.format_as_markdown("alice", msgs)
        assert "# Conversation export — `alice`" in rendered
        assert "**Messages:** 3" in rendered
        assert "### user · 2025-01-01T10:00:00+00:00" in rendered
        # Multi-line content quoted with leading "> "
        assert "> multi" in rendered
        assert "> line" in rendered
        assert "> msg" in rendered

    def test_handles_empty_messages(self) -> None:
        rendered = export_cli.format_as_markdown("ghost", [])
        assert "# Conversation export — `ghost`" in rendered
        assert "**Messages:** 0" in rendered


class TestExportMain:
    def _patch_db_path(self, monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
        monkeypatch.setattr(export_cli, "_resolve_db_path", lambda _config: db)

    def test_list_users_when_no_user_id(
        self,
        populated_db: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._patch_db_path(monkeypatch, populated_db)
        rc = export_cli.export_main([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "alice" in out and "bob" in out

    def test_export_json_to_stdout(
        self,
        populated_db: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._patch_db_path(monkeypatch, populated_db)
        rc = export_cli.export_main(["--user-id", "alice", "--format", "json"])
        out = capsys.readouterr().out
        assert rc == 0
        payload = json.loads(out)
        assert payload["user_id"] == "alice"
        assert payload["message_count"] == 3

    def test_export_md_to_file(
        self,
        populated_db: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._patch_db_path(monkeypatch, populated_db)
        out_file = tmp_path / "out.md"
        rc = export_cli.export_main(
            [
                "--user-id",
                "bob",
                "--format",
                "md",
                "--output",
                str(out_file),
            ]
        )
        assert rc == 0
        text = out_file.read_text(encoding="utf-8")
        assert "# Conversation export — `bob`" in text
        assert "how are you" in text
        assert "great" in text

    def test_missing_db_returns_2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ghost = tmp_path / "missing.sqlite"
        monkeypatch.setattr(export_cli, "_resolve_db_path", lambda _config: ghost)
        rc = export_cli.export_main(["--user-id", "alice"])
        assert rc == 2

    def test_since_and_until_flags(
        self,
        populated_db: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._patch_db_path(monkeypatch, populated_db)
        rc = export_cli.export_main(
            [
                "--user-id",
                "alice",
                "--since",
                "2025-01-01",
                "--until",
                "2025-01-02",
                "--format",
                "json",
            ]
        )
        out = capsys.readouterr().out
        assert rc == 0
        payload = json.loads(out)
        assert payload["message_count"] == 2

    def test_no_users_prints_to_stderr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        empty = tmp_path / "empty.sqlite"
        conn = sqlite3.connect(str(empty))
        try:
            conn.execute(
                "CREATE TABLE conversation_messages ("
                " id INTEGER PRIMARY KEY, user_id TEXT, role TEXT, "
                " content TEXT, adapter_id TEXT, created_at TEXT)"
            )
            conn.commit()
        finally:
            conn.close()
        monkeypatch.setattr(export_cli, "_resolve_db_path", lambda _config: empty)
        rc = export_cli.export_main([])
        captured = capsys.readouterr()
        assert rc == 0
        assert "No conversations" in captured.err


# Silence "unused import" lint for module-only side effects.
_ = io
