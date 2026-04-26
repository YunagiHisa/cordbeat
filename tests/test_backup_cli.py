"""Tests for the cordbeat-backup / cordbeat-restore CLIs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

from cordbeat import backup_cli


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump({"memory": {"sqlite_path": str(db_path)}}),
        encoding="utf-8",
    )
    return cfg


def _seed_db(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS marker (v TEXT)")
        conn.execute("DELETE FROM marker")
        conn.execute("INSERT INTO marker (v) VALUES (?)", (value,))
        conn.commit()
    finally:
        conn.close()


def _read_marker(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.execute("SELECT v FROM marker")
        row = cur.fetchone()
        return str(row[0])
    finally:
        conn.close()


def test_backup_creates_copy_with_explicit_destination(tmp_path):
    db = tmp_path / "src.db"
    _seed_db(db, "hello")
    cfg = _write_config(tmp_path, db)
    dst = tmp_path / "out" / "snapshot.db"

    rc = backup_cli.backup_main(["--config", str(cfg), str(dst)])

    assert rc == 0
    assert dst.exists()
    assert _read_marker(dst) == "hello"


def test_backup_default_destination_is_timestamped(tmp_path):
    db = tmp_path / "src.db"
    _seed_db(db, "x")
    cfg = _write_config(tmp_path, db)

    rc = backup_cli.backup_main(["--config", str(cfg)])

    assert rc == 0
    backups = list(db.parent.glob(f"{db.name}.backup-*.db"))
    assert len(backups) == 1
    assert _read_marker(backups[0]) == "x"


def test_backup_returns_nonzero_when_source_missing(tmp_path):
    cfg = _write_config(tmp_path, tmp_path / "nope.db")
    rc = backup_cli.backup_main(["--config", str(cfg), str(tmp_path / "out.db")])
    assert rc == 2


def test_restore_overwrites_destination(tmp_path):
    db = tmp_path / "live.db"
    _seed_db(db, "old")
    cfg = _write_config(tmp_path, db)

    snap = tmp_path / "snap.db"
    _seed_db(snap, "new")

    rc = backup_cli.restore_main(["--config", str(cfg), "--yes", str(snap)])

    assert rc == 0
    assert _read_marker(db) == "new"
    pre = db.with_suffix(db.suffix + ".pre-restore")
    assert pre.exists()
    assert _read_marker(pre) == "old"


def test_restore_returns_nonzero_when_source_missing(tmp_path):
    cfg = _write_config(tmp_path, tmp_path / "live.db")
    rc = backup_cli.restore_main(
        ["--config", str(cfg), "--yes", str(tmp_path / "missing.db")]
    )
    assert rc == 2


def test_restore_aborts_when_user_declines(tmp_path, monkeypatch):
    db = tmp_path / "live.db"
    _seed_db(db, "keep")
    cfg = _write_config(tmp_path, db)

    snap = tmp_path / "snap.db"
    _seed_db(snap, "discarded")

    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("no\n"))

    rc = backup_cli.restore_main(["--config", str(cfg), str(snap)])

    assert rc == 1
    assert _read_marker(db) == "keep"


@pytest.mark.parametrize(
    "entrypoint", [backup_cli.backup_main, backup_cli.restore_main]
)
def test_help_exits_cleanly(entrypoint, capsys):
    with pytest.raises(SystemExit) as exc:
        entrypoint(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "cordbeat" in captured.out.lower()
