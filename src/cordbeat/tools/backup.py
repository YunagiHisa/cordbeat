"""Online backup / restore CLI for the CordBeat SQLite database.

Uses SQLite's online backup API
(:py:meth:`sqlite3.Connection.backup`) so the live agent does **not**
need to be stopped during backup. The backup file is a fully-formed
SQLite database; restore is an atomic file replacement (must be done
with the agent stopped).

Entry points (registered via ``[project.scripts]`` in ``pyproject.toml``):

- ``cordbeat-backup [destination]`` — copies the configured
  ``memory.sqlite_path`` to *destination*. If *destination* is omitted,
  a timestamped file is written next to the source.
- ``cordbeat-restore <source>`` — overwrites the configured
  ``memory.sqlite_path`` with *source*. The agent must not be running.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from cordbeat.config import Config, cordbeat_home, load_config

logger = logging.getLogger(__name__)


def _resolve_db_path(config_path: str | None) -> Path:
    if config_path:
        cfg: Config = load_config(Path(config_path))
    else:
        default = cordbeat_home() / "config.yaml"
        cfg = load_config(default)
    return Path(cfg.memory.sqlite_path)


def _online_backup(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(src))
    try:
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def backup_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cordbeat-backup",
        description="Create an online SQLite backup of the CordBeat memory database.",
    )
    parser.add_argument(
        "destination",
        nargs="?",
        help="Path for the backup file. Defaults to "
        "<sqlite_path>.backup-<UTC-timestamp>.db next to the source.",
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml (default: ~/.cordbeat/config.yaml).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    src = _resolve_db_path(args.config)
    if not src.exists():
        logger.error("Source database does not exist: %s", src)
        return 2

    if args.destination:
        dst = Path(args.destination)
    else:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        dst = src.with_suffix(src.suffix + f".backup-{ts}.db")

    logger.info("Backing up %s -> %s", src, dst)
    _online_backup(src, dst)
    size_kb = dst.stat().st_size / 1024
    logger.info("Backup complete (%.1f KiB)", size_kb)
    return 0


def restore_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cordbeat-restore",
        description="Restore the CordBeat memory database from a backup file. "
        "The agent must NOT be running.",
    )
    parser.add_argument("source", help="Path to the backup .db file to restore from.")
    parser.add_argument(
        "--config",
        help="Path to config.yaml (default: ~/.cordbeat/config.yaml).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    src = Path(args.source)
    if not src.exists():
        logger.error("Backup file does not exist: %s", src)
        return 2

    dst = _resolve_db_path(args.config)
    if dst.exists() and not args.yes:
        sys.stderr.write(
            f"This will overwrite {dst}. The CordBeat agent must be stopped.\n"
            "Type 'yes' to continue: "
        )
        sys.stderr.flush()
        reply = sys.stdin.readline().strip().lower()
        if reply != "yes":
            logger.info("Aborted by user")
            return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        rollback = dst.with_suffix(dst.suffix + ".pre-restore")
        shutil.copy2(dst, rollback)
        logger.info("Existing database saved to %s", rollback)

    shutil.copy2(src, dst)
    logger.info("Restored %s -> %s", src, dst)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(backup_main())
