"""Schema migration framework for the memory database.

Migrations are versioned, append-only, and applied in order on
:meth:`MemoryStore.initialize`. The latest applied version is recorded
in the ``schema_version`` table so subsequent boots are no-ops.

Adding a new migration:

1. Append a :class:`Migration` to :data:`MIGRATIONS` with the next
   integer version. **Never re-number or remove existing entries.**
2. Provide either ``sql`` (a string of one or more statements suitable
   for ``executescript``) or ``callable`` (an async function receiving
   the open connection) — not both.
3. If your migration touches data, write an idempotent statement so
   re-running on a partial failure remains safe.

Existing pre-migration databases (deployed before this framework
landed) are detected via the presence of the ``users`` table and
fast-forwarded to version 1 without re-running its statements.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import aiosqlite

from cordbeat._memory_common import SCHEMA, VECTOR_SCHEMA

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Migration:
    """A single, atomically-applied schema change."""

    version: int
    description: str
    sql: str | None = None
    callable: Callable[[aiosqlite.Connection], Awaitable[None]] | None = None

    def __post_init__(self) -> None:
        if (self.sql is None) == (self.callable is None):
            raise ValueError(
                "Migration must specify exactly one of sql/callable "
                f"(version={self.version})"
            )


MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        description="initial schema (users, platform_links, profiles, "
        "records, conversations, sqlite-vec virtual tables)",
        sql=SCHEMA + VECTOR_SCHEMA,
    ),
]


async def _ensure_version_table(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version    INTEGER PRIMARY KEY,
            applied_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    await conn.commit()


async def _current_version(conn: aiosqlite.Connection) -> int:
    cur = await conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    row = await cur.fetchone()
    await cur.close()
    return int(row[0]) if row and row[0] is not None else 0


async def _table_exists(conn: aiosqlite.Connection, name: str) -> bool:
    cur = await conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    )
    row = await cur.fetchone()
    await cur.close()
    return row is not None


async def apply_migrations(conn: aiosqlite.Connection) -> int:
    """Apply all pending migrations and return the resulting version.

    Pre-existing databases that pre-date the migration framework
    (have ``users`` but not ``schema_version``) are silently
    fast-forwarded to version 1 without re-executing the initial
    schema, since ``CREATE TABLE IF NOT EXISTS`` already produced an
    equivalent state.
    """
    await _ensure_version_table(conn)
    current = await _current_version(conn)

    if current == 0 and await _table_exists(conn, "users"):
        # Legacy pre-framework DB — schema is already at version 1.
        logger.info(
            "Detected legacy schema with no schema_version row; stamping as version 1"
        )
        await conn.execute(
            "INSERT INTO schema_version (version) VALUES (1)",
        )
        await conn.commit()
        current = 1

    pending = [m for m in MIGRATIONS if m.version > current]
    if not pending:
        logger.debug("Schema up to date at version %d", current)
        return current

    pending.sort(key=lambda m: m.version)
    for m in pending:
        logger.info("Applying schema migration %d: %s", m.version, m.description)
        if m.sql is not None:
            await conn.executescript(m.sql)
        else:
            assert m.callable is not None
            await m.callable(conn)
        await conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (m.version,),
        )
        await conn.commit()
        current = m.version

    logger.info("Schema migrated to version %d", current)
    return current


def latest_version() -> int:
    """The highest version present in :data:`MIGRATIONS`."""
    return max((m.version for m in MIGRATIONS), default=0)
