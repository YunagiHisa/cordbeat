"""Tests for the schema migration framework."""

from __future__ import annotations

import aiosqlite
import pytest
import sqlite_vec

from cordbeat._memory_migrations import (
    MIGRATIONS,
    Migration,
    apply_migrations,
    latest_version,
)


async def _open(db_path: str) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.enable_load_extension(True)
    await conn.load_extension(sqlite_vec.loadable_path())
    await conn.enable_load_extension(False)
    return conn


async def test_apply_migrations_on_empty_db_creates_full_schema(tmp_path):
    db = tmp_path / "fresh.db"
    conn = await _open(str(db))
    try:
        version = await apply_migrations(conn)
        assert version == latest_version()

        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('users','platform_links','core_profiles',"
            "'certain_records','conversation_messages','semantic_memory',"
            "'episodic_memory','schema_version')"
        )
        rows = await cur.fetchall()
        await cur.close()
        names = {r["name"] for r in rows}
        for required in (
            "users",
            "platform_links",
            "core_profiles",
            "certain_records",
            "conversation_messages",
            "semantic_memory",
            "episodic_memory",
            "schema_version",
        ):
            assert required in names

        cur = await conn.execute("SELECT MAX(version) FROM schema_version")
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == latest_version()
    finally:
        await conn.close()


async def test_apply_migrations_idempotent(tmp_path):
    db = tmp_path / "twice.db"
    conn = await _open(str(db))
    try:
        v1 = await apply_migrations(conn)
        v2 = await apply_migrations(conn)
        assert v1 == v2 == latest_version()

        cur = await conn.execute("SELECT COUNT(*) FROM schema_version")
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == 1
    finally:
        await conn.close()


async def test_legacy_db_without_schema_version_is_fast_forwarded(tmp_path):
    db = tmp_path / "legacy.db"
    conn = await _open(str(db))
    try:
        # Simulate a pre-framework DB: create the users table directly.
        await conn.execute(
            "CREATE TABLE users (user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL)"
        )
        await conn.commit()

        version = await apply_migrations(conn)
        assert version == latest_version()

        cur = await conn.execute("SELECT version FROM schema_version ORDER BY version")
        rows = await cur.fetchall()
        await cur.close()
        # Only the fast-forward stamp; no re-execution of v1.
        assert [r[0] for r in rows] == [1]
    finally:
        await conn.close()


def test_migration_requires_exactly_one_of_sql_or_callable():
    with pytest.raises(ValueError, match="exactly one"):
        Migration(version=99, description="bad")
    with pytest.raises(ValueError, match="exactly one"):
        Migration(
            version=99,
            description="bad",
            sql="SELECT 1",
            callable=lambda c: None,  # type: ignore[arg-type, return-value]
        )


def test_migrations_list_is_strictly_increasing():
    versions = [m.version for m in MIGRATIONS]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)
    assert versions[0] == 1
