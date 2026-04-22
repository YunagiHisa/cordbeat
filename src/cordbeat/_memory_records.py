"""Immutable records: core profiles, proposals, link tokens (private)."""

from __future__ import annotations

import json
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite

from cordbeat._memory_common import ensure_aware


class RecordStore:
    """SQLite-backed immutable records (core profiles + certain records)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def set_core_profile(
        self,
        user_id: str,
        key: str,
        value: str,
    ) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO core_profiles (user_id, key, value) "
            "VALUES (?, ?, ?)",
            (user_id, key, value),
        )
        await self._db.commit()

    async def get_core_profile(self, user_id: str) -> dict[str, str]:
        cursor = await self._db.execute(
            "SELECT key, value FROM core_profiles WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}

    async def add_certain_record(
        self,
        user_id: str,
        content: str,
        record_type: str = "log",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        record_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO certain_records "
            "(id, user_id, content, record_type, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record_id,
                user_id,
                content,
                record_type,
                datetime.now(tz=UTC).isoformat(),
                json.dumps(metadata or {}),
            ),
        )
        await self._db.commit()
        return record_id

    async def get_certain_records(
        self,
        user_id: str,
        record_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if record_type:
            cursor = await self._db.execute(
                "SELECT * FROM certain_records "
                "WHERE user_id = ? AND record_type = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, record_type, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM certain_records "
                "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        """Retrieve a single proposal by ID."""
        cursor = await self._db.execute(
            "SELECT * FROM certain_records WHERE id = ? AND record_type = 'proposal'",
            (proposal_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    # Valid proposal status transitions
    _VALID_TRANSITIONS: dict[str, set[str]] = {
        "pending": {"approved", "rejected", "expired"},
        "approved": {"executed", "expired"},
    }

    async def update_proposal_status(
        self,
        proposal_id: str,
        status: str,
    ) -> bool:
        """Atomically update a proposal's status with transition validation.

        Uses a conditional SQL UPDATE so that two concurrent callers racing
        on the same proposal cannot both succeed (lost-update). The UPDATE
        only matches rows whose current ``metadata.status`` is one of the
        valid predecessors for the requested *status*. If the row exists
        but no transition matches we distinguish:

        * ``False`` — proposal does not exist
        * ``ValueError`` — proposal exists, but transition is disallowed
          (either because the current status is terminal or because another
          concurrent caller already moved it away)

        Returns
        -------
        bool
            ``True`` if the UPDATE affected a row; ``False`` if the proposal
            id does not exist.
        """
        # Reverse map: which current statuses may transition to *status*?
        valid_sources: list[str] = [
            current
            for current, targets in self._VALID_TRANSITIONS.items()
            if status in targets
        ]
        if not valid_sources:
            # The target status has no predecessor in the state machine.
            # Treat this the same as any other disallowed transition so
            # existing callers still see ``Invalid proposal transition``.
            exists_cursor = await self._db.execute(
                "SELECT "
                "COALESCE(json_extract(metadata, '$.status'), 'pending') AS st "
                "FROM certain_records "
                "WHERE id = ? AND record_type = 'proposal'",
                (proposal_id,),
            )
            row = await exists_cursor.fetchone()
            if row is None:
                return False
            current = row["st"]
            raise ValueError(
                f"Invalid proposal transition: '{current}' → '{status}' "
                f"(allowed: set())"
            )

        placeholders = ",".join("?" for _ in valid_sources)
        # COALESCE so that rows whose metadata has no explicit status
        # (treated as 'pending') are also matched.
        sql = (
            "UPDATE certain_records "
            "SET metadata = json_set("
            "    COALESCE(metadata, '{}'),"
            "    '$.status', ?"
            ") "
            "WHERE id = ? AND record_type = 'proposal' "
            f"AND COALESCE(json_extract(metadata, '$.status'), 'pending') "
            f"IN ({placeholders})"
        )
        cursor = await self._db.execute(
            sql,
            (status, proposal_id, *valid_sources),
        )
        await self._db.commit()
        if cursor.rowcount and cursor.rowcount > 0:
            return True

        # No row affected: is it because the proposal is missing, or because
        # the current state disallows the transition?
        exists_cursor = await self._db.execute(
            "SELECT "
            "COALESCE(json_extract(metadata, '$.status'), 'pending') AS st "
            "FROM certain_records "
            "WHERE id = ? AND record_type = 'proposal'",
            (proposal_id,),
        )
        row = await exists_cursor.fetchone()
        if row is None:
            return False

        current = row["st"]
        allowed = self._VALID_TRANSITIONS.get(current, set())
        raise ValueError(
            f"Invalid proposal transition: '{current}' → '{status}' "
            f"(allowed: {allowed})"
        )

    async def get_pending_proposals(
        self,
        user_id: str | None = None,
        status: str = "pending",
    ) -> list[dict[str, Any]]:
        """Return all proposals matching the given status.

        Uses SQLite's ``json_extract()`` to filter at the database level
        instead of loading every proposal into Python.
        """
        if user_id:
            cursor = await self._db.execute(
                "SELECT * FROM certain_records "
                "WHERE record_type = 'proposal' AND user_id = ? "
                "AND json_extract(metadata, '$.status') = ? "
                "ORDER BY created_at ASC",
                (user_id, status),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM certain_records "
                "WHERE record_type = 'proposal' "
                "AND json_extract(metadata, '$.status') = ? "
                "ORDER BY created_at ASC",
                (status,),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def expire_old_proposals(self, max_age_days: int = 7) -> int:
        """Expire PENDING proposals older than max_age_days.

        Returns the number of proposals expired.
        """
        cutoff = (datetime.now(tz=UTC) - timedelta(days=max_age_days)).isoformat()
        cursor = await self._db.execute(
            "SELECT id, metadata FROM certain_records "
            "WHERE record_type = 'proposal' AND created_at < ?",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        count = 0
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            if meta.get("status") == "pending":
                meta["status"] = "expired"
                await self._db.execute(
                    "UPDATE certain_records SET metadata = ? WHERE id = ?",
                    (json.dumps(meta), row["id"]),
                )
                count += 1
        if count > 0:
            await self._db.commit()
        return count

    async def store_link_token(
        self,
        requester_adapter_id: str,
        requester_platform_user_id: str,
        token_expiry_minutes: int = 10,
    ) -> str:
        """Generate and store a link token for cross-platform account linking.

        Returns the generated token string.
        """
        token = secrets.token_urlsafe(16)
        expiry = (
            datetime.now(tz=UTC) + timedelta(minutes=token_expiry_minutes)
        ).isoformat()
        metadata = {
            "requester_adapter_id": requester_adapter_id,
            "requester_platform_user_id": requester_platform_user_id,
            "expires_at": expiry,
            "used": False,
        }
        record_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO certain_records "
            "(id, user_id, content, record_type, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record_id,
                "__system__",
                token,
                "link_token",
                datetime.now(tz=UTC).isoformat(),
                json.dumps(metadata),
            ),
        )
        await self._db.commit()
        return token

    async def verify_link_token(
        self,
        token: str,
    ) -> dict[str, str] | None:
        """Verify a link token and return requester info if valid.

        Returns a dict with ``requester_adapter_id`` and
        ``requester_platform_user_id`` if the token is valid and not
        expired. Returns ``None`` otherwise. Marks the token as used.
        """
        cursor = await self._db.execute(
            "SELECT * FROM certain_records "
            "WHERE record_type = 'link_token' AND content = ?",
            (token,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        meta = json.loads(row["metadata"]) if row["metadata"] else {}

        # Check if already used
        if meta.get("used"):
            return None

        # Check expiry
        expires_at = meta.get("expires_at", "")
        if expires_at and ensure_aware(
            datetime.fromisoformat(expires_at)
        ) < datetime.now(tz=UTC):
            return None

        # Mark as used
        meta["used"] = True
        await self._db.execute(
            "UPDATE certain_records SET metadata = ? WHERE id = ?",
            (json.dumps(meta), row["id"]),
        )
        await self._db.commit()

        return {
            "requester_adapter_id": meta["requester_adapter_id"],
            "requester_platform_user_id": meta["requester_platform_user_id"],
        }
