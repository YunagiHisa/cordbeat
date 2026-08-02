"""SQLite-backed conversation history store (private)."""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo

import aiosqlite

from .time_window import local_date_bounds_utc, today_bounds_utc


def _scope_conditions(
    user_id: str,
    channel_id: str | None,
    is_dm: bool | None,
    adapter_id: str | None,
    channel_wide: bool,
    prefix: str = "",
) -> tuple[list[str], list[object]]:
    """Build the WHERE fragments that scope a history query.

    ``channel_wide`` drops the ``user_id`` filter so the result covers every
    participant of the channel rather than one user's slice of it.  It is
    honoured only when a concrete ``channel_id`` is given: without one there
    is no room to widen to, and dropping ``user_id`` would return the whole
    database instead.
    """
    conditions: list[str] = []
    params: list[object] = []
    scoped_to_channel = channel_id is not None and channel_id != ""
    if not (channel_wide and scoped_to_channel):
        conditions.append(f"{prefix}user_id = ?")
        params.append(user_id)
    if scoped_to_channel:
        conditions.append(f"{prefix}channel_id = ?")
        params.append(channel_id)
    if is_dm is not None:
        conditions.append(f"{prefix}is_dm = ?")
        params.append(1 if is_dm else 0)
    if adapter_id is not None and adapter_id != "":
        conditions.append(f"{prefix}adapter_id = ?")
        params.append(adapter_id)
    return conditions, params


class ConversationStore:
    """SQLite-backed conversation message storage."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        adapter_id: str = "",
        channel_id: str = "",
        is_dm: bool = True,
        created_at: datetime | None = None,
        received_at: datetime | None = None,
    ) -> int:
        stored_created_at = created_at or datetime.now(tz=UTC)
        stored_received_at = received_at or stored_created_at
        cursor = await self._db.execute(
            "INSERT INTO conversation_messages "
            "(user_id, role, content, adapter_id, channel_id, is_dm, created_at, "
            "received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                role,
                content,
                adapter_id,
                channel_id,
                1 if is_dm else 0,
                stored_created_at.isoformat(),
                stored_received_at.isoformat(),
            ),
        )
        await self._db.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("conversation message insert returned no row id")
        return int(cursor.lastrowid)

    async def add_media_observation(
        self,
        message_id: int,
        *,
        summary: str,
        relation: str,
        mime_type: str = "",
        content_sha256: str = "",
        source_ref: str = "",
    ) -> None:
        await self._db.execute(
            "INSERT INTO conversation_media_observations "
            "(message_id, media_kind, relation, mime_type, content_sha256, "
            "summary, source_ref, created_at) VALUES (?, 'image', ?, ?, ?, ?, ?, ?)",
            (
                message_id,
                relation,
                mime_type,
                content_sha256,
                summary,
                source_ref,
                datetime.now(tz=UTC).isoformat(),
            ),
        )
        await self._db.commit()

    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 20,
        channel_id: str | None = None,
        is_dm: bool | None = None,
        adapter_id: str | None = None,
        channel_wide: bool = False,
    ) -> list[dict[str, str]]:
        """Return the *limit* most recent messages for *user_id*.

        When ``channel_id`` is given (non-empty), results are scoped to that
        channel.  When ``is_dm`` is given, results are scoped to DM vs
        non-DM history.  When ``adapter_id`` is given (non-empty), results
        are scoped to that adapter (e.g. ``discord``, ``telegram``, ``cli``)
        so cross-platform linked accounts don't see each other's history.
        Passing more arguments narrows further.  Pass none to keep the
        legacy "all history for this user" behaviour (used by tools and
        backward-compat call sites).

        ``channel_wide`` (requires ``channel_id``) widens instead: it returns
        every participant's messages in that channel, which is what a shared
        room actually looks like to the people in it.  Each row carries a
        ``speaker`` display name so callers can tell them apart.
        """
        conditions, params = _scope_conditions(
            user_id, channel_id, is_dm, adapter_id, channel_wide, prefix="m."
        )
        where_clause = " AND ".join(conditions)
        params.append(limit)
        cursor = await self._db.execute(
            "SELECT role, content, created_at, received_at, speaker FROM ("
            "  SELECT m.role, m.content, m.created_at, m.received_at, "
            "         u.display_name AS speaker "
            "  FROM conversation_messages m "
            "  LEFT JOIN users u ON u.user_id = m.user_id "
            f"  WHERE {where_clause} "
            "  ORDER BY m.created_at DESC LIMIT ?"
            ") sub ORDER BY created_at ASC",
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "received_at": row["received_at"],
                "speaker": row["speaker"] or "",
            }
            for row in rows
        ]

    async def get_recent_messages_with_media(
        self,
        user_id: str,
        limit: int = 20,
        channel_id: str | None = None,
        is_dm: bool | None = None,
        adapter_id: str | None = None,
        channel_wide: bool = False,
    ) -> list[dict[str, object]]:
        conditions, params = _scope_conditions(
            user_id, channel_id, is_dm, adapter_id, channel_wide, prefix="m."
        )
        params.append(limit)
        cursor = await self._db.execute(
            "SELECT id, role, content, created_at, received_at, speaker FROM ("
            "SELECT m.id, m.role, m.content, m.created_at, m.received_at, "
            "       u.display_name AS speaker "
            "FROM conversation_messages m "
            "LEFT JOIN users u ON u.user_id = m.user_id "
            f"WHERE {' AND '.join(conditions)} ORDER BY m.created_at DESC LIMIT ?) "
            "sub ORDER BY created_at ASC",
            tuple(params),
        )
        rows = await cursor.fetchall()
        messages: list[dict[str, object]] = []
        for row in rows:
            message_id = int(row["id"])
            media_cursor = await self._db.execute(
                "SELECT relation, mime_type, content_sha256, summary, source_ref "
                "FROM conversation_media_observations WHERE message_id = ? "
                "ORDER BY id ASC",
                (message_id,),
            )
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "received_at": row["received_at"],
                    "speaker": row["speaker"] or "",
                    "media_observations": [
                        dict(media_row) for media_row in await media_cursor.fetchall()
                    ],
                }
            )
        return messages

    async def get_recent_channels(
        self,
        user_id: str,
        adapter_id: str,
        limit: int = 4,
    ) -> list[dict[str, object]]:
        """Return channels the user recently talked in, most recent first.

        Each item has ``channel_id``, ``is_dm``, and ``last_message_at``.
        Rows without a channel id (legacy messages) are skipped.
        """
        cursor = await self._db.execute(
            "SELECT channel_id, is_dm, MAX(created_at) AS last_message_at "
            "FROM conversation_messages "
            "WHERE user_id = ? AND adapter_id = ? AND channel_id != '' "
            "GROUP BY channel_id, is_dm "
            "ORDER BY last_message_at DESC LIMIT ?",
            (user_id, adapter_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "channel_id": row["channel_id"],
                "is_dm": bool(row["is_dm"]),
                "last_message_at": row["last_message_at"],
            }
            for row in rows
        ]

    async def get_todays_messages(
        self,
        user_id: str,
        timezone: str | tzinfo | None = UTC,
    ) -> list[dict[str, object]]:
        _, start_iso, end_iso = today_bounds_utc(timezone)
        return await self.get_messages_between(user_id, start_iso, end_iso)

    async def get_messages_between(
        self,
        user_id: str,
        start_iso: str,
        end_iso: str,
    ) -> list[dict[str, object]]:
        cursor = await self._db.execute(
            "SELECT role, content, channel_id, is_dm FROM conversation_messages "
            "WHERE user_id = ? "
            "AND datetime(created_at) >= datetime(?) "
            "AND datetime(created_at) < datetime(?) "
            "ORDER BY created_at ASC",
            (user_id, start_iso, end_iso),
        )
        rows = await cursor.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "channel_id": row["channel_id"] or "",
                "is_dm": bool(row["is_dm"]),
            }
            for row in rows
        ]

    async def get_messages_on_date(
        self,
        user_id: str,
        date_str: str,
        timezone: str | tzinfo | None = UTC,
    ) -> list[dict[str, object]]:
        """Get all messages on a specific date (YYYY-MM-DD)."""
        start_iso, end_iso = local_date_bounds_utc(date_str, timezone)
        return await self.get_messages_between(user_id, start_iso, end_iso)

    async def trim_old_messages(
        self,
        user_id: str,
        keep: int = 100,
    ) -> int:
        await self._db.execute(
            "DELETE FROM conversation_media_observations WHERE message_id IN ("
            "SELECT id FROM conversation_messages WHERE user_id = ? "
            "AND id NOT IN (SELECT id FROM conversation_messages "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?))",
            (user_id, user_id, keep),
        )
        cursor = await self._db.execute(
            "DELETE FROM conversation_messages WHERE user_id = ? "
            "AND id NOT IN ("
            "  SELECT id FROM conversation_messages "
            "  WHERE user_id = ? ORDER BY created_at DESC LIMIT ?"
            ")",
            (user_id, user_id, keep),
        )
        await self._db.commit()
        return cursor.rowcount

    async def count_messages(self, user_id: str) -> int:
        """Return the total number of stored messages for *user_id*."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM conversation_messages WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def get_oldest_messages(
        self,
        user_id: str,
        limit: int,
        channel_id: str | None = None,
        is_dm: bool | None = None,
    ) -> list[dict[str, object]]:
        """Return the *limit* oldest messages for *user_id* (ascending).

        ``channel_id``/``is_dm`` restrict the result to one conversation.
        Callers that turn these rows into a memory need that: a summary
        spanning a DM and a public channel cannot be labelled with an
        origin, and so cannot be kept out of the wrong room later.
        """
        conditions = ["user_id = ?"]
        params: list[object] = [user_id]
        if channel_id is not None:
            conditions.append("channel_id = ?")
            params.append(channel_id)
        if is_dm is not None:
            conditions.append("is_dm = ?")
            params.append(1 if is_dm else 0)
        params.append(limit)
        cursor = await self._db.execute(
            "SELECT id, role, content, created_at, channel_id, is_dm "
            "FROM conversation_messages "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY created_at ASC LIMIT ?",
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": str(row["id"]),
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "channel_id": row["channel_id"] or "",
                "is_dm": bool(row["is_dm"]),
            }
            for row in rows
        ]

    async def delete_messages_with_ids(self, ids: list[str]) -> int:
        """Delete messages by primary-key IDs.  Returns deleted count."""
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cursor = await self._db.execute(
            f"DELETE FROM conversation_messages WHERE id IN ({placeholders})",
            ids,
        )
        await self._db.commit()
        return cursor.rowcount

    async def clear_conversation_history(self, user_id: str) -> int:
        """Delete all conversation messages for a user. Returns row count deleted."""
        await self._db.execute(
            "DELETE FROM conversation_media_observations WHERE message_id IN "
            "(SELECT id FROM conversation_messages WHERE user_id = ?)",
            (user_id,),
        )
        cursor = await self._db.execute(
            "DELETE FROM conversation_messages WHERE user_id = ?",
            (user_id,),
        )
        await self._db.commit()
        return cursor.rowcount
