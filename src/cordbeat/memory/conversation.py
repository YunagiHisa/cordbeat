"""SQLite-backed conversation history store (private)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import aiosqlite


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
    ) -> None:
        await self._db.execute(
            "INSERT INTO conversation_messages "
            "(user_id, role, content, adapter_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, role, content, adapter_id, datetime.now(tz=UTC).isoformat()),
        )
        await self._db.commit()

    async def get_recent_messages(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        cursor = await self._db.execute(
            "SELECT role, content FROM ("
            "  SELECT role, content, created_at "
            "  FROM conversation_messages "
            "  WHERE user_id = ? "
            "  ORDER BY created_at DESC LIMIT ?"
            ") sub ORDER BY created_at ASC",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def get_todays_messages(
        self,
        user_id: str,
    ) -> list[dict[str, str]]:
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        cursor = await self._db.execute(
            "SELECT role, content FROM conversation_messages "
            "WHERE user_id = ? AND created_at >= ? "
            "ORDER BY created_at ASC",
            (user_id, today),
        )
        rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def get_messages_on_date(
        self,
        user_id: str,
        date_str: str,
    ) -> list[dict[str, str]]:
        """Get all messages on a specific date (YYYY-MM-DD)."""
        next_day = (datetime.fromisoformat(date_str) + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        cursor = await self._db.execute(
            "SELECT role, content FROM conversation_messages "
            "WHERE user_id = ? AND created_at >= ? AND created_at < ? "
            "ORDER BY created_at ASC",
            (user_id, date_str, next_day),
        )
        rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def trim_old_messages(
        self,
        user_id: str,
        keep: int = 100,
    ) -> int:
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
    ) -> list[dict[str, str]]:
        """Return the *limit* oldest messages for *user_id* (ascending)."""
        cursor = await self._db.execute(
            "SELECT id, role, content, created_at "
            "FROM conversation_messages "
            "WHERE user_id = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": str(row["id"]),
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
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
        cursor = await self._db.execute(
            "DELETE FROM conversation_messages WHERE user_id = ?",
            (user_id,),
        )
        await self._db.commit()
        return cursor.rowcount
