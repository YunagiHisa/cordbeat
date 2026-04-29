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
