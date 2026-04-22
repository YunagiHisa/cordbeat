"""User CRUD and platform-link resolution (private)."""

from __future__ import annotations

from datetime import UTC, datetime

import aiosqlite

from cordbeat._memory_common import ensure_aware
from cordbeat.models import UserSummary


class UserStore:
    """Handles user CRUD and platform link resolution."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def get_or_create_user(
        self,
        user_id: str,
        display_name: str,
    ) -> UserSummary:
        cursor = await self._db.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row:
            return UserSummary(
                user_id=row["user_id"],
                display_name=row["display_name"],
                last_talked_at=ensure_aware(
                    datetime.fromisoformat(row["last_talked_at"])
                )
                if row["last_talked_at"]
                else None,
                last_platform=row["last_platform"],
                last_topic=row["last_topic"] or "",
                emotional_tone=row["emotional_tone"] or "",
                attention_score=row["attention_score"] or 0.5,
                preferred_platform=row["preferred_platform"],
            )
        await self._db.execute(
            "INSERT INTO users (user_id, display_name) VALUES (?, ?)",
            (user_id, display_name),
        )
        await self._db.commit()
        return UserSummary(user_id=user_id, display_name=display_name)

    async def update_user_summary(self, summary: UserSummary) -> None:
        await self._db.execute(
            "UPDATE users SET display_name=?, last_talked_at=?, "
            "last_platform=?, last_topic=?, emotional_tone=?, "
            "attention_score=?, preferred_platform=? WHERE user_id=?",
            (
                summary.display_name,
                summary.last_talked_at.isoformat() if summary.last_talked_at else None,
                summary.last_platform,
                summary.last_topic,
                summary.emotional_tone,
                summary.attention_score,
                summary.preferred_platform,
                summary.user_id,
            ),
        )
        await self._db.commit()

    async def get_all_user_summaries(self) -> list[UserSummary]:
        cursor = await self._db.execute("SELECT * FROM users")
        rows = await cursor.fetchall()
        return [
            UserSummary(
                user_id=row["user_id"],
                display_name=row["display_name"],
                last_talked_at=ensure_aware(
                    datetime.fromisoformat(row["last_talked_at"])
                )
                if row["last_talked_at"]
                else None,
                last_platform=row["last_platform"],
                last_topic=row["last_topic"] or "",
                emotional_tone=row["emotional_tone"] or "",
                attention_score=row["attention_score"] or 0.5,
                preferred_platform=row["preferred_platform"],
            )
            for row in rows
        ]

    async def link_platform(
        self,
        user_id: str,
        adapter_id: str,
        platform_user_id: str,
    ) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO platform_links "
            "(user_id, adapter_id, platform_user_id, linked_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, adapter_id, platform_user_id, datetime.now(tz=UTC).isoformat()),
        )
        await self._db.commit()

    async def unlink_platform(
        self,
        user_id: str,
        adapter_id: str,
    ) -> bool:
        """Remove a platform link. Returns True if a row was deleted."""
        cursor = await self._db.execute(
            "DELETE FROM platform_links WHERE user_id = ? AND adapter_id = ?",
            (user_id, adapter_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def get_linked_platforms(
        self,
        user_id: str,
    ) -> list[dict[str, str]]:
        """Return all platform links for a user."""
        cursor = await self._db.execute(
            "SELECT adapter_id, platform_user_id FROM platform_links WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "adapter_id": row["adapter_id"],
                "platform_user_id": row["platform_user_id"],
            }
            for row in rows
        ]

    async def resolve_user(
        self,
        adapter_id: str,
        platform_user_id: str,
    ) -> str | None:
        cursor = await self._db.execute(
            "SELECT user_id FROM platform_links "
            "WHERE adapter_id = ? AND platform_user_id = ?",
            (adapter_id, platform_user_id),
        )
        row = await cursor.fetchone()
        return row["user_id"] if row else None

    async def resolve_platform_user(
        self,
        user_id: str,
        adapter_id: str,
    ) -> str | None:
        """Reverse-lookup: internal user_id + adapter → platform_user_id."""
        cursor = await self._db.execute(
            "SELECT platform_user_id FROM platform_links "
            "WHERE user_id = ? AND adapter_id = ?",
            (user_id, adapter_id),
        )
        row = await cursor.fetchone()
        return row["platform_user_id"] if row else None
