"""timer skill — set a reminder for a user."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


async def execute(
    *,
    user_id: str,
    message: str,
    minutes: int = 30,
    context: Any = None,
) -> dict[str, Any]:
    """Store a reminder that HEARTBEAT will deliver later."""
    if context is None or context.memory is None:
        return {"error": "Memory access not available"}

    remind_at = (datetime.now() + timedelta(minutes=minutes)).isoformat()

    memory = context.memory
    record_id = await memory.add_certain_record(
        user_id=user_id,
        content=message,
        record_type="reminder",
        metadata={
            "remind_at": remind_at,
            "status": "pending",
        },
    )

    return {
        "status": "scheduled",
        "record_id": record_id,
        "remind_at": remind_at,
        "message": message,
    }
