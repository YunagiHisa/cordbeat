"""timer skill — set a reminder for a user."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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

    requested_minutes = minutes
    scheduled_minutes = max(1, min(minutes, 60 * 24 * 30))
    remind_at = (
        datetime.now(tz=UTC) + timedelta(minutes=scheduled_minutes)
    ).isoformat()

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

    result: dict[str, Any] = {
        "status": "scheduled",
        "record_id": record_id,
        "remind_at": remind_at,
        "message": message,
        "minutes": scheduled_minutes,
    }
    if scheduled_minutes != requested_minutes:
        result["clamped"] = True
        result["clamp_note"] = (
            f"Requested {requested_minutes} minutes; scheduled "
            f"{scheduled_minutes} minutes within the supported range."
        )
    return result
