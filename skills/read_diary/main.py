"""read_diary skill — retrieve diary entries and logs from memory."""

from __future__ import annotations

from typing import Any


async def execute(
    *,
    user_id: str,
    record_type: str = "diary",
    limit: int = 5,
    context: Any = None,
) -> dict[str, Any]:
    """Fetch diary entries or other certain records for a user."""
    if context is None or context.memory is None:
        return {"error": "Memory access not available"}

    memory = context.memory
    records = await memory.get_certain_records(
        user_id, record_type=record_type, limit=limit
    )

    entries = []
    for record in records:
        entries.append(
            {
                "content": record.get("content", ""),
                "created_at": record.get("created_at", ""),
            }
        )

    return {
        "user_id": user_id,
        "record_type": record_type,
        "count": len(entries),
        "entries": entries,
    }
