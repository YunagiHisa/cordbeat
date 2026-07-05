"""Timezone-aware date windows for memory queries."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def resolve_timezone(value: str | tzinfo | None) -> tzinfo:
    if isinstance(value, tzinfo):
        return value
    if not value:
        return UTC
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError:
        return UTC


def local_date_for_timezone(
    timezone: str | tzinfo | None,
    *,
    now: datetime | None = None,
) -> str:
    tz = resolve_timezone(timezone)
    current = now or datetime.now(tz=UTC)
    return current.astimezone(tz).date().isoformat()


def local_date_bounds_utc(
    date_str: str,
    timezone: str | tzinfo | None,
) -> tuple[str, str]:
    tz = resolve_timezone(timezone)
    local_day = date.fromisoformat(date_str)
    start_local = datetime.combine(local_day, datetime.min.time(), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return (
        start_local.astimezone(UTC).isoformat(),
        end_local.astimezone(UTC).isoformat(),
    )


def today_bounds_utc(
    timezone: str | tzinfo | None,
    *,
    now: datetime | None = None,
) -> tuple[str, str, str]:
    date_str = local_date_for_timezone(timezone, now=now)
    start_iso, end_iso = local_date_bounds_utc(date_str, timezone)
    return date_str, start_iso, end_iso
