"""Per-skill rate limiting (token-bucket).

A small, dependency-free rate limiter for the skill subsystem. The bucket
is keyed by skill name and refills at ``capacity`` tokens per minute
(default; configurable). Exhausting the bucket raises
:class:`cordbeat.exceptions.SkillRateLimitError`, which surfaces a
``retry_after_seconds`` attribute so callers (e.g., the heartbeat loop
or proposal executor) can decide whether to defer or skip the call.

Design notes
------------
* All buckets share a single :class:`asyncio.Lock` for refill bookkeeping
  to keep the implementation simple and predictable. Per-skill locks are
  unnecessary because token accounting is O(1) and contention is
  inherently low (skills run sequentially per heartbeat tick).
* ``time.monotonic`` is used so that wall-clock jumps (NTP sync, DST)
  do not corrupt the bucket state.
* ``per_minute=0`` is treated as "unlimited" (the limiter is a no-op
  for that skill). This matches the config-default semantics.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from cordbeat.exceptions import SkillRateLimitError

__all__ = ["SkillRateLimiter"]


@dataclass
class _Bucket:
    capacity: float
    tokens: float
    refill_per_second: float
    last_refill: float


class SkillRateLimiter:
    """Token-bucket rate limiter shared across a :class:`SkillRegistry`.

    The bucket for a given skill is created lazily on first
    :meth:`acquire`. The effective limit is, in priority order, the
    skill's own ``rate_limit_per_minute`` (``SkillMeta``), then the
    registry-wide default passed to ``__init__``. ``0`` (or ``None``)
    means *unlimited* and short-circuits the limiter for that skill.
    """

    def __init__(self, default_per_minute: int = 0) -> None:
        self._default = max(0, int(default_per_minute))
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    def _resolve_limit(self, skill_limit: int | None) -> int:
        if skill_limit is not None and skill_limit >= 0:
            return int(skill_limit)
        return self._default

    async def acquire(self, skill_name: str, skill_limit: int | None = None) -> None:
        """Consume one token for ``skill_name`` or raise.

        Raises
        ------
        SkillRateLimitError
            If the bucket is empty. ``retry_after_seconds`` is set to
            the time until the next token will be available.
        """
        limit = self._resolve_limit(skill_limit)
        if limit <= 0:
            return  # unlimited

        async with self._lock:
            now = time.monotonic()
            bucket = self._buckets.get(skill_name)
            if bucket is None or bucket.capacity != limit:
                bucket = _Bucket(
                    capacity=float(limit),
                    tokens=float(limit),
                    refill_per_second=float(limit) / 60.0,
                    last_refill=now,
                )
                self._buckets[skill_name] = bucket
            else:
                elapsed = now - bucket.last_refill
                if elapsed > 0:
                    bucket.tokens = min(
                        bucket.capacity,
                        bucket.tokens + elapsed * bucket.refill_per_second,
                    )
                    bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return

            needed = 1.0 - bucket.tokens
            retry_after = needed / bucket.refill_per_second
            raise SkillRateLimitError(
                f"Skill {skill_name!r} rate limit exceeded "
                f"({limit}/min); retry in {retry_after:.2f}s",
                retry_after_seconds=retry_after,
                skill_name=skill_name,
                limit_per_minute=limit,
            )

    def reset(self, skill_name: str | None = None) -> None:
        """Drop bucket state. ``None`` clears all skills (test helper)."""
        if skill_name is None:
            self._buckets.clear()
        else:
            self._buckets.pop(skill_name, None)
