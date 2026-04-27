"""Tests for the per-skill rate limiter (D-6)."""

from __future__ import annotations

import pytest

from cordbeat.exceptions import SkillRateLimitError
from cordbeat.skill_rate_limit import SkillRateLimiter


class TestSkillRateLimiter:
    async def test_unlimited_when_default_zero_and_no_override(self) -> None:
        limiter = SkillRateLimiter(default_per_minute=0)
        for _ in range(100):
            await limiter.acquire("free", None)

    async def test_per_skill_override_overrides_default(self) -> None:
        limiter = SkillRateLimiter(default_per_minute=0)
        for _ in range(2):
            await limiter.acquire("capped", skill_limit=2)
        with pytest.raises(SkillRateLimitError) as exc_info:
            await limiter.acquire("capped", skill_limit=2)
        assert exc_info.value.skill_name == "capped"
        assert exc_info.value.limit_per_minute == 2
        assert exc_info.value.retry_after_seconds > 0

    async def test_default_limit_enforced(self) -> None:
        limiter = SkillRateLimiter(default_per_minute=3)
        for _ in range(3):
            await limiter.acquire("default_capped", None)
        with pytest.raises(SkillRateLimitError):
            await limiter.acquire("default_capped", None)

    async def test_skill_zero_means_unlimited(self) -> None:
        limiter = SkillRateLimiter(default_per_minute=2)
        # Per-skill 0 must NOT be treated as "unlimited" override —
        # _resolve_limit only short-circuits when the *resolved* limit
        # is <=0. 0 from the skill is treated as a valid value (so
        # falls through to limit=0 = unlimited). Document that behavior.
        for _ in range(50):
            await limiter.acquire("override_zero", skill_limit=0)

    async def test_negative_skill_value_falls_back_to_default(self) -> None:
        limiter = SkillRateLimiter(default_per_minute=2)
        # Negative is treated as "unset" → use default
        for _ in range(2):
            await limiter.acquire("neg", skill_limit=-1)
        with pytest.raises(SkillRateLimitError):
            await limiter.acquire("neg", skill_limit=-1)

    async def test_buckets_are_per_skill(self) -> None:
        limiter = SkillRateLimiter(default_per_minute=1)
        await limiter.acquire("a", None)
        # 'b' has its own bucket
        await limiter.acquire("b", None)
        with pytest.raises(SkillRateLimitError):
            await limiter.acquire("a", None)

    async def test_reset_releases_bucket(self) -> None:
        limiter = SkillRateLimiter(default_per_minute=1)
        await limiter.acquire("x", None)
        with pytest.raises(SkillRateLimitError):
            await limiter.acquire("x", None)
        limiter.reset("x")
        await limiter.acquire("x", None)

    async def test_reset_all(self) -> None:
        limiter = SkillRateLimiter(default_per_minute=1)
        await limiter.acquire("x", None)
        await limiter.acquire("y", None)
        limiter.reset()
        await limiter.acquire("x", None)
        await limiter.acquire("y", None)

    async def test_changing_capacity_rebuilds_bucket(self) -> None:
        limiter = SkillRateLimiter(default_per_minute=1)
        await limiter.acquire("recap", skill_limit=1)
        # Same call with a higher cap rebuilds the bucket fresh
        await limiter.acquire("recap", skill_limit=5)

    async def test_retry_after_reflects_refill_rate(self) -> None:
        # 60/min -> 1 token per second, so retry_after for 1 token ≈ 1s
        limiter = SkillRateLimiter(default_per_minute=60)
        for _ in range(60):
            await limiter.acquire("burst", None)
        with pytest.raises(SkillRateLimitError) as exc_info:
            await limiter.acquire("burst", None)
        assert 0.5 <= exc_info.value.retry_after_seconds <= 2.0
