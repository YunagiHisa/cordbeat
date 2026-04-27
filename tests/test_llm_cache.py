"""Tests for the LLM response cache."""

from __future__ import annotations

from typing import Any

import pytest

from cordbeat.ai_backend import AIBackend, create_backend
from cordbeat.config import AIBackendConfig, LLMCacheConfig
from cordbeat.llm_cache import CACHE_HITS, CACHE_MISSES, CachingBackend
from cordbeat.metrics import REGISTRY


class FakeBackend(AIBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, int]] = []

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append((prompt, system, temperature, max_tokens))
        return f"RESP[{len(self.calls)}]:{prompt}"


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    REGISTRY.reset()
    CACHE_HITS._values.clear()
    CACHE_MISSES._values.clear()


def _make(
    enabled: bool = True,
    max_entries: int = 4,
    ttl_seconds: int = 3600,
    max_temperature: float = 0.5,
    time_func: Any = None,
) -> tuple[CachingBackend, FakeBackend]:
    inner = FakeBackend()
    cfg = LLMCacheConfig(
        enabled=enabled,
        max_entries=max_entries,
        ttl_seconds=ttl_seconds,
        max_temperature=max_temperature,
    )
    cb = CachingBackend(
        inner=inner,
        config=cfg,
        model="test-model",
        backend_name="test",
        time_func=time_func or (lambda: 0.0),
    )
    return cb, inner


@pytest.mark.asyncio
async def test_cache_hit_returns_same_value() -> None:
    cb, inner = _make()
    a = await cb.generate("hi", system="s", temperature=0.0)
    b = await cb.generate("hi", system="s", temperature=0.0)
    assert a == b
    assert len(inner.calls) == 1
    assert CACHE_HITS.value({"backend": "test"}) == 1.0


@pytest.mark.asyncio
async def test_cache_skips_high_temperature() -> None:
    cb, inner = _make(max_temperature=0.2)
    await cb.generate("hi", temperature=0.9)
    await cb.generate("hi", temperature=0.9)
    assert len(inner.calls) == 2
    assert CACHE_MISSES.value({"backend": "test", "reason": "high_temperature"}) == 2.0


@pytest.mark.asyncio
async def test_cache_disabled_passes_through() -> None:
    cb, inner = _make(enabled=False)
    await cb.generate("hi", temperature=0.0)
    await cb.generate("hi", temperature=0.0)
    assert len(inner.calls) == 2
    # No cache metrics recorded when disabled
    assert CACHE_HITS.value({"backend": "test"}) == 0.0


@pytest.mark.asyncio
async def test_cache_lru_eviction() -> None:
    cb, inner = _make(max_entries=2)
    await cb.generate("a", temperature=0.0)
    await cb.generate("b", temperature=0.0)
    await cb.generate("c", temperature=0.0)  # evicts "a"
    assert cb.cache_size() == 2
    await cb.generate("a", temperature=0.0)  # miss again
    assert len(inner.calls) == 4


@pytest.mark.asyncio
async def test_cache_ttl_expiry() -> None:
    clock = {"t": 0.0}

    def now() -> float:
        return clock["t"]

    cb, inner = _make(ttl_seconds=10, time_func=now)
    await cb.generate("hi", temperature=0.0)
    clock["t"] = 5.0
    await cb.generate("hi", temperature=0.0)
    assert len(inner.calls) == 1
    clock["t"] = 100.0
    await cb.generate("hi", temperature=0.0)
    assert len(inner.calls) == 2
    assert CACHE_MISSES.value({"backend": "test", "reason": "expired"}) == 1.0


@pytest.mark.asyncio
async def test_cache_keys_distinct_by_temperature() -> None:
    cb, inner = _make(max_temperature=1.0)
    await cb.generate("hi", temperature=0.0)
    await cb.generate("hi", temperature=0.1)
    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_cache_keys_distinct_by_system_prompt() -> None:
    cb, inner = _make()
    await cb.generate("hi", system="s1", temperature=0.0)
    await cb.generate("hi", system="s2", temperature=0.0)
    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_create_backend_wraps_when_enabled() -> None:
    cfg = AIBackendConfig(
        provider="ollama",
        cache=LLMCacheConfig(enabled=True),
    )
    backend = create_backend(cfg)
    assert isinstance(backend, CachingBackend)
    await backend.aclose()


@pytest.mark.asyncio
async def test_create_backend_no_wrap_when_disabled() -> None:
    cfg = AIBackendConfig(provider="ollama")
    backend = create_backend(cfg)
    assert not isinstance(backend, CachingBackend)
    await backend.aclose()
