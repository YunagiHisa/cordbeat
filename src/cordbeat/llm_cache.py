"""LRU-with-TTL response cache for LLM ``generate()`` calls.

Wraps an :class:`cordbeat.ai_backend.AIBackend` with a thin decorator
that memoizes deterministic-ish responses (temperature below a configured
threshold) keyed on the full request signature. The cache is opt-in via
:class:`cordbeat.config.LLMCacheConfig`.

This is a pragmatic, dependency-free first cut: an ``OrderedDict``-based
LRU with per-entry TTL, single-process only. It is intentionally **not**
embedding-based — semantic similarity caching is left for v1.0+ once we
have an evaluator that can verify cache hits don't degrade output
quality.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

from cordbeat.ai_backend import AIBackend
from cordbeat.config import LLMCacheConfig
from cordbeat.metrics import REGISTRY, inc_counter

logger = logging.getLogger(__name__)


CACHE_HITS = REGISTRY.counter(
    "cordbeat_llm_cache_hits_total",
    "Total LLM cache hits (labels: backend).",
)
CACHE_MISSES = REGISTRY.counter(
    "cordbeat_llm_cache_misses_total",
    "Total LLM cache misses (labels: backend, reason).",
)


def _hash_key(
    prompt: str, system: str, model: str, temperature: float, max_tokens: int
) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x1f")
    h.update(f"{temperature:.4f}".encode())
    h.update(b"\x1f")
    h.update(str(max_tokens).encode("ascii"))
    h.update(b"\x1f")
    h.update(system.encode("utf-8"))
    h.update(b"\x1f")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


class CachingBackend(AIBackend):
    """Decorator that memoizes ``generate()`` calls on an inner backend."""

    def __init__(
        self,
        inner: AIBackend,
        config: LLMCacheConfig,
        model: str,
        backend_name: str,
        time_func: Any = time.monotonic,
    ) -> None:
        self._inner = inner
        self._config = config
        self._model = model
        self._backend_name = backend_name
        self._time = time_func
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

    @property
    def inner(self) -> AIBackend:
        return self._inner

    def cache_size(self) -> int:
        return len(self._cache)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        if not self._config.enabled:
            return await self._inner.generate(prompt, system, temperature, max_tokens)

        if temperature > self._config.max_temperature:
            inc_counter(
                CACHE_MISSES,
                {"backend": self._backend_name, "reason": "high_temperature"},
            )
            return await self._inner.generate(prompt, system, temperature, max_tokens)

        key = _hash_key(prompt, system, self._model, temperature, max_tokens)
        now = self._time()
        entry = self._cache.get(key)
        if entry is not None:
            stored_at, value = entry
            if now - stored_at <= self._config.ttl_seconds:
                self._cache.move_to_end(key)
                inc_counter(CACHE_HITS, {"backend": self._backend_name})
                return value
            # Expired
            del self._cache[key]
            inc_counter(
                CACHE_MISSES,
                {"backend": self._backend_name, "reason": "expired"},
            )
        else:
            inc_counter(CACHE_MISSES, {"backend": self._backend_name, "reason": "miss"})

        value = await self._inner.generate(prompt, system, temperature, max_tokens)
        self._cache[key] = (now, value)
        self._cache.move_to_end(key)
        if len(self._cache) > self._config.max_entries:
            self._cache.popitem(last=False)
        return value
