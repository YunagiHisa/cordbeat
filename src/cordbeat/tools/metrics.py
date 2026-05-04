"""In-process metrics registry for CordBeat.

A small, dependency-free metrics layer suitable for a single-event-loop
agent. Provides :class:`Counter` (monotonic) and :class:`Histogram`
(latency / size buckets), all keyed by an immutable label tuple.

The :data:`REGISTRY` singleton collects all metric series and can render
them in Prometheus 0.0.4 text exposition format via
:func:`render_prometheus`. The :func:`time_block` async context manager
records elapsed seconds into a histogram.

Metrics can be globally disabled via :class:`cordbeat.config.MetricsConfig`
(``enabled=False``). When disabled, recording is a no-op and the renderer
returns an empty string.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


# Default histogram buckets (seconds). Covers the full operational range
# from sub-millisecond memory hits to multi-second LLM completions.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


def _label_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


def _format_labels(key: tuple[tuple[str, str], ...]) -> str:
    if not key:
        return ""
    parts = [f'{name}="{_escape(value)}"' for name, value in key]
    return "{" + ",".join(parts) + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@dataclass
class Counter:
    """Monotonic counter."""

    name: str
    description: str
    _values: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if amount < 0:
            raise ValueError("Counter increments must be non-negative")
        key = _label_key(labels)
        self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, labels: dict[str, str] | None = None) -> float:
        return self._values.get(_label_key(labels), 0.0)

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.description}"
        yield f"# TYPE {self.name} counter"
        for key, val in sorted(self._values.items()):
            yield f"{self.name}{_format_labels(key)} {val}"


@dataclass
class _HistogramSeries:
    counts: list[int]
    sum_seconds: float = 0.0
    total: int = 0


@dataclass
class Histogram:
    """Cumulative histogram with fixed bucket boundaries."""

    name: str
    description: str
    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    _series: dict[tuple[tuple[str, str], ...], _HistogramSeries] = field(
        default_factory=dict
    )

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = _label_key(labels)
        series = self._series.get(key)
        if series is None:
            series = _HistogramSeries(counts=[0] * len(self.buckets))
            self._series[key] = series
        # Increment only the smallest bucket containing the value;
        # render() accumulates these into Prometheus cumulative form.
        for i, boundary in enumerate(self.buckets):
            if value <= boundary:
                series.counts[i] += 1
                break
        series.sum_seconds += value
        series.total += 1

    def total(self, labels: dict[str, str] | None = None) -> int:
        series = self._series.get(_label_key(labels))
        return 0 if series is None else series.total

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.description}"
        yield f"# TYPE {self.name} histogram"
        for key, series in sorted(self._series.items()):
            cumulative = 0
            for boundary, count in zip(self.buckets, series.counts, strict=True):
                cumulative += count
                bucket_labels = key + (("le", _format_le(boundary)),)
                yield (
                    f"{self.name}_bucket{_format_labels(bucket_labels)} {cumulative}"
                )
            inf_labels = key + (("le", "+Inf"),)
            yield f"{self.name}_bucket{_format_labels(inf_labels)} {series.total}"
            yield f"{self.name}_sum{_format_labels(key)} {series.sum_seconds}"
            yield f"{self.name}_count{_format_labels(key)} {series.total}"


def _format_le(value: float) -> str:
    if value == int(value):
        return f"{int(value)}"
    return f"{value}"


class MetricsRegistry:
    """Thread-safe singleton holding all Counters and Histograms."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def counter(self, name: str, description: str) -> Counter:
        with self._lock:
            existing = self._counters.get(name)
            if existing is None:
                existing = Counter(name=name, description=description)
                self._counters[name] = existing
            return existing

    def histogram(
        self,
        name: str,
        description: str,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> Histogram:
        with self._lock:
            existing = self._histograms.get(name)
            if existing is None:
                existing = Histogram(
                    name=name, description=description, buckets=buckets
                )
                self._histograms[name] = existing
            return existing

    def reset(self) -> None:
        """Clear all metric series. Test-only convenience."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._enabled = True

    def render_prometheus(self) -> str:
        if not self._enabled:
            return ""
        lines: list[str] = []
        with self._lock:
            for counter in self._counters.values():
                lines.extend(counter.render())
            for histogram in self._histograms.values():
                lines.extend(histogram.render())
        if not lines:
            return ""
        return "\n".join(lines) + "\n"


REGISTRY = MetricsRegistry()


# ── High-level helpers exposed to callers ──────────────────────────


def render_prometheus() -> str:
    """Render the global registry in Prometheus text format."""
    return REGISTRY.render_prometheus()


@asynccontextmanager
async def time_block(
    histogram: Histogram, labels: dict[str, str] | None = None
) -> AsyncIterator[None]:
    """Async context manager that records elapsed wall-clock seconds.

    The observation is recorded even on exception (the timer always
    fires in the finally block), so failure latencies are visible too.
    """
    if not REGISTRY.enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        try:
            histogram.observe(elapsed, labels)
        except Exception:
            logger.debug("metrics: histogram.observe failed", exc_info=True)


def inc_counter(counter: Counter, labels: dict[str, str] | None = None) -> None:
    """Increment a counter by 1, no-op when metrics are disabled."""
    if not REGISTRY.enabled:
        return
    try:
        counter.inc(1.0, labels)
    except Exception:
        logger.debug("metrics: counter.inc failed", exc_info=True)


# ── Predefined metrics ─────────────────────────────────────────────


HEARTBEAT_TICK_LATENCY = REGISTRY.histogram(
    "cordbeat_heartbeat_tick_seconds",
    "Wall-clock duration of one HEARTBEAT tick.",
)
HEARTBEAT_TICK_TOTAL = REGISTRY.counter(
    "cordbeat_heartbeat_tick_total",
    "Total HEARTBEAT ticks executed (labels: outcome=ok|error).",
)

MEMORY_QUERY_LATENCY = REGISTRY.histogram(
    "cordbeat_memory_query_seconds",
    "Latency of memory queries (labels: kind=semantic|episodic).",
)

SKILL_EXEC_LATENCY = REGISTRY.histogram(
    "cordbeat_skill_exec_seconds",
    "Skill execution latency (labels: skill, safety_level).",
)
SKILL_EXEC_TOTAL = REGISTRY.counter(
    "cordbeat_skill_exec_total",
    "Total skill executions (labels: skill, safety_level, outcome).",
)

LLM_GENERATE_LATENCY = REGISTRY.histogram(
    "cordbeat_llm_generate_seconds",
    "LLM generate() latency (labels: backend, model).",
)
LLM_GENERATE_TOTAL = REGISTRY.counter(
    "cordbeat_llm_generate_total",
    "Total LLM generate() calls (labels: backend, outcome).",
)
