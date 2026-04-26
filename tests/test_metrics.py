"""Tests for the in-process metrics module."""

from __future__ import annotations

import pytest

from cordbeat.metrics import (
    REGISTRY,
    Counter,
    Histogram,
    inc_counter,
    render_prometheus,
    time_block,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    REGISTRY.reset()


def test_counter_basic() -> None:
    c = Counter(name="x_total", description="x")
    c.inc()
    c.inc(2.5, labels={"a": "1"})
    c.inc(1.0, labels={"a": "1"})
    assert c.value() == 1.0
    assert c.value({"a": "1"}) == 3.5


def test_counter_rejects_negative() -> None:
    c = Counter(name="x", description="x")
    with pytest.raises(ValueError):
        c.inc(-1)


def test_histogram_observe_and_render() -> None:
    h = Histogram(name="latency_seconds", description="lat", buckets=(0.1, 1.0, 10.0))
    h.observe(0.05)
    h.observe(0.5)
    h.observe(2.0)
    h.observe(20.0)
    rendered = "\n".join(h.render())
    # Prometheus cumulative buckets:
    assert 'latency_seconds_bucket{le="0.1"} 1' in rendered
    assert 'latency_seconds_bucket{le="1"} 2' in rendered
    assert 'latency_seconds_bucket{le="10"} 3' in rendered
    assert 'latency_seconds_bucket{le="+Inf"} 4' in rendered
    assert "latency_seconds_count " in rendered
    assert "latency_seconds_sum " in rendered


def test_registry_singletons() -> None:
    a = REGISTRY.counter("test_a", "A")
    b = REGISTRY.counter("test_a", "A")
    assert a is b


def test_render_prometheus_disabled_returns_empty() -> None:
    REGISTRY.set_enabled(False)
    c = REGISTRY.counter("x_total", "x")
    c.inc()
    assert render_prometheus() == ""


@pytest.mark.asyncio
async def test_time_block_records() -> None:
    h = REGISTRY.histogram("blk_seconds", "block latency")
    async with time_block(h, {"op": "demo"}):
        pass
    assert h.total({"op": "demo"}) == 1


@pytest.mark.asyncio
async def test_time_block_records_on_exception() -> None:
    h = REGISTRY.histogram("blk2_seconds", "fail latency")
    with pytest.raises(RuntimeError):
        async with time_block(h):
            raise RuntimeError("boom")
    assert h.total() == 1


@pytest.mark.asyncio
async def test_time_block_noop_when_disabled() -> None:
    REGISTRY.set_enabled(False)
    h = REGISTRY.histogram("blk3_seconds", "noop")
    async with time_block(h):
        pass
    assert h.total() == 0


def test_inc_counter_helper() -> None:
    c = REGISTRY.counter("hctr_total", "h")
    inc_counter(c, {"k": "v"})
    inc_counter(c, {"k": "v"})
    assert c.value({"k": "v"}) == 2.0


def test_render_prometheus_escapes_label_values() -> None:
    c = REGISTRY.counter("escape_total", "esc")
    c.inc(labels={"k": 'a"b\\c'})
    out = render_prometheus()
    assert 'k="a\\"b\\\\c"' in out


def test_label_ordering_is_stable() -> None:
    c = REGISTRY.counter("order_total", "ord")
    c.inc(labels={"b": "2", "a": "1"})
    out = render_prometheus()
    assert 'order_total{a="1",b="2"} 1' in out
