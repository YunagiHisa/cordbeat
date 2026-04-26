"""Integration test for the optional Prometheus HTTP exporter."""

from __future__ import annotations

import asyncio

import pytest

from cordbeat.metrics import REGISTRY
from cordbeat.metrics_server import PrometheusServer


async def _http_get(host: str, port: int, path: str) -> tuple[int, bytes]:
    reader, writer = await asyncio.open_connection(host, port)
    request = (
        f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    writer.write(request)
    await writer.drain()
    raw = await reader.read(-1)
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:  # noqa: BLE001 - best-effort
        pass
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0]
    code = int(status_line.split(b" ")[1])
    return code, body


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    REGISTRY.reset()


@pytest.mark.asyncio
async def test_prometheus_server_serves_metrics() -> None:
    counter = REGISTRY.counter("served_total", "served")
    counter.inc(3)

    server = PrometheusServer(host="127.0.0.1", port=0)
    await server.start()
    assert server._server is not None  # noqa: SLF001 - test introspection
    sock = next(iter(server._server.sockets))  # noqa: SLF001
    port = sock.getsockname()[1]

    try:
        code, body = await _http_get("127.0.0.1", port, "/metrics")
        assert code == 200
        text = body.decode("utf-8")
        assert "served_total 3" in text
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_prometheus_server_404_for_other_paths() -> None:
    server = PrometheusServer(host="127.0.0.1", port=0)
    await server.start()
    assert server._server is not None  # noqa: SLF001
    sock = next(iter(server._server.sockets))  # noqa: SLF001
    port = sock.getsockname()[1]
    try:
        code, _ = await _http_get("127.0.0.1", port, "/")
        assert code == 404
    finally:
        await server.stop()
