"""Optional Prometheus text-format HTTP exporter.

Implements a tiny stdlib-only HTTP/1.1 server bound to a configurable
host/port that responds to ``GET /metrics`` with the output of
:func:`cordbeat.metrics.render_prometheus`. Default bind is loopback,
matching :class:`cordbeat.config.GatewayConfig`'s security posture.

Anything other than ``GET /metrics`` returns 404 with no body. The
server has no authentication: deploy behind a reverse proxy or scrape
locally only.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable

from cordbeat.metrics import render_prometheus

logger = logging.getLogger(__name__)

_NOT_FOUND = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
_BAD_REQUEST = (
    b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
)


def _build_metrics_response(body: bytes) -> bytes:
    headers = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
        b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
        b"Connection: close\r\n"
        b"\r\n"
    )
    return headers + body


class PrometheusServer:
    """Minimal asyncio TCP server speaking just enough HTTP/1.1."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, host=self._host, port=self._port
        )

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        try:
            wait_closed: Awaitable[None] = self._server.wait_closed()
            await asyncio.wait_for(wait_closed, timeout=5.0)
        except TimeoutError:
            logger.warning("metrics_server: wait_closed timed out")
        finally:
            self._server = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not request_line:
                writer.write(_BAD_REQUEST)
                return
            try:
                method, path, _ = request_line.decode("iso-8859-1").split(" ", 2)
            except ValueError:
                writer.write(_BAD_REQUEST)
                return

            # Drain remaining headers (we ignore them).
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line in (b"\r\n", b"\n", b""):
                    break

            if method.upper() != "GET" or path.split("?", 1)[0] != "/metrics":
                writer.write(_NOT_FOUND)
                return

            body = render_prometheus().encode("utf-8")
            writer.write(_build_metrics_response(body))
        except TimeoutError:
            logger.debug("metrics_server: client read timeout", exc_info=True)
        except Exception:
            logger.exception("metrics_server: handler error")
        finally:
            try:
                await writer.drain()
            except Exception:  # noqa: BLE001 - best-effort drain on shutdown
                pass
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001 - best-effort close
                pass
