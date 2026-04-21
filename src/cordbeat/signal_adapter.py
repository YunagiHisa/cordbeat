"""Signal adapter — bridges Signal Messenger to CordBeat Core via WebSocket.

This is a v1.0+ scaffold. Signal has no official Python SDK, so this
adapter talks to a local ``signal-cli`` instance running in JSON-RPC
HTTP daemon mode (see https://github.com/AsamK/signal-cli).

Setup outline:

1. Install and register ``signal-cli`` with your phone number
2. Run: ``signal-cli -a +1234567890 daemon --http localhost:8088``
3. Configure CordBeat:

   .. code-block:: yaml

       adapters:
         signal:
           options:
             rpc_url: "http://localhost:8088/api/v1/rpc"
             phone_number: "+1234567890"
             poll_interval: 2

Install with::

    uv sync --extra signal
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from cordbeat.config import AdapterConfig
from cordbeat.gateway import RetryableConnection

logger = logging.getLogger(__name__)

ADAPTER_ID = "signal"


class SignalAdapter(RetryableConnection):
    """Signal adapter using ``signal-cli`` JSON-RPC daemon."""

    adapter_id = ADAPTER_ID

    def __init__(self, config: AdapterConfig) -> None:
        self._config = config
        self._ws_url = config.core_ws_url
        opts = config.options
        self._rpc_url: str = opts.get(
            "rpc_url", "http://localhost:8088/api/v1/rpc"
        )
        self._phone_number: str = opts.get("phone_number", "")
        self._poll_interval: float = float(opts.get("poll_interval", 2))
        self._http_client: Any = None
        self._ws: Any = None
        self._running = False
        self._max_backoff = config.reconnect_max_backoff
        self._request_id = 0

    async def start(self) -> None:
        try:
            import httpx
        except ImportError:
            logger.error(
                "httpx is not installed. Install with: uv sync --extra signal"
            )
            return

        if not self._phone_number:
            logger.error(
                "Signal phone_number not configured in adapters.signal.options"
            )
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Connect to Core in background
        asyncio.create_task(self._connect_to_core())
        # Poll signal-cli for inbound messages
        asyncio.create_task(self._poll_loop())

        logger.info("Signal adapter started (RPC: %s)", self._rpc_url)

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._http_client:
            await self._http_client.aclose()

    async def _dispatch_core_message(self, platform_user_id: str, content: str) -> None:
        await self._send_to_signal(platform_user_id, content)

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        if not self._http_client:
            return None
        req = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
            "params": params,
        }
        resp = await self._http_client.post(self._rpc_url, json=req)
        resp.raise_for_status()
        return resp.json().get("result")

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                result = await self._rpc(
                    "receive", {"account": self._phone_number, "timeout": 1}
                )
                for envelope in result or []:
                    msg = (envelope.get("envelope", {}) or {}).get(
                        "dataMessage", {}
                    ) or {}
                    text = msg.get("message") or ""
                    source = (envelope.get("envelope", {}) or {}).get("source") or ""
                    if text and source:
                        await self._forward_to_core(user_id=source, text=text)
            except Exception:
                logger.exception("Signal poll error")
            await asyncio.sleep(self._poll_interval)

    async def _forward_to_core(self, *, user_id: str, text: str) -> None:
        if self._ws is None or not user_id:
            return
        payload = json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": user_id,
                "content": text,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "metadata": {},
            }
        )
        try:
            await self._ws.send(payload)
        except Exception:
            logger.exception("Failed to forward message to Core")

    async def _send_to_signal(self, platform_user_id: str, content: str) -> None:
        if not self._http_client or not platform_user_id:
            return
        try:
            await self._rpc(
                "send",
                {
                    "account": self._phone_number,
                    "recipient": [platform_user_id],
                    "message": content,
                },
            )
        except Exception:
            logger.exception(
                "Failed to send message to Signal user %s", platform_user_id
            )
