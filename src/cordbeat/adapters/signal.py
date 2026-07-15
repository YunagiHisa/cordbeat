"""Signal adapter — bridges Signal Messenger to CordBeat Core via WebSocket.

This is an optional scaffold. Signal has no official Python SDK, so this
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

from cordbeat.adapters._utils import (
    AdapterFilter,
    normalize_inbound_text,
    parse_unix_timestamp,
)
from cordbeat.config import AdapterConfig
from cordbeat.core.gateway import RetryableConnection

logger = logging.getLogger(__name__)

ADAPTER_ID = "signal"


class SignalAdapter(RetryableConnection):
    """Signal adapter using ``signal-cli`` JSON-RPC daemon."""

    adapter_id = ADAPTER_ID

    def __init__(self, config: AdapterConfig) -> None:
        self._config = config
        self._ws_url = config.core_ws_url
        self._auth_token = config.auth_token
        opts = config.options
        self._rpc_url: str = opts.get("rpc_url", "http://localhost:8088/api/v1/rpc")
        self._phone_number: str = opts.get("phone_number", "")
        self._poll_interval: float = float(opts.get("poll_interval", 2))
        self._http_timeout: float = float(opts.get("http_timeout", 30.0))
        self._http_client: Any = None
        self._ws: Any = None
        self._running = False
        self._max_backoff = config.reconnect_max_backoff
        self._request_id = 0
        self._filter = AdapterFilter.from_options(config.options)

    async def start(self) -> None:
        try:
            import httpx
        except ImportError:
            logger.error("httpx is not installed. Install with: uv sync --extra signal")
            return

        if not self._phone_number:
            logger.error(
                "Signal phone_number not configured in adapters.signal.options"
            )
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=self._http_timeout)

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

    async def _dispatch_core_message(
        self,
        platform_user_id: str,
        content: str,
        images: list[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
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
        failures = 0
        last_warning_at = 0.0
        while self._running:
            try:
                result = await self._rpc(
                    "receive", {"account": self._phone_number, "timeout": 1}
                )
                for envelope in result or []:
                    envelope_data = envelope.get("envelope", {}) or {}
                    msg = envelope_data.get(
                        "dataMessage", {}
                    ) or {}
                    text = msg.get("message") or ""
                    source = envelope_data.get("source") or ""
                    if text and source:
                        sent_at = parse_unix_timestamp(
                            envelope_data.get("timestamp"), milliseconds=True
                        )
                        await self._forward_to_core(
                            user_id=source, text=text, sent_at=sent_at
                        )
                failures = 0
            except Exception:
                failures += 1
                loop_time = asyncio.get_running_loop().time()
                if failures == 1:
                    logger.exception("Signal poll error")
                    last_warning_at = loop_time
                elif loop_time - last_warning_at >= 30.0:
                    logger.warning("Signal poll error continues", exc_info=True)
                    last_warning_at = loop_time
            delay = min(self._poll_interval * 2**failures, 60.0)
            await asyncio.sleep(delay)

    async def _forward_to_core(
        self, *, user_id: str, text: str, sent_at: datetime | None = None
    ) -> None:
        if not user_id:
            return
        normalized = normalize_inbound_text(text, adapter_id=ADAPTER_ID)
        if normalized is None:
            return
        text = normalized

        # Signal is 1:1 — always a DM; no channel filters needed.
        # respond_mode and ai_decision_keywords still apply.
        if not self._filter.should_respond(user_id=user_id, is_dm=True):
            return

        payload = json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": user_id,
                "content": text,
                "timestamp": (sent_at or datetime.now(tz=UTC)).isoformat(),
                "metadata": {
                    "channel_id": user_id,
                    "is_dm": True,
                },
            }
        )
        # Buffered for resend if Core is unreachable (bounded outbox).
        await self._send_to_core(payload)

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
