"""WhatsApp adapter — bridges WhatsApp Cloud API to CordBeat Core.

This is a v1.0+ scaffold. The Meta WhatsApp Cloud API is webhook-based
(no official Python SDK). This adapter uses raw ``httpx`` for outbound
messages and ``aiohttp`` to receive inbound webhook events.

Configure in ``adapters.whatsapp.options``:

* ``access_token``      — permanent or short-lived access token
* ``phone_number_id``   — Meta phone number ID (numeric string)
* ``verify_token``      — user-chosen string used for webhook subscription verify
* ``webhook_host``      — bind host (default ``0.0.0.0``)
* ``webhook_port``      — bind port (default ``8081``)
* ``webhook_path``      — URL path (default ``/webhook``)

This scaffold handles only text messages; media support is a future task.

Install with::

    uv sync --extra whatsapp
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

ADAPTER_ID = "whatsapp"

_GRAPH_API_BASE = "https://graph.facebook.com/v20.0"


class WhatsAppAdapter(RetryableConnection):
    """WhatsApp Cloud API adapter that forwards messages to CordBeat Core."""

    adapter_id = ADAPTER_ID

    def __init__(self, config: AdapterConfig) -> None:
        self._config = config
        self._ws_url = config.core_ws_url
        opts = config.options
        self._access_token: str = opts.get("access_token", "")
        self._phone_number_id: str = opts.get("phone_number_id", "")
        self._verify_token: str = opts.get("verify_token", "")
        self._webhook_host: str = opts.get("webhook_host", "0.0.0.0")
        self._webhook_port: int = int(opts.get("webhook_port", 8081))
        self._webhook_path: str = opts.get("webhook_path", "/webhook")
        self._http_runner: Any = None
        self._http_client: Any = None
        self._ws: Any = None
        self._running = False
        self._max_backoff = config.reconnect_max_backoff

    async def start(self) -> None:
        try:
            import httpx
            from aiohttp import web
        except ImportError:
            logger.error(
                "httpx / aiohttp not installed. Install with: uv sync --extra whatsapp"
            )
            return

        if not self._access_token or not self._phone_number_id:
            logger.error(
                "WhatsApp credentials not configured in adapters.whatsapp.options "
                "(access_token + phone_number_id required)"
            )
            return

        self._running = True
        self._http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._access_token}"},
            timeout=30.0,
        )

        async def handle_verify(request: web.Request) -> web.Response:
            # Meta webhook verification GET
            mode = request.query.get("hub.mode")
            token = request.query.get("hub.verify_token")
            challenge = request.query.get("hub.challenge", "")
            if mode == "subscribe" and token == self._verify_token:
                return web.Response(text=challenge)
            return web.Response(status=403, text="Verification failed")

        async def handle_webhook(request: web.Request) -> web.Response:
            try:
                payload = await request.json()
            except Exception:
                logger.exception("Invalid WhatsApp webhook payload")
                return web.Response(status=400)

            for entry in payload.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    for message in value.get("messages", []):
                        if message.get("type") != "text":
                            continue
                        await self._forward_to_core(
                            user_id=message.get("from", ""),
                            text=message.get("text", {}).get("body", ""),
                        )
            return web.Response(text="OK")

        app = web.Application()
        app.router.add_get(self._webhook_path, handle_verify)
        app.router.add_post(self._webhook_path, handle_webhook)
        self._http_runner = web.AppRunner(app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, self._webhook_host, self._webhook_port)
        await site.start()
        logger.info(
            "WhatsApp webhook listening on http://%s:%d%s",
            self._webhook_host,
            self._webhook_port,
            self._webhook_path,
        )

        # Connect to Core in background
        asyncio.create_task(self._connect_to_core())

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._http_runner:
            await self._http_runner.cleanup()
        if self._http_client:
            await self._http_client.aclose()

    async def _dispatch_core_message(self, platform_user_id: str, content: str) -> None:
        await self._send_to_whatsapp(platform_user_id, content)

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

    async def _send_to_whatsapp(self, platform_user_id: str, content: str) -> None:
        if not self._http_client or not platform_user_id:
            return
        url = f"{_GRAPH_API_BASE}/{self._phone_number_id}/messages"
        body = {
            "messaging_product": "whatsapp",
            "to": platform_user_id,
            "type": "text",
            "text": {"body": content},
        }
        try:
            resp = await self._http_client.post(url, json=body)
            resp.raise_for_status()
        except Exception:
            logger.exception(
                "Failed to send message to WhatsApp user %s", platform_user_id
            )
