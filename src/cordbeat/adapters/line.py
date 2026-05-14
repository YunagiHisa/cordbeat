"""LINE adapter — bridges LINE Messaging API to CordBeat Core via WebSocket.

This is a v1.0+ scaffold. LINE Messaging API is webhook-based (no long-poll),
so this adapter runs a small ``aiohttp`` HTTP server to receive webhook
events from LINE, and uses ``line-bot-sdk`` (v3 async API) to push replies.

Configure in ``adapters.line.options``:

* ``channel_access_token`` — long-lived Channel Access Token
* ``channel_secret``       — channel secret (used to verify webhook signature)
* ``webhook_host``         — host to bind webhook server (default ``0.0.0.0``)
* ``webhook_port``         — port to bind webhook server (default ``8080``)
* ``webhook_path``         — URL path to serve (default ``/webhook``)

Install with::

    uv sync --extra line
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from cordbeat.adapters._utils import AdapterFilter
from cordbeat.config import AdapterConfig
from cordbeat.core.gateway import RetryableConnection

logger = logging.getLogger(__name__)

ADAPTER_ID = "line"


class LineAdapter(RetryableConnection):
    """LINE bot that forwards messages to CordBeat Core via WebSocket."""

    adapter_id = ADAPTER_ID

    def __init__(self, config: AdapterConfig) -> None:
        self._config = config
        self._ws_url = config.core_ws_url
        self._auth_token = config.auth_token
        opts = config.options
        self._channel_access_token: str = opts.get("channel_access_token", "")
        self._channel_secret: str = opts.get("channel_secret", "")
        self._webhook_host: str = opts.get("webhook_host", "0.0.0.0")
        self._webhook_port: int = int(opts.get("webhook_port", 8080))
        self._webhook_path: str = opts.get("webhook_path", "/webhook")
        self._http_runner: Any = None
        self._messaging_api: Any = None
        self._parser: Any = None
        self._ws: Any = None
        self._running = False
        self._max_backoff = config.reconnect_max_backoff
        self._filter = AdapterFilter.from_options(config.options)

    async def start(self) -> None:
        try:
            from aiohttp import web
            from linebot.v3 import WebhookParser
            from linebot.v3.messaging import (
                AsyncApiClient,
                AsyncMessagingApi,
                Configuration,
                PushMessageRequest,
                TextMessage,
            )
            from linebot.v3.webhooks import MessageEvent, TextMessageContent
        except ImportError:
            logger.error(
                "line-bot-sdk / aiohttp not installed. "
                "Install with: uv sync --extra line"
            )
            return

        if not self._channel_access_token or not self._channel_secret:
            logger.error(
                "LINE credentials not configured in adapters.line.options "
                "(channel_access_token + channel_secret required)"
            )
            return

        self._running = True
        line_config = Configuration(access_token=self._channel_access_token)
        api_client = AsyncApiClient(line_config)
        self._messaging_api = AsyncMessagingApi(api_client)
        self._parser = WebhookParser(self._channel_secret)

        # Expose PushMessageRequest/TextMessage for the reply path
        self._PushMessageRequest = PushMessageRequest
        self._TextMessage = TextMessage

        async def handle_webhook(request: web.Request) -> web.Response:
            signature = request.headers.get("X-Line-Signature", "")
            body = await request.text()
            try:
                events = self._parser.parse(body, signature)
            except Exception:
                logger.exception("Invalid LINE webhook signature or payload")
                return web.Response(status=400, text="Invalid signature")

            for event in events:
                if isinstance(event, MessageEvent) and isinstance(
                    event.message, TextMessageContent
                ):
                    await self._forward_to_core(
                        user_id=event.source.user_id or "",
                        text=event.message.text,
                        is_group=getattr(event.source, "type", "user") != "user",
                    )
            return web.Response(text="OK")

        app = web.Application()
        app.router.add_post(self._webhook_path, handle_webhook)
        self._http_runner = web.AppRunner(app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, self._webhook_host, self._webhook_port)
        await site.start()
        logger.info(
            "LINE webhook listening on http://%s:%d%s",
            self._webhook_host,
            self._webhook_port,
            self._webhook_path,
        )

        # Connect to Core in background
        asyncio.create_task(self._connect_to_core())

        # Keep coroutine alive until stop()
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._http_runner:
            await self._http_runner.cleanup()

    async def _dispatch_core_message(
        self, platform_user_id: str, content: str, images: list[str]
    ) -> None:
        await self._send_to_line(platform_user_id, content)

    async def _forward_to_core(
        self, *, user_id: str, text: str, is_group: bool = False
    ) -> None:
        if self._ws is None or not user_id:
            return

        # ── E-4 response filtering ────────────────────────────────────
        # LINE groups don't have a platform @mention; treat keyword-match as "mentioned"
        is_dm = not is_group
        is_mentioned = False
        if is_group:
            from cordbeat.adapters._utils import _read_soul_keywords

            kws = list(self._filter.ai_keywords) or _read_soul_keywords()
            if kws:
                is_mentioned = any(kw.lower() in text.lower() for kw in kws)
            else:
                is_mentioned = True  # no keywords → treat as mentioned
        if not self._filter.should_respond(
            user_id=user_id,
            channel_id="",  # LINE groups have no stable channel_id for filtering
            is_dm=is_dm,
            is_mentioned=is_mentioned,
            text=text,
        ):
            return
        # ─────────────────────────────────────────────────────────────

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

    async def _send_to_line(self, platform_user_id: str, content: str) -> None:
        if not self._messaging_api or not platform_user_id:
            return
        try:
            request = self._PushMessageRequest(
                to=platform_user_id,
                messages=[self._TextMessage(text=content)],
            )
            await self._messaging_api.push_message(request)
        except Exception:
            logger.exception("Failed to send message to LINE user %s", platform_user_id)
