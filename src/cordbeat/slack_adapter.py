"""Slack adapter — bridges Slack workspace to CordBeat Core via WebSocket.

This is a v1.0+ scaffold. The CordBeat-side plumbing (Core WebSocket
connection, message forwarding, reply dispatch) follows the same pattern
as :mod:`cordbeat.discord_adapter` / :mod:`cordbeat.telegram_adapter`.

Platform-side integration uses ``slack-sdk``'s Socket Mode
(``SocketModeClient``) so that the adapter does not need a public
webhook endpoint. Configure two tokens in ``adapters.slack.options``:

* ``bot_token``   — ``xoxb-...`` (Bot User OAuth Token)
* ``app_token``   — ``xapp-...`` (App-Level Token with ``connections:write``)

Install with::

    uv sync --extra slack
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

ADAPTER_ID = "slack"


class SlackAdapter(RetryableConnection):
    """Slack bot that forwards messages to CordBeat Core via WebSocket."""

    adapter_id = ADAPTER_ID

    def __init__(self, config: AdapterConfig) -> None:
        self._config = config
        self._ws_url = config.core_ws_url
        self._bot_token: str = config.options.get("bot_token", "")
        self._app_token: str = config.options.get("app_token", "")
        self._socket_client: Any = None
        self._web_client: Any = None
        self._ws: Any = None
        self._running = False
        self._max_backoff = config.reconnect_max_backoff
        # Map platform_user_id → channel for reply routing
        self._user_channels: dict[str, str] = {}

    async def start(self) -> None:
        try:
            from slack_sdk.socket_mode.aiohttp import SocketModeClient
            from slack_sdk.socket_mode.request import SocketModeRequest
            from slack_sdk.socket_mode.response import SocketModeResponse
            from slack_sdk.web.async_client import AsyncWebClient
        except ImportError:
            logger.error(
                "slack-sdk is not installed. Install with: uv sync --extra slack"
            )
            return

        if not self._bot_token or not self._app_token:
            logger.error(
                "Slack tokens not configured in adapters.slack.options "
                "(bot_token + app_token required)"
            )
            return

        self._running = True
        self._web_client = AsyncWebClient(token=self._bot_token)
        self._socket_client = SocketModeClient(
            app_token=self._app_token,
            web_client=self._web_client,
        )

        async def handle_socket(
            client: SocketModeClient, req: SocketModeRequest
        ) -> None:
            # Ack every request first
            await client.send_socket_mode_response(
                SocketModeResponse(envelope_id=req.envelope_id)
            )
            if req.type != "events_api":
                return
            event = req.payload.get("event", {})
            if event.get("type") != "message" or event.get("bot_id"):
                return
            await self._forward_to_core(
                user_id=event.get("user", ""),
                text=event.get("text", ""),
                channel=event.get("channel", ""),
            )

        self._socket_client.socket_mode_request_listeners.append(handle_socket)

        # Connect to Core in background
        asyncio.create_task(self._connect_to_core())

        # Connect to Slack Socket Mode (long-running)
        await self._socket_client.connect()
        logger.info("Slack adapter started")

        # Keep the coroutine alive until stop() is called
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._socket_client:
            await self._socket_client.disconnect()
            await self._socket_client.close()

    async def _dispatch_core_message(self, platform_user_id: str, content: str) -> None:
        await self._send_to_slack(platform_user_id, content)

    async def _forward_to_core(self, *, user_id: str, text: str, channel: str) -> None:
        if self._ws is None or not user_id:
            return

        self._user_channels[user_id] = channel

        payload = json.dumps(
            {
                "type": "message",
                "adapter_id": ADAPTER_ID,
                "platform_user_id": user_id,
                "content": text,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "metadata": {"channel": channel},
            }
        )
        try:
            await self._ws.send(payload)
        except Exception:
            logger.exception("Failed to forward message to Core")

    async def _send_to_slack(self, platform_user_id: str, content: str) -> None:
        if not self._web_client or not platform_user_id:
            return
        channel = self._user_channels.get(platform_user_id)
        if not channel:
            # Fall back to opening a DM
            try:
                resp = await self._web_client.conversations_open(users=platform_user_id)
                channel = resp["channel"]["id"]
            except Exception:
                logger.exception(
                    "Failed to open Slack DM for user %s", platform_user_id
                )
                return
        try:
            await self._web_client.chat_postMessage(channel=channel, text=content)
        except Exception:
            logger.exception(
                "Failed to send message to Slack user %s", platform_user_id
            )
