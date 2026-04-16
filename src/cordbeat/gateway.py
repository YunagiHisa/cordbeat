"""Gateway — WebSocket server and adapter base class."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import websockets
from websockets.asyncio.server import Server, ServerConnection

from cordbeat.config import GatewayConfig
from cordbeat.models import GatewayMessage, MessageType

logger = logging.getLogger(__name__)

_MAX_BACKOFF = 60


class BaseAdapter(ABC):
    """Base class for platform adapters (Discord, Telegram, CLI, etc.)."""

    adapter_id: str

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the CordBeat Core gateway."""

    @abstractmethod
    async def on_message(self, message: GatewayMessage) -> None:
        """Handle an incoming message from the platform and forward to Core."""

    @abstractmethod
    async def send_message(
        self,
        platform_user_id: str,
        content: str,
        **kwargs: Any,
    ) -> None:
        """Send a message from Core to the platform user."""

    @abstractmethod
    async def on_disconnect(self) -> None:
        """Handle disconnection from Core."""


class RetryableConnection(ABC):
    """Mixin / helper for adapters that connect to Core via WebSocket.

    Handles exponential-backoff reconnection and incoming message dispatch.
    Subclasses implement ``_dispatch_core_message`` to handle inbound data.
    """

    _ws: Any
    _running: bool
    _ws_url: str
    adapter_id: str

    async def _connect_to_core(self) -> None:
        """Maintain a persistent WebSocket connection to Core with retry."""
        backoff = 1
        while self._running:
            try:
                self._ws = await websockets.connect(self._ws_url)
                await self._ws.send(json.dumps({"adapter_id": self.adapter_id}))
                ack = json.loads(await self._ws.recv())
                logger.info("Connected to Core: %s", ack.get("content", "OK"))
                backoff = 1
                await self._listen_core()
            except Exception:
                if not self._running:
                    break
                logger.warning(
                    "Core connection failed, retrying in %ds...",
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, getattr(self, "_max_backoff", _MAX_BACKOFF))

    async def _listen_core(self) -> None:
        try:
            async for raw in self._ws:
                data = json.loads(raw)
                msg_type = data.get("type", "")
                content = data.get("content", "")
                platform_user_id = data.get("platform_user_id", "")
                if msg_type in ("message", "heartbeat_message", "error"):
                    await self._dispatch_core_message(platform_user_id, content)
        except websockets.ConnectionClosed:
            logger.info("Core connection closed")

    @abstractmethod
    async def _dispatch_core_message(self, platform_user_id: str, content: str) -> None:
        """Override in subclass to route messages to the platform."""


# ── Message queue ─────────────────────────────────────────────────────


class MessageQueue:
    """Global message queue — single-threaded processing for local AI."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[GatewayMessage] = asyncio.Queue()
        self._handler: Any = None

    def set_handler(self, handler: Any) -> None:
        """Set the message handler (typically the Core engine)."""
        self._handler = handler

    async def put(self, message: GatewayMessage) -> None:
        await self._queue.put(message)

    async def process_loop(self) -> None:
        """Process messages one at a time (no parallel AI inference)."""
        while True:
            message = await self._queue.get()
            try:
                if self._handler:
                    await self._handler(message)
            except Exception:
                logger.exception("Error processing message: %s", message)
            finally:
                self._queue.task_done()


# ── WebSocket Gateway Server ──────────────────────────────────────────


class GatewayServer:
    """WebSocket server that adapters connect to."""

    def __init__(self, config: GatewayConfig, queue: MessageQueue) -> None:
        self._config = config
        self._queue = queue
        self._connections: dict[str, ServerConnection] = {}
        self._server: Server | None = None

    async def start(self) -> None:
        self._server = await websockets.serve(
            self._handle_connection,
            self._config.host,
            self._config.port,
        )
        logger.info(
            "Gateway server started on ws://%s:%d",
            self._config.host,
            self._config.port,
        )

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Gateway server stopped")

    async def send_to_adapter(
        self,
        adapter_id: str,
        message: GatewayMessage,
    ) -> None:
        ws = self._connections.get(adapter_id)
        if ws is None:
            logger.warning("Adapter '%s' not connected", adapter_id)
            return

        payload = json.dumps(
            {
                "type": message.type.value,
                "adapter_id": message.adapter_id,
                "platform_user_id": message.platform_user_id,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata,
            }
        )
        await ws.send(payload)

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        adapter_id: str | None = None
        try:
            # First message must identify the adapter
            raw = await asyncio.wait_for(
                websocket.recv(), timeout=self._config.handshake_timeout
            )
            data = json.loads(raw)
            adapter_id = data.get("adapter_id")
            if not adapter_id:
                await websocket.close(1008, "Missing adapter_id")
                return

            self._connections[adapter_id] = websocket
            logger.info("Adapter connected: %s", adapter_id)

            # Send ACK
            await websocket.send(
                json.dumps(
                    {
                        "type": MessageType.ACK.value,
                        "adapter_id": "core",
                        "content": f"Welcome {adapter_id}",
                    }
                )
            )

            async for raw_msg in websocket:
                try:
                    msg_data = json.loads(raw_msg)
                    message = GatewayMessage(
                        type=MessageType(msg_data.get("type", "message")),
                        adapter_id=msg_data.get("adapter_id", adapter_id),
                        platform_user_id=msg_data.get("platform_user_id", ""),
                        content=msg_data.get("content", ""),
                        timestamp=datetime.fromisoformat(msg_data["timestamp"])
                        if "timestamp" in msg_data
                        else datetime.now(),
                        metadata=msg_data.get("metadata", {}),
                    )
                    await self._queue.put(message)
                except (json.JSONDecodeError, ValueError, KeyError) as exc:
                    logger.warning(
                        "Invalid message from %s: %s",
                        adapter_id,
                        exc,
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": MessageType.ERROR.value,
                                "content": "Invalid message format",
                            }
                        )
                    )

        except TimeoutError:
            logger.warning("Adapter handshake timeout")
        except websockets.ConnectionClosed:
            logger.info("Adapter disconnected: %s", adapter_id)
        finally:
            if adapter_id and adapter_id in self._connections:
                del self._connections[adapter_id]
