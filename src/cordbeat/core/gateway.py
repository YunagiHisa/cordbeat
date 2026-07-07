"""Gateway — WebSocket server and adapter base class."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import websockets
from websockets.asyncio.server import Server, ServerConnection

from cordbeat.config import GatewayConfig
from cordbeat.models import GatewayMessage, MessageType

logger = logging.getLogger(__name__)

# Probe connections (raw TCP / health checks) cause a harmless
# "did not receive a valid HTTP request" error inside the websockets library.
# Raise its threshold to WARNING so these don't pollute the user's log.
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
logging.getLogger("websockets.asyncio.server").setLevel(logging.CRITICAL)

_MAX_BACKOFF = 60
_DEFAULT_WS_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
# Bounded resend buffer for user messages typed while Core is unreachable.
_OUTBOX_MAX = 50


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
    _auth_token: str
    _pending_outbox: deque[str]
    adapter_id: str

    async def _connect_to_core(self) -> None:
        """Maintain a persistent WebSocket connection to Core with retry."""
        backoff = 1
        while self._running:
            try:
                self._ws = await websockets.connect(
                    self._ws_url,
                    max_size=getattr(
                        self,
                        "_ws_max_message_bytes",
                        _DEFAULT_WS_MAX_MESSAGE_BYTES,
                    ),
                )
                handshake: dict[str, str] = {"adapter_id": self.adapter_id}
                if getattr(self, "_auth_token", ""):
                    handshake["auth_token"] = self._auth_token
                await self._ws.send(json.dumps(handshake))
                ack = json.loads(await self._ws.recv())
                logger.info("Connected to Core: %s", ack.get("content", "OK"))
                backoff = 1
                await self._flush_outbox()
                await self._listen_core()
            except Exception:
                self._ws = None  # ensure stale WS is not used by _forward_to_core
                if not self._running:
                    break
                logger.warning(
                    "Core connection failed, retrying in %ds...",
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, getattr(self, "_max_backoff", _MAX_BACKOFF))

    @property
    def _outbox(self) -> deque[str]:
        box: deque[str] | None = getattr(self, "_pending_outbox", None)
        if box is None:
            box = deque(maxlen=_OUTBOX_MAX)
            self._pending_outbox = box
        return box

    async def _send_to_core(self, payload: str) -> bool:
        """Send *payload* to Core, buffering it for resend on failure.

        Returns True when the payload was sent immediately. Buffered
        payloads are flushed in order after the next successful reconnect
        (oldest entries are dropped beyond the buffer limit).
        """
        ws = getattr(self, "_ws", None)
        if ws is not None:
            try:
                await ws.send(payload)
                return True
            except Exception:
                logger.warning("Send to Core failed; buffering message for resend")
        else:
            logger.warning("Not connected to Core; buffering message for resend")
        self._outbox.append(payload)
        return False

    async def _flush_outbox(self) -> None:
        """Resend any messages buffered while Core was unreachable."""
        box = self._outbox
        if not box:
            return
        logger.info("Resending %d buffered message(s) to Core", len(box))
        while box:
            payload = box.popleft()
            try:
                await self._ws.send(payload)
            except Exception:
                box.appendleft(payload)
                logger.warning(
                    "Outbox flush interrupted; %d message(s) still pending",
                    len(box),
                )
                return

    async def _listen_core(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from Core: %r", raw)
                    continue
                msg_type = data.get("type", "")
                content = data.get("content", "")
                platform_user_id = data.get("platform_user_id", "")
                images: list[str] = data.get("images") or []
                metadata: dict[str, Any] = data.get("metadata") or {}
                if msg_type == "skill_confirm":
                    await self._dispatch_skill_confirm(platform_user_id, data)
                elif msg_type in ("message", "heartbeat_message", "ack", "error"):
                    await self._dispatch_core_message(
                        platform_user_id, content, images, metadata=metadata
                    )
        except websockets.ConnectionClosed:
            logger.info("Core connection closed")

    async def _dispatch_skill_confirm(
        self, platform_user_id: str, data: dict[str, Any]
    ) -> None:
        """Handle a skill confirmation request from Core.

        Default implementation falls back to a plain text message.
        Override in subclasses to show a richer confirmation UI (e.g. buttons).
        """
        content = data.get("content", "")
        await self._dispatch_core_message(platform_user_id, content, [])

    @abstractmethod
    async def _dispatch_core_message(
        self,
        platform_user_id: str,
        content: str,
        images: list[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Override in subclass to route messages to the platform.

        ``metadata`` carries optional adapter-specific routing hints from
        Core (e.g. ``{"channel_id": "...", "is_dm": False}``). Adapters
        may ignore it for backward compatibility.
        """


# ── Message queue ─────────────────────────────────────────────────────


@runtime_checkable
class MessageQueueProtocol(Protocol):
    """Swappable message-queue contract.

    CordBeat uses a single serialized queue by default because local AI
    inference is a heavy, non-parallelizable operation on most hardware.
    The contract is kept behind this Protocol so deployments that *do*
    have headroom for concurrent inference (multi-GPU, Ollama with
    parallelism enabled, remote backends) can substitute an alternative
    implementation — e.g. a worker-pool or priority queue — without
    touching :class:`GatewayServer` or :class:`~cordbeat.heartbeat.HeartbeatLoop`.
    """

    def set_handler(self, handler: Any) -> None:
        """Register the coroutine that will consume dequeued messages."""

    async def put(self, message: GatewayMessage) -> None:
        """Enqueue *message* for eventual handler dispatch."""

    async def process_loop(self) -> None:
        """Run forever, passing each dequeued message to the handler."""

    def is_busy(self) -> bool:
        """Return True if the queue is currently dispatching a message
        (i.e. the handler is actively running).

        Used by heartbeat / proactive subsystems to defer work that
        would otherwise contend with the active inference call. A
        non-empty pending queue ALSO counts as busy because the next
        message will be picked up immediately after the current one.
        """


class MessageQueue:
    """Default :class:`MessageQueueProtocol` — single-threaded FIFO."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[GatewayMessage] = asyncio.Queue()
        self._handler: Any = None
        self._processing: bool = False

    def set_handler(self, handler: Any) -> None:
        """Set the message handler (typically the Core engine)."""
        self._handler = handler

    async def put(self, message: GatewayMessage) -> None:
        await self._queue.put(message)

    def is_busy(self) -> bool:
        """True while a message is being handled or any are pending.

        Single-GPU local inference cannot tolerate parallel calls; the
        heartbeat loop checks this before kicking off its own LLM run.
        """
        return self._processing or not self._queue.empty()

    async def process_loop(self) -> None:
        """Process messages one at a time (no parallel AI inference)."""
        while True:
            message = await self._queue.get()
            self._processing = True
            try:
                if self._handler:
                    await self._handler(message)
            except Exception:
                logger.exception("Error processing message: %s", message)
            finally:
                self._processing = False
                self._queue.task_done()


# ── WebSocket Gateway Server ──────────────────────────────────────────


class GatewayServer:
    """WebSocket server that adapters connect to."""

    def __init__(self, config: GatewayConfig, queue: MessageQueueProtocol) -> None:
        self._config = config
        self._queue = queue
        self._connections: dict[str, ServerConnection] = {}
        self._server: Server | None = None

    async def start(self) -> None:
        self._server = await websockets.serve(
            self._handle_connection,
            self._config.host,
            self._config.port,
            max_size=self._config.max_message_bytes,
        )
        logger.info(
            "Gateway server started on ws://%s:%d (max_message_bytes=%d)",
            self._config.host,
            self._config.port,
            self._config.max_message_bytes,
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
                "images": message.images,
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
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Rejected connection: invalid handshake JSON")
                await websocket.close(1008, "Invalid handshake")
                return
            if not isinstance(data, dict):
                logger.warning("Rejected connection: handshake is not an object")
                await websocket.close(1008, "Invalid handshake")
                return
            adapter_id = data.get("adapter_id")
            if not adapter_id:
                await websocket.close(1008, "Missing adapter_id")
                return

            # Validate auth token if configured
            if self._config.auth_token:
                token = data.get("auth_token", "")
                if not hmac.compare_digest(token, self._config.auth_token):
                    await websocket.close(1008, "Invalid auth_token")
                    logger.warning(
                        "Rejected adapter '%s': invalid auth token", adapter_id
                    )
                    return

            old_websocket = self._connections.get(adapter_id)
            if old_websocket is not None and old_websocket is not websocket:
                await old_websocket.close(1012, "Replaced by new connection")
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
                        else datetime.now(tz=UTC),
                        metadata=msg_data.get("metadata", {}),
                        images=msg_data.get("images", []),
                        is_voice=bool(msg_data.get("is_voice", False)),
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
            if adapter_id and self._connections.get(adapter_id) is websocket:
                del self._connections[adapter_id]
