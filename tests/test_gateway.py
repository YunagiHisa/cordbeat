"""Tests for gateway message queue and server."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from cordbeat.config import GatewayConfig
from cordbeat.gateway import GatewayServer, MessageQueue, RetryableConnection
from cordbeat.models import GatewayMessage, MessageType


class _AsyncIter:
    """Helper to create an async iterator from a list."""

    def __init__(self, items: list[str]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


class TestMessageQueue:
    async def test_put_and_process(self) -> None:
        queue = MessageQueue()
        received: list[GatewayMessage] = []

        async def handler(msg: GatewayMessage) -> None:
            received.append(msg)

        queue.set_handler(handler)

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="u1",
            content="hello",
        )
        await queue.put(msg)

        task = asyncio.create_task(queue.process_loop())
        await asyncio.sleep(0.05)
        task.cancel()

        assert len(received) == 1
        assert received[0].content == "hello"

    async def test_no_handler_does_not_crash(self) -> None:
        queue = MessageQueue()  # no handler set
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="u1",
            content="hello",
        )
        await queue.put(msg)

        task = asyncio.create_task(queue.process_loop())
        await asyncio.sleep(0.05)
        task.cancel()

    async def test_handler_error_does_not_stop_loop(self) -> None:
        queue = MessageQueue()
        call_count = 0

        async def failing_handler(msg: GatewayMessage) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("test error")

        queue.set_handler(failing_handler)

        for i in range(3):
            await queue.put(
                GatewayMessage(
                    type=MessageType.MESSAGE,
                    adapter_id="test",
                    platform_user_id="u1",
                    content=f"msg{i}",
                )
            )

        task = asyncio.create_task(queue.process_loop())
        await asyncio.sleep(0.1)
        task.cancel()

        assert call_count == 3

    async def test_multiple_messages_processed_sequentially(self) -> None:
        queue = MessageQueue()
        order: list[str] = []

        async def handler(msg: GatewayMessage) -> None:
            order.append(msg.content)

        queue.set_handler(handler)

        for content in ["first", "second", "third"]:
            await queue.put(
                GatewayMessage(
                    type=MessageType.MESSAGE,
                    adapter_id="test",
                    platform_user_id="u1",
                    content=content,
                )
            )

        task = asyncio.create_task(queue.process_loop())
        await asyncio.sleep(0.1)
        task.cancel()

        assert order == ["first", "second", "third"]


class TestGatewayServer:
    def test_init(self) -> None:
        config = GatewayConfig(host="127.0.0.1", port=9999)
        queue = MessageQueue()
        server = GatewayServer(config, queue)
        assert server._config.host == "127.0.0.1"
        assert server._config.port == 9999

    async def test_send_to_adapter_not_connected(self) -> None:
        """send_to_adapter with unknown adapter_id → warning, no crash."""
        config = GatewayConfig()
        queue = MessageQueue()
        server = GatewayServer(config, queue)
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="unknown",
            platform_user_id="u1",
            content="hello",
        )
        # Should not raise
        await server.send_to_adapter("unknown", msg)

    async def test_send_to_adapter_connected(self) -> None:
        """send_to_adapter with connected adapter → sends JSON."""
        config = GatewayConfig()
        queue = MessageQueue()
        server = GatewayServer(config, queue)

        mock_ws = AsyncMock()
        server._connections["test_adapter"] = mock_ws

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test_adapter",
            platform_user_id="u1",
            content="hello",
        )
        await server.send_to_adapter("test_adapter", msg)
        mock_ws.send.assert_called_once()
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload["content"] == "hello"
        assert payload["type"] == "message"

    async def test_start_and_stop(self) -> None:
        config = GatewayConfig(host="127.0.0.1", port=0)
        queue = MessageQueue()
        server = GatewayServer(config, queue)
        await server.start()
        assert server._server is not None
        await server.stop()

    async def test_stop_when_not_started(self) -> None:
        config = GatewayConfig()
        queue = MessageQueue()
        server = GatewayServer(config, queue)
        # Should not raise
        await server.stop()

    async def test_handle_connection_valid_handshake(self) -> None:
        config = GatewayConfig()
        queue = MessageQueue()
        server = GatewayServer(config, queue)

        mock_ws = AsyncMock()
        # First recv returns adapter ID, then iterator yields a message
        handshake = json.dumps({"adapter_id": "test_adapter"})
        msg_data = json.dumps(
            {
                "type": "message",
                "adapter_id": "test_adapter",
                "platform_user_id": "u1",
                "content": "hello",
            }
        )
        mock_ws.recv = AsyncMock(return_value=handshake)
        mock_ws.__aiter__ = lambda self: _AsyncIter([msg_data])
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        await server._handle_connection(mock_ws)

        # Adapter should have been registered then cleaned up
        assert "test_adapter" not in server._connections
        # ACK should have been sent
        assert mock_ws.send.call_count >= 1
        ack = json.loads(mock_ws.send.call_args_list[0][0][0])
        assert ack["type"] == "ack"

    async def test_handle_connection_missing_adapter_id(self) -> None:
        config = GatewayConfig()
        queue = MessageQueue()
        server = GatewayServer(config, queue)

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({}))
        mock_ws.close = AsyncMock()

        await server._handle_connection(mock_ws)
        mock_ws.close.assert_called_once_with(1008, "Missing adapter_id")

    async def test_handle_connection_handshake_timeout(self) -> None:
        config = GatewayConfig()
        queue = MessageQueue()
        server = GatewayServer(config, queue)

        mock_ws = AsyncMock()

        async def slow_recv() -> str:
            await asyncio.sleep(20)
            return "{}"

        mock_ws.recv = slow_recv

        # Patch wait_for timeout to be very short
        with patch("cordbeat.gateway.asyncio.wait_for", side_effect=TimeoutError):
            await server._handle_connection(mock_ws)
        # Should not crash

    async def test_handle_connection_invalid_message(self) -> None:
        config = GatewayConfig()
        queue = MessageQueue()
        server = GatewayServer(config, queue)

        mock_ws = AsyncMock()
        handshake = json.dumps({"adapter_id": "test_adapter"})
        invalid_msg = "not valid json"
        mock_ws.recv = AsyncMock(return_value=handshake)
        mock_ws.__aiter__ = lambda self: _AsyncIter([invalid_msg])
        mock_ws.send = AsyncMock()

        await server._handle_connection(mock_ws)
        # Error response should have been sent (after ACK)
        calls = mock_ws.send.call_args_list
        assert len(calls) >= 2
        error_payload = json.loads(calls[-1][0][0])
        assert error_payload["type"] == "error"

    async def test_queue_receives_valid_message(self) -> None:
        config = GatewayConfig()
        queue = MessageQueue()
        server = GatewayServer(config, queue)

        mock_ws = AsyncMock()
        handshake = json.dumps({"adapter_id": "test_adapter"})
        msg_data = json.dumps(
            {
                "type": "message",
                "adapter_id": "test_adapter",
                "platform_user_id": "u1",
                "content": "hello world",
            }
        )
        mock_ws.recv = AsyncMock(return_value=handshake)
        mock_ws.__aiter__ = lambda self: _AsyncIter([msg_data])
        mock_ws.send = AsyncMock()

        await server._handle_connection(mock_ws)
        assert not queue._queue.empty()
        queued_msg = await queue._queue.get()
        assert queued_msg.content == "hello world"


class _ConcreteConnection(RetryableConnection):
    """Concrete subclass for testing the ABC."""

    async def _dispatch_core_message(self, platform_user_id: str, content: str) -> None:
        raise NotImplementedError


class TestRetryableConnection:
    async def test_dispatch_core_message_raises(self) -> None:
        """Base _dispatch_core_message raises NotImplementedError."""
        conn = _ConcreteConnection()
        try:
            await conn._dispatch_core_message("u1", "hello")
            raise AssertionError("Should have raised")  # noqa: TRY301
        except NotImplementedError:
            pass

    async def test_listen_core_dispatches_message(self) -> None:
        conn = _ConcreteConnection()
        dispatched: list[tuple[str, str]] = []

        async def fake_dispatch(uid: str, content: str) -> None:
            dispatched.append((uid, content))

        conn._dispatch_core_message = fake_dispatch  # type: ignore[assignment]
        msg = json.dumps(
            {
                "type": "message",
                "platform_user_id": "u1",
                "content": "hi",
            }
        )

        # Simulate a WebSocket that yields one message then closes
        mock_ws = AsyncMock()
        mock_ws.__aiter__ = lambda self: _AsyncIter([msg])
        conn._ws = mock_ws

        await conn._listen_core()
        assert dispatched == [("u1", "hi")]

    async def test_listen_core_ignores_unknown_types(self) -> None:
        conn = _ConcreteConnection()
        dispatched: list[tuple[str, str]] = []

        async def fake_dispatch(uid: str, content: str) -> None:
            dispatched.append((uid, content))

        conn._dispatch_core_message = fake_dispatch  # type: ignore[assignment]
        msg = json.dumps(
            {
                "type": "unknown_type",
                "platform_user_id": "u1",
                "content": "hi",
            }
        )
        mock_ws = AsyncMock()
        mock_ws.__aiter__ = lambda self: _AsyncIter([msg])
        conn._ws = mock_ws

        await conn._listen_core()
        assert dispatched == []
