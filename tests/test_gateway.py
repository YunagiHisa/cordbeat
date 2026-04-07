"""Tests for gateway message queue and server."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

from cordbeat.config import GatewayConfig
from cordbeat.gateway import GatewayServer, MessageQueue
from cordbeat.models import GatewayMessage, MessageType


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
