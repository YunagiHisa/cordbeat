"""Tests for gateway message queue."""

from __future__ import annotations

import asyncio

from cordbeat.gateway import MessageQueue
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
