"""Tests for core engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cordbeat.config import MemoryConfig
from cordbeat.engine import CoreEngine
from cordbeat.memory import MemoryStore
from cordbeat.models import GatewayMessage, MessageType
from cordbeat.skills import SkillRegistry
from cordbeat.soul import Soul


@pytest.fixture
def soul(tmp_path: Path) -> Soul:
    return Soul(tmp_path / "soul")


@pytest.fixture
async def memory(tmp_path: Path) -> MemoryStore:
    config = MemoryConfig(
        sqlite_path=str(tmp_path / "test.db"),
        chroma_path=str(tmp_path / "chroma"),
    )
    store = MemoryStore(config)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def mock_ai() -> AsyncMock:
    ai = AsyncMock()
    ai.generate = AsyncMock(return_value="Hello there!")
    return ai


@pytest.fixture
def mock_gateway() -> AsyncMock:
    gw = AsyncMock()
    gw.send_to_adapter = AsyncMock()
    return gw


@pytest.fixture
def skills(tmp_path: Path) -> SkillRegistry:
    return SkillRegistry(tmp_path / "skills")


@pytest.fixture
def engine(
    mock_ai: AsyncMock,
    soul: Soul,
    memory: MemoryStore,
    skills: SkillRegistry,
    mock_gateway: AsyncMock,
) -> CoreEngine:
    return CoreEngine(
        ai=mock_ai,
        soul=soul,
        memory=memory,
        skills=skills,
        gateway=mock_gateway,
    )


class TestCoreEngine:
    async def test_handle_message_calls_ai(
        self,
        engine: CoreEngine,
        mock_ai: AsyncMock,
    ) -> None:
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hi!",
        )
        await engine.handle_message(msg)
        mock_ai.generate.assert_awaited_once()

    async def test_handle_message_sends_reply(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hi!",
        )
        await engine.handle_message(msg)
        mock_gateway.send_to_adapter.assert_awaited_once()
        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert reply.content == "Hello there!"
        assert reply.adapter_id == "test"

    async def test_handle_message_creates_user(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
    ) -> None:
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="new_user",
            content="First message",
        )
        await engine.handle_message(msg)
        # User should now exist
        user_id = await memory.resolve_user("test", "new_user")
        assert user_id is not None

    async def test_handle_message_ignores_non_message_types(
        self,
        engine: CoreEngine,
        mock_ai: AsyncMock,
    ) -> None:
        msg = GatewayMessage(
            type=MessageType.ACK,
            adapter_id="test",
            platform_user_id="user1",
            content="ack",
        )
        await engine.handle_message(msg)
        mock_ai.generate.assert_not_awaited()

    async def test_handle_message_ai_failure_returns_fallback(
        self,
        engine: CoreEngine,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        mock_ai.generate = AsyncMock(side_effect=RuntimeError("AI down"))
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hi!",
        )
        await engine.handle_message(msg)
        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert reply.content == "..."

    async def test_handle_message_includes_soul_in_prompt(
        self,
        engine: CoreEngine,
        mock_ai: AsyncMock,
    ) -> None:
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="What's your name?",
        )
        await engine.handle_message(msg)
        system_arg = mock_ai.generate.call_args[1]["system"]
        assert "CordBeat" in system_arg
        assert "Never harm a user" in system_arg
