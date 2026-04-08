"""Tests for core engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cordbeat.config import MemoryConfig
from cordbeat.engine import CoreEngine
from cordbeat.memory import MemoryStore
from cordbeat.models import Emotion, GatewayMessage, MessageType
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

    async def _generate(**kwargs: object) -> str:
        prompt = kwargs.get("prompt", "")
        if isinstance(prompt, str) and "what emotion" in prompt.lower():
            return '{"emotion": "joy", "intensity": 0.7}'
        return "Hello there!"

    ai.generate = AsyncMock(side_effect=_generate)
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
        # Called twice: once for response, once for emotion inference
        assert mock_ai.generate.await_count == 2

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

    async def test_handle_message_ai_failure_sends_error(
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
        assert reply.type == MessageType.ERROR
        assert "failed" in reply.content.lower()

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
        # First call is the main generate, not the emotion inference
        system_arg = mock_ai.generate.call_args_list[0][1]["system"]
        assert "CordBeat" in system_arg
        assert "Never harm a user" in system_arg

    async def test_handle_message_stores_conversation(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
    ) -> None:
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Remember this!",
        )
        await engine.handle_message(msg)
        user_id = await memory.resolve_user("test", "user1")
        assert user_id is not None
        msgs = await memory.get_recent_messages(user_id)
        assert len(msgs) == 2

    async def test_handle_message_updates_emotion(
        self,
        engine: CoreEngine,
        soul: Soul,
    ) -> None:
        """After a message, emotion should be updated via AI inference."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Great news!",
        )
        await engine.handle_message(msg)
        assert soul.emotion.primary == Emotion.JOY
        assert soul.emotion.primary_intensity == pytest.approx(0.7)

    async def test_emotion_inference_failure_does_not_crash(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """If emotion inference returns bad JSON, message still works."""
        ai = AsyncMock()

        async def _bad_infer(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return "not valid json!"
            return "Hello there!"

        ai.generate = AsyncMock(side_effect=_bad_infer)
        eng = CoreEngine(
            ai=ai,
            soul=soul,
            memory=memory,
            skills=skills,
            gateway=mock_gateway,
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hi!",
        )
        await eng.handle_message(msg)
        # Should not crash — emotion stays at default
        assert soul.emotion.primary == Emotion.CALM
        mock_gateway.send_to_adapter.assert_awaited_once()

    async def test_handle_message_includes_history_in_prompt(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        # Send first message
        msg1 = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="My name is Alice",
        )
        await engine.handle_message(msg1)

        # Send second message — history should be included
        msg2 = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="What did I just say?",
        )
        await engine.handle_message(msg2)

        # The main generate call for the second message (index 2)
        # Index 0: msg1 generate, 1: msg1 emotion, 2: msg2 generate
        main_call = mock_ai.generate.call_args_list[2]
        prompt_arg = main_call[1].get(
            "prompt",
            main_call[0][0] if main_call[0] else "",
        )
        assert "My name is Alice" in prompt_arg
