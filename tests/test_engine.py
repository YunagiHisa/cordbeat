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
        if isinstance(prompt, str) and "extract memory" in prompt.lower():
            return (
                '{"topic": "greeting", "emotional_tone": "neutral",'
                ' "facts": [], "episode_summary": ""}'
            )
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
        # Called three times: response, emotion inference, memory extraction
        assert mock_ai.generate.await_count == 3

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
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "", "emotional_tone": "",'
                    ' "facts": [], "episode_summary": ""}'
                )
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

        # The main generate call for the second message
        # Index 0: msg1 response, 1: msg1 emotion, 2: msg1 extraction,
        # 3: msg2 response
        main_call = mock_ai.generate.call_args_list[3]
        prompt_arg = main_call[1].get(
            "prompt",
            main_call[0][0] if main_call[0] else "",
        )
        assert "My name is Alice" in prompt_arg

    async def test_high_intensity_creates_flashbulb(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """High emotion intensity (>=0.8) should create a flashbulb memory."""
        ai = AsyncMock()

        async def _high_emotion(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.9}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "marriage", "emotional_tone": "ecstatic",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "That's wonderful!"

        ai.generate = AsyncMock(side_effect=_high_emotion)
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
            content="I just got married!",
        )
        await eng.handle_message(msg)

        # Emotion should be updated
        assert soul.emotion.primary == Emotion.JOY

        # Flashbulb memory should have been created
        user_id = await memory.resolve_user("test", "user1")
        assert user_id is not None
        results = await memory.search_episodic(user_id, "married")
        assert len(results) >= 1
        assert results[0]["metadata"]["flashbulb"] == "True"

    async def test_low_intensity_no_flashbulb(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
    ) -> None:
        """Below threshold intensity should NOT create flashbulb."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="The weather is nice.",
        )
        await engine.handle_message(msg)
        user_id = await memory.resolve_user("test", "user1")
        assert user_id is not None
        results = await memory.search_episodic(user_id, "weather")
        # No flashbulb created (intensity 0.7 < 0.8)
        assert len(results) == 0

    async def test_handle_message_calls_memory_extraction(
        self,
        engine: CoreEngine,
        mock_ai: AsyncMock,
    ) -> None:
        """After a message, memory extraction should be called."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hi!",
        )
        await engine.handle_message(msg)
        # 3 calls: main generate, emotion inference, memory extraction
        assert mock_ai.generate.await_count == 3

    async def test_memory_extraction_stores_facts(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Extracted facts should be stored as semantic memories."""
        ai = AsyncMock()

        async def _generate(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "curiosity", "intensity": 0.5}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "programming", "emotional_tone": "curious",'
                    ' "facts": ["User enjoys Python programming"],'
                    ' "episode_summary": ""}'
                )
            return "That's cool!"

        ai.generate = AsyncMock(side_effect=_generate)
        eng = CoreEngine(
            ai=ai, soul=soul, memory=memory, skills=skills, gateway=mock_gateway
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="I love Python!",
        )
        await eng.handle_message(msg)

        user_id = await memory.resolve_user("test", "user1")
        assert user_id is not None
        results = await memory.search_semantic(user_id, "Python")
        assert len(results) >= 1
        assert "Python" in results[0]["content"]

    async def test_memory_extraction_stores_episode(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Notable episodes should be stored as episodic memories."""
        ai = AsyncMock()

        async def _generate(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "warmth", "intensity": 0.6}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "career", "emotional_tone": "excited",'
                    ' "facts": [],'
                    ' "episode_summary": '
                    '"User shared they got a new job as a developer"}'
                )
            return "Congratulations!"

        ai.generate = AsyncMock(side_effect=_generate)
        eng = CoreEngine(
            ai=ai, soul=soul, memory=memory, skills=skills, gateway=mock_gateway
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="I got a new job!",
        )
        await eng.handle_message(msg)

        user_id = await memory.resolve_user("test", "user1")
        assert user_id is not None
        results = await memory.search_episodic(user_id, "new job")
        assert len(results) >= 1
        assert "developer" in results[0]["content"]

    async def test_memory_extraction_updates_user_summary(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Extracted topic and tone should update UserSummary."""
        ai = AsyncMock()

        async def _generate(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "calm", "intensity": 0.5}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "cooking recipes", "emotional_tone": "happy",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "Nice recipe!"

        ai.generate = AsyncMock(side_effect=_generate)
        eng = CoreEngine(
            ai=ai, soul=soul, memory=memory, skills=skills, gateway=mock_gateway
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Check out this recipe!",
        )
        await eng.handle_message(msg)

        user_id = await memory.resolve_user("test", "user1")
        assert user_id is not None
        summaries = await memory.get_all_user_summaries()
        user = next(u for u in summaries if u.user_id == user_id)
        assert user.last_topic == "cooking recipes"
        assert user.emotional_tone == "happy"

    async def test_memory_extraction_failure_does_not_crash(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Bad extraction JSON should not crash message handling."""
        ai = AsyncMock()

        async def _bad_extract(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "calm", "intensity": 0.5}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return "not valid json!"
            return "Hello!"

        ai.generate = AsyncMock(side_effect=_bad_extract)
        eng = CoreEngine(
            ai=ai, soul=soul, memory=memory, skills=skills, gateway=mock_gateway
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Test",
        )
        await eng.handle_message(msg)
        mock_gateway.send_to_adapter.assert_awaited_once()
