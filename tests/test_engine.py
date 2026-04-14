"""Tests for core engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cordbeat.config import MemoryConfig
from cordbeat.engine import CoreEngine
from cordbeat.memory import MemoryStore
from cordbeat.models import (
    Emotion,
    GatewayMessage,
    MemoryEntry,
    MemoryLayer,
    MessageType,
    ProposalStatus,
)
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
        if isinstance(prompt, str) and "recall keywords" in prompt.lower():
            return '{"keywords": []}'
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
        # Called four times: recall keywords, response, emotion, extraction
        assert mock_ai.generate.await_count == 4

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
        # First call is recall keywords, second is the main generate
        system_arg = mock_ai.generate.call_args_list[1][1]["system"]
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
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": []}'
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
        # Index 0: msg1 recall, 1: msg1 response, 2: msg1 emotion,
        # 3: msg1 extraction, 4: msg2 recall, 5: msg2 response
        main_call = mock_ai.generate.call_args_list[5]
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
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": []}'
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
        # 4 calls: recall keywords, main generate, emotion, extraction
        assert mock_ai.generate.await_count == 4

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
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": []}'
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
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": []}'
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
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": []}'
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
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": []}'
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


class TestAccountLinking:
    """Tests for cross-platform account linking via tokens."""

    async def test_link_request_generates_token(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """LINK_REQUEST should return a token in the reply."""
        msg = GatewayMessage(
            type=MessageType.LINK_REQUEST,
            adapter_id="telegram",
            platform_user_id="tg_user_1",
            content="",
        )
        await engine.handle_message(msg)
        mock_gateway.send_to_adapter.assert_awaited_once()
        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert reply.type == MessageType.ACK
        assert "Link token generated:" in reply.content

    async def test_link_confirm_success(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Valid token from an existing user links the new platform."""
        # Set up existing user on Discord
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_alice")

        # Generate token from Telegram (new platform)
        token = await memory.store_link_token("telegram", "tg_alice")

        # Confirm from Discord (existing platform)
        msg = GatewayMessage(
            type=MessageType.LINK_CONFIRM,
            adapter_id="discord",
            platform_user_id="discord_alice",
            content=token,
        )
        await engine.handle_message(msg)

        # Telegram should now be linked to u1
        user_id = await memory.resolve_user("telegram", "tg_alice")
        assert user_id == "u1"

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert reply.type == MessageType.ACK
        assert "linked" in reply.content.lower()

    async def test_link_confirm_invalid_token(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Invalid token returns an error."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_alice")

        msg = GatewayMessage(
            type=MessageType.LINK_CONFIRM,
            adapter_id="discord",
            platform_user_id="discord_alice",
            content="bogus_token",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert reply.type == MessageType.ERROR
        assert "invalid" in reply.content.lower()

    async def test_link_confirm_no_existing_account(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Confirm from unknown user returns an error."""
        token = await memory.store_link_token("telegram", "tg_alice")

        msg = GatewayMessage(
            type=MessageType.LINK_CONFIRM,
            adapter_id="discord",
            platform_user_id="unknown_user",
            content=token,
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert reply.type == MessageType.ERROR
        assert "existing account" in reply.content.lower()

    async def test_link_confirm_token_used_twice(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """A token can only be used once."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_alice")

        token = await memory.store_link_token("telegram", "tg_alice")

        # First use — success
        msg1 = GatewayMessage(
            type=MessageType.LINK_CONFIRM,
            adapter_id="discord",
            platform_user_id="discord_alice",
            content=token,
        )
        await engine.handle_message(msg1)
        reply1 = mock_gateway.send_to_adapter.call_args[0][1]
        assert reply1.type == MessageType.ACK

        # Second use — fail
        msg2 = GatewayMessage(
            type=MessageType.LINK_CONFIRM,
            adapter_id="discord",
            platform_user_id="discord_alice",
            content=token,
        )
        await engine.handle_message(msg2)
        reply2 = mock_gateway.send_to_adapter.call_args[0][1]
        assert reply2.type == MessageType.ERROR

    async def test_link_request_does_not_trigger_ai(
        self,
        engine: CoreEngine,
        mock_ai: AsyncMock,
    ) -> None:
        """LINK_REQUEST should not call the AI backend."""
        msg = GatewayMessage(
            type=MessageType.LINK_REQUEST,
            adapter_id="telegram",
            platform_user_id="tg_user_1",
            content="",
        )
        await engine.handle_message(msg)
        mock_ai.generate.assert_not_awaited()

    async def test_link_confirm_does_not_trigger_ai(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        """LINK_CONFIRM should not call the AI backend."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_alice")
        token = await memory.store_link_token("telegram", "tg_alice")

        msg = GatewayMessage(
            type=MessageType.LINK_CONFIRM,
            adapter_id="discord",
            platform_user_id="discord_alice",
            content=token,
        )
        await engine.handle_message(msg)
        mock_ai.generate.assert_not_awaited()


class TestRecallModel:
    """Tests for the multi-phase recall model (Phase2 + Phase3)."""

    async def test_recall_keywords_expand_memory_search(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Phase2: AI-extracted keywords should find additional memories."""
        ai = AsyncMock()

        async def _generate(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": ["Python", "Linux"]}'
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "calm", "intensity": 0.5}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "", "emotional_tone": "",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "I recall that!"

        ai.generate = AsyncMock(side_effect=_generate)

        # Pre-populate a semantic memory that won't match "How are you?"
        # but will match the recall keyword "Python"
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        entry = MemoryEntry(
            id="mem_python",
            user_id="u1",
            layer=MemoryLayer.SEMANTIC,
            content="User enjoys Python programming",
        )
        await memory.add_semantic_memory(entry)

        eng = CoreEngine(
            ai=ai, soul=soul, memory=memory, skills=skills, gateway=mock_gateway
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="How are you?",
        )
        await eng.handle_message(msg)

        # The prompt should include the recalled memory
        main_call = ai.generate.call_args_list[1]
        prompt_arg = main_call[1].get("prompt", "")
        assert "Python" in prompt_arg

    async def test_recall_keywords_deduplicate(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Phase2: Duplicate memories should not appear twice."""
        ai = AsyncMock()

        async def _generate(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": ["cats"]}'
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "calm", "intensity": 0.5}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "", "emotional_tone": "",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "Meow!"

        ai.generate = AsyncMock(side_effect=_generate)

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        entry = MemoryEntry(
            id="mem_cats",
            user_id="u1",
            layer=MemoryLayer.SEMANTIC,
            content="User loves cats",
        )
        await memory.add_semantic_memory(entry)

        eng = CoreEngine(
            ai=ai, soul=soul, memory=memory, skills=skills, gateway=mock_gateway
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="I love cats!",
        )
        await eng.handle_message(msg)

        # "cats" should appear only once in the prompt
        main_call = ai.generate.call_args_list[1]
        prompt_arg = main_call[1].get("prompt", "")
        assert prompt_arg.count("User loves cats") == 1

    async def test_recall_keyword_failure_does_not_crash(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Phase2: AI failure in keyword extraction should not crash."""
        ai = AsyncMock()

        async def _generate(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return "not valid json!"
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "calm", "intensity": 0.5}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "", "emotional_tone": "",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "Hello!"

        ai.generate = AsyncMock(side_effect=_generate)
        eng = CoreEngine(
            ai=ai, soul=soul, memory=memory, skills=skills, gateway=mock_gateway
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hi!",
        )
        await eng.handle_message(msg)
        mock_gateway.send_to_adapter.assert_awaited_once()

    async def test_emotion_recall_finds_matching_memories(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Phase3: Current emotion should trigger emotion-tagged recall."""
        ai = AsyncMock()

        async def _generate(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.7}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "", "emotional_tone": "",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "I remember that happy time!"

        ai.generate = AsyncMock(side_effect=_generate)

        # Set emotion to joy before the message
        soul.update_emotion(Emotion.JOY, 0.8)

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        # Store episodic memory with joy emotion tag
        entry = MemoryEntry(
            id="mem_happy",
            user_id="u1",
            layer=MemoryLayer.EPISODIC,
            content="Had a wonderful birthday party together",
            metadata={"emotional_tone": "joy"},
        )
        await memory.add_episodic_memory(entry)

        eng = CoreEngine(
            ai=ai, soul=soul, memory=memory, skills=skills, gateway=mock_gateway
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="I feel great today!",
        )
        await eng.handle_message(msg)

        # The emotion-matched memory should appear in prompt
        main_call = ai.generate.call_args_list[1]
        prompt_arg = main_call[1].get("prompt", "")
        assert "birthday party" in prompt_arg

    async def test_calm_emotion_skips_emotion_recall(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        soul: Soul,
        mock_ai: AsyncMock,
    ) -> None:
        """Phase3: Calm emotion should NOT trigger emotion recall."""
        assert soul.emotion.primary == Emotion.CALM

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        entry = MemoryEntry(
            id="mem_calm",
            user_id="u1",
            layer=MemoryLayer.EPISODIC,
            content="A calm afternoon reading",
            metadata={"emotional_tone": "calm"},
        )
        await memory.add_episodic_memory(entry)

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hello",
        )
        await engine.handle_message(msg)

    async def test_phase4_recall_hints_included_in_prompt(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Phase4: Precomputed recall hints appear in the prompt."""
        ai = AsyncMock()

        async def _generate(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "calm", "intensity": 0.5}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "", "emotional_tone": "",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "I remember!"

        ai.generate = AsyncMock(side_effect=_generate)

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        # Store a precomputed recall hint
        await memory.store_recall_hint(
            user_id="u1",
            hint_type="temporal",
            content="7 days ago Alice talked about: OSS design",
            metadata={"original_date": "2025-01-01"},
        )

        eng = CoreEngine(
            ai=ai, soul=soul, memory=memory, skills=skills, gateway=mock_gateway
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="What were we talking about?",
        )
        await eng.handle_message(msg)

        # The recall hint should appear in the prompt
        main_call = ai.generate.call_args_list[1]
        prompt_arg = main_call[1].get("prompt", "")
        assert "OSS design" in prompt_arg

    async def test_phase4_recall_hints_failure_graceful(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Phase4: If recall hints lookup fails, response still works."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        # Monkey-patch to raise
        original = memory.get_recall_hints
        memory.get_recall_hints = AsyncMock(side_effect=RuntimeError("DB error"))

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hello",
        )
        await engine.handle_message(msg)
        mock_ai.generate.assert_called()

        # Restore
        memory.get_recall_hints = original

    async def test_phase4_chain_recall_adds_linked_memories(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
    ) -> None:
        """Phase4: Chain recall should add pre-linked memories to context."""
        ai = AsyncMock()

        async def _generate(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "calm", "intensity": 0.5}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "", "emotional_tone": "",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "I remember the chain!"

        ai.generate = AsyncMock(side_effect=_generate)

        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        # Store a semantic memory and a pre-linked chain
        entry = MemoryEntry(
            id="mem_cats",
            user_id="u1",
            layer=MemoryLayer.SEMANTIC,
            content="User loves cats",
        )
        await memory.add_semantic_memory(entry)

        # Pre-link: when "mem_cats" is recalled, also surface this
        await memory.store_chain_link(
            user_id="u1",
            source_memory_id="mem_cats",
            linked_content="User adopted a kitten last month",
            linked_memory_id="mem_kitten",
        )

        eng = CoreEngine(
            ai=ai, soul=soul, memory=memory, skills=skills, gateway=mock_gateway
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="I love cats!",
        )
        await eng.handle_message(msg)

        # The chain-linked memory should appear in the prompt
        main_call = ai.generate.call_args_list[1]
        prompt_arg = main_call[1].get("prompt", "")
        assert "kitten" in prompt_arg

    async def test_phase4_chain_recall_failure_graceful(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Phase4: If chain recall fails, response still works."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        original = memory.get_chain_links
        memory.get_chain_links = AsyncMock(side_effect=RuntimeError("DB error"))

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hello",
        )
        await engine.handle_message(msg)
        mock_ai.generate.assert_called()

        memory.get_chain_links = original


class TestProposalCommands:
    """Tests for /approve, /reject, /proposals slash commands."""

    async def _create_proposal(
        self,
        memory: MemoryStore,
        user_id: str,
        content: str = "Test proposal",
    ) -> str:
        """Helper to create a pending proposal."""
        return await memory.add_certain_record(
            user_id=user_id,
            content=content,
            record_type="proposal",
            metadata={
                "status": ProposalStatus.PENDING,
                "proposal_type": "general",
            },
        )

    async def test_approve_proposal(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Users can approve their own pending proposals."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        pid = await self._create_proposal(memory, "u1", "Add emoji reactions")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content=f"/approve {pid}",
        )
        await engine.handle_message(msg)

        # Verify proposal status changed
        proposal = await memory.get_proposal(pid)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.APPROVED

        # Verify reply was sent
        reply_call = mock_gateway.send_to_adapter.call_args
        assert "approved" in reply_call[0][1].content.lower()

    async def test_reject_proposal(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Users can reject their own pending proposals."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        pid = await self._create_proposal(memory, "u1", "Change personality")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content=f"/reject {pid}",
        )
        await engine.handle_message(msg)

        proposal = await memory.get_proposal(pid)
        assert proposal is not None
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.REJECTED

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "rejected" in reply_call[0][1].content.lower()

    async def test_approve_nonexistent_proposal(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Approving a nonexistent proposal returns error."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/approve fake-id-123",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "not found" in reply_call[0][1].content.lower()

    async def test_approve_other_users_proposal(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Users cannot approve proposals belonging to other users."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        await memory.get_or_create_user("u2", "Bob")
        await memory.link_platform("u2", "test", "user2")

        pid = await self._create_proposal(memory, "u2", "Bob's proposal")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content=f"/approve {pid}",
        )
        await engine.handle_message(msg)

        # Should not approve — ownership check fails
        proposal = await memory.get_proposal(pid)
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.PENDING

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "not found" in reply_call[0][1].content.lower()

    async def test_approve_already_approved(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Approving an already approved proposal returns error."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        pid = await self._create_proposal(memory, "u1")
        await memory.update_proposal_status(pid, ProposalStatus.APPROVED)

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content=f"/approve {pid}",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "cannot approve" in reply_call[0][1].content.lower()

    async def test_approve_no_id(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Approve without proposal ID shows usage."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/approve",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "usage" in reply_call[0][1].content.lower()

    async def test_list_proposals(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Users can list their pending proposals."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        await self._create_proposal(memory, "u1", "Proposal A")
        await self._create_proposal(memory, "u1", "Proposal B")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/proposals",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        response = reply_call[0][1].content
        assert "Proposal A" in response
        assert "Proposal B" in response

    async def test_list_proposals_empty(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Listing with no pending proposals returns none message."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/proposals",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "no pending" in reply_call[0][1].content.lower()

    async def test_unknown_command_falls_through(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        """Unknown slash commands are processed as normal messages."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/unknown_command",
        )
        await engine.handle_message(msg)
        # Should fall through to normal AI processing
        mock_ai.generate.assert_called()

    async def test_reject_no_id(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Reject without proposal ID shows usage."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/reject",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "usage" in reply_call[0][1].content.lower()

    async def test_reject_nonexistent_proposal(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Rejecting a nonexistent proposal returns error."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/reject fake-id-123",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "not found" in reply_call[0][1].content.lower()

    async def test_reject_other_users_proposal(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Users cannot reject proposals belonging to other users."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        await memory.get_or_create_user("u2", "Bob")
        await memory.link_platform("u2", "test", "user2")

        pid = await self._create_proposal(memory, "u2", "Bob's idea")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content=f"/reject {pid}",
        )
        await engine.handle_message(msg)

        # Proposal should remain pending
        proposal = await memory.get_proposal(pid)
        meta = json.loads(proposal["metadata"])
        assert meta["status"] == ProposalStatus.PENDING

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "not found" in reply_call[0][1].content.lower()

    async def test_reject_already_rejected(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Rejecting an already rejected proposal returns error."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        pid = await self._create_proposal(memory, "u1")
        await memory.update_proposal_status(pid, ProposalStatus.REJECTED)

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content=f"/reject {pid}",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "cannot reject" in reply_call[0][1].content.lower()

    async def test_approve_unknown_user(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """Approving from an unknown user returns error."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="nobody",
            content="/approve some-id",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "not found" in reply_call[0][1].content.lower()

    async def test_reject_unknown_user(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """Rejecting from an unknown user returns error."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="nobody",
            content="/reject some-id",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "not found" in reply_call[0][1].content.lower()

    async def test_proposals_unknown_user(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """Listing proposals from unknown user returns error."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="nobody",
            content="/proposals",
        )
        await engine.handle_message(msg)

        reply_call = mock_gateway.send_to_adapter.call_args
        assert "not found" in reply_call[0][1].content.lower()


class TestLinkCommands:
    """Tests for /link and /unlink text commands."""

    async def test_link_command_generates_token(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """/link should generate a link token and reply."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="/link",
        )
        await engine.handle_message(msg)

        mock_gateway.send_to_adapter.assert_awaited()
        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "Link token generated:" in reply.content

    async def test_link_command_does_not_trigger_ai(
        self,
        engine: CoreEngine,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        """/link should not call the AI backend."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="/link",
        )
        await engine.handle_message(msg)
        mock_ai.generate.assert_not_awaited()

    async def test_link_command_creates_audit_record(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """/link should create an audit record."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="/link",
        )
        await engine.handle_message(msg)

        records = await memory.get_certain_records(
            "__system__", record_type="link_audit"
        )
        assert len(records) >= 1
        assert records[-1]["content"] == "Token issued on discord"

    async def test_unlink_command_removes_platform(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """/unlink should remove a linked platform."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "user1")
        await memory.link_platform("u1", "telegram", "tg_user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="/unlink telegram",
        )
        await engine.handle_message(msg)

        # Verify telegram is unlinked
        user_id = await memory.resolve_user("telegram", "tg_user1")
        assert user_id is None

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "unlinked" in reply.content.lower()

    async def test_unlink_command_no_platform_arg(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """/unlink without a platform name shows usage."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="/unlink",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "usage" in reply.content.lower()

    async def test_unlink_command_unknown_user(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """/unlink from an unknown user returns error."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="nobody",
            content="/unlink telegram",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "not found" in reply.content.lower()

    async def test_unlink_command_last_platform_blocked(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """/unlink should block if only one platform is linked."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="/unlink discord",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "at least one" in reply.content.lower()

    async def test_unlink_command_current_platform_blocked(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """/unlink cannot remove the platform being used."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "user1")
        await memory.link_platform("u1", "telegram", "tg_user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="/unlink discord",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "currently using" in reply.content.lower()

        # Discord should still be linked
        user_id = await memory.resolve_user("discord", "user1")
        assert user_id == "u1"

    async def test_unlink_command_nonexistent_platform(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """/unlink with a platform that isn't linked returns error."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "user1")
        await memory.link_platform("u1", "telegram", "tg_user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="/unlink slack",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "not linked" in reply.content.lower()

    async def test_unlink_command_creates_audit_record(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """/unlink should record an audit entry."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "user1")
        await memory.link_platform("u1", "telegram", "tg_user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="/unlink telegram",
        )
        await engine.handle_message(msg)

        records = await memory.get_certain_records("u1", record_type="link_audit")
        assert len(records) >= 1
        assert "unlink" in records[-1]["content"].lower()

    async def test_link_confirm_creates_audit_record(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """LINK_CONFIRM should also create an audit record."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "discord", "discord_alice")
        token = await memory.store_link_token("telegram", "tg_alice")

        msg = GatewayMessage(
            type=MessageType.LINK_CONFIRM,
            adapter_id="discord",
            platform_user_id="discord_alice",
            content=token,
        )
        await engine.handle_message(msg)

        records = await memory.get_certain_records("u1", record_type="link_audit")
        assert len(records) >= 1
        assert "link_confirm" in json.loads(records[-1]["metadata"]).get("action", "")


class TestSoulCommands:
    """Tests for /name, /quiet, /prefer text commands."""

    async def test_name_command_shows_current(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """/name without args shows current name."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/name",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "current name" in reply.content.lower()

    async def test_name_command_updates_name(
        self,
        engine: CoreEngine,
        soul: Soul,
        mock_gateway: AsyncMock,
    ) -> None:
        """/name <new> updates the SOUL name."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/name Athena",
        )
        await engine.handle_message(msg)

        assert soul.name == "Athena"
        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "athena" in reply.content.lower()

    async def test_quiet_command_shows_current(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """/quiet without args shows current quiet hours."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/quiet",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "current quiet hours" in reply.content.lower()

    async def test_quiet_command_updates_hours(
        self,
        engine: CoreEngine,
        soul: Soul,
        mock_gateway: AsyncMock,
    ) -> None:
        """/quiet <start> <end> updates quiet hours."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/quiet 23:00 08:00",
        )
        await engine.handle_message(msg)

        assert soul.quiet_hours == ("23:00", "08:00")
        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "23:00" in reply.content

    async def test_quiet_command_invalid_format(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """/quiet with bad format shows error."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/quiet abc xyz",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "invalid" in reply.content.lower()

    async def test_quiet_command_missing_end(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """/quiet with only start shows usage."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/quiet 01:00",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "usage" in reply.content.lower()

    async def test_prefer_command_shows_current(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """/prefer without args shows current preference."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/prefer",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "not set" in reply.content.lower()

    async def test_prefer_command_sets_platform(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """/prefer <platform> sets the preferred platform."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        await memory.link_platform("u1", "discord", "disc_alice")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/prefer discord",
        )
        await engine.handle_message(msg)

        user = await memory.get_or_create_user("u1", "Alice")
        assert user.preferred_platform == "discord"

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "discord" in reply.content.lower()

    async def test_prefer_command_unlinked_platform(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """/prefer with an unlinked platform returns error."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/prefer slack",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "not linked" in reply.content.lower()

    async def test_prefer_command_clear(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """/prefer clear removes the preferred platform."""
        await memory.get_or_create_user("u1", "Alice")
        await memory.link_platform("u1", "test", "user1")
        await memory.link_platform("u1", "discord", "disc_alice")

        # Set preference first
        user = await memory.get_or_create_user("u1", "Alice")
        user.preferred_platform = "discord"
        await memory.update_user_summary(user)

        # Clear it
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="/prefer clear",
        )
        await engine.handle_message(msg)

        user = await memory.get_or_create_user("u1", "Alice")
        assert user.preferred_platform is None

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "cleared" in reply.content.lower()

    async def test_prefer_command_unknown_user(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        """/prefer from unknown user returns error."""
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="nobody",
            content="/prefer discord",
        )
        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "not found" in reply.content.lower()
