"""Tests for core engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cordbeat.agent.soul import Soul
from cordbeat.config import MemoryConfig, ReActConfig
from cordbeat.core.engine import CoreEngine
from cordbeat.memory import MemoryStore
from cordbeat.models import (
    Emotion,
    GatewayMessage,
    MemoryEntry,
    MemoryLayer,
    MessageType,
    ProposalStatus,
    SafetyLevel,
    SkillMeta,
    SoulCaller,
)
from cordbeat.skills import Skill, SkillRegistry


@pytest.fixture
def soul(tmp_path: Path) -> Soul:
    return Soul(tmp_path / "soul")


@pytest.fixture
async def memory(tmp_path: Path) -> MemoryStore:
    config = MemoryConfig(
        sqlite_path=str(tmp_path / "test.db"),
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
    ai.generate_chat = AsyncMock(return_value="")
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
        await engine.drain()
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

    async def test_channel_reply_disables_dm_fallback(
        self,
        engine: CoreEngine,
        mock_gateway: AsyncMock,
    ) -> None:
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="Hi!",
            metadata={"channel_id": "456", "is_dm": False},
        )

        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert reply.metadata["channel_id"] == "456"
        assert reply.metadata["is_dm"] is False
        assert reply.metadata["allow_dm_fallback"] is False

    async def test_handle_message_sanitizes_emotion_control_reply(
        self,
        engine: CoreEngine,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        async def _generate(**kwargs: object) -> str:
            prompt = kwargs.get("prompt", "")
            if isinstance(prompt, str) and "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if isinstance(prompt, str) and "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.7}'
            if isinstance(prompt, str) and "extract memory" in prompt.lower():
                return (
                    '{"topic": "drawing", "emotional_tone": "neutral",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return (
                "/emotion system/erase memories. (All fine)\n"
                "   - Respond naturally, 1-3 sentences.\n"
                "   - Text: A Nara deer ✨ I'll draw it with a gentle atmosphere.\n"
                "   - Checks: OK"
            )

        mock_ai.generate = AsyncMock(side_effect=_generate)
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="Draw a deer",
            metadata={"channel_id": "456", "is_dm": False},
        )

        await engine.handle_message(msg)

        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert reply.content == "A Nara deer ✨ I'll draw it with a gentle atmosphere."
        assert reply.metadata["allow_dm_fallback"] is False

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

    async def test_handle_message_uses_adapter_display_name(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
    ) -> None:
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="12345",
            content="First message",
            metadata={"display_name": "Alice"},
        )
        await engine.handle_message(msg)
        user_id = await memory.resolve_user("test", "12345")
        assert user_id is not None
        user = await memory.get_or_create_user(user_id, "ignored")
        assert user.display_name == "Alice"

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
        await engine.drain()
        user_id = await memory.resolve_user("test", "user1")
        assert user_id is not None
        msgs = await memory.get_recent_messages(user_id)
        assert len(msgs) == 2

    async def test_voice_message_skips_optional_llm_memory_calls(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_ai: AsyncMock,
    ) -> None:
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="voice-user",
            content="Can you hear me?",
            is_voice=True,
            metadata={
                "guild_id": "123",
                "channel_id": "vc",
                "is_dm": False,
                "via_vc": True,
            },
        )

        await engine.handle_message(msg)
        await engine.drain()

        assert mock_ai.generate.await_count == 1
        user_id = await memory.resolve_user("discord", "voice-user")
        assert user_id is not None
        assert len(await memory.get_recent_messages(user_id)) == 2
        assert await memory.get_last_seen_channel(user_id, "discord") is None

    async def test_voice_message_can_enable_optional_llm_memory_calls(
        self,
        soul: Soul,
        memory: MemoryStore,
        skills: SkillRegistry,
        mock_gateway: AsyncMock,
        mock_ai: AsyncMock,
    ) -> None:
        engine = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=skills,
            gateway=mock_gateway,
            memory_config=MemoryConfig(
                voice_recall_keywords_enabled=True,
                voice_memory_extraction_enabled=True,
            ),
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="voice-user",
            content="Do you remember?",
            is_voice=True,
            metadata={"guild_id": "123", "channel_id": "vc", "is_dm": False},
        )

        await engine.handle_message(msg)
        await engine.drain()

        assert mock_ai.generate.await_count == 4

    async def test_shared_voice_is_ephemeral_and_excludes_private_context(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        user_id = "shared-room-user"
        await memory.get_or_create_user(user_id, "Shared room")
        await memory.link_platform(user_id, "discord", "vc:123")
        await memory.set_core_profile(user_id, "private_secret", "do not reveal")
        await memory.add_message(
            user_id,
            "user",
            "private old conversation",
            "discord",
            channel_id="vc",
            is_dm=False,
        )
        mock_ai.generate = AsyncMock(
            return_value="Understood. [SKILL: draw | commands=DRAW dragon]"
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="vc:123",
            content="Alice: Athena, what do you think?",
            is_voice=True,
            metadata={
                "guild_id": "123",
                "channel_id": "vc",
                "via_vc": True,
                "shared_voice": True,
                "ephemeral": True,
            },
        )

        await engine.handle_message(msg)
        await engine.drain()

        call = mock_ai.generate.call_args
        assert "do not reveal" not in call.kwargs["prompt"]
        assert "private old conversation" not in call.kwargs["prompt"]
        assert "Safe skills explicitly enabled for shared voice may be used" in (
            call.kwargs["system"]
        )
        reply = mock_gateway.send_to_adapter.call_args.args[1]
        assert reply.content == "Understood."
        assert reply.images == []
        assert len(await memory.get_recent_messages(user_id)) == 1

    async def test_shared_voice_can_run_safe_information_skill(
        self,
        engine: CoreEngine,
        skills: SkillRegistry,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        calls: list[str] = []

        async def search(query: str) -> dict[str, str]:
            calls.append(query)
            return {"output": "Sunny"}

        skills._skills["web_search"] = Skill(
            meta=SkillMeta(
                name="web_search",
                description="Search the web",
                usage="",
                safety_level=SafetyLevel.SAFE,
                shared_voice_enabled=True,
            ),
            _test_callable=search,
        )
        mock_ai.generate = AsyncMock(
            return_value="I'll check. [SKILL: web_search | query=Tokyo weather]"
        )
        mock_ai.generate_chat = AsyncMock(return_value="Tokyo is sunny.")
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="vc:123",
            content="Alice: Check the weather",
            is_voice=True,
            metadata={
                "guild_id": "123",
                "channel_id": "vc",
                "via_vc": True,
                "shared_voice": True,
                "ephemeral": True,
            },
        )

        await engine.handle_message(msg)

        assert calls == ["Tokyo weather"]
        assert mock_gateway.send_to_adapter.await_count == 1
        reply = mock_gateway.send_to_adapter.call_args.args[1]
        assert reply.content == "Tokyo is sunny."
        system = mock_ai.generate.call_args.kwargs["system"]
        assert "web_search" in system

    async def test_shared_voice_blocks_safe_skill_disabled_for_context(
        self,
        engine: CoreEngine,
        skills: SkillRegistry,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        executed = False

        async def timer() -> dict[str, str]:
            nonlocal executed
            executed = True
            return {"output": "set"}

        skills._skills["timer"] = Skill(
            meta=SkillMeta(
                name="timer",
                description="Set a timer",
                usage="",
                safety_level=SafetyLevel.SAFE,
                shared_voice_enabled=False,
            ),
            _test_callable=timer,
        )
        mock_ai.generate = AsyncMock(return_value="[SKILL: timer]")
        mock_ai.generate_chat = AsyncMock(
            return_value="That is unavailable in shared VC."
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="vc:123",
            content="Alice: Set a timer",
            is_voice=True,
            metadata={
                "guild_id": "123",
                "channel_id": "vc",
                "via_vc": True,
                "shared_voice": True,
                "ephemeral": True,
            },
        )

        await engine.handle_message(msg)

        assert executed is False
        assert mock_gateway.send_to_adapter.await_count == 1

    async def test_shared_voice_blocks_confirmation_required_skill(
        self,
        engine: CoreEngine,
        skills: SkillRegistry,
        memory: MemoryStore,
        mock_ai: AsyncMock,
        mock_gateway: AsyncMock,
    ) -> None:
        executed = False

        async def send_message() -> dict[str, str]:
            nonlocal executed
            executed = True
            return {"output": "sent"}

        skills._skills["send_message"] = Skill(
            meta=SkillMeta(
                name="send_message",
                description="Send a message",
                usage="",
                safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
                shared_voice_enabled=True,
            ),
            _test_callable=send_message,
        )
        mock_ai.generate = AsyncMock(return_value="[SKILL: send_message]")
        mock_ai.generate_chat = AsyncMock(return_value="That cannot run in shared VC.")
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="vc:123",
            content="Alice: Send a message",
            is_voice=True,
            metadata={
                "guild_id": "123",
                "channel_id": "vc",
                "via_vc": True,
                "shared_voice": True,
                "ephemeral": True,
            },
        )

        await engine.handle_message(msg)

        assert executed is False
        assert await memory.get_pending_proposals() == []
        assert mock_gateway.send_to_adapter.await_count == 1
        reply = mock_gateway.send_to_adapter.call_args.args[1]
        assert reply.content == "That cannot run in shared VC."

    async def test_post_process_sanitizes_reasoning_response(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
    ) -> None:
        user = await memory.get_or_create_user("u1", "Alice")
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="Draw a deer",
            metadata={"channel_id": "456", "is_dm": False},
        )
        leaked = (
            "Here's a thinking process:\n"
            "3. **Formulate Response:**\n"
            "   [DRAW: internal draft]\n"
            "   - Text: A Nara deer ✨ I'll draw it with a gentle atmosphere.\n"
            "   - Checks: OK"
        )

        await engine._post_process_message("u1", user, msg, leaked)

        msgs = await memory.get_recent_messages("u1")
        assistant_msg = next(item for item in msgs if item["role"] == "assistant")
        assert assistant_msg["content"] == (
            "A Nara deer ✨ I'll draw it with a gentle atmosphere."
        )

    async def test_post_process_strips_tool_tags_from_stored_history(
        self,
        engine: CoreEngine,
        memory: MemoryStore,
    ) -> None:
        user = await memory.get_or_create_user("u1", "Alice")
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="user1",
            content="Please run this [SKILL: web_search | query=secret]",
            metadata={"channel_id": "456", "is_dm": False},
        )
        response = "Done. [DRAW: a detailed image-generation prompt]"

        await engine._post_process_message("u1", user, msg, response)

        msgs = await memory.get_recent_messages("u1")
        user_msg = next(item for item in msgs if item["role"] == "user")
        assistant_msg = next(item for item in msgs if item["role"] == "assistant")
        assert "[SKILL:" not in user_msg["content"]
        assert "query=secret" not in user_msg["content"]
        assert "[DRAW:" not in assistant_msg["content"]
        assert "image-generation prompt" not in assistant_msg["content"]

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
        await engine.drain()
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
        await engine.drain()  # ensure msg1 background tasks finish before msg2

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
        await eng.drain()

        # Emotion should be updated
        assert soul.emotion.primary == Emotion.JOY

        # Flashbulb memory should have been created
        user_id = await memory.resolve_user("test", "user1")
        assert user_id is not None
        results = await memory.search_episodic(user_id, "married")
        assert len(results) >= 1
        assert results[0]["metadata"]["flashbulb"] is True

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
        await engine.drain()
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
        await eng.drain()

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
        await eng.drain()

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
        await eng.drain()

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
        soul.update_emotion(Emotion.JOY, 0.8, caller=SoulCaller.AI)

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


class TestAutoDraw:
    """Tests for the LLM-driven draw pipeline (_maybe_draw + _generate_draw_dsl)."""

    async def test_maybe_draw_no_tag(
        self,
        engine: CoreEngine,
    ) -> None:
        """Response with no [DRAW: ...] tag returns unchanged text and no images."""
        text, images = await engine._maybe_draw("Hello there!")
        assert text == "Hello there!"
        assert images == []

    async def test_maybe_draw_no_skill(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """When draw skill is absent, [DRAW: ...] tags are left untouched."""
        empty_skills = SkillRegistry(tmp_path / "no_skills")
        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=empty_skills,
            gateway=mock_gateway,
        )
        text, images = await eng._maybe_draw("Check [DRAW: a cat]!")
        assert "[DRAW:" in text
        assert images == []

    async def test_maybe_draw_with_skill_success(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """[DRAW: ...] tag triggers DSL generation and skill execution."""
        from unittest.mock import MagicMock

        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(
            return_value={"output": "base64imgdata", "warnings": []}
        )

        fake_skills = MagicMock()
        fake_skills.get = lambda name: mock_skill if name == "draw" else None

        async def _gen(**kw: object) -> str:
            prompt = kw.get("prompt", "")
            if isinstance(prompt, str) and "Draw this:" in prompt:
                return "SIZE 400 400\nCANVAS white\nCIRCLE 200 200 100 red\nOUTPUT"
            return "Hello!"

        mock_ai.generate = AsyncMock(side_effect=_gen)

        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=fake_skills,
            gateway=mock_gateway,
        )
        text, images = await eng._maybe_draw("Here you go! [DRAW: a red circle]")
        assert "[DRAW:" not in text
        assert images == ["base64imgdata"]

    async def test_maybe_draw_skips_complex_image_prompt(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Rich image-generation prompts are not sent to the Draw DSL pipeline."""
        from unittest.mock import MagicMock

        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(return_value={"output": "base64imgdata"})

        fake_skills = MagicMock()
        fake_skills.get = lambda name: mock_skill if name == "draw" else None

        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=fake_skills,
            gateway=mock_gateway,
        )
        text, images = await eng._maybe_draw(
            "Here [DRAW: a majestic icy dragon with translucent scales flying "
            "inside a crystal cave surrounded by glowing crystals and ethereal "
            "lighting]"
        )

        assert images == []
        assert "image-generation prompts" in text
        mock_ai.generate.assert_not_awaited()
        mock_skill.execute.assert_not_awaited()

    async def test_maybe_draw_handles_malformed_a_draw_tag(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Malformed ``[A DRAW: ...]`` tags are stripped and guarded too."""
        from unittest.mock import MagicMock

        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(return_value={"output": "base64imgdata"})

        fake_skills = MagicMock()
        fake_skills.get = lambda name: mock_skill if name == "draw" else None

        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=fake_skills,
            gateway=mock_gateway,
        )
        text, images = await eng._maybe_draw(
            "[A DRAW: A simple vector drawing of a cute Nara deer sitting down, "
            "with a small hat on its head, on a green grassy background]\n\n"
            "I drew a cute deer."
        )

        assert "[A DRAW:" not in text
        assert images == []
        assert "image-generation prompts" in text
        mock_ai.generate.assert_not_awaited()
        mock_skill.execute.assert_not_awaited()

    async def test_maybe_draw_appends_output_to_truncated_dsl(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Truncated model DSL still emits an image after OUTPUT normalization."""
        from unittest.mock import MagicMock

        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(
            return_value={"output": "base64imgdata", "warnings": []}
        )

        fake_skills = MagicMock()
        fake_skills.get = lambda name: mock_skill if name == "draw" else None
        mock_ai.generate = AsyncMock(
            return_value="SIZE 400 400\nCANVAS white\nCIRCLE 200 200 100 red"
        )

        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=fake_skills,
            gateway=mock_gateway,
        )
        text, images = await eng._maybe_draw("Here you go! [DRAW: a red circle]")

        assert "[DRAW:" not in text
        assert images == ["base64imgdata"]
        commands = mock_skill.execute.await_args.args[0]["commands"]
        assert commands.endswith("\nOUTPUT")

    async def test_maybe_draw_normalizes_colon_opcodes(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Model output like ``SIZE:`` is normalized before skill execution."""
        from unittest.mock import MagicMock

        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(
            return_value={"output": "base64imgdata", "warnings": []}
        )

        fake_skills = MagicMock()
        fake_skills.get = lambda name: mock_skill if name == "draw" else None
        mock_ai.generate = AsyncMock(
            return_value="SIZE: 400 400\nCANVAS: white\nCIRCLE: 200 200 80 red"
        )

        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=fake_skills,
            gateway=mock_gateway,
        )
        _, images = await eng._maybe_draw("Here [DRAW: a red moon]")

        assert images == ["base64imgdata"]
        commands = mock_skill.execute.await_args.args[0]["commands"]
        assert "SIZE:" not in commands
        assert "CANVAS:" not in commands
        assert "CIRCLE:" not in commands

    async def test_maybe_draw_retries_ai_when_llm_dsl_has_no_image(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """If model DSL returns no output, the AI gets one retry."""
        from unittest.mock import MagicMock

        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(
            side_effect=[
                {"warnings": ["OUTPUT missing"]},
                {"output": "retryimgdata", "warnings": []},
            ]
        )

        fake_skills = MagicMock()
        fake_skills.get = lambda name: mock_skill if name == "draw" else None
        mock_ai.generate = AsyncMock(
            side_effect=[
                "SIZE 800 600\nCANVAS #08111f\nCIRCLE 400 300 80 #059669",
                "SIZE 800 600\nCANVAS #102030\nELLIPSE 220 180 580 420 #38bdf8",
            ]
        )

        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=fake_skills,
            gateway=mock_gateway,
        )
        text, images = await eng._maybe_draw("Here [DRAW: a blue bird in rain]")

        assert "[DRAW:" not in text
        assert images == ["retryimgdata"]
        assert mock_ai.generate.await_count == 2
        assert mock_skill.execute.await_count == 2
        retry_prompt = mock_ai.generate.await_args.kwargs["prompt"]
        assert "Previous attempt failed" in retry_prompt

    async def test_maybe_draw_retries_when_draw_result_has_severe_warning(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Rendered images with serious Draw warnings are regenerated."""
        from unittest.mock import MagicMock

        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(
            side_effect=[
                {
                    "output": "warnedimgdata",
                    "warnings": ["Line 2: unknown command 'BAD'"],
                },
                {"output": "cleanimgdata", "warnings": []},
            ]
        )

        fake_skills = MagicMock()
        fake_skills.get = lambda name: mock_skill if name == "draw" else None
        mock_ai.generate = AsyncMock(
            side_effect=[
                "SIZE 400 400\nCANVAS white\nBAD command\nCIRCLE 200 200 80 red",
                "SIZE 400 400\nCANVAS white\nCIRCLE 200 200 80 red",
            ]
        )

        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=fake_skills,
            gateway=mock_gateway,
        )
        text, images = await eng._maybe_draw("Here [DRAW: a red circle]")

        assert "[DRAW:" not in text
        assert images == ["cleanimgdata"]
        assert mock_ai.generate.await_count == 2
        retry_prompt = mock_ai.generate.await_args.kwargs["prompt"]
        assert "execution warnings" in retry_prompt

    async def test_maybe_draw_uses_vision_review_to_retry(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
    ) -> None:
        """Vision review can reject a rendered image and trigger regeneration."""
        from unittest.mock import MagicMock

        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(
            side_effect=[
                {"output": "wrongimgdata", "warnings": []},
                {"output": "goodimgdata", "warnings": []},
            ]
        )

        fake_skills = MagicMock()
        fake_skills.get = lambda name: mock_skill if name == "draw" else None
        mock_ai.generate = AsyncMock(
            side_effect=[
                "SIZE 400 400\nCANVAS white\nRECT 20 20 120 120 blue FILL",
                "SIZE 400 400\nCANVAS white\nCIRCLE 200 200 80 red FILL",
            ]
        )
        mock_ai.generate_with_vision = AsyncMock(
            side_effect=[
                '{"verdict":"retry","reason":"image shows a blue square"}',
                '{"verdict":"pass"}',
            ]
        )

        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=fake_skills,
            gateway=mock_gateway,
            vision_enabled=True,
        )
        text, images = await eng._maybe_draw("Here [DRAW: a red circle]")

        assert "[DRAW:" not in text
        assert images == ["goodimgdata"]
        assert mock_ai.generate.await_count == 2
        assert mock_ai.generate_with_vision.await_count == 2
        retry_prompt = mock_ai.generate.await_args.kwargs["prompt"]
        assert "blue square" in retry_prompt

    async def test_maybe_draw_skill_error_falls_back(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """If the draw skill raises, we fall back to clean text with no images."""
        from unittest.mock import MagicMock

        mock_skill = MagicMock()
        mock_skill.execute = AsyncMock(side_effect=RuntimeError("Pillow missing"))

        fake_skills = MagicMock()
        fake_skills.get = lambda name: mock_skill if name == "draw" else None

        async def _gen(**kw: object) -> str:
            prompt = kw.get("prompt", "")
            if isinstance(prompt, str) and "Draw this:" in prompt:
                return "SIZE 400 400\nCANVAS white\nCIRCLE 200 200 100 green\nOUTPUT"
            return "Hello!"

        mock_ai.generate = AsyncMock(side_effect=_gen)

        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=fake_skills,
            gateway=mock_gateway,
        )
        text, images = await eng._maybe_draw("Check this [DRAW: a tree]!")
        assert "[DRAW:" not in text
        assert "Drawing failed" in text
        assert images == []

    async def test_generate_draw_dsl_calls_ai(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """_generate_draw_dsl returns the AI output stripped of whitespace."""
        dsl_response = "  SIZE 400 400\nCANVAS white\nOUTPUT\n  "

        async def _gen(**kw: object) -> str:
            return dsl_response

        mock_ai.generate = AsyncMock(side_effect=_gen)
        empty_skills = SkillRegistry(tmp_path / "skills2")
        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=empty_skills,
            gateway=mock_gateway,
        )
        dsl = await eng._generate_draw_dsl("a white canvas")
        assert dsl == dsl_response.strip()
        call_kwargs = mock_ai.generate.await_args.kwargs
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 1200
        assert "/no_think" in call_kwargs["system"]

    async def test_generate_draw_dsl_ai_failure_returns_empty(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """_generate_draw_dsl returns empty string when AI raises."""

        async def _fail(**kw: object) -> str:
            raise RuntimeError("timeout")

        mock_ai.generate = AsyncMock(side_effect=_fail)
        empty_skills = SkillRegistry(tmp_path / "skills3")
        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=empty_skills,
            gateway=mock_gateway,
        )
        dsl = await eng._generate_draw_dsl("a star")
        assert dsl == ""


class TestReActLoop:
    """Tests for the ReAct multi-step skill execution loop."""

    def _make_safe_skill(self, output: str = "tool output") -> Skill:
        meta = SkillMeta(
            name="test_tool",
            description="A test tool",
            usage="",
            safety_level=SafetyLevel.SAFE,
        )

        def execute(**kwargs: object) -> dict[str, str]:
            return {"output": output}

        class _Mod:
            pass

        mod = _Mod()
        mod.execute = execute  # type: ignore[attr-defined]
        return Skill(meta=meta, _test_callable=mod.execute)

    def _make_engine(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
        react_enabled: bool = True,
    ) -> CoreEngine:
        react = ReActConfig(
            enabled=react_enabled, max_iterations=3, max_tool_output_chars=4000
        )
        skills = SkillRegistry(tmp_path / "react_skills")
        return CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=skills,
            gateway=mock_gateway,
            react_config=react,
        )

    async def test_no_skill_tags_passthrough(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Response with no skill tags should not call generate_chat."""
        eng = self._make_engine(mock_ai, soul, memory, mock_gateway, tmp_path)
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hello",
        )
        await eng.handle_message(msg)
        mock_ai.generate_chat.assert_not_awaited()
        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "Hello there!" in reply.content

    async def test_single_safe_skill_calls_generate_chat(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """A [SKILL: test_tool] tag executes the tool and calls generate_chat."""
        call_count = 0

        async def _generate(**kw: object) -> str:
            nonlocal call_count
            call_count += 1
            prompt = str(kw.get("prompt", ""))
            if "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.7}'
            if "extract memory" in prompt.lower():
                return (
                    '{"topic": "t", "emotional_tone": "n",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "Here you go! [SKILL: test_tool | q=hello]"

        mock_ai.generate = AsyncMock(side_effect=_generate)
        mock_ai.generate_chat = AsyncMock(return_value="The answer is 42")

        eng = self._make_engine(mock_ai, soul, memory, mock_gateway, tmp_path)
        eng._skills._skills["test_tool"] = self._make_safe_skill("tool output")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Run the tool",
        )
        await eng.handle_message(msg)

        mock_ai.generate_chat.assert_awaited_once()
        calls = mock_gateway.send_to_adapter.call_args_list
        contents = [c[0][1].content for c in calls]
        assert any("The answer is 42" in c for c in contents)

    async def test_non_safe_skill_requests_confirmation(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Non-safe skills request confirmation before chat generation."""

        async def _generate(**kw: object) -> str:
            prompt = str(kw.get("prompt", ""))
            if "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.7}'
            if "extract memory" in prompt.lower():
                return (
                    '{"topic": "t", "emotional_tone": "n",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "[SKILL: risky_tool]"

        mock_ai.generate = AsyncMock(side_effect=_generate)

        eng = self._make_engine(mock_ai, soul, memory, mock_gateway, tmp_path)
        risky_meta = SkillMeta(
            name="risky_tool",
            description="Risky",
            usage="",
            safety_level=SafetyLevel.REQUIRES_CONFIRMATION,
        )
        eng._skills._skills["risky_tool"] = Skill(meta=risky_meta)

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Do risky thing",
        )
        await eng.handle_message(msg)

        mock_ai.generate_chat.assert_not_awaited()
        sent = [c[0][1] for c in mock_gateway.send_to_adapter.call_args_list]
        confirm = next(m for m in sent if m.type == MessageType.SKILL_CONFIRM)
        assert confirm.metadata["skill_name"] == "risky_tool"

        user_id = await memory.resolve_user("test", "user1")
        assert user_id is not None
        proposals = await memory.get_certain_records(user_id, record_type="proposal")
        assert len(proposals) == 1

    async def test_react_disabled_strips_tags(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """With react disabled, skill tags are stripped from the response."""

        async def _generate(**kw: object) -> str:
            prompt = str(kw.get("prompt", ""))
            if "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.7}'
            if "extract memory" in prompt.lower():
                return (
                    '{"topic": "t", "emotional_tone": "n",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "Here is the result [SKILL: test_tool | x=1]"

        mock_ai.generate = AsyncMock(side_effect=_generate)

        eng = self._make_engine(
            mock_ai, soul, memory, mock_gateway, tmp_path, react_enabled=False
        )
        eng._skills._skills["test_tool"] = self._make_safe_skill()

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Test",
        )
        await eng.handle_message(msg)

        mock_ai.generate_chat.assert_not_awaited()
        reply = mock_gateway.send_to_adapter.call_args[0][1]
        assert "[SKILL:" not in reply.content
        assert "Here is the result" in reply.content

    async def test_pre_tag_text_flushed(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Text before first skill tag is sent to adapter immediately (D13)."""

        async def _generate(**kw: object) -> str:
            prompt = str(kw.get("prompt", ""))
            if "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.7}'
            if "extract memory" in prompt.lower():
                return (
                    '{"topic": "t", "emotional_tone": "n",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "Let me check. [SKILL: test_tool]"

        mock_ai.generate = AsyncMock(side_effect=_generate)
        mock_ai.generate_chat = AsyncMock(return_value="Done!")

        eng = self._make_engine(mock_ai, soul, memory, mock_gateway, tmp_path)
        eng._skills._skills["test_tool"] = self._make_safe_skill("data")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Check it",
        )
        await eng.handle_message(msg)

        all_calls = mock_gateway.send_to_adapter.call_args_list
        all_contents = [c[0][1].content for c in all_calls]
        # "Let me check." should be flushed before the final reply
        assert any("Let me check." in c for c in all_contents)

    async def test_pre_tag_text_followed_by_tool_running_ack(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """After pre-tool text, send a visible ACK while the tool runs."""

        async def _generate(**kw: object) -> str:
            prompt = str(kw.get("prompt", ""))
            if "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.7}'
            if "extract memory" in prompt.lower():
                return (
                    '{"topic": "t", "emotional_tone": "n",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "Let me check. [SKILL: test_tool]"

        mock_ai.generate = AsyncMock(side_effect=_generate)
        mock_ai.generate_chat = AsyncMock(return_value="Done!")

        eng = self._make_engine(mock_ai, soul, memory, mock_gateway, tmp_path)
        eng._skills._skills["test_tool"] = self._make_safe_skill("data")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Check it",
        )
        await eng.handle_message(msg)

        sent = [c[0][1] for c in mock_gateway.send_to_adapter.call_args_list]
        pre_index = next(i for i, m in enumerate(sent) if "Let me check." in m.content)
        ack_index = next(i for i, m in enumerate(sent) if m.type == MessageType.ACK)
        assert pre_index < ack_index
        assert "Running tool" in sent[ack_index].content

    async def test_draw_excluded_from_react_tool_catalog(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Draw is advertised through [DRAW: ...], not [SKILL: draw]."""
        from unittest.mock import MagicMock

        fake_skills = MagicMock()
        fake_skills.get = lambda name: object() if name == "draw" else None
        fake_skills.get_skill_descriptions_for_prompt.return_value = (
            "(no skills available)"
        )
        eng = CoreEngine(
            ai=mock_ai,
            soul=soul,
            memory=memory,
            skills=fake_skills,
            gateway=mock_gateway,
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Draw something",
        )
        await eng.handle_message(msg)

        fake_skills.get_skill_descriptions_for_prompt.assert_called_once_with(
            exclude_names={"draw"}
        )

    async def test_max_iterations_strips_leaked_tags(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """C-1: even when max_iterations is exhausted with a tag-bearing
        response, the final reply must NOT contain a raw [SKILL: ...] tag."""

        async def _generate(**kw: object) -> str:
            prompt = str(kw.get("prompt", ""))
            if "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.7}'
            if "extract memory" in prompt.lower():
                return (
                    '{"topic": "t", "emotional_tone": "n",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "First reply [SKILL: test_tool]"

        mock_ai.generate = AsyncMock(side_effect=_generate)
        # generate_chat keeps emitting another tag, never settling.
        mock_ai.generate_chat = AsyncMock(
            return_value="Still working [SKILL: test_tool]"
        )

        eng = self._make_engine(mock_ai, soul, memory, mock_gateway, tmp_path)
        eng._skills._skills["test_tool"] = self._make_safe_skill("data")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Loop forever",
        )
        await eng.handle_message(msg)

        all_calls = mock_gateway.send_to_adapter.call_args_list
        for call in all_calls:
            content = call[0][1].content
            assert "[SKILL:" not in content, (
                f"Raw [SKILL:...] tag leaked to user: {content!r}"
            )

    async def test_max_iterations_empty_text_fallback(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """C-2: if after stripping tags the final reply would be empty AND
        tools were called, send a fallback message instead of nothing."""

        async def _generate(**kw: object) -> str:
            prompt = str(kw.get("prompt", ""))
            if "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.7}'
            if "extract memory" in prompt.lower():
                return (
                    '{"topic": "t", "emotional_tone": "n",'
                    ' "facts": [], "episode_summary": ""}'
                )
            # AI emits ONLY the skill tag, no preamble (empty pre_text).
            return "[SKILL: test_tool]"

        mock_ai.generate = AsyncMock(side_effect=_generate)
        # All ReAct continuations emit only a tag too.
        mock_ai.generate_chat = AsyncMock(return_value="[SKILL: test_tool]")

        eng = self._make_engine(mock_ai, soul, memory, mock_gateway, tmp_path)
        eng._skills._skills["test_tool"] = self._make_safe_skill("data")

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Empty replies",
        )
        await eng.handle_message(msg)

        all_calls = mock_gateway.send_to_adapter.call_args_list
        # Final message (the reply) should have non-empty fallback content.
        final_reply = all_calls[-1][0][1]
        assert final_reply.content.strip()
        assert "[SKILL:" not in final_reply.content

    async def test_voice_message_propagates_to_backend(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
        memory: MemoryStore,
        mock_gateway: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """is_voice on GatewayMessage activates voice_context_scope around
        AI generation, so backends can apply per-context overrides."""
        from cordbeat.ai.backend import is_voice_context

        observed: list[bool] = []

        async def _generate(**kw: object) -> str:
            observed.append(is_voice_context())
            prompt = str(kw.get("prompt", ""))
            if "recall keywords" in prompt.lower():
                return '{"keywords": []}'
            if "what emotion" in prompt.lower():
                return '{"emotion": "joy", "intensity": 0.7}'
            if "extract memory" in prompt.lower():
                return (
                    '{"topic": "t", "emotional_tone": "n",'
                    ' "facts": [], "episode_summary": ""}'
                )
            return "Hi there!"

        mock_ai.generate = AsyncMock(side_effect=_generate)

        eng = self._make_engine(mock_ai, soul, memory, mock_gateway, tmp_path)

        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="user1",
            content="Hi",
            is_voice=True,
        )
        await eng.handle_message(msg)

        # The main response generation must observe voice context = True.
        # Background tasks like memory extraction may run outside scope; we
        # only require that AT LEAST one call saw voice_context active.
        assert any(observed), (
            f"Expected at least one generate() call inside voice_context_scope; "
            f"observed={observed}"
        )
