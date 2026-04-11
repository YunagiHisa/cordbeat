"""Tests for memory extraction module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cordbeat.config import MemoryConfig
from cordbeat.extraction import MemoryExtractor
from cordbeat.memory import MemoryStore
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
    yield store  # type: ignore[misc]
    await store.close()


@pytest.fixture
def mock_ai() -> AsyncMock:
    ai = AsyncMock()
    return ai


@pytest.fixture
def extractor(mock_ai: AsyncMock, soul: Soul, memory: MemoryStore) -> MemoryExtractor:
    return MemoryExtractor(ai=mock_ai, soul=soul, memory=memory)


class TestInferAndUpdateEmotion:
    @pytest.mark.anyio
    async def test_updates_emotion(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock, soul: Soul
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value='{"emotion": "joy", "intensity": 0.6}'
        )
        await extractor.infer_and_update_emotion("u1", "Great news!", "Wonderful!")
        snap = soul.get_soul_snapshot()
        assert snap["emotion"]["primary"] == "joy"

    @pytest.mark.anyio
    async def test_high_intensity_creates_flashbulb(
        self,
        extractor: MemoryExtractor,
        mock_ai: AsyncMock,
        memory: MemoryStore,
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value='{"emotion": "excitement", "intensity": 0.9}'
        )
        await memory.get_or_create_user("u1", "Alice")
        await extractor.infer_and_update_emotion("u1", "I won!", "Amazing!")
        # Flashbulb memory should be searchable
        results = await memory.search_episodic("u1", "won", n_results=5)
        assert len(results) > 0

    @pytest.mark.anyio
    async def test_parse_failure_does_not_raise(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(return_value="not json at all")
        # Should not raise
        await extractor.infer_and_update_emotion("u1", "hi", "hello")

    @pytest.mark.anyio
    async def test_ai_error_does_not_raise(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(side_effect=RuntimeError("timeout"))
        await extractor.infer_and_update_emotion("u1", "hi", "hello")


class TestExtractAndStoreMemories:
    @pytest.mark.anyio
    async def test_stores_semantic_facts(
        self,
        extractor: MemoryExtractor,
        mock_ai: AsyncMock,
        memory: MemoryStore,
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value=json.dumps(
                {
                    "topic": "programming",
                    "emotional_tone": "curious",
                    "facts": ["User likes Python", "User uses Linux"],
                    "episode_summary": "",
                }
            )
        )
        await memory.get_or_create_user("u1", "Alice")
        await extractor.extract_and_store_memories(
            "u1", "Alice", "I use Python", "Cool!"
        )
        results = await memory.search_semantic("u1", "Python", n_results=5)
        assert len(results) > 0

    @pytest.mark.anyio
    async def test_stores_episodic_summary(
        self,
        extractor: MemoryExtractor,
        mock_ai: AsyncMock,
        memory: MemoryStore,
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value=json.dumps(
                {
                    "topic": "big event",
                    "emotional_tone": "excited",
                    "facts": [],
                    "episode_summary": "User announced they got a new job today.",
                }
            )
        )
        await memory.get_or_create_user("u1", "Bob")
        await extractor.extract_and_store_memories(
            "u1", "Bob", "Got a new job!", "Congrats!"
        )
        results = await memory.search_episodic("u1", "job", n_results=5)
        assert len(results) > 0

    @pytest.mark.anyio
    async def test_updates_user_topic_and_tone(
        self,
        extractor: MemoryExtractor,
        mock_ai: AsyncMock,
        memory: MemoryStore,
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value=json.dumps(
                {
                    "topic": "cooking",
                    "emotional_tone": "happy",
                    "facts": [],
                    "episode_summary": "",
                }
            )
        )
        await memory.get_or_create_user("u1", "Alice")
        await extractor.extract_and_store_memories(
            "u1", "Alice", "I love cooking", "Great!"
        )
        user = await memory.get_or_create_user("u1", "Alice")
        assert user.last_topic == "cooking"
        assert user.emotional_tone == "happy"

    @pytest.mark.anyio
    async def test_ai_failure_does_not_raise(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(side_effect=RuntimeError("boom"))
        await extractor.extract_and_store_memories("u1", "Bob", "hi", "hello")

    @pytest.mark.anyio
    async def test_caps_facts_at_five(
        self,
        extractor: MemoryExtractor,
        mock_ai: AsyncMock,
        memory: MemoryStore,
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value=json.dumps(
                {
                    "topic": "",
                    "emotional_tone": "",
                    "facts": [f"fact {i}" for i in range(10)],
                    "episode_summary": "",
                }
            )
        )
        await memory.get_or_create_user("u1", "Alice")
        await extractor.extract_and_store_memories("u1", "Alice", "many facts", "ok")
        results = await memory.search_semantic("u1", "fact", n_results=20)
        assert len(results) <= 5

    @pytest.mark.anyio
    async def test_stores_emotional_tone_metadata(
        self,
        extractor: MemoryExtractor,
        mock_ai: AsyncMock,
        memory: MemoryStore,
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value=json.dumps(
                {
                    "topic": "music",
                    "emotional_tone": "happy",
                    "facts": ["User plays guitar"],
                    "episode_summary": "User talked about learning guitar recently.",
                }
            )
        )
        await memory.get_or_create_user("u1", "Alice")
        await extractor.extract_and_store_memories(
            "u1", "Alice", "I play guitar", "Cool!"
        )
        semantic = await memory.search_semantic("u1", "guitar", n_results=5)
        assert len(semantic) > 0
        assert semantic[0]["metadata"]["emotional_tone"] == "happy"

        episodic = await memory.search_episodic("u1", "guitar", n_results=5)
        assert len(episodic) > 0
        assert episodic[0]["metadata"]["emotional_tone"] == "happy"


class TestExtractRecallKeywords:
    @pytest.mark.anyio
    async def test_extracts_keywords(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value='{"keywords": ["Python", "OSS", "tired"]}'
        )
        keywords = await extractor.extract_recall_keywords("How are you?")
        assert keywords == ["Python", "OSS", "tired"]

    @pytest.mark.anyio
    async def test_includes_history_context(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(return_value='{"keywords": ["CordBeat"]}')
        history = [
            {"role": "user", "content": "Let's talk about CordBeat"},
            {"role": "assistant", "content": "Sure!"},
        ]
        keywords = await extractor.extract_recall_keywords("Go on", history)
        assert len(keywords) >= 1
        # Verify history was passed in the prompt
        call_prompt = mock_ai.generate.call_args[1]["prompt"]
        assert "CordBeat" in call_prompt

    @pytest.mark.anyio
    async def test_caps_at_three_keywords(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value='{"keywords": ["a", "bb", "cc", "dd", "ee"]}'
        )
        keywords = await extractor.extract_recall_keywords("test")
        assert len(keywords) <= 3

    @pytest.mark.anyio
    async def test_filters_short_keywords(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(return_value='{"keywords": ["a", "ok", "Python"]}')
        keywords = await extractor.extract_recall_keywords("test")
        # "a" is too short (<2 chars), should be filtered
        assert "a" not in keywords
        assert "ok" in keywords
        assert "Python" in keywords

    @pytest.mark.anyio
    async def test_parse_failure_returns_empty(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(return_value="not json")
        keywords = await extractor.extract_recall_keywords("test")
        assert keywords == []

    @pytest.mark.anyio
    async def test_ai_error_returns_empty(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(side_effect=RuntimeError("timeout"))
        keywords = await extractor.extract_recall_keywords("test")
        assert keywords == []
