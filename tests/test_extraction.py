"""Tests for memory extraction module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from cordbeat.agent.soul import Soul
from cordbeat.ai.backend import is_internal_context
from cordbeat.ai.extraction import MemoryExtractor
from cordbeat.ai.reasoning import parse_json_object
from cordbeat.config import MemoryConfig
from cordbeat.memory import MemoryStore


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
    yield store  # type: ignore[misc]
    await store.close()


@pytest.fixture
def mock_ai() -> AsyncMock:
    ai = AsyncMock()
    return ai


@pytest.fixture
def extractor(mock_ai: AsyncMock, soul: Soul, memory: MemoryStore) -> MemoryExtractor:
    return MemoryExtractor(ai=mock_ai, soul=soul, memory=memory)


def test_parse_json_object_accepts_fenced_model_output() -> None:
    result = parse_json_object(
        'Here is the result:\n```json\n{"keywords": ["CordBeat"]}\n```'
    )
    assert result == {"keywords": ["CordBeat"]}


class TestInferAndUpdateEmotion:
    @pytest.mark.anyio
    async def test_runs_backend_call_in_internal_scope(
        self,
        extractor: MemoryExtractor,
        mock_ai: AsyncMock,
    ) -> None:
        async def observe_scope(**_kwargs: object) -> str:
            assert is_internal_context() is True
            return '{"emotion": "calm", "intensity": 0.2}'

        mock_ai.generate = AsyncMock(side_effect=observe_scope)

        await extractor.infer_and_update_emotion("u1", "Hello", "Hi")

        assert is_internal_context() is False

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
        assert "/no_think" not in mock_ai.generate.await_args.kwargs["system"]


class TestServerSharedNoteExtraction:
    async def test_accepts_exact_evidence_in_internal_scope(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        async def observe_scope(**_kwargs: object) -> str:
            assert is_internal_context() is True
            return json.dumps(
                {
                    "kind": "decision",
                    "summary": "The release moved to Friday.",
                    "evidence": "release moved to Friday",
                }
            )

        mock_ai.generate = AsyncMock(side_effect=observe_scope)

        note = await extractor.extract_server_shared_note(
            "The release moved to Friday after review."
        )

        assert note == {
            "kind": "decision",
            "summary": "The release moved to Friday.",
            "evidence": "release moved to Friday",
        }
        assert is_internal_context() is False

    @pytest.mark.parametrize(
        "payload",
        [
            {"kind": "none", "summary": "", "evidence": ""},
            {
                "kind": "decision",
                "summary": "A fabricated decision.",
                "evidence": "words absent from the source",
            },
        ],
    )
    async def test_rejects_non_notes_and_ungrounded_evidence(
        self,
        extractor: MemoryExtractor,
        mock_ai: AsyncMock,
        payload: dict[str, str],
    ) -> None:
        mock_ai.generate = AsyncMock(return_value=json.dumps(payload))

        assert await extractor.extract_server_shared_note("Casual conversation") is None

    async def test_backend_failure_is_fail_soft(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(side_effect=RuntimeError("offline"))

        assert await extractor.extract_server_shared_note("Release decided") is None


class TestInferAndUpdateEmotionContinued:

    @pytest.mark.anyio
    async def test_accepts_fenced_json(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock, soul: Soul
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value='```json\n{"emotion": "joy", "intensity": 0.6}\n```'
        )
        await extractor.infer_and_update_emotion("u1", "Great news!", "Wonderful!")
        assert soul.get_soul_snapshot()["emotion"]["primary"] == "joy"

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
        assert "/no_think" not in mock_ai.generate.await_args.kwargs["system"]

    @pytest.mark.anyio
    async def test_accepts_fenced_json(
        self,
        extractor: MemoryExtractor,
        mock_ai: AsyncMock,
        memory: MemoryStore,
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value=(
                "```json\n"
                '{"topic":"programming","emotional_tone":"curious",'
                '"facts":["User likes Python"],"episode_summary":""}'
                "\n```"
            )
        )
        await memory.get_or_create_user("u1", "Alice")
        await extractor.extract_and_store_memories(
            "u1", "Alice", "I like Python", "Cool!"
        )
        assert await memory.search_semantic("u1", "Python", n_results=5)

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
    async def test_extraction_prompt_forbids_unconfirmed_ai_work_claims(
        self,
        extractor: MemoryExtractor,
        mock_ai: AsyncMock,
        memory: MemoryStore,
    ) -> None:
        """The extractor LLM is instructed not to pin unverified AI claims."""
        mock_ai.generate = AsyncMock(
            return_value=json.dumps(
                {
                    "topic": "",
                    "emotional_tone": "neutral",
                    "facts": [],
                    "episode_summary": "",
                }
            )
        )
        await memory.get_or_create_user("u1", "Bob")

        await extractor.extract_and_store_memories("u1", "Bob", "hi", "hello")

        prompt = mock_ai.generate.await_args.kwargs["prompt"]
        assert "Do not store AI claims about hidden progress" in prompt
        assert "unless the user independently confirmed" in prompt

    @pytest.mark.anyio
    async def test_caps_fact_and_episode_length(
        self,
        mock_ai: AsyncMock,
        soul: Soul,
    ) -> None:
        """Runaway LLM output is capped before it reaches embeddings/DB."""
        memory = MagicMock()
        memory.add_semantic_memory = AsyncMock()
        memory.add_episodic_memory = AsyncMock()
        extractor = MemoryExtractor(ai=mock_ai, soul=soul, memory=memory)
        mock_ai.generate = AsyncMock(
            return_value=json.dumps(
                {
                    "topic": "",
                    "emotional_tone": "",
                    "facts": ["f" * 2000],
                    "episode_summary": "e" * 2000,
                }
            )
        )

        await extractor.extract_and_store_memories("u1", "Bob", "hi", "hello")

        fact_entry = memory.add_semantic_memory.await_args.args[0]
        assert len(fact_entry.content) == 500
        episode_entry = memory.add_episodic_memory.await_args.args[0]
        assert len(episode_entry.content) == 500

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
        assert "/no_think" in mock_ai.generate.await_args.kwargs["system"]

    @pytest.mark.anyio
    async def test_accepts_fenced_json(
        self, extractor: MemoryExtractor, mock_ai: AsyncMock
    ) -> None:
        mock_ai.generate = AsyncMock(
            return_value='```json\n{"keywords": ["Python", "OSS"]}\n```'
        )
        assert await extractor.extract_recall_keywords("test") == ["Python", "OSS"]

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
