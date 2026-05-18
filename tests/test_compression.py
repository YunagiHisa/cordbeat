"""Tests for ConversationCompressor."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cordbeat.ai.compression import ConversationCompressor


def _make_history(n: int, content_len: int = 100) -> list[dict[str, str]]:
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": "x" * content_len,
            "created_at": f"2026-04-{(i % 28) + 1:02d}T10:00:00+00:00",
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_compress_in_memory_below_threshold() -> None:
    ai = AsyncMock()
    compressor = ConversationCompressor(ai)
    history = _make_history(4, content_len=50)  # total ~200 chars
    result = await compressor.compress_in_memory(history, chars_threshold=4000)
    assert result == history
    ai.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_compress_in_memory_above_threshold() -> None:
    ai = AsyncMock()
    ai.generate = AsyncMock(return_value="Earlier in this conversation: stuff.")
    compressor = ConversationCompressor(ai)
    history = _make_history(10, content_len=500)  # total 5000 chars
    result = await compressor.compress_in_memory(
        history, chars_threshold=4000, soul_name="Aria"
    )
    # First entry is the summary
    assert len(result) < len(history)
    assert result[0]["role"] == "system"
    assert result[0]["content"].startswith("Earlier")
    # Subsequent entries are the more recent originals
    assert result[1] == history[5]
    ai.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_compress_in_memory_too_short() -> None:
    """Fewer than 4 messages → no compression even if over chars threshold."""
    ai = AsyncMock()
    compressor = ConversationCompressor(ai)
    history = _make_history(3, content_len=5000)
    result = await compressor.compress_in_memory(history, chars_threshold=100)
    assert result == history
    ai.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_compress_chunk_to_text() -> None:
    ai = AsyncMock()
    ai.generate = AsyncMock(return_value="  Earlier: summary text.  ")
    compressor = ConversationCompressor(ai)
    history = _make_history(5)
    result = await compressor.compress_chunk_to_text(history)
    assert result == "Earlier: summary text."


@pytest.mark.asyncio
async def test_compress_chunk_to_text_empty() -> None:
    ai = AsyncMock()
    compressor = ConversationCompressor(ai)
    result = await compressor.compress_chunk_to_text([])
    assert result is None
    ai.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_compress_in_memory_llm_failure() -> None:
    """LLM raising an exception → original history returned unchanged."""
    ai = AsyncMock()
    ai.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
    compressor = ConversationCompressor(ai)
    history = _make_history(10, content_len=500)
    result = await compressor.compress_in_memory(history, chars_threshold=4000)
    assert result == history


@pytest.mark.asyncio
async def test_compress_in_memory_empty_llm_response() -> None:
    """LLM returning empty string → original history returned unchanged."""
    ai = AsyncMock()
    ai.generate = AsyncMock(return_value="   ")
    compressor = ConversationCompressor(ai)
    history = _make_history(10, content_len=500)
    result = await compressor.compress_in_memory(history, chars_threshold=4000)
    assert result == history
