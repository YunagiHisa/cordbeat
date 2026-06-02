"""Tests for ``respond_mode: ai_decision_llm`` (opt-in LLM judge)."""

from __future__ import annotations

from typing import Any

import pytest

from cordbeat.adapters._utils import (
    AdapterFilter,
    get_judge_backend,
    set_judge_backend,
)


class _FakeBackend:
    """Minimal AIBackend stub returning a canned answer."""

    def __init__(self, answer: str = "yes") -> None:
        self.answer = answer
        self.calls: list[str] = []

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        return self.answer

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_backend() -> Any:
    set_judge_backend(None)
    yield
    set_judge_backend(None)


@pytest.mark.asyncio
async def test_ai_decision_llm_yes_allows_message() -> None:
    set_judge_backend(_FakeBackend("yes"))  # type: ignore[arg-type]
    f = AdapterFilter(respond_mode="ai_decision_llm")
    assert await f.should_respond_async(
        user_id="u", channel_id="c", is_dm=False, text="hello bot"
    )


@pytest.mark.asyncio
async def test_ai_decision_llm_no_blocks_message() -> None:
    set_judge_backend(_FakeBackend("no"))  # type: ignore[arg-type]
    f = AdapterFilter(respond_mode="ai_decision_llm")
    assert not await f.should_respond_async(
        user_id="u", channel_id="c", is_dm=False, text="hi all"
    )


@pytest.mark.asyncio
async def test_ai_decision_llm_dm_skips_judge() -> None:
    fake = _FakeBackend("no")
    set_judge_backend(fake)  # type: ignore[arg-type]
    f = AdapterFilter(respond_mode="ai_decision_llm")
    # DMs always go through without consulting the judge backend.
    assert await f.should_respond_async(user_id="u", is_dm=True, text="hi")
    assert fake.calls == []


@pytest.mark.asyncio
async def test_ai_decision_llm_no_backend_falls_back_to_keywords() -> None:
    f = AdapterFilter(respond_mode="ai_decision_llm", ai_keywords=["botty"])
    assert await f.should_respond_async(
        user_id="u", channel_id="c", is_dm=False, text="hey botty!"
    )
    assert not await f.should_respond_async(
        user_id="u", channel_id="c", is_dm=False, text="unrelated chatter"
    )


@pytest.mark.asyncio
async def test_ai_decision_llm_backend_exception_allows_message() -> None:
    class Boom:
        async def generate(self, prompt: str, **kwargs: Any) -> str:
            raise RuntimeError("boom")

        async def aclose(self) -> None: ...

    set_judge_backend(Boom())  # type: ignore[arg-type]
    f = AdapterFilter(respond_mode="ai_decision_llm")
    assert await f.should_respond_async(
        user_id="u", channel_id="c", is_dm=False, text="hi"
    )


@pytest.mark.asyncio
async def test_existing_keyword_mode_unaffected() -> None:
    set_judge_backend(_FakeBackend("no"))  # type: ignore[arg-type]
    f = AdapterFilter(respond_mode="ai_decision", ai_keywords=["bot"])
    assert await f.should_respond_async(
        user_id="u", channel_id="c", is_dm=False, text="hey bot"
    )
    assert not await f.should_respond_async(
        user_id="u", channel_id="c", is_dm=False, text="hey"
    )


def test_set_and_get_judge_backend_roundtrip() -> None:
    b = _FakeBackend()
    set_judge_backend(b)  # type: ignore[arg-type]
    assert get_judge_backend() is b
    set_judge_backend(None)
    assert get_judge_backend() is None
