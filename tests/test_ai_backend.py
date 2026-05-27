"""Tests for AI backend abstraction layer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cordbeat.ai.backend import (
    OllamaBackend,
    OpenAICompatBackend,
    create_backend,
)
from cordbeat.config import AIBackendConfig
from cordbeat.exceptions import AIBackendError

# ── Factory ───────────────────────────────────────────────────────────


class TestCreateBackend:
    def test_ollama(self) -> None:
        cfg = AIBackendConfig(provider="ollama")
        backend = create_backend(cfg)
        assert isinstance(backend, OllamaBackend)

    def test_openai(self) -> None:
        cfg = AIBackendConfig(provider="openai")
        backend = create_backend(cfg)
        assert isinstance(backend, OpenAICompatBackend)

    def test_openai_compat(self) -> None:
        cfg = AIBackendConfig(provider="openai_compat")
        backend = create_backend(cfg)
        assert isinstance(backend, OpenAICompatBackend)

    def test_unknown_provider(self) -> None:
        cfg = AIBackendConfig(provider="llama_cpp")
        with pytest.raises(ValueError, match="Unknown AI backend provider"):
            create_backend(cfg)


# ── generate_json ─────────────────────────────────────────────────────


class TestGenerateJson:
    async def test_plain_json(self) -> None:
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)
        backend.generate = AsyncMock(  # type: ignore[method-assign]
            return_value='{"action": "none"}'
        )
        result = await backend.generate_json("test")
        assert result == {"action": "none"}

    async def test_json_in_code_block(self) -> None:
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)
        backend.generate = AsyncMock(  # type: ignore[method-assign]
            return_value='```json\n{"action": "message"}\n```'
        )
        result = await backend.generate_json("test")
        assert result == {"action": "message"}

    async def test_invalid_json_raises(self) -> None:
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)
        backend.generate = AsyncMock(  # type: ignore[method-assign]
            return_value="not json at all"
        )
        with pytest.raises(json.JSONDecodeError):
            await backend.generate_json("test")

    async def test_think_tags_stripped(self) -> None:
        """Qwen3 / DeepSeek-R1 CoT models prepend <think>...</think> blocks."""
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)
        backend.generate = AsyncMock(  # type: ignore[method-assign]
            return_value='<think>\nLet me think...\n</think>\n{"action": "none"}'
        )
        result = await backend.generate_json("test")
        assert result == {"action": "none"}

    async def test_think_tags_multiline_stripped(self) -> None:
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)
        think_block = "<think>\nStep 1: ...\nStep 2: ...\n</think>"
        backend.generate = AsyncMock(  # type: ignore[method-assign]
            return_value=f'{think_block}\n```json\n{{"key": "val"}}\n```'
        )
        result = await backend.generate_json("test")
        assert result == {"key": "val"}


# ── OllamaBackend ────────────────────────────────────────────────────


class TestOllamaBackend:
    async def test_generate_calls_api(self) -> None:
        cfg = AIBackendConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model="test-model",
        )
        backend = OllamaBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hello!"}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.generate("test prompt", system="sys")

        assert result == "Hello!"
        backend._client.post.assert_called_once()
        call_kwargs = backend._client.post.call_args
        assert "/api/generate" in call_kwargs[0][0]

    async def test_generate_empty_response(self) -> None:
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.generate("test")

        assert result == ""

    async def test_aclose(self) -> None:
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)
        backend._client = AsyncMock()
        await backend.aclose()
        backend._client.aclose.assert_called_once()


# ── OpenAICompatBackend ───────────────────────────────────────────────


class TestOpenAICompatBackend:
    async def test_generate_calls_chat_completions(self) -> None:
        cfg = AIBackendConfig(
            provider="openai",
            base_url="http://localhost:8000",
            model="gpt-test",
            options={"api_key": "sk-test"},
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.generate("test prompt", system="sys")

        assert result == "Hi!"
        call_kwargs = backend._client.post.call_args
        assert "/chat/completions" in call_kwargs[0][0]

    async def test_unexpected_response_format(self) -> None:
        cfg = AIBackendConfig(provider="openai")
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"bad": "format"}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(AIBackendError, match="Unexpected response format"):
            await backend.generate("test")

    async def test_reasoning_content_retry_uses_min_4096_tokens(self) -> None:
        """Thinking retry should ensure max_tokens>=4096 even when doubling.

        If caller uses default max_tokens=1024, doubling to 2048 is still too
        small for many thinking models that produce 3000+ chars in reasoning.
        The retry should bump to at least 4096.
        """
        cfg = AIBackendConfig(provider="openai_compat")
        backend = OpenAICompatBackend(cfg)

        first_response = MagicMock()
        first_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "x" * 3000,
                    }
                }
            ]
        }
        first_response.raise_for_status = MagicMock()

        retry_response = MagicMock()
        retry_response.json.return_value = {
            "choices": [{"message": {"content": "answer"}}]
        }
        retry_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=[first_response, retry_response])

        result = await backend.generate("test", max_tokens=1024)
        assert result == "answer"

        # Inspect the retry call payload — must use >=8192
        retry_call_payload = backend._client.post.call_args_list[1][1]["json"]
        assert retry_call_payload["max_tokens"] >= 8192

    async def test_reasoning_content_only_returns_empty(self) -> None:
        """When content=null but reasoning_content is present, return empty string.

        reasoning_content is internal chain-of-thought; leaking it to the user
        would expose the model's raw thinking process verbatim.
        """
        cfg = AIBackendConfig(provider="openai_compat")
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "Here's a thinking process:\n1. ...",
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.generate("test")
        assert result == ""

    async def test_enable_thinking_false_sent_in_payload(self) -> None:
        """enable_thinking: false should be included in the API payload."""
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"enable_thinking": False},
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test")

        call_kwargs = backend._client.post.call_args
        payload = call_kwargs[1]["json"]
        assert payload.get("enable_thinking") is False

    async def test_enable_thinking_not_sent_by_default(self) -> None:
        """enable_thinking should NOT be sent when not configured.

        Avoids breaking non-thinking backends that don't recognise the param.
        """
        cfg = AIBackendConfig(provider="openai_compat")
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test")

        call_kwargs = backend._client.post.call_args
        payload = call_kwargs[1]["json"]
        assert "enable_thinking" not in payload


class TestGenerateJsonEmptyResponse:
    async def test_empty_raw_raises_json_decode_error(self) -> None:
        """Empty LLM response should raise JSONDecodeError with a helpful message."""
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)
        backend.generate = AsyncMock(return_value="")  # type: ignore[method-assign]

        with pytest.raises(
            json.JSONDecodeError, match="Empty AI response|no JSON content"
        ):
            await backend.generate_json("test")

    async def test_think_only_response_raises_json_decode_error(self) -> None:
        """Response with ONLY <think> block (no JSON) should raise JSONDecodeError."""
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)
        backend.generate = AsyncMock(  # type: ignore[method-assign]
            return_value="<think>I should respond with JSON but I won't</think>"
        )

        with pytest.raises(json.JSONDecodeError):
            await backend.generate_json("test")
