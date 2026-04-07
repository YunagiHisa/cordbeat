"""Tests for AI backend abstraction layer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cordbeat.ai_backend import (
    OllamaBackend,
    OpenAICompatBackend,
    create_backend,
)
from cordbeat.config import AIBackendConfig

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

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await backend.generate("test prompt", system="sys")

        assert result == "Hello!"
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "/api/generate" in call_kwargs[0][0]

    async def test_generate_empty_response(self) -> None:
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await backend.generate("test")

        assert result == ""


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

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await backend.generate("test prompt", system="sys")

        assert result == "Hi!"
        call_kwargs = mock_client.post.call_args
        assert "/v1/chat/completions" in call_kwargs[0][0]

    async def test_unexpected_response_format(self) -> None:
        cfg = AIBackendConfig(provider="openai")
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"bad": "format"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Unexpected response format"):
                await backend.generate("test")
