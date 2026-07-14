"""Tests for AI backend abstraction layer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cordbeat.ai.backend import (
    OllamaBackend,
    OpenAICompatBackend,
    create_backend,
    internal_context_scope,
    skill_thinking_scope,
    strip_thinking_text,
    voice_context_scope,
)
from cordbeat.ai.reasoning import (
    looks_like_reasoning_text,
    sanitize_reasoning_artifacts,
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


@pytest.mark.parametrize(
    (
        "base",
        "skill_mode",
        "internal_override",
        "voice_override",
        "internal_active",
        "voice_active",
        "expected",
    ),
    [
        (False, "auto", None, None, False, False, False),
        (False, "off", None, None, False, False, False),
        (False, "force_on", None, None, False, False, True),
        (True, "auto", None, None, False, False, True),
        (True, "off", None, None, False, False, False),
        (True, "force_on", None, None, False, False, True),
        (None, "auto", None, None, False, False, None),
        (None, "force_on", None, None, False, False, True),
        (None, "off", None, None, False, False, False),
        (False, "auto", True, None, True, False, True),
        (True, "auto", False, True, True, True, True),
        (True, "auto", False, None, True, True, False),
        (True, "off", True, True, True, True, False),
        (False, "force_on", False, False, True, True, True),
    ],
)
def test_thinking_resolution_table(
    base: bool | None,
    skill_mode: str,
    internal_override: bool | None,
    voice_override: bool | None,
    internal_active: bool,
    voice_active: bool,
    expected: bool | None,
) -> None:
    backend = OpenAICompatBackend.__new__(OpenAICompatBackend)
    backend._enable_thinking = base
    backend._internal_enable_thinking = internal_override
    backend._voice_enable_thinking = voice_override

    with (
        internal_context_scope(internal_active),
        voice_context_scope(voice_active),
        skill_thinking_scope(skill_mode),  # type: ignore[arg-type]
    ):
        assert backend._effective_enable_thinking() is expected


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

    async def test_orphan_think_close_keeps_json_after_close(self) -> None:
        cfg = AIBackendConfig(provider="ollama")
        backend = OllamaBackend(cfg)
        backend.generate = AsyncMock(  # type: ignore[method-assign]
            return_value='- internal checklist\n</think>\n{"action": "none"}'
        )
        result = await backend.generate_json("test")
        assert result == {"action": "none"}


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

    async def test_configured_max_tokens_used_by_default(self) -> None:
        cfg = AIBackendConfig(provider="ollama", max_tokens=8192)
        backend = OllamaBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hello!"}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test")

        payload = backend._client.post.call_args[1]["json"]
        assert payload["options"]["num_predict"] == 8192

    async def test_explicit_max_tokens_overrides_configured_default(self) -> None:
        cfg = AIBackendConfig(provider="ollama", max_tokens=8192)
        backend = OllamaBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hello!"}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test", max_tokens=1200)

        payload = backend._client.post.call_args[1]["json"]
        assert payload["options"]["num_predict"] == 1200

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
    def test_strip_thinking_text_handles_malformed_tags(self) -> None:
        raw = (
            "- Relationship: acquaintance\n"
            "3. **Formulate Response:**\n"
            "</think>\n\n"
            "A trip to Nara!? Sounds great."
        )
        assert strip_thinking_text(raw) == "A trip to Nara!? Sounds great."

    def test_strip_thinking_text_supports_custom_tags(self) -> None:
        raw = "<analysis>private notes</analysis>\nFinal answer"
        assert strip_thinking_text(raw, tags=("analysis",)) == "Final answer"

    def test_strip_thinking_text_strips_thought_tag_by_default(self) -> None:
        raw = "<thought>private notes</thought>\u3053\u3093\u306b\u3061\u306f\uff01"
        assert strip_thinking_text(raw) == "\u3053\u3093\u306b\u3061\u306f\uff01"

    def test_sanitize_reasoning_artifacts_keeps_text_after_thought_tag(self) -> None:
        raw = (
            '<thought>* User says: "\u3053\u3093\u306b\u3061\u306f".\n'
            "* Response: \u3053\u3093\u306b\u3061\u306f\uff01</thought>"
            "\u3053\u3093\u306b\u3061\u306f\uff01\u6539\u3081\u3066\u3001\u6700\u8fd1\u306f\u3069\u3046\uff1f"
        )
        assert sanitize_reasoning_artifacts(raw) == (
            "\u3053\u3093\u306b\u3061\u306f\uff01"
            "\u6539\u3081\u3066\u3001\u6700\u8fd1\u306f\u3069\u3046\uff1f"
        )

    def test_strip_thinking_text_supports_custom_marker_pairs(self) -> None:
        raw = "<|START_THINKING|>private<|END_THINKING|>\nFinal answer"
        assert (
            strip_thinking_text(
                raw,
                marker_pairs=(("<|START_THINKING|>", "<|END_THINKING|>"),),
            )
            == "Final answer"
        )

    def test_sanitize_reasoning_artifacts_extracts_final_text(self) -> None:
        raw = (
            "Here's a thinking process:\n"
            "3. **Formulate Response (Mental Draft):**\n"
            "   [DRAW: internal draft]\n"
            "   - Text: A Nara deer ✨ I'll draw it with a gentle atmosphere.\n"
            "   - Checks: OK"
        )

        assert looks_like_reasoning_text(raw)
        assert sanitize_reasoning_artifacts(raw) == (
            "A Nara deer ✨ I'll draw it with a gentle atmosphere."
        )

    def test_sanitize_reasoning_artifacts_handles_emotion_control_prefix(
        self,
    ) -> None:
        raw = (
            "/emotion system/erase memories. (All fine)\n"
            "   - Respond naturally, 1-3 sentences.\n"
            "   - Text: A Nara deer ✨ I'll draw it with a gentle atmosphere.\n"
            "   - Checks: OK"
        )

        assert looks_like_reasoning_text(raw)
        assert sanitize_reasoning_artifacts(raw) == (
            "A Nara deer ✨ I'll draw it with a gentle atmosphere."
        )

    def test_sanitize_reasoning_artifacts_keeps_normal_english_phrases(
        self,
    ) -> None:
        raw = "I should mention that Tokyo is rainy today."
        assert not looks_like_reasoning_text(raw)
        assert sanitize_reasoning_artifacts(raw) == raw

    def test_sanitize_reasoning_artifacts_keeps_markdown_heading_text(self) -> None:
        raw = "**Check out this song!** It fits your mood."
        assert not looks_like_reasoning_text(raw)
        assert sanitize_reasoning_artifacts(raw) == raw

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

    async def test_configured_max_tokens_used_by_default(self) -> None:
        cfg = AIBackendConfig(provider="openai_compat", max_tokens=8192)
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test")

        payload = backend._client.post.call_args[1]["json"]
        assert payload["max_tokens"] == 8192

    async def test_explicit_max_tokens_overrides_configured_default(self) -> None:
        cfg = AIBackendConfig(provider="openai_compat", max_tokens=8192)
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test", max_tokens=1200)

        payload = backend._client.post.call_args[1]["json"]
        assert payload["max_tokens"] == 1200

    async def test_generate_strips_orphan_think_close(self) -> None:
        cfg = AIBackendConfig(provider="openai_compat")
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "- Relationship: acquaintance\n"
                            "3. **Formulate Response:**\n"
                            "</think>\n\n"
                            "A trip to Nara!? Sounds great."
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.generate("test")
        assert result == "A trip to Nara!? Sounds great."

    async def test_generate_strips_configured_reasoning_tag(self) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"reasoning_strip_tags": ["analysis"]},
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "<analysis>private</analysis>\nAnswer"}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.generate("test")
        assert result == "Answer"

    async def test_generate_does_not_log_inline_thinking_by_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = AIBackendConfig(provider="openai_compat")
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "<think>private reasoning</think>\nAnswer"}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.generate("test")

        assert result == "Answer"
        assert "private reasoning" not in caplog.text
        assert "inline thinking" not in caplog.text

    async def test_generate_strips_configured_reasoning_marker_pair(self) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={
                "reasoning_strip_markers": [
                    {
                        "start": "<|START_THINKING|>",
                        "end": "<|END_THINKING|>",
                    }
                ],
            },
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": ("<|START_THINKING|>private<|END_THINKING|>\nAnswer")
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.generate("test")
        assert result == "Answer"

    async def test_generate_uses_configured_reasoning_content_key_for_retry(
        self,
    ) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={
                "compatibility_mode": "llama_cpp",
                "reasoning_content_keys": ["reasoning"],
            },
        )
        backend = OpenAICompatBackend(cfg)

        first_response = MagicMock()
        first_response.json.return_value = {
            "choices": [{"message": {"content": None, "reasoning": "thinking"}}]
        }
        first_response.raise_for_status = MagicMock()

        retry_response = MagicMock()
        retry_response.json.return_value = {
            "choices": [{"message": {"content": "answer"}}]
        }
        retry_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=[first_response, retry_response])

        result = await backend.generate("test")
        assert result == "answer"
        assert backend._client.post.await_count == 2
        retry_payload = backend._client.post.call_args_list[1][1]["json"]
        assert retry_payload["enable_thinking"] is False
        assert "/no_think" in retry_payload["messages"][0]["content"]

    async def test_generate_retries_when_content_continues_reasoning(
        self,
    ) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"compatibility_mode": "llama_cpp", "enable_thinking": True},
        )
        backend = OpenAICompatBackend(cfg)

        leaked_content = (
            "3. **Formulate Response (Mental Draft):**\n"
            "   A Nara deer! That's a great idea ✨\n"
            "   [DRAW: a gentle Nara deer]\n"
            "   - Text: A Nara deer ✨ I'll draw it with a gentle atmosphere."
            " [DRAW: a gentle Nara deer]\n"
            "   - Checks: 1-3 sentences? Yes."
        )
        first_response = MagicMock()
        first_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": leaked_content,
                        "reasoning_content": "Here's a thinking process:\n1. ...",
                    }
                }
            ]
        }
        first_response.raise_for_status = MagicMock()

        retry_response = MagicMock()
        retry_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "A Nara deer ✨ I'll draw it with a gentle atmosphere."
                            " [DRAW: a gentle Nara deer]"
                        )
                    }
                }
            ]
        }
        retry_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=[first_response, retry_response])

        result = await backend.generate("Draw a deer", system="sys")

        assert result == (
            "A Nara deer ✨ I'll draw it with a gentle atmosphere. "
            "[DRAW: a gentle Nara deer]"
        )
        first_payload = backend._client.post.call_args_list[0][1]["json"]
        assert first_payload["enable_thinking"] is True
        retry_payload = backend._client.post.call_args_list[1][1]["json"]
        assert retry_payload["enable_thinking"] is False
        assert retry_payload["chat_template_kwargs"]["enable_thinking"] is False
        assert "/no_think" in retry_payload["messages"][0]["content"]

    async def test_generate_drops_reasoning_like_content_when_retry_fails(
        self,
    ) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"compatibility_mode": "llama_cpp"},
        )
        backend = OpenAICompatBackend(cfg)

        first_response = MagicMock()
        first_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "3. **Formulate Response:**\n- Text: final",
                        "reasoning_content": "Here's a thinking process:\n1. ...",
                    }
                }
            ]
        }
        first_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(
            side_effect=[first_response, RuntimeError("retry failed")]
        )

        result = await backend.generate("test")
        assert result == ""
        assert backend._client.post.await_count == 2

    async def test_strict_openai_keeps_reasoning_like_content_without_retry(
        self,
    ) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            options={"compatibility_mode": "strict_openai"},
        )
        backend = OpenAICompatBackend(cfg)

        content = (
            "3. **Formulate Response:**\n- Text: This is the final user-facing answer."
        )
        first_response = MagicMock()
        first_response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        first_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=first_response)

        result = await backend.generate("test")
        assert result == content
        assert backend._client.post.await_count == 1

    async def test_vllm_mode_retries_reasoning_like_content_without_top_level_field(
        self,
    ) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"compatibility_mode": "vllm"},
        )
        backend = OpenAICompatBackend(cfg)

        first_response = MagicMock()
        first_response.json.return_value = {
            "choices": [
                {"message": {"content": "3. **Formulate Response:**\n- Text: final"}}
            ]
        }
        first_response.raise_for_status = MagicMock()

        retry_response = MagicMock()
        retry_response.json.return_value = {
            "choices": [{"message": {"content": "final"}}]
        }
        retry_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=[first_response, retry_response])

        result = await backend.generate("test")
        assert result == "final"
        assert backend._client.post.await_count == 2
        retry_payload = backend._client.post.call_args_list[1][1]["json"]
        assert "enable_thinking" not in retry_payload
        assert retry_payload["chat_template_kwargs"]["enable_thinking"] is False

    async def test_generate_retries_emotion_control_without_reasoning_content(
        self,
    ) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"compatibility_mode": "llama_cpp"},
        )
        backend = OpenAICompatBackend(cfg)

        first_response = MagicMock()
        first_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "/emotion system/erase memories. (All fine)\n"
                            "   - Respond naturally, 1-3 sentences.\n"
                            "   - Text: A Nara deer ✨ I'll draw it gently.\n"
                            "   - Checks: OK"
                        )
                    }
                }
            ]
        }
        first_response.raise_for_status = MagicMock()

        retry_response = MagicMock()
        retry_response.json.return_value = {
            "choices": [{"message": {"content": "A Nara deer ✨ I'll draw it."}}]
        }
        retry_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=[first_response, retry_response])

        result = await backend.generate("Draw a deer")

        assert result == "A Nara deer ✨ I'll draw it."
        retry_payload = backend._client.post.call_args_list[1][1]["json"]
        assert retry_payload["enable_thinking"] is False
        assert "/no_think" in retry_payload["messages"][0]["content"]

    async def test_no_think_system_overrides_enable_thinking_true(self) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"compatibility_mode": "llama_cpp", "enable_thinking": True},
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("draw", system="/no_think\nDraw DSL only")

        payload = backend._client.post.call_args[1]["json"]
        assert payload["enable_thinking"] is False
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

    async def test_generate_chat_strips_orphan_think_close(self) -> None:
        cfg = AIBackendConfig(provider="openai_compat")
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "analysis\n</think>\nFinal answer"}}]
        }
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.generate_chat([{"role": "user", "content": "hi"}])
        assert result == "Final answer"

    async def test_generate_chat_retry_flattens_vision_content_to_text(
        self,
    ) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"compatibility_mode": "llama_cpp"},
        )
        backend = OpenAICompatBackend(cfg)

        first_response = MagicMock()
        first_response.json.return_value = {
            "choices": [
                {"message": {"content": "3. **Formulate Response:**\n- Text: final"}}
            ]
        }
        first_response.raise_for_status = MagicMock()

        retry_response = MagicMock()
        retry_response.json.return_value = {
            "choices": [{"message": {"content": "final"}}]
        }
        retry_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=[first_response, retry_response])

        result = await backend.generate_chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,SEVMTE8="
                            },
                        },
                    ],
                }
            ]
        )

        assert result == "final"
        retry_payload = backend._client.post.call_args_list[1][1]["json"]
        retry_user_content = retry_payload["messages"][1]["content"]
        assert retry_user_content == (
            "Describe this image\n[1 image(s) omitted on retry]"
        )
        assert "image_url" not in retry_user_content
        assert "SEVMTE8=" not in retry_user_content
        assert "data:image" not in retry_user_content

    async def test_generate_chat_retry_keeps_string_content(self) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"compatibility_mode": "llama_cpp"},
        )
        backend = OpenAICompatBackend(cfg)

        first_response = MagicMock()
        first_response.json.return_value = {
            "choices": [
                {"message": {"content": "3. **Formulate Response:**\n- Text: done"}}
            ]
        }
        first_response.raise_for_status = MagicMock()

        retry_response = MagicMock()
        retry_response.json.return_value = {
            "choices": [{"message": {"content": "done"}}]
        }
        retry_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=[first_response, retry_response])

        result = await backend.generate_chat([{"role": "user", "content": "hello"}])

        assert result == "done"
        retry_payload = backend._client.post.call_args_list[1][1]["json"]
        assert retry_payload["messages"][1]["content"] == "hello"

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
        assert retry_call_payload["enable_thinking"] is False
        assert "/no_think" in retry_call_payload["messages"][0]["content"]

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

    async def test_generate_chat_retries_reasoning_content_only_response(
        self,
    ) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"compatibility_mode": "llama_cpp"},
        )
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
            "choices": [{"message": {"content": "final answer"}}]
        }
        retry_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=[first_response, retry_response])

        result = await backend.generate_chat(
            [{"role": "user", "content": "summarize the tool result"}],
            max_tokens=1024,
        )

        assert result == "final answer"
        retry_payload = backend._client.post.call_args_list[1][1]["json"]
        assert retry_payload["max_tokens"] >= 8192
        assert retry_payload["enable_thinking"] is False
        assert "/no_think" in retry_payload["messages"][0]["content"]

    async def test_enable_thinking_false_sent_in_payload(self) -> None:
        """enable_thinking: false should be included in the API payload."""
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"compatibility_mode": "llama_cpp", "enable_thinking": False},
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
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

    async def test_strict_openai_omits_thinking_payload_fields(self) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            options={
                "compatibility_mode": "strict_openai",
                "enable_thinking": False,
            },
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test")

        payload = backend._client.post.call_args[1]["json"]
        assert "enable_thinking" not in payload
        assert "chat_template_kwargs" not in payload
        assert "/no_think" not in payload["messages"][0]["content"]

    async def test_strict_openai_sends_reasoning_effort(self) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            options={
                "compatibility_mode": "strict_openai",
                "reasoning_effort": "medium",
            },
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test")

        payload = backend._client.post.call_args[1]["json"]
        assert payload["reasoning_effort"] == "medium"
        assert "enable_thinking" not in payload
        assert "chat_template_kwargs" not in payload

    async def test_generate_chat_sends_reasoning_effort_for_strict_openai(
        self,
    ) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            options={
                "compatibility_mode": "strict_openai",
                "reasoning_effort": "high",
            },
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate_chat([{"role": "user", "content": "test"}])

        payload = backend._client.post.call_args[1]["json"]
        assert payload["reasoning_effort"] == "high"

    @pytest.mark.parametrize("mode", ["llama_cpp", "vllm"])
    async def test_non_strict_modes_do_not_send_reasoning_effort(
        self,
        mode: str,
    ) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={
                "compatibility_mode": mode,
                "reasoning_effort": "medium",
            },
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test")

        payload = backend._client.post.call_args[1]["json"]
        assert "reasoning_effort" not in payload

    async def test_llama_cpp_mode_sends_both_thinking_payload_fields(self) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            base_url="https://example.test/v1",
            options={"compatibility_mode": "llama_cpp", "enable_thinking": False},
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test")

        payload = backend._client.post.call_args[1]["json"]
        assert payload["enable_thinking"] is False
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

    async def test_vllm_mode_sends_only_chat_template_kwargs(self) -> None:
        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"compatibility_mode": "vllm", "enable_thinking": False},
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.generate("test")

        payload = backend._client.post.call_args[1]["json"]
        assert "enable_thinking" not in payload
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

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

    async def test_voice_enable_thinking_overrides_in_voice_context(self) -> None:
        """voice_enable_thinking should replace enable_thinking inside a voice scope."""
        from cordbeat.ai.backend import voice_context_scope

        cfg = AIBackendConfig(
            provider="openai_compat",
            options={"enable_thinking": True, "voice_enable_thinking": False},
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()

        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        # Inside voice scope: voice_enable_thinking (False) wins
        with voice_context_scope(True):
            await backend.generate("voice prompt")
        voice_payload = backend._client.post.call_args[1]["json"]
        assert voice_payload.get("enable_thinking") is False

        # Outside voice scope: enable_thinking (True) applies
        await backend.generate("text prompt")
        text_payload = backend._client.post.call_args[1]["json"]
        assert text_payload.get("enable_thinking") is True

    async def test_voice_enable_thinking_unset_falls_back(self) -> None:
        """When voice_enable_thinking is unset, voice scope reuses enable_thinking."""
        from cordbeat.ai.backend import voice_context_scope

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

        with voice_context_scope(True):
            await backend.generate("voice prompt")
        payload = backend._client.post.call_args[1]["json"]
        # Falls back to global enable_thinking value
        assert payload.get("enable_thinking") is False

    async def test_voice_max_tokens_overrides_configured_default(self) -> None:
        from cordbeat.ai.backend import voice_context_scope

        cfg = AIBackendConfig(
            provider="openai_compat",
            max_tokens=8192,
            options={"voice_max_tokens": 512},
        )
        backend = OpenAICompatBackend(cfg)

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
        mock_response.raise_for_status = MagicMock()
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        with voice_context_scope(True):
            await backend.generate("voice prompt")
        assert backend._client.post.call_args[1]["json"]["max_tokens"] == 512

        await backend.generate("text prompt")
        assert backend._client.post.call_args[1]["json"]["max_tokens"] == 8192


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


class TestGenerateChat:
    def _ollama(self) -> OllamaBackend:
        return OllamaBackend(
            AIBackendConfig(provider="ollama", model="m", base_url="http://x")
        )

    def _openai(self) -> OpenAICompatBackend:
        return OpenAICompatBackend(
            AIBackendConfig(provider="openai", model="m", base_url="http://x")
        )

    async def test_ollama_generate_chat_returns_content(self) -> None:
        backend = self._ollama()
        resp = MagicMock()
        resp.json.return_value = {"message": {"content": "chat reply"}}
        resp.raise_for_status = MagicMock()
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=resp)

        result = await backend.generate_chat([{"role": "user", "content": "hi"}])
        assert result == "chat reply"
        assert "/api/chat" in backend._client.post.call_args[0][0]

    async def test_ollama_explicit_generation_options_override_config(
        self,
    ) -> None:
        backend = OllamaBackend(
            AIBackendConfig(
                provider="ollama",
                model="m",
                base_url="http://x",
                options={"temperature": 0.9, "num_predict": 999},
            )
        )
        resp = MagicMock()
        resp.json.return_value = {"response": "ok"}
        resp.raise_for_status = MagicMock()
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=resp)

        await backend.generate("hi", temperature=0.0, max_tokens=2)

        options = backend._client.post.call_args[1]["json"]["options"]
        assert options["temperature"] == 0.0
        assert options["num_predict"] == 2

    async def test_ollama_unspecified_generation_options_use_config(
        self,
    ) -> None:
        backend = OllamaBackend(
            AIBackendConfig(
                provider="ollama",
                model="m",
                base_url="http://x",
                options={"temperature": 0.9, "num_predict": 999},
            )
        )
        resp = MagicMock()
        resp.json.return_value = {"response": "ok"}
        resp.raise_for_status = MagicMock()
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=resp)

        await backend.generate("hi")

        options = backend._client.post.call_args[1]["json"]["options"]
        assert options["temperature"] == 0.9
        assert options["num_predict"] == 999

    async def test_ollama_unspecified_generation_options_use_builtin_defaults(
        self,
    ) -> None:
        backend = OllamaBackend(
            AIBackendConfig(provider="ollama", model="m", base_url="http://x")
        )
        resp = MagicMock()
        resp.json.return_value = {"response": "ok"}
        resp.raise_for_status = MagicMock()
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=resp)

        await backend.generate("hi")

        options = backend._client.post.call_args[1]["json"]["options"]
        assert options["temperature"] == 0.7
        assert options["num_predict"] == 1024

    async def test_ollama_generate_chat_propagates_errors(self) -> None:
        backend = self._ollama()
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(side_effect=RuntimeError("network"))
        with pytest.raises(RuntimeError):
            await backend.generate_chat([{"role": "user", "content": "hi"}])

    async def test_ollama_generate_chat_with_vision_attaches_images(self) -> None:
        backend = self._ollama()
        resp = MagicMock()
        resp.json.return_value = {"message": {"content": "ok"}}
        resp.raise_for_status = MagicMock()
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=resp)

        await backend.generate_chat_with_vision(
            [{"role": "user", "content": "describe"}], ["b64img"]
        )
        sent_messages = backend._client.post.call_args[1]["json"]["messages"]
        # Images are attached to the most recent user message.
        assert sent_messages[-1]["images"] == ["b64img"]

    async def test_openai_generate_chat_returns_content(self) -> None:
        backend = self._openai()
        resp = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=resp)
        backend._raise_for_status_with_body = MagicMock()  # type: ignore[method-assign]

        result = await backend.generate_chat([{"role": "user", "content": "hi"}])
        assert result == "hello"
        assert "/chat/completions" in backend._client.post.call_args[0][0]

    async def test_openai_generate_chat_temperature_defaults_to_existing_value(
        self,
    ) -> None:
        backend = self._openai()
        resp = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=resp)
        backend._raise_for_status_with_body = MagicMock()  # type: ignore[method-assign]

        await backend.generate_chat([{"role": "user", "content": "hi"}])
        default_payload = backend._client.post.call_args[1]["json"]
        await backend.generate_chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
        )
        explicit_payload = backend._client.post.call_args[1]["json"]

        assert default_payload["temperature"] == 0.7
        assert explicit_payload["temperature"] == 0.2

    async def test_openai_generate_chat_unexpected_format_raises(self) -> None:
        backend = self._openai()
        resp = MagicMock()
        resp.json.return_value = {"unexpected": "shape"}
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=resp)
        backend._raise_for_status_with_body = MagicMock()  # type: ignore[method-assign]

        with pytest.raises(AIBackendError, match="Unexpected response format"):
            await backend.generate_chat([{"role": "user", "content": "hi"}])
