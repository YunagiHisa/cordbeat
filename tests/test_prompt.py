"""Tests for prompt building utilities."""

from __future__ import annotations

from cordbeat.ai.prompt import (
    MAX_USER_INPUT_LEN,
    build_context,
    build_soul_system_prompt,
    sanitize,
    sanitize_tool_artifacts,
)


class TestSanitize:
    def test_strips_control_chars(self) -> None:
        assert sanitize("hello\x00world\x08") == "helloworld"

    def test_preserves_newlines(self) -> None:
        assert sanitize("hello\nworld") == "hello\nworld"

    def test_strict_strips_hash_and_newlines(self) -> None:
        result = sanitize("bad\nuser#name\r", strict=True)
        assert "#" not in result
        assert "\n" not in result
        assert "\r" not in result
        assert result == "badusername"

    def test_truncates_long_input(self) -> None:
        long_text = "a" * (MAX_USER_INPUT_LEN + 100)
        result = sanitize(long_text)
        assert len(result) == MAX_USER_INPUT_LEN

    def test_sanitize_tool_artifacts_strips_draw_tags(self) -> None:
        result = sanitize_tool_artifacts(
            "I'll draw it. [DRAW: a detailed fantasy image prompt"
        )
        assert result == "I'll draw it."

    def test_sanitize_tool_artifacts_strips_malformed_a_draw_tags(self) -> None:
        result = sanitize_tool_artifacts(
            "Reply [A DRAW: A simple vector drawing with prompt-like details]"
        )
        assert result == "Reply"


class TestBuildSoulSystemPrompt:
    def test_includes_name_and_traits(self) -> None:
        snap = {
            "name": "TestBot",
            "traits": ["curious", "caring"],
            "emotion": {"primary": "calm", "intensity": 0.5},
            "immutable_rules": ["Never lie"],
        }
        result = build_soul_system_prompt(snap)
        assert "TestBot" in result
        assert "curious" in result
        assert "Never lie" in result

    def test_includes_secondary_emotion(self) -> None:
        snap = {
            "name": "Bot",
            "traits": ["brave"],
            "emotion": {
                "primary": "joy",
                "intensity": 0.8,
                "secondary": "curiosity",
                "secondary_intensity": 0.3,
            },
            "immutable_rules": [],
        }
        result = build_soul_system_prompt(snap)
        assert "secondary: curiosity" in result

    def test_includes_soul_notes(self) -> None:
        snap = {
            "name": "TestBot",
            "traits": ["curious"],
            "emotion": {"primary": "calm", "intensity": 0.5},
            "immutable_rules": [],
            "notes": "Speak casually with emoji",
        }
        result = build_soul_system_prompt(snap)
        assert "Character notes:" in result
        assert "Speak casually with emoji" in result

    def test_empty_notes_excluded(self) -> None:
        snap = {
            "name": "TestBot",
            "traits": ["curious"],
            "emotion": {"primary": "calm", "intensity": 0.5},
            "immutable_rules": [],
            "notes": "",
        }
        result = build_soul_system_prompt(snap)
        assert "Character notes:" not in result

    def test_no_notes_key_excluded(self) -> None:
        snap = {
            "name": "TestBot",
            "traits": ["curious"],
            "emotion": {"primary": "calm", "intensity": 0.5},
            "immutable_rules": [],
        }
        result = build_soul_system_prompt(snap)
        assert "Character notes:" not in result

    def test_default_tone_is_friend_like(self) -> None:
        """Default prompt nudges the model away from butler-speak."""
        snap = {
            "name": "TestBot",
            "traits": ["curious"],
            "emotion": {"primary": "calm", "intensity": 0.5},
            "immutable_rules": [],
        }
        result = build_soul_system_prompt(snap)
        assert "talk like a friend" in result.lower()
        assert "butler-speak" in result

    def test_familiarity_stages(self) -> None:
        """Relationship stage label tracks message count."""
        snap = {
            "name": "TestBot",
            "traits": ["curious"],
            "emotion": {"primary": "calm", "intensity": 0.5},
            "immutable_rules": [],
        }
        assert "stranger" in build_soul_system_prompt(snap, user_message_count=0)
        assert "acquaintance" in build_soul_system_prompt(snap, user_message_count=20)
        assert "friend" in build_soul_system_prompt(snap, user_message_count=200)
        assert "close friend" in build_soul_system_prompt(snap, user_message_count=1000)

    def test_familiarity_omitted_when_none(self) -> None:
        """Without a message count, no relationship-stage section is added."""
        snap = {
            "name": "TestBot",
            "traits": ["curious"],
            "emotion": {"primary": "calm", "intensity": 0.5},
            "immutable_rules": [],
        }
        result = build_soul_system_prompt(snap)
        assert "Relationship stage" not in result

    def test_language_rule_allows_task_suitable_tool_arguments(self) -> None:
        snap = {
            "name": "TestBot",
            "traits": ["curious"],
            "emotion": {"primary": "calm", "intensity": 0.5},
            "immutable_rules": [],
            "language": "ja",
        }
        result = build_soul_system_prompt(snap)
        assert "Always respond to the user in ja." in result
        assert "Tool arguments may use the language best suited" in result


class TestBuildContext:
    def test_minimal_context(self) -> None:
        result = build_context(user_display_name="Alice")
        assert "[BEGIN USER CONTEXT]" in result
        assert "User: Alice" in result
        assert "[END USER CONTEXT]" in result

    def test_with_profile(self) -> None:
        result = build_context(
            user_display_name="Alice",
            profile={"age": "25", "hobby": "coding"},
        )
        assert "Known info:" in result
        assert "age=25" in result
        assert "[END USER CONTEXT]" in result

    def test_with_memories(self) -> None:
        result = build_context(
            user_display_name="Alice",
            semantic_memories=[{"content": "Likes Python"}],
            episodic_memories=[{"content": "Got a new job"}],
        )
        assert "[BEGIN RECALLED FACTS]" in result
        assert "Likes Python" in result
        assert "[BEGIN RECALLED EPISODES]" in result
        assert "Got a new job" in result

    def test_recalled_episodes_remove_response_and_near_duplicates(self) -> None:
        result = build_context(
            user_display_name="Alice",
            episodic_memories=[
                {
                    "content": (
                        "[joy] User: 自画像描いてって / Response: "
                        "I'll create an image prompt."
                    )
                },
                {"content": "[joy] User: <@123> アテナの自画像を描いて"},
                {"content": "Alice decided to publish CordBeat."},
            ],
        )

        assert "image prompt" not in result
        assert result.count("自画像") == 1
        assert "Alice decided to publish CordBeat." in result

    def test_recalled_episode_limit_reduces_context_weight(self) -> None:
        result = build_context(
            user_display_name="Alice",
            episodic_memories=[
                {"content": "Alice adopted a cat"},
                {"content": "Alice released CordBeat"},
                {"content": "Alice visited Kyoto"},
            ],
            recalled_episode_limit=2,
        )

        assert "Alice adopted a cat" in result
        assert "Alice released CordBeat" in result
        assert "Alice visited Kyoto" not in result

    def test_recalled_draw_tags_are_removed(self) -> None:
        result = build_context(
            user_display_name="Alice",
            episodic_memories=[
                {
                    "content": (
                        "User asked for art / Response: Sure [DRAW: a majestic "
                        "dragon with cinematic lighting]"
                    )
                }
            ],
        )

        assert "[DRAW:" not in result
        assert "cinematic lighting" not in result
        assert "User asked for art" in result

    def test_with_recall_hints(self) -> None:
        result = build_context(
            user_display_name="Alice",
            recall_hints=["7 days ago Alice talked about: OSS design"],
        )
        assert "[BEGIN RECALL HINTS]" in result
        assert "OSS design" in result

    def test_recall_hints_none_omitted(self) -> None:
        result = build_context(
            user_display_name="Alice",
            recall_hints=None,
        )
        assert "Recall hints:" not in result

    def test_with_history(self) -> None:
        result = build_context(
            user_display_name="Alice",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            soul_name="CordBeat",
        )
        assert "[BEGIN CONVERSATION HISTORY]" in result
        assert "Conversation history:" in result
        assert "User: Hello" in result
        assert "CordBeat: Hi!" in result
        assert "[END CONVERSATION HISTORY]" in result

    def test_history_sanitizes_assistant_reasoning_leak(self) -> None:
        result = build_context(
            user_display_name="Alice",
            history=[
                {"role": "user", "content": "Draw a deer"},
                {
                    "role": "assistant",
                    "content": (
                        "Here's a thinking process:\n"
                        "3. **Formulate Response (Mental Draft):**\n"
                        "   [DRAW: internal draft]\n"
                        "   - Text: A Nara deer ✨ I'll draw it with a gentle "
                        "atmosphere.\n"
                        "   - Checks: OK"
                    ),
                },
            ],
            soul_name="CordBeat",
        )

        assert (
            "CordBeat: A Nara deer ✨ I'll draw it with a gentle atmosphere."
            in result
        )
        assert "thinking process" not in result
        assert "Formulate Response" not in result
        assert "[DRAW: internal draft]" not in result

    def test_none_values_excluded(self) -> None:
        result = build_context(
            user_display_name="Bob",
            profile=None,
            semantic_memories=None,
            episodic_memories=None,
            history=None,
        )
        assert "Known info:" not in result
        assert "Conversation history:" not in result
