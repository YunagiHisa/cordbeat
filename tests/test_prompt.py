"""Tests for prompt building utilities."""

from __future__ import annotations

from cordbeat.prompt import (
    MAX_USER_INPUT_LEN,
    build_context,
    build_soul_system_prompt,
    sanitize,
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


class TestBuildContext:
    def test_minimal_context(self) -> None:
        result = build_context(user_display_name="Alice")
        assert "User: Alice" in result

    def test_with_profile(self) -> None:
        result = build_context(
            user_display_name="Alice",
            profile={"age": "25", "hobby": "coding"},
        )
        assert "Known info:" in result
        assert "age=25" in result

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
        assert "Conversation history:" in result
        assert "User: Hello" in result
        assert "CordBeat: Hi!" in result

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
