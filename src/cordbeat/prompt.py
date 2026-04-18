"""Prompt building utilities — shared by engine and heartbeat."""

from __future__ import annotations

import re
from typing import Any

# Strip control characters that could manipulate prompt structure
_SANITIZE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Stricter pattern that also strips # and newlines (for embedded user data)
_SANITIZE_STRICT_RE = re.compile(r"[#\n\r\x00-\x1f]")

MAX_USER_INPUT_LEN = 2000


def sanitize(
    text: str,
    *,
    strict: bool = False,
    max_len: int = MAX_USER_INPUT_LEN,
) -> str:
    """Remove control characters and truncate for safe prompt use.

    Args:
        text: Raw text to sanitize.
        strict: If True, also strip ``#`` and newlines (for user-controlled
            data embedded inside a prompt section).
        max_len: Maximum length of the returned string.
    """
    pattern = _SANITIZE_STRICT_RE if strict else _SANITIZE_RE
    return pattern.sub("", text)[:max_len]


def build_soul_system_prompt(soul_snap: dict[str, Any]) -> str:
    """Build a system prompt from a soul snapshot."""
    emotion_desc = (
        f"Current emotion: {soul_snap['emotion']['primary']} "
        f"(intensity: {soul_snap['emotion']['intensity']})"
    )
    if "secondary" in soul_snap["emotion"]:
        emotion_desc += (
            f", secondary: {soul_snap['emotion']['secondary']} "
            f"(intensity: {soul_snap['emotion']['secondary_intensity']})"
        )

    prompt = (
        f"You are {soul_snap['name']}. "
        f"Personality: {', '.join(soul_snap['traits'])}. "
        f"{emotion_desc}. "
        f"\nImmutable rules:\n"
        + "\n".join(f"- {r}" for r in soul_snap["immutable_rules"])
        + "\n\nRespond naturally to the user's message. "
        "Keep your response concise."
    )

    language = soul_snap.get("language", "en")
    if language != "en":
        prompt += f"\n\nAlways respond in {language}."

    notes = soul_snap.get("notes", "").strip()
    if notes:
        prompt += f"\n\nCharacter notes:\n{notes}"

    return prompt


def build_context(
    *,
    user_display_name: str,
    profile: dict[str, str] | None = None,
    semantic_memories: list[dict[str, Any]] | None = None,
    episodic_memories: list[dict[str, Any]] | None = None,
    recall_hints: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
    soul_name: str = "",
    max_user_input_len: int = MAX_USER_INPUT_LEN,
) -> str:
    """Assemble the context block from memory and conversation data."""
    parts = [f"User: {sanitize(user_display_name, max_len=max_user_input_len)}"]

    if profile:
        sanitized = ", ".join(
            f"{k}={sanitize(str(v), max_len=max_user_input_len)}"
            for k, v in profile.items()
        )
        parts.append(f"Known info: {sanitized}")

    if semantic_memories:
        parts.append("\nKnown preferences/facts:")
        for mem in semantic_memories:
            parts.append(f"  - {mem['content']}")

    if episodic_memories:
        parts.append("\nRelated past moments:")
        for mem in episodic_memories:
            parts.append(f"  - {mem['content']}")

    if recall_hints:
        parts.append("\nRecall hints:")
        for hint in recall_hints:
            parts.append(f"  - {hint}")

    if history:
        parts.append("\nConversation history:")
        for msg in history:
            prefix = "User" if msg["role"] == "user" else (soul_name or "AI")
            sanitized = sanitize(msg["content"], max_len=max_user_input_len)
            parts.append(f"  {prefix}: {sanitized}")

    return "\n".join(parts)
