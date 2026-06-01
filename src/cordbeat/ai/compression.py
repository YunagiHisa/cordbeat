"""Conversation context compression utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cordbeat.ai.backend import AIBackend

logger = logging.getLogger(__name__)

DEFAULT_COMPRESS_TEMPERATURE = 0.3
DEFAULT_COMPRESS_MAX_TOKENS = 256

_COMPRESS_SYSTEM_PROMPT = """\
You are summarizing an older portion of a conversation between a user and an
AI assistant. Create a concise summary that preserves the key information:
topics discussed, decisions made, user preferences or facts mentioned, and the
emotional tone.
Write the summary in third-person, past tense, 3-5 sentences max.
Start with: "Earlier in this conversation:"
Respond with ONLY the summary text, no JSON or extra formatting.
"""


class ConversationCompressor:
    """Summarises conversation history chunks using LLM."""

    def __init__(
        self,
        ai: AIBackend,
        *,
        temperature: float = DEFAULT_COMPRESS_TEMPERATURE,
        max_tokens: int = DEFAULT_COMPRESS_MAX_TOKENS,
    ) -> None:
        self._ai = ai
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def compress_in_memory(
        self,
        history: list[dict[str, str]],
        chars_threshold: int,
        soul_name: str = "AI",
    ) -> list[dict[str, str]]:
        """Return a (possibly shorter) history list suitable for prompt injection.

        If the total character count of *history* exceeds *chars_threshold*,
        the oldest half is summarised and replaced with a single summary
        entry.  The DB is not modified — this is purely for the current request.
        """
        total_chars = sum(len(m.get("content", "")) for m in history)
        if total_chars <= chars_threshold or len(history) < 4:
            return history

        split = len(history) // 2
        old_chunk = history[:split]
        recent = history[split:]

        summary = await self._summarise_chunk(old_chunk, soul_name)
        if not summary:
            return history  # summarisation failed — return original

        summary_entry: dict[str, str] = {
            "role": "system",
            "content": summary,
        }
        logger.debug(
            "Compressed %d old messages into summary (%d chars → %d chars)",
            len(old_chunk),
            sum(len(m.get("content", "")) for m in old_chunk),
            len(summary),
        )
        return [summary_entry] + recent

    async def compress_chunk_to_text(
        self,
        messages: list[dict[str, str]],
        soul_name: str = "AI",
    ) -> str | None:
        """Summarise *messages* and return summary text, or None on failure."""
        return await self._summarise_chunk(messages, soul_name)

    async def _summarise_chunk(
        self,
        messages: list[dict[str, str]],
        soul_name: str,
    ) -> str | None:
        if not messages:
            return None

        first_ts = messages[0].get("created_at", "")
        last_ts = messages[-1].get("created_at", "")
        time_range = f" ({first_ts[:10]} ~ {last_ts[:10]})" if first_ts else ""

        conversation_text = "\n".join(
            f"{'User' if m['role'] == 'user' else soul_name}: {m.get('content', '')}"
            for m in messages
        )
        prompt = f"Conversation to summarise{time_range}:\n\n{conversation_text}"

        try:
            summary = await self._ai.generate(
                prompt=prompt,
                system=_COMPRESS_SYSTEM_PROMPT,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            return summary.strip() or None
        except Exception:
            logger.exception("Conversation compression LLM call failed")
            return None
