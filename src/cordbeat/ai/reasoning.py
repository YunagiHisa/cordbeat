"""Helpers for handling model reasoning artifacts."""

from __future__ import annotations

import json
import re
from typing import Any

_DEFAULT_REASONING_STRIP_TAGS = ("think", "thought")
_REASONING_LEAK_RE = re.compile(
    r"(?is)"
    r"(^\s*/(?:emotion|memory|system)\b|"
    r"\*\*?(analy[sz]e|check|formulate|refine|draft|self-correction)|"
    r"\b(mental draft|thinking process|self-correction|output matches response|"
    r"check constraints|check rules|tool usage|the prompt says|"
    r"i (must|should|need to|will just)|respond naturally)\b)"
)
_FINAL_TEXT_RE = re.compile(
    r"(?ims)(?:^|\n)\s*-\s*Text:\s*(.+?)(?:\n\s*-\s*Checks:|\Z)"
)


def strip_thinking_text(
    raw: str,
    *,
    tags: tuple[str, ...] = _DEFAULT_REASONING_STRIP_TAGS,
    marker_pairs: tuple[tuple[str, str], ...] = (),
) -> str:
    """Remove complete and malformed thinking-tag output from model text."""

    text = raw
    for start, end in marker_pairs:
        block_re = re.compile(
            f"{re.escape(start)}.*?{re.escape(end)}",
            re.DOTALL | re.IGNORECASE,
        )
        text = block_re.sub("", text)
        if end.lower() in text.lower():
            text = re.split(re.escape(end), text, flags=re.IGNORECASE)[-1]
        if start.lower() in text.lower():
            text = re.split(re.escape(start), text, maxsplit=1, flags=re.IGNORECASE)[0]

    for tag in tags:
        escaped_tag = re.escape(tag)
        block_re = re.compile(
            rf"<{escaped_tag}\b[^>]*>.*?</{escaped_tag}\s*>",
            re.DOTALL | re.IGNORECASE,
        )
        close_re = re.compile(rf"</{escaped_tag}\s*>", re.IGNORECASE)
        open_re = re.compile(rf"<{escaped_tag}\b[^>]*>", re.IGNORECASE)
        text = block_re.sub("", text)
        if close_re.search(text):
            text = close_re.split(text)[-1]
        if open_re.search(text):
            text = open_re.split(text, maxsplit=1)[0]
    return text.strip()


def looks_like_reasoning_text(text: str) -> bool:
    """Return True when text appears to be private reasoning/checklist output."""

    return bool(_REASONING_LEAK_RE.search(text))


def sanitize_reasoning_artifacts(raw: str) -> str:
    """Remove leaked reasoning artifacts from text used in prompts/history."""

    text = strip_thinking_text(raw)
    if not looks_like_reasoning_text(text):
        return text
    match = _FINAL_TEXT_RE.search(text)
    if match:
        candidate = strip_thinking_text(match.group(1))
        if candidate and not looks_like_reasoning_text(candidate):
            return candidate
    return ""


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object from plain, fenced, or explanatory output."""

    cleaned = sanitize_reasoning_artifacts(raw).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise json.JSONDecodeError("No JSON object found", cleaned, 0)
