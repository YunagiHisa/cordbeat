"""Generic speech styling, text preparation, and WAV helpers."""

from __future__ import annotations

import io
import json
import re
import unicodedata
import wave
from dataclasses import dataclass
from typing import Any

from cordbeat.config import SpeechDirectionConfig, VoiceProfileConfig

SPEECH_DIRECTOR_SYSTEM_PROMPT = """\
You are a speech director for a conversational voice assistant.

Create a short delivery direction for the assistant's completed response using:
- the recent conversation,
- the user's latest message,
- the assistant's completed response,
- the assistant's current emotional and personality state,
- and the configured voice profile.

The direction must reflect both the emotional context and the communicative
purpose of the response. Consider whether the assistant is explaining,
reassuring, celebrating, empathizing, warning, joking, asking, informing, or
reflecting.

Rules:
- Do not rewrite or summarize the assistant's response.
- Do not change the configured voice identity, gender, age, accent, or language.
- Keep the delivery natural and conversational.
- Avoid theatrical or exaggerated acting.
- Use one overall direction for the entire response.
- Return valid JSON only, without a Markdown code fence.

Return this exact shape:
{
  "intent": "one allowed intent",
  "tone": "a short descriptive phrase",
  "pace": "slow | slightly_slow | normal | slightly_fast | fast",
  "energy": 0.0,
  "direction": "a concise natural-language delivery instruction"
}
"""

_INTENTS = frozenset(
    {
        "explain",
        "reassure",
        "celebrate",
        "empathize",
        "warn",
        "joke",
        "ask",
        "inform",
        "reflect",
    }
)
_PACES = frozenset({"slow", "slightly_slow", "normal", "slightly_fast", "fast"})
_LANGUAGE_NAMES = {
    "ja": "Japanese",
    "jp": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
}


@dataclass(frozen=True)
class SpeechStyle:
    """Backend-independent directions attached to one assistant response."""

    language: str = ""
    intent: str = "inform"
    tone: str = "natural and conversational"
    pace: str = "normal"
    energy: float = 0.5
    direction: str = "Speak in a natural, conversational manner."

    def as_metadata(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "intent": self.intent,
            "tone": self.tone,
            "pace": self.pace,
            "energy": self.energy,
            "direction": self.direction,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> SpeechStyle | None:
        if not isinstance(metadata, dict):
            return None
        raw = metadata.get("speech_direction")
        if not isinstance(raw, dict):
            return None
        try:
            energy = max(0.0, min(1.0, float(raw.get("energy", 0.5))))
        except (TypeError, ValueError):
            energy = 0.5
        return cls(
            language=str(raw.get("language") or "").strip(),
            intent=str(raw.get("intent") or "inform").strip(),
            tone=str(raw.get("tone") or "natural and conversational").strip(),
            pace=str(raw.get("pace") or "normal").strip(),
            energy=energy,
            direction=str(
                raw.get("direction") or "Speak in a natural, conversational manner."
            ).strip(),
        )


def resolve_speech_language(
    profile: VoiceProfileConfig,
    soul_language: str,
) -> str:
    source = (profile.language_source or "soul").strip().casefold()
    if source == "auto":
        return ""
    if source == "fixed":
        return profile.language.strip()
    return (soul_language or profile.language).strip()


def build_speech_director_prompt(
    *,
    voice_profile: str,
    soul_snapshot: dict[str, Any],
    recent_conversation: list[dict[str, Any]],
    user_message: str,
    assistant_response: str,
) -> str:
    emotion = soul_snapshot.get("emotion") or {}
    history_lines: list[str] = []
    for item in recent_conversation:
        role = str(item.get("role") or "unknown").strip().title()
        content = str(item.get("content") or "").strip().replace("\x00", "")
        if content:
            history_lines.append(f"{role}: {content[:1000]}")
    recent = "\n".join(history_lines) or "(none)"
    traits = ", ".join(str(value) for value in soul_snapshot.get("traits") or [])
    secondary = str(emotion.get("secondary") or "none")
    secondary_intensity = float(emotion.get("secondary_intensity") or 0.0)
    return f"""\
Voice profile:
{voice_profile}

Assistant state:
- Personality traits: {traits or "unspecified"}
- Primary emotion: {emotion.get("primary", "calm")}
- Primary intensity: {float(emotion.get("intensity") or 0.0):.2f}
- Secondary emotion: {secondary}
- Secondary intensity: {secondary_intensity:.2f}

Recent conversation:
{recent}

Latest user message:
{user_message[:2000]}

Completed assistant response:
{assistant_response[:4000]}

Create the delivery direction.
"""


def parse_speech_direction(
    raw: str,
    *,
    language: str,
    config: SpeechDirectionConfig,
) -> SpeechStyle:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("speech direction response must be a JSON object")
    intent = str(data.get("intent") or "inform").strip().casefold()
    if intent not in _INTENTS:
        intent = "inform"
    pace = str(data.get("pace") or "normal").strip().casefold()
    if pace not in _PACES:
        pace = "normal"
    try:
        energy = max(0.0, min(1.0, float(data.get("energy", 0.5))))
    except (TypeError, ValueError):
        energy = 0.5
    tone = _single_line(str(data.get("tone") or "natural and conversational"))[:80]
    direction = _single_line(str(data.get("direction") or config.fallback))[
        : max(1, config.max_direction_chars)
    ]
    if not direction:
        direction = config.fallback
    return SpeechStyle(
        language=language,
        intent=intent,
        tone=tone,
        pace=pace,
        energy=energy,
        direction=direction,
    )


def fallback_speech_style(
    *,
    language: str,
    soul_snapshot: dict[str, Any],
    assistant_response: str,
    fallback: str,
) -> SpeechStyle:
    emotion = str((soul_snapshot.get("emotion") or {}).get("primary") or "calm")
    directions = {
        "joy": "Speak with genuine warmth and a light sense of happiness.",
        "excitement": "Speak with lively interest while staying natural and clear.",
        "curiosity": (
            "Speak with engaged curiosity and a natural conversational rhythm."
        ),
        "warmth": "Speak gently, warmly, and with an approachable tone.",
        "worry": "Speak carefully and reassuringly without sounding alarming.",
        "loneliness": (
            "Speak softly with quiet warmth and appreciation for the conversation."
        ),
        "sadness": "Speak softly and sincerely with restrained emotion.",
        "boredom": "Speak in a calm, low-energy manner without sounding dismissive.",
        "calm": fallback,
    }
    intent = "ask" if assistant_response.rstrip().endswith(("?", "？")) else "inform"
    direction = directions.get(emotion, fallback)
    return SpeechStyle(
        language=language,
        intent=intent,
        tone="natural and conversational",
        pace="normal",
        energy=0.5,
        direction=direction,
    )


def compile_voice_prompt(profile: VoiceProfileConfig, style: SpeechStyle | None) -> str:
    parts = [_single_line(profile.description)]
    language = style.language if style is not None else ""
    if language:
        normalized = language.strip().casefold().split("-", 1)[0]
        name = _LANGUAGE_NAMES.get(normalized, language.strip())
        parts.append(f"Speak naturally and fluently in {name}.")
    if style is not None and style.direction:
        parts.append(_single_line(style.direction))
    return " ".join(part for part in parts if part).strip()


def normalize_for_speech(text: str) -> str:
    """Remove chat markup that should not be spoken aloud."""

    normalized = re.sub(r"```[\s\S]*?```", " ", text)
    normalized = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", normalized)
    normalized = re.sub(r"https?://\S+", "link", normalized)
    normalized = re.sub(r"<@!?\d+>", "mention", normalized)
    normalized = re.sub(r"`([^`]*)`", r"\1", normalized)
    normalized = re.sub(r"[*_~#>]", "", normalized)
    normalized = "".join(
        char for char in normalized if not unicodedata.category(char).startswith("So")
    )
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\s*\n\s*", "。", normalized)
    normalized = re.sub(r"。{2,}", "。", normalized)
    return normalized.strip()


def chunk_speech_text(text: str, *, min_chars: int, max_chars: int) -> list[str]:
    """Split normalized speech text at natural Japanese/Latin boundaries."""

    minimum = max(1, int(min_chars))
    maximum = max(minimum, int(max_chars))
    normalized = normalize_for_speech(text)
    if not normalized:
        return []
    sentences = [
        item.strip()
        for item in re.findall(r"[^。！？!?]+[。！？!?]?", normalized)
        if item.strip()
    ]
    chunks: list[str] = []
    pending = ""
    for sentence in sentences:
        for piece in _split_long_piece(sentence, maximum):
            if not pending:
                pending = piece
            elif len(pending) + len(piece) <= maximum:
                pending += piece
            else:
                chunks.append(pending)
                pending = piece
    if pending:
        chunks.append(pending)
    if len(chunks) > 1 and len(chunks[-1]) < minimum:
        tail = chunks.pop()
        if len(chunks[-1]) + len(tail) <= maximum:
            chunks[-1] += tail
        else:
            chunks.append(tail)
    return chunks


def merge_wav_chunks(chunks: list[bytes]) -> bytes:
    """Join compatible PCM WAV files by frames, never by binary concatenation."""

    if not chunks:
        return b""
    output = io.BytesIO()
    expected: tuple[int, int, int, str, str] | None = None
    frames: list[bytes] = []
    for payload in chunks:
        with wave.open(io.BytesIO(payload), "rb") as reader:
            params = (
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
                reader.getcomptype(),
                reader.getcompname(),
            )
            if expected is None:
                expected = params
            elif params != expected:
                raise ValueError("incompatible WAV chunk formats")
            frames.append(reader.readframes(reader.getnframes()))
    assert expected is not None
    with wave.open(output, "wb") as writer:
        writer.setnchannels(expected[0])
        writer.setsampwidth(expected[1])
        writer.setframerate(expected[2])
        writer.setcomptype(expected[3], expected[4])
        for frame_data in frames:
            writer.writeframes(frame_data)
    return output.getvalue()


def _split_long_piece(text: str, maximum: int) -> list[str]:
    pieces: list[str] = []
    remaining = text
    break_chars = "、，, "
    while len(remaining) > maximum:
        boundary = max(remaining.rfind(char, 0, maximum + 1) for char in break_chars)
        if boundary < max(1, maximum // 2):
            boundary = maximum
        elif remaining[boundary] in break_chars:
            boundary += 1
        pieces.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _single_line(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\x00", " ")).strip()
