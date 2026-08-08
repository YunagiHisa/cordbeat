"""Tests for generic speech styling and chunk preparation."""

from __future__ import annotations

import io
import wave

from cordbeat.ai.speech import (
    SpeechStyle,
    chunk_speech_text,
    compile_voice_prompt,
    merge_wav_chunks,
    normalize_for_speech,
    parse_speech_direction,
)
from cordbeat.config import SpeechDirectionConfig, VoiceProfileConfig


def _wav(frames: bytes, *, rate: int = 48_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return output.getvalue()


def test_normalize_for_speech_removes_chat_markup() -> None:
    text = "**Important:** see [docs](https://example.com) and `config.yaml` 😊"

    assert normalize_for_speech(text) == "Important: see docs and config.yaml"


def test_chunk_speech_text_uses_sentence_boundaries_and_maximum() -> None:
    text = "First one here! Second is longer? Last!"

    chunks = chunk_speech_text(text, min_chars=8, max_chars=22)

    # Sentences are stripped before being regrouped, so the spaces that
    # separated them do not survive the round trip.
    assert "".join(chunks) == text.replace("! ", "!").replace("? ", "?")
    assert all(len(chunk) <= 22 for chunk in chunks)
    assert chunks[0].endswith("!")


def test_chunk_speech_text_preserves_decimal_and_shortens_url() -> None:
    chunks = chunk_speech_text(
        "Version 4.2 is documented at https://example.com/very/long/path. Continue.",
        min_chars=5,
        max_chars=30,
    )

    joined = "".join(chunks)
    assert "4.2" in joined
    assert "https://" not in joined
    assert "link" in joined


def test_compile_voice_prompt_keeps_identity_and_adds_language_and_direction() -> None:
    profile = VoiceProfileConfig(description="A warm and approachable adult voice.")
    style = SpeechStyle(
        language="ja",
        direction="Explain the conclusion clearly and gently.",
    )

    prompt = compile_voice_prompt(profile, style)

    assert prompt.startswith("A warm and approachable adult voice.")
    assert "Speak naturally and fluently in Japanese." in prompt
    assert prompt.endswith("Explain the conclusion clearly and gently.")


def test_parse_speech_direction_validates_and_limits_output() -> None:
    config = SpeechDirectionConfig(max_direction_chars=20)

    style = parse_speech_direction(
        '{"intent":"reassure","tone":"gentle","pace":"slightly_slow",'
        '"energy":1.5,"direction":"Speak gently and make the answer clear."}',
        language="ja",
        config=config,
    )

    assert style.intent == "reassure"
    assert style.pace == "slightly_slow"
    assert style.energy == 1.0
    assert len(style.direction) == 20


def test_merge_wav_chunks_combines_pcm_frames() -> None:
    merged = merge_wav_chunks([_wav(b"\x01\x00" * 4), _wav(b"\x02\x00" * 6)])

    with wave.open(io.BytesIO(merged), "rb") as reader:
        assert reader.getframerate() == 48_000
        assert reader.getnchannels() == 1
        assert reader.getnframes() == 10
