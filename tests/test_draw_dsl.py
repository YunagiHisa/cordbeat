"""Tests for the Draw skill contract used by the core integration."""

import importlib.util
from pathlib import Path

from cordbeat.skills import draw_dsl
from cordbeat.skills.draw_dsl import (
    build_capability_prompt,
    build_generation_request,
    normalize,
)


def _load_renderer():
    """Import the sandboxed draw renderer the way the skill runner would."""

    path = Path(__file__).parent.parent / "skills" / "draw" / "main.py"
    spec = importlib.util.spec_from_file_location("_draw_renderer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_reports_only_lost_dsl_commands() -> None:
    result = normalize(
        "Here is the drawing:\n"
        "SIZE matters in this drawing.\n"
        "```draw\n"
        "# layered shape\n"
        "SIZE 400 400\n"
        "CANVAS white\n"
        "CIRCLE 200 200\n"
        "BAD_COMMAND 1 2 3\n"
        "CIRCLE 200 200 80 red FILL\n"
        "```\n"
        "Done."
    )

    assert "CIRCLE 200 200 80 red FILL" in result.normalized_dsl
    assert result.normalized_dsl.endswith("\nOUTPUT")
    assert len(result.validation_issues) == 2
    assert "CIRCLE has missing arguments" in result.validation_issues[0]
    assert "unknown Draw command" in result.validation_issues[1]


def test_capability_prompt_requests_renderer_spec_not_image_prompt() -> None:
    prompt = build_capability_prompt()

    assert "intermediate renderer specification" in prompt
    assert "not an image-generation prompt" in prompt
    assert "subject=<main subject>" in prompt


def test_generation_request_owns_dsl_grammar_and_retry_feedback() -> None:
    system, prompt = build_generation_request(
        "subject=red circle; layout=centered",
        retry_reason="CIRCLE has missing arguments",
    )

    assert "Available commands" in system
    assert "CIRCLE <cx> <cy> <radius>" in system
    assert "never exceed 200 total" in system
    assert "Previous attempt failed" in prompt
    assert "CIRCLE has missing arguments" in prompt


def test_generation_prompt_lists_every_canonical_command() -> None:
    system, _ = build_generation_request("subject=red circle")

    # The prompt's command list is derived from _COMMAND_SPECS, so every
    # opcode the normalizer accepts must be advertised to the generator.
    for opcode in draw_dsl._SAFE_OPCODES:
        assert opcode in system


def test_renderer_dispatch_matches_canonical_spec() -> None:
    """The sandboxed renderer must implement exactly the canonical grammar.

    The renderer (skills/draw/main.py) cannot import draw_dsl at runtime, so
    this contract test is what keeps the two grammar definitions from drifting.
    """
    renderer = _load_renderer()
    dispatch_opcodes = set(renderer._DrawDSL()._dispatch)

    # The renderer dispatches every safe opcode except the control-flow words
    # (REPEAT/END are expanded before dispatch) and adds the renderer-only
    # opcodes (SAVE) the normalizer intentionally strips.
    expected = (
        draw_dsl._SAFE_OPCODES - draw_dsl._CONTROL_FLOW_OPCODES
    ) | draw_dsl.RENDERER_ONLY_OPCODES
    assert dispatch_opcodes == expected


def test_oversized_dsl_is_truncated_not_rejected() -> None:
    raw = "\n".join(f"CIRCLE {i} {i} 5 red FILL" for i in range(400))
    result = normalize(raw)

    assert result.truncated is True
    assert result.validation_issues == ()  # truncation is not a content-loss issue
    assert result.normalized_dsl.endswith("\nOUTPUT")
    # Capped near MAX_AUTO_LINES (plus inserted SIZE/CANVAS/OUTPUT scaffolding).
    body = [ln for ln in result.normalized_dsl.splitlines() if ln.startswith("CIRCLE")]
    assert len(body) == draw_dsl.MAX_AUTO_LINES


def test_truncated_repeat_block_is_closed_without_validation_issue() -> None:
    raw = "\n".join(
        ["REPEAT 3"] + [f"CIRCLE {i} {i} 5 red FILL" for i in range(400)]
    )

    result = normalize(raw)

    assert result.truncated is True
    assert not any(
        "REPEAT without matching END" in issue
        for issue in result.validation_issues
    )
    lines = result.normalized_dsl.splitlines()
    assert lines.count("REPEAT 3") == lines.count("END")
    assert lines[-1] == "OUTPUT"


def test_balanced_repeat_under_limit_is_unchanged() -> None:
    result = normalize(
        "SIZE 100 100\n"
        "CANVAS white\n"
        "REPEAT 2\n"
        "CIRCLE 10 10 5 red FILL\n"
        "END\n"
    )

    assert result.truncated is False
    assert result.validation_issues == ()
    assert "REPEAT 2\nCIRCLE 10 10 5 red FILL\nEND" in result.normalized_dsl


def test_normalize_reports_unbalanced_repeat_blocks() -> None:
    result = normalize(
        "REPEAT 3\n"
        "CIRCLE 100 100 20 red FILL\n"
    )

    assert "REPEAT without matching END" in result.validation_issues[0]


def test_normalize_rejects_polygon_with_too_few_points() -> None:
    result = normalize("POLYGON 10 10 20 20 red FILL")

    assert result.normalized_dsl == ""
    assert "POLYGON has missing arguments" in result.validation_issues[0]
