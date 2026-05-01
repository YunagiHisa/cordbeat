"""Tests for the draw skill DSL interpreter."""

from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path
from typing import Any

# Load the draw skill directly via importlib to avoid sys.path conflicts.
_SKILL_DIR = Path(__file__).parent.parent / "skills" / "draw"
_spec = importlib.util.spec_from_file_location(
    "_draw_skill_main", _SKILL_DIR / "main.py"
)
assert _spec is not None and _spec.loader is not None
draw_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(draw_main)  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(commands: str) -> dict[str, Any]:
    """Run the draw DSL synchronously via the public execute() coroutine."""
    import asyncio

    return asyncio.run(draw_main.execute(commands=commands))


# ---------------------------------------------------------------------------
# Basic canvas + output
# ---------------------------------------------------------------------------


def test_empty_commands_returns_error() -> None:
    result = run("   \n  ")
    assert "error" in result


def test_output_returns_base64_png() -> None:
    result = run("SIZE 100 100\nCANVAS white\nOUTPEUT PNG\nOUTPUT PNG")
    # One of the OUTPUT commands succeeds; warnings collect the typo
    assert "output" in result
    data = base64.b64decode(result["output"])
    assert data[:4] == b"\x89PNG"


def test_output_jpeg() -> None:
    result = run("SIZE 64 64\nCANVAS blue\nOUTPUT JPEG")
    assert result.get("format") == "JPEG"
    assert "output" in result
    data = base64.b64decode(result["output"])
    # JPEG magic bytes
    assert data[:2] == b"\xff\xd8"


def test_size_defaults_to_512() -> None:
    result = run("OUTPUT PNG")
    assert "output" in result
    # No warnings about SIZE
    assert all("SIZE" not in w for w in result.get("warnings", []))


def test_canvas_changes_background() -> None:
    result = run("SIZE 10 10\nCANVAS red\nOUTPUT PNG")
    assert "output" in result
    assert not result.get("warnings")


# ---------------------------------------------------------------------------
# Shape commands
# ---------------------------------------------------------------------------


def test_circle_no_fill() -> None:
    result = run("SIZE 200 200\nCANVAS white\nCIRCLE 100 100 40 black\nOUTPUT PNG")
    assert "output" in result
    assert not result.get("warnings")


def test_circle_fill() -> None:
    result = run("SIZE 200 200\nCIRCLE 100 100 40 red FILL\nOUTPUT PNG")
    assert "output" in result


def test_rect_outline() -> None:
    result = run("SIZE 200 200\nRECT 10 10 100 80 blue\nOUTPUT PNG")
    assert "output" in result
    assert not result.get("warnings")


def test_rect_fill() -> None:
    result = run("SIZE 200 200\nRECT 10 10 100 80 green FILL\nOUTPUT PNG")
    assert "output" in result


def test_ellipse() -> None:
    result = run("SIZE 200 200\nELLIPSE 20 20 180 120 purple FILL\nOUTPUT PNG")
    assert "output" in result
    assert not result.get("warnings")


def test_line() -> None:
    result = run("SIZE 200 200\nLINE 0 0 200 200 black 3\nOUTPUT PNG")
    assert "output" in result
    assert not result.get("warnings")


def test_polygon() -> None:
    result = run("SIZE 200 200\nPOLYGON 10 10 100 10 55 90 orange FILL\nOUTPUT PNG")
    assert "output" in result
    assert not result.get("warnings")


def test_star() -> None:
    result = run("SIZE 300 300\nSTAR 150 150 100 50 5 gold FILL\nOUTPUT PNG")
    assert "output" in result
    assert not result.get("warnings")


def test_spiral() -> None:
    result = run("SIZE 300 300\nSPIRAL 150 150 3 120 navy 2\nOUTPUT PNG")
    assert "output" in result
    assert not result.get("warnings")


# ---------------------------------------------------------------------------
# Text command
# ---------------------------------------------------------------------------


def test_text_command() -> None:
    cmds = 'SIZE 200 100\nCANVAS white\nTEXT 10 40 "hello world" black 14\nOUTPUT PNG'
    result = run(cmds)
    assert "output" in result
    assert not result.get("warnings")


def test_text_missing_quote_warns() -> None:
    result = run("SIZE 200 100\nTEXT 10 40 hello black\nOUTPUT PNG")
    assert any("TEXT" in w for w in result.get("warnings", []))


# ---------------------------------------------------------------------------
# Unknown / bad commands → warnings, not errors
# ---------------------------------------------------------------------------


def test_unknown_command_warns() -> None:
    result = run("FOOBAR 1 2 3\nOUTPUT PNG")
    warnings = result.get("warnings", [])
    assert any("FOOBAR" in w for w in warnings)


def test_comment_lines_are_ignored() -> None:
    result = run("# This is a comment\n# Another comment\nOUTPUT PNG")
    assert "output" in result
    assert not result.get("warnings")


def test_size_out_of_range_warns() -> None:
    result = run("SIZE 99999 99999\nCANVAS white\nOUTPUT PNG")
    warnings = result.get("warnings", [])
    assert any("SIZE" in w for w in warnings)


def test_size_missing_args_warns() -> None:
    result = run("SIZE 100\nOUTPUT PNG")
    assert any("SIZE" in w for w in result.get("warnings", []))


# ---------------------------------------------------------------------------
# SAVE command
# ---------------------------------------------------------------------------


def test_save_png(tmp_path: Path) -> None:
    outfile = tmp_path / "test_draw.png"
    result = run(f"SIZE 50 50\nCANVAS yellow\nSAVE {outfile}")
    assert result.get("saved") == str(outfile)
    assert outfile.exists()


def test_save_jpeg(tmp_path: Path) -> None:
    outfile = tmp_path / "test_draw.jpg"
    result = run(f"SIZE 50 50\nCANVAS cyan\nSAVE {outfile}")
    assert result.get("saved") == str(outfile)
    assert outfile.exists()


# ---------------------------------------------------------------------------
# Validator: skill source passes static analysis
# ---------------------------------------------------------------------------


def test_skill_source_passes_validator() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cordbeat.skills.validator",
        Path(__file__).parent.parent / "src" / "cordbeat" / "skills" / "validator.py",
    )
    assert spec is not None and spec.loader is not None
    importlib.util.module_from_spec(spec)
    # Stub out the relative import of exceptions
    import types

    stub_exceptions = types.ModuleType("cordbeat.exceptions")

    class _SkillError(Exception):
        pass

    stub_exceptions.SkillError = _SkillError  # type: ignore[attr-defined]
    sys.modules.setdefault("cordbeat", types.ModuleType("cordbeat"))
    sys.modules.setdefault("cordbeat.exceptions", stub_exceptions)
    # Also make the package-relative import work via the installed path
    # Prefer the already-imported validator from the installed package if available
    from cordbeat.skills.validator import validate_skill_source

    src = (Path(__file__).parent.parent / "skills" / "draw" / "main.py").read_text(
        encoding="utf-8"
    )
    validate_skill_source(src, "draw")  # must not raise
