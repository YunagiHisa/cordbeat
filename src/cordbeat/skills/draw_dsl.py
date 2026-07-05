"""Draw skill contract, normalization, and LLM-facing guidance."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cordbeat.ai.prompt import sanitize

DRAW_TAG_RE = re.compile(
    r"\[(?:A\s+)?DRAW:\s*(.+?)\]",
    re.DOTALL | re.IGNORECASE,
)

# Hard truncation cap for generated drawings. Sits above the prompt's stated
# budget so the model has margin before its work is trimmed, while still
# bounding the flat command list (each line is a single cheap draw call). It is
# also below the generator's ~350-line token ceiling (max_tokens=3000), so
# detailed portraits that previously hit the old 120 cap now render in full.
MAX_AUTO_LINES = 220
MAX_AUTO_ATTEMPTS = 3


@dataclass(frozen=True)
class _CommandSpec:
    """Canonical definition of one Draw DSL opcode.

    This is the single source of truth for the grammar the normalizer
    enforces and the grammar the generator prompt advertises. The renderer
    (``skills/draw/main.py``) runs in an import-isolated subprocess and so
    cannot share this object at runtime; ``tests/test_draw_dsl.py`` instead
    asserts the renderer's dispatch table matches these opcodes, which keeps
    the two definitions from drifting.
    """

    signature: str
    """Human/LLM-facing form, e.g. ``CIRCLE <cx> <cy> <radius> <color> [FILL]``."""
    min_tokens: int
    """Fewest whitespace tokens (opcode included) a usable line must have."""
    produces_content: bool
    """True when the command can draw something the user should see."""
    numeric_first_arg: bool
    """True when arg 1 must be a number — lets prose be told apart from DSL."""


# Ordered so the generated prompt lists commands in a sensible teaching order.
_COMMAND_SPECS: dict[str, _CommandSpec] = {
    "SIZE": _CommandSpec("SIZE <width> <height>", 3, False, True),
    "CANVAS": _CommandSpec("CANVAS <color>", 2, False, False),
    "CIRCLE": _CommandSpec("CIRCLE <cx> <cy> <radius> <color> [FILL]", 5, True, True),
    "RECT": _CommandSpec("RECT <x1> <y1> <x2> <y2> <color> [FILL]", 6, True, True),
    "ELLIPSE": _CommandSpec(
        "ELLIPSE <x1> <y1> <x2> <y2> <color> [FILL]", 6, True, True
    ),
    "LINE": _CommandSpec("LINE <x1> <y1> <x2> <y2> <color> [width]", 6, True, True),
    "POLYGON": _CommandSpec(
        "POLYGON <x1> <y1> <x2> <y2> <x3> <y3> ... <color> [FILL]",
        8,
        True,
        True,
    ),
    "TEXT": _CommandSpec('TEXT <x> <y> "<text>" <color> [size]', 5, True, True),
    "STAR": _CommandSpec(
        "STAR <cx> <cy> <outer_r> <inner_r> <points> <color> [FILL]", 7, True, True
    ),
    "SPIRAL": _CommandSpec(
        "SPIRAL <cx> <cy> <turns> <max_radius> <color> [width]", 6, True, True
    ),
    "ARC": _CommandSpec(
        "ARC <cx> <cy> <radius> <start_deg> <end_deg> <color> [FILL]", 7, True, True
    ),
    "BEZIER": _CommandSpec(
        "BEZIER <x1> <y1> <cx1> <cy1> <cx2> <cy2> <x2> <y2> <color> [width]",
        10,
        True,
        True,
    ),
    "GRADIENT": _CommandSpec(
        "GRADIENT <x1> <y1> <x2> <y2> <color1> <color2> [horizontal|vertical|radial]",
        7,
        True,
        True,
    ),
    "DOTS": _CommandSpec(
        "DOTS <x1> <y1> <x2> <y2> <count> <color> [radius]", 7, True, True
    ),
    "TURTLE": _CommandSpec("TURTLE <x> <y>", 3, True, True),
    "HEADING": _CommandSpec("HEADING <degrees>", 2, False, True),
    "PENCOLOR": _CommandSpec("PENCOLOR <color>", 2, False, False),
    "PENWIDTH": _CommandSpec("PENWIDTH <width>", 2, False, True),
    "PENUP": _CommandSpec("PENUP", 1, False, False),
    "PENDOWN": _CommandSpec("PENDOWN", 1, False, False),
    "FORWARD": _CommandSpec("FORWARD <distance>", 2, True, True),
    "BACKWARD": _CommandSpec("BACKWARD <distance>", 2, True, True),
    "RIGHT": _CommandSpec("RIGHT <degrees>", 2, True, True),
    "LEFT": _CommandSpec("LEFT <degrees>", 2, True, True),
    "REPEAT": _CommandSpec("REPEAT <count> ... END", 2, False, True),
    "END": _CommandSpec("END", 1, False, True),
    "OUTPUT": _CommandSpec("OUTPUT [PNG]", 1, False, False),
}

# Opcodes the renderer interprets but the spec deliberately omits from the
# normalizer's safe set (SAVE writes files; the auto-draw path forbids it).
RENDERER_ONLY_OPCODES = frozenset({"SAVE"})
# REPEAT/END are control flow expanded before dispatch, so the renderer has no
# direct handler for them even though they are valid DSL.
_CONTROL_FLOW_OPCODES = frozenset({"REPEAT", "END"})

_SAFE_OPCODES = frozenset(_COMMAND_SPECS)
_CONTENT_OPCODES = frozenset(
    op for op, spec in _COMMAND_SPECS.items() if spec.produces_content
)
_NUMERIC_FIRST_ARG_OPCODES = frozenset(
    op for op, spec in _COMMAND_SPECS.items() if spec.numeric_first_arg
)
_MINIMUM_TOKENS = {op: spec.min_tokens for op, spec in _COMMAND_SPECS.items()}
_DSL_LIKE_LINE_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,}:?(?:\s|$)")


@dataclass(frozen=True)
class NormalizedDrawDSL:
    """Safe Draw DSL plus issues that would cause visible content loss."""

    normalized_dsl: str
    validation_issues: tuple[str, ...] = ()
    truncated: bool = False
    """True when output was capped at ``MAX_AUTO_LINES`` (still renderable)."""


def build_capability_prompt() -> str:
    """Return chat-model guidance for emitting an intermediate Draw spec."""

    return (
        "\n\nYou can create procedural illustrations for the user"
        " through CordBeat's Draw DSL renderer. Draw is not a diffusion"
        " image generator. The [DRAW: ...] tag is an intermediate"
        " renderer specification, not an image-generation prompt."
        " Describe concrete placements, primitive-friendly shapes,"
        " colors tied to objects, and a few essential identifying"
        " details. Do not write cinematic, quality, lighting, mood,"
        " texture, particle, or art-style prompt language. For a"
        " difficult request, preserve its essential visual"
        " relationships with simpler geometry. Do not refuse merely"
        " because the subject is complex. When the user asks for a"
        " drawing, include exactly one concise tag in this format:"
        " [DRAW: subject=<main subject>; layout=<where each major part"
        " goes>; geometry=<large shapes and curves>; colors=<object"
        " colors>; details=<few recognition-critical details>]."
        " Example: [DRAW: subject=celestial maiden bust portrait;"
        " layout=head centered, shoulders lower center, hair framing"
        " both sides; geometry=oval face, curved hair masses, simple"
        " shoulders; colors=indigo hair, warm face, dark blue"
        " background; details=silver star hairpin and gentle eyes]."
        " The tag will be converted to Draw DSL, rendered, and sent"
        " with your reply."
        " No image exists unless this reply contains the tag: if you"
        " say you will draw, are drawing, or have drawn something,"
        " the SAME reply MUST contain the [DRAW: ...] tag. Never claim"
        " a drawing is attached, finished, or on its way without the"
        " tag in this reply."
        " Do not call the draw skill directly with [SKILL: draw];"
        " use the [DRAW: ...] tag instead."
    )


def _command_reference() -> str:
    """Render the indented opcode list for the generator prompt from specs."""

    return "\n".join(f"  {spec.signature}" for spec in _COMMAND_SPECS.values())


def build_generation_request(
    description: str,
    *,
    retry_reason: str | None = None,
) -> tuple[str, str]:
    """Build the Draw DSL generator system prompt and user prompt."""

    retry_line = ""
    if retry_reason:
        retry_line = (
            "\nPrevious attempt failed: "
            f"{sanitize(retry_reason, strict=True, max_len=500)}"
            "\nProduce a complete alternative for the same request. Correct "
            "the stated failure without deleting requested essential detail."
        )
    system = (
        "/no_think\n"
        "You are a drawing DSL generator. "
        "Given an intermediate renderer specification, output ONLY valid Draw DSL"
        " commands — no prose, no markdown fences.\n"
        "Treat vague aesthetic words as non-binding and translate the subject, "
        "layout, geometry, colors, and essential details into concrete shapes. "
        "Do not invent dense particles, textures, scales, or decorative detail. "
        "Silently plan the composition before emitting commands. Use an "
        "800x600 canvas unless another aspect ratio is clearly better. "
        "Place the main subject in a large focal region, then layer filled "
        "silhouettes, interior shapes, outlines, and small identifying "
        "details from back to front. Keep important shapes inside the canvas. "
        "Use contrast between subject and background. Stylize difficult "
        "subjects into recognizable geometric forms instead of refusing. "
        "Prefer 40-110 meaningful command lines and never exceed 200 total "
        "command lines including SIZE, CANVAS, and OUTPUT. Reserve "
        "the final line for OUTPUT and finish the composition before it. "
        "Never use SAVE. Always end with OUTPUT as the final line.\n"
        "Available commands (one per line):\n"
        f"{_command_reference()}\n"
        "Colors: named colors (white, red, blue, ...) or #RRGGBB."
        " Use curves and gradients only when they improve the main silhouette."
        " Always end with OUTPUT.\n"
        "Composition patterns:\n"
        "- Character/animal: large head/body silhouettes first, then limbs, "
        "face, markings, and a simple ground/background.\n"
        "- Landscape: background gradient, distant silhouettes, foreground "
        "subject, then highlights and texture.\n"
        "- Icon/diagram: strong central geometry, consistent line widths, "
        "minimal labels.\n"
        "Example of valid layering:\n"
        "SIZE 800 600\n"
        "GRADIENT 0 0 800 600 #102040 #6aaed6 vertical\n"
        "ELLIPSE 220 170 580 520 #263238 FILL\n"
        "CIRCLE 330 290 18 white FILL\n"
        "CIRCLE 470 290 18 white FILL\n"
        "ARC 330 300 140 20 160 #f5c16c\n"
        "OUTPUT"
    )
    return system, f"Draw this: {sanitize(description, max_len=2000)}{retry_line}"


def _known_line_is_dsl_like(opcode: str, parts: list[str]) -> bool:
    if opcode not in _NUMERIC_FIRST_ARG_OPCODES or len(parts) < 2:
        return True
    try:
        float(parts[1].split(maxsplit=1)[0])
    except ValueError:
        return False
    return True


def _validation_issue(lineno: int, reason: str, line: str) -> str:
    return f"line {lineno} {reason}: {sanitize(line, strict=True, max_len=100)}"


def normalize(raw_dsl: str) -> NormalizedDrawDSL:
    """Keep safe Draw DSL lines and report commands that could not be preserved."""

    normalized: list[str] = []
    validation_issues: list[str] = []
    has_size = False
    has_canvas = False
    has_content = False
    truncated = False
    repeat_stack: list[int] = []

    for lineno, raw_line in enumerate(raw_dsl.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(("```", "#")):
            continue
        parts = line.split(maxsplit=1)
        opcode = parts[0].upper().rstrip(":")
        if opcode not in _SAFE_OPCODES:
            if _DSL_LIKE_LINE_RE.match(line):
                validation_issues.append(
                    _validation_issue(lineno, "uses unknown Draw command", line)
                )
            continue
        if not _known_line_is_dsl_like(opcode, parts):
            continue
        normalized_line = f"{opcode} {parts[1]}".strip() if len(parts) > 1 else opcode
        if len(normalized_line.split()) < _MINIMUM_TOKENS.get(opcode, 1):
            validation_issues.append(
                _validation_issue(lineno, f"{opcode} has missing arguments", line)
            )
            continue
        if opcode == "REPEAT":
            repeat_stack.append(lineno)
        elif opcode == "END":
            if not repeat_stack:
                validation_issues.append(
                    _validation_issue(lineno, "has unmatched END", line)
                )
                continue
            repeat_stack.pop()
        if opcode == "OUTPUT":
            continue
        if opcode == "SIZE":
            if has_size:
                continue
            has_size = True
        elif opcode == "CANVAS":
            if has_canvas:
                continue
            has_canvas = True
        elif opcode == "GRADIENT":
            has_canvas = True
        if opcode in _CONTENT_OPCODES:
            has_content = True
        normalized.append(normalized_line)
        if len(normalized) >= MAX_AUTO_LINES:
            # A too-long drawing is a soft cap, not lost content: render the
            # capped commands instead of forcing a retry that ends in total
            # failure. (Observed in production: long character portraits were
            # rejected three times and the user got no image at all.)
            truncated = True
            break

    for repeat_lineno in repeat_stack:
        validation_issues.append(
            f"line {repeat_lineno} has REPEAT without matching END"
        )

    if not has_content:
        return NormalizedDrawDSL("", tuple(validation_issues), truncated)
    if not has_size:
        normalized.insert(0, "SIZE 800 600")
    if not has_canvas:
        canvas_index = (
            1 if normalized and normalized[0].upper().startswith("SIZE") else 0
        )
        normalized.insert(canvas_index, "CANVAS #f8fafc")
    normalized.append("OUTPUT")
    return NormalizedDrawDSL("\n".join(normalized), tuple(validation_issues), truncated)
