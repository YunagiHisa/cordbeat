"""draw skill — execute a simple drawing DSL and return an image.

The skill accepts a ``commands`` string containing one instruction per line:

    SIZE <width> <height>          — set canvas dimensions (default 512×512)
    CANVAS <color>                 — fill background with a solid colour
    CIRCLE <cx> <cy> <r> <color> [FILL]
    RECT <x1> <y1> <x2> <y2> <color> [FILL]
    ELLIPSE <x1> <y1> <x2> <y2> <color> [FILL]
    LINE <x1> <y1> <x2> <y2> <color> [<width>]
    POLYGON <x1> <y1> <x2> <y2> ... <color> [FILL]
    TEXT <x> <y> <"quoted text"> <color> [<font_size>]
    SAVE <filename>                — write to a file (relative to cwd)
    OUTPUT [<format>]              — emit base64-encoded image (PNG/JPEG)

Colors can be named CSS colour strings (``red``, ``white``, ``#ff00aa``).
Each command that is not recognised or is missing arguments is silently skipped
and recorded in the ``warnings`` list of the result.

Return value (dict):
    - ``output``: base64-encoded PNG/JPEG when OUTPUT is used
    - ``saved``: path written when SAVE is used
    - ``warnings``: list of parse/execution warnings
    - ``format``: image format string (``PNG`` / ``JPEG``)
"""

from __future__ import annotations

import base64
import io
import math
import re
from collections.abc import Callable
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# ── Helpers ────────────────────────────────────────────────────────────


def _parse_colour(token: str) -> str:
    """Return *token* unchanged; invalid colours are caught by Pillow."""
    return token


def _parse_coords(tokens: list[str], count: int) -> list[float] | None:
    """Parse *count* numeric tokens; return None on failure."""
    if len(tokens) < count:
        return None
    try:
        return [float(t) for t in tokens[:count]]
    except ValueError:
        return None


def _int_coords(tokens: list[str], count: int) -> list[int] | None:
    coords = _parse_coords(tokens, count)
    if coords is None:
        return None
    return [int(round(c)) for c in coords]


def _parse_quoted(tokens: list[str]) -> tuple[str, list[str]] | None:
    """Parse a quoted string from the remaining token stream.

    Tokens are rejoined so quotes spanning multiple tokens are handled.
    Returns ``(text, remaining_tokens)`` or ``None``.
    """
    rejoined = " ".join(tokens)
    m = re.match(r'^"(.*?)"', rejoined)
    if not m:
        return None
    consumed = m.group(0)
    remaining = rejoined[len(consumed):].split()
    return m.group(1), remaining


# ── DSL interpreter ────────────────────────────────────────────────────


class _DrawDSL:
    """Stateful interpreter for the CordBeat drawing DSL."""

    def __init__(self) -> None:
        self._width = 512
        self._height = 512
        self._bg: str = "white"
        self._img: Image.Image | None = None
        self._draw: ImageDraw.ImageDraw | None = None
        self.warnings: list[str] = []
        self._saved: str | None = None
        self._output_b64: str | None = None
        self._output_fmt: str = "PNG"
        # Build dispatch table from bound methods (avoids getattr at call time)
        self._dispatch: dict[str, Callable[[list[str]], None]] = {
            "SIZE": self._cmd_size,
            "CANVAS": self._cmd_canvas,
            "CIRCLE": self._cmd_circle,
            "RECT": self._cmd_rect,
            "ELLIPSE": self._cmd_ellipse,
            "LINE": self._cmd_line,
            "POLYGON": self._cmd_polygon,
            "TEXT": self._cmd_text,
            "SAVE": self._cmd_save,
            "OUTPUT": self._cmd_output,
            "STAR": self._cmd_star,
            "SPIRAL": self._cmd_spiral,
        }

    # ── canvas init ──────────────────────────────────────────────────

    def _ensure_canvas(self) -> None:
        if self._img is None:
            self._img = Image.new("RGBA", (self._width, self._height), self._bg)
            self._draw = ImageDraw.Draw(self._img)

    @property
    def _d(self) -> ImageDraw.ImageDraw:
        self._ensure_canvas()
        assert self._draw is not None
        return self._draw

    # ── command handlers ─────────────────────────────────────────────

    def _cmd_size(self, args: list[str]) -> None:
        coords = _int_coords(args, 2)
        if not coords:
            self.warnings.append("SIZE requires width height")
            return
        w, h = coords
        if w <= 0 or h <= 0 or w > 4096 or h > 4096:
            self.warnings.append(f"SIZE values out of range: {w}×{h}")
            return
        self._width, self._height = w, h

    def _cmd_canvas(self, args: list[str]) -> None:
        if not args:
            self.warnings.append("CANVAS requires a colour")
            return
        colour = _parse_colour(args[0])
        self._bg = colour
        # If canvas was already created, replace it
        try:
            self._img = Image.new("RGBA", (self._width, self._height), colour)
            self._draw = ImageDraw.Draw(self._img)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"CANVAS colour error: {exc}")

    def _cmd_circle(self, args: list[str]) -> None:
        coords = _int_coords(args, 3)
        if not coords or len(args) < 4:
            self.warnings.append("CIRCLE requires cx cy r color [FILL]")
            return
        cx, cy, r = coords
        colour = args[3]
        fill = len(args) > 4 and args[4].upper() == "FILL"
        bbox = [cx - r, cy - r, cx + r, cy + r]
        try:
            if fill:
                self._d.ellipse(bbox, fill=colour)
            else:
                self._d.ellipse(bbox, outline=colour)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"CIRCLE error: {exc}")

    def _cmd_rect(self, args: list[str]) -> None:
        coords = _int_coords(args, 4)
        if not coords or len(args) < 5:
            self.warnings.append("RECT requires x1 y1 x2 y2 color [FILL]")
            return
        x1, y1, x2, y2 = coords
        colour = args[4]
        fill = len(args) > 5 and args[5].upper() == "FILL"
        try:
            if fill:
                self._d.rectangle([x1, y1, x2, y2], fill=colour)
            else:
                self._d.rectangle([x1, y1, x2, y2], outline=colour)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"RECT error: {exc}")

    def _cmd_ellipse(self, args: list[str]) -> None:
        coords = _int_coords(args, 4)
        if not coords or len(args) < 5:
            self.warnings.append("ELLIPSE requires x1 y1 x2 y2 color [FILL]")
            return
        x1, y1, x2, y2 = coords
        colour = args[4]
        fill = len(args) > 5 and args[5].upper() == "FILL"
        try:
            if fill:
                self._d.ellipse([x1, y1, x2, y2], fill=colour)
            else:
                self._d.ellipse([x1, y1, x2, y2], outline=colour)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"ELLIPSE error: {exc}")

    def _cmd_line(self, args: list[str]) -> None:
        coords = _int_coords(args, 4)
        if not coords or len(args) < 5:
            self.warnings.append("LINE requires x1 y1 x2 y2 color [width]")
            return
        x1, y1, x2, y2 = coords
        colour = args[4]
        width = 1
        if len(args) > 5:
            try:
                width = int(args[5])
            except ValueError:
                pass
        try:
            self._d.line([x1, y1, x2, y2], fill=colour, width=width)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"LINE error: {exc}")

    def _cmd_polygon(self, args: list[str]) -> None:
        # args: x1 y1 x2 y2 ... color [FILL]
        # Find where colour starts (non-numeric)
        coord_end = 0
        for i, tok in enumerate(args):
            try:
                float(tok)
                coord_end = i + 1
            except ValueError:
                break
        if coord_end < 4 or coord_end % 2 != 0:
            self.warnings.append("POLYGON requires paired x/y coords then color")
            return
        if len(args) <= coord_end:
            self.warnings.append("POLYGON requires a colour after coordinates")
            return
        raw = _int_coords(args, coord_end)
        if not raw:
            self.warnings.append("POLYGON could not parse coordinates")
            return
        points = [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]
        colour = args[coord_end]
        fill = len(args) > coord_end + 1 and args[coord_end + 1].upper() == "FILL"
        try:
            if fill:
                self._d.polygon(points, fill=colour)
            else:
                self._d.polygon(points, outline=colour)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"POLYGON error: {exc}")

    def _cmd_text(self, args: list[str]) -> None:
        coords = _int_coords(args, 2)
        if not coords:
            self.warnings.append("TEXT requires x y \"text\" color [font_size]")
            return
        x, y = coords
        rest = args[2:]
        parsed = _parse_quoted(rest)
        if parsed is None:
            self.warnings.append("TEXT: quoted text not found")
            return
        text, remaining = parsed
        if not remaining:
            self.warnings.append("TEXT requires a colour after the quoted text")
            return
        colour = remaining[0]
        font_size = 16
        if len(remaining) > 1:
            try:
                font_size = int(remaining[1])
            except ValueError:
                pass
        try:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
            self._d.text((x, y), text, fill=colour, font=font)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"TEXT error: {exc}")

    def _cmd_save(self, args: list[str]) -> None:
        if not args:
            self.warnings.append("SAVE requires a filename")
            return
        self._ensure_canvas()
        assert self._img is not None
        filename = args[0]
        fmt = "PNG" if not filename.lower().endswith((".jpg", ".jpeg")) else "JPEG"
        try:
            img = self._img.convert("RGB") if fmt == "JPEG" else self._img
            img.save(filename, format=fmt)
            self._saved = filename
        except Exception as exc:
            self.warnings.append(f"SAVE error: {exc}")

    def _cmd_output(self, args: list[str]) -> None:
        self._ensure_canvas()
        assert self._img is not None
        fmt = (args[0].upper() if args else "PNG")
        if fmt not in ("PNG", "JPEG", "JPG", "WEBP"):
            fmt = "PNG"
        if fmt == "JPG":
            fmt = "JPEG"
        self._output_fmt = fmt
        buf = io.BytesIO()
        try:
            img = self._img.convert("RGB") if fmt == "JPEG" else self._img
            img.save(buf, format=fmt)
            self._output_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception as exc:
            self.warnings.append(f"OUTPUT error: {exc}")

    # ── star / spiral helpers ────────────────────────────────────────

    def _cmd_star(self, args: list[str]) -> None:
        """STAR cx cy outer_r inner_r points color [FILL]"""
        coords = _int_coords(args, 5)
        if not coords or len(args) < 6:
            self.warnings.append("STAR requires cx cy outer inner points color [FILL]")
            return
        cx, cy, outer_r, inner_r, n_points = coords
        colour = args[5]
        fill = len(args) > 6 and args[6].upper() == "FILL"
        pts: list[tuple[int, int]] = []
        for i in range(int(n_points) * 2):
            angle = math.pi * i / n_points - math.pi / 2
            r = outer_r if i % 2 == 0 else inner_r
            pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        try:
            if fill:
                self._d.polygon(pts, fill=colour)
            else:
                self._d.polygon(pts, outline=colour)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"STAR error: {exc}")

    def _cmd_spiral(self, args: list[str]) -> None:
        """SPIRAL cx cy turns max_r color [width]"""
        coords = _parse_coords(args, 4)
        if not coords or len(args) < 5:
            self.warnings.append("SPIRAL requires cx cy turns max_r color [width]")
            return
        cx, cy, turns, max_r = coords
        colour = args[4]
        width = 1
        if len(args) > 5:
            try:
                width = int(args[5])
            except ValueError:
                pass
        steps = max(int(turns * 60), 4)
        pts: list[tuple[int, int]] = []
        for i in range(steps + 1):
            t = i / steps
            angle = t * turns * 2 * math.pi
            r = t * max_r
            pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        if len(pts) < 2:
            return
        try:
            self._d.line(pts, fill=colour, width=width)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"SPIRAL error: {exc}")

    # ── main dispatch ────────────────────────────────────────────────

    def run(self, commands: str) -> dict[str, Any]:
        """Execute all lines in *commands* and return the result dict."""
        for lineno, raw_line in enumerate(commands.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            cmd = tokens[0].upper()
            handler = self._dispatch.get(cmd)
            if handler is None:
                self.warnings.append(f"Line {lineno}: unknown command {cmd!r}")
                continue
            try:
                handler(tokens[1:])
            except Exception as exc:
                self.warnings.append(f"Line {lineno}: {cmd} raised {exc}")

        result: dict[str, Any] = {"warnings": self.warnings}
        if self._output_b64 is not None:
            result["output"] = self._output_b64
            result["format"] = self._output_fmt
        if self._saved is not None:
            result["saved"] = self._saved
        return result


# ── Public entry point ─────────────────────────────────────────────────


async def execute(
    *,
    commands: str,
    context: Any = None,
) -> dict[str, Any]:
    """Run the drawing DSL and return the result.

    Parameters
    ----------
    commands:
        Newline-separated drawing instructions (see module docstring).
    context:
        Ignored; present for the skill runner interface.
    """
    if not commands.strip():
        return {"error": "commands is empty", "warnings": []}
    dsl = _DrawDSL()
    return dsl.run(commands)
