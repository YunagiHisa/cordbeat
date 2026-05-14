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
    ARC <cx> <cy> <r> <start> <end> <color> [FILL]
    BEZIER <x1> <y1> <cx1> <cy1> <cx2> <cy2> <x2> <y2> <color> [<width>]
    GRADIENT <x1> <y1> <x2> <y2> <color1> <color2> [horizontal|vertical|radial]
    DOTS <x1> <y1> <x2> <y2> <count> <color> [<radius>]
    STAR <cx> <cy> <outer_r> <inner_r> <points> <color> [FILL]
    SPIRAL <cx> <cy> <turns> <max_r> <color> [<width>]
    SAVE <filename>                — write to a file (relative to cwd)
    OUTPUT [<format>]              — emit base64-encoded image (PNG/JPEG)

Turtle graphics (stateful pen):
    TURTLE <x> <y>                 — move turtle to position (pen up)
    HEADING <degrees>              — set turtle direction (0=right, 90=down)
    PENCOLOR <color>               — set pen colour
    PENWIDTH <width>               — set pen width (pixels)
    PENUP                          — lift pen (moves without drawing)
    PENDOWN                        — lower pen (draws on move)
    FORWARD <distance>             — move forward; draws if pen down
    BACKWARD <distance>            — move backward; draws if pen down
    RIGHT <degrees>                — turn right
    LEFT <degrees>                 — turn left
    REPEAT <n>                     — begin a repeat block
    END                            — end a repeat block

Colors can be named CSS colour strings (``red``, ``white``, ``#ff00aa``).
HSL colors are supported via ``hsl(h,s%,l%)`` syntax.
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
import colorsys
import io
import math
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# ── Helpers ────────────────────────────────────────────────────────────


def _parse_colour(token: str) -> str:
    """Parse a colour token.

    Supports CSS named colours, hex strings, and ``hsl(h,s%,l%)`` syntax.
    Invalid values are returned unchanged (Pillow will raise on use).
    """
    t = token.strip()
    # hsl(h, s%, l%) — convert to hex
    m = re.match(
        r"^hsl\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)%\s*,\s*(\d+(?:\.\d+)?)%\s*\)$",
        t,
        re.IGNORECASE,
    )
    if m:
        h = float(m.group(1)) / 360.0
        s = float(m.group(2)) / 100.0
        l_ = float(m.group(3)) / 100.0
        r, g, b = colorsys.hls_to_rgb(h, l_, s)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return t


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
        # Turtle state
        self._tx: float = 256.0
        self._ty: float = 256.0
        self._heading: float = 0.0  # degrees; 0 = right, 90 = down
        self._pen_down: bool = True
        self._pen_color: str = "black"
        self._pen_width: int = 1
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
            "ARC": self._cmd_arc,
            "BEZIER": self._cmd_bezier,
            "GRADIENT": self._cmd_gradient,
            "DOTS": self._cmd_dots,
            "SAVE": self._cmd_save,
            "OUTPUT": self._cmd_output,
            "STAR": self._cmd_star,
            "SPIRAL": self._cmd_spiral,
            # Turtle
            "TURTLE": self._cmd_turtle,
            "HEADING": self._cmd_heading,
            "PENCOLOR": self._cmd_pencolor,
            "PENWIDTH": self._cmd_penwidth,
            "PENUP": self._cmd_penup,
            "PENDOWN": self._cmd_pendown,
            "FORWARD": self._cmd_forward,
            "BACKWARD": self._cmd_backward,
            "RIGHT": self._cmd_right,
            "LEFT": self._cmd_left,
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
        # Strip all directory components so the AI cannot perform path traversal
        # (e.g. SAVE ../../../etc/passwd).  Files are always written to the
        # cordbeat draw-output directory.
        basename = Path(args[0]).name
        if not basename:
            self.warnings.append("SAVE: invalid filename")
            return
        safe_dir = Path.home() / ".cordbeat" / "draw_output"
        safe_dir.mkdir(parents=True, exist_ok=True)
        filepath = (safe_dir / basename).resolve()
        # Defend against symlink-based escapes as well.
        try:
            filepath.relative_to(safe_dir.resolve())
        except ValueError:
            self.warnings.append("SAVE: path escaped safe directory")
            return
        fmt = "PNG" if not basename.lower().endswith((".jpg", ".jpeg")) else "JPEG"
        try:
            img = self._img.convert("RGB") if fmt == "JPEG" else self._img
            img.save(str(filepath), format=fmt)
            self._saved = str(filepath)
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

    # ── arc / bezier / gradient / dots ──────────────────────────────
    def _cmd_arc(self, args: list[str]) -> None:
        """ARC cx cy r start_deg end_deg color [FILL]"""
        coords = _int_coords(args, 5)
        if not coords or len(args) < 6:
            self.warnings.append("ARC requires cx cy r start end color [FILL]")
            return
        cx, cy, r, start, end = coords
        colour = _parse_colour(args[5])
        fill = len(args) > 6 and args[6].upper() == "FILL"
        bbox = [cx - r, cy - r, cx + r, cy + r]
        try:
            if fill:
                self._d.pieslice(bbox, start=start, end=end, fill=colour)
            else:
                self._d.arc(bbox, start=start, end=end, fill=colour)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"ARC error: {exc}")

    def _cmd_bezier(self, args: list[str]) -> None:
        """BEZIER x1 y1 cx1 cy1 cx2 cy2 x2 y2 color [width]"""
        coords = _parse_coords(args, 8)
        if not coords or len(args) < 9:
            self.warnings.append(
                "BEZIER requires x1 y1 cx1 cy1 cx2 cy2 x2 y2 color [width]"
            )
            return
        x1, y1, cx1, cy1, cx2, cy2, x2, y2 = coords
        colour = _parse_colour(args[8])
        width = 1
        if len(args) > 9:
            try:
                width = int(args[9])
            except ValueError:
                pass
        # Approximate cubic bezier with line segments
        steps = max(int(math.dist((x1, y1), (x2, y2)) / 2), 20)
        pts: list[tuple[int, int]] = []
        for i in range(steps + 1):
            t = i / steps
            mt = 1 - t
            bx = mt**3 * x1 + 3 * mt**2 * t * cx1 + 3 * mt * t**2 * cx2 + t**3 * x2
            by = mt**3 * y1 + 3 * mt**2 * t * cy1 + 3 * mt * t**2 * cy2 + t**3 * y2
            pts.append((int(bx), int(by)))
        try:
            self._d.line(pts, fill=colour, width=width)
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"BEZIER error: {exc}")

    def _cmd_gradient(self, args: list[str]) -> None:
        """GRADIENT x1 y1 x2 y2 color1 color2 [horizontal|vertical|radial]"""
        coords = _int_coords(args, 4)
        if not coords or len(args) < 6:
            self.warnings.append(
                "GRADIENT requires x1 y1 x2 y2 color1 color2 [direction]"
            )
            return
        x1, y1, x2, y2 = coords
        c1 = _parse_colour(args[4])
        c2 = _parse_colour(args[5])
        direction = args[6].lower() if len(args) > 6 else "horizontal"
        self._ensure_canvas()
        assert self._img is not None
        try:
            rgb1 = self._img.getpixel((0, 0))[:3]  # dummy to trigger conversion
            # Use Pillow to parse colours
            tmp = Image.new("RGB", (1, 1), c1)
            r1, g1, b1 = tmp.getpixel((0, 0))
            tmp = Image.new("RGB", (1, 1), c2)
            r2, g2, b2 = tmp.getpixel((0, 0))
            _ = rgb1  # unused but mypy happy
        except Exception as exc:
            self.warnings.append(f"GRADIENT colour error: {exc}")
            return
        w = abs(x2 - x1) or 1
        h = abs(y2 - y1) or 1
        cx_mid = (x1 + x2) / 2
        cy_mid = (y1 + y2) / 2
        r_max = math.dist((x1, y1), (x2, y2)) / 2
        try:
            for gy in range(int(y1), int(y2)):
                for gx in range(int(x1), int(x2)):
                    if direction == "vertical":
                        t = (gy - y1) / h
                    elif direction == "radial":
                        dist = math.dist((gx, gy), (cx_mid, cy_mid))
                        t = min(dist / r_max, 1.0)
                    else:  # horizontal (default)
                        t = (gx - x1) / w
                    t = max(0.0, min(1.0, t))
                    r = int(r1 + (r2 - r1) * t)
                    g = int(g1 + (g2 - g1) * t)
                    b = int(b1 + (b2 - b1) * t)
                    self._img.putpixel((gx, gy), (r, g, b, 255))
        except Exception as exc:
            self.warnings.append(f"GRADIENT render error: {exc}")

    def _cmd_dots(self, args: list[str]) -> None:
        """DOTS x1 y1 x2 y2 count color [radius]"""
        coords = _int_coords(args, 4)
        if not coords or len(args) < 6:
            self.warnings.append("DOTS requires x1 y1 x2 y2 count color [radius]")
            return
        x1, y1, x2, y2 = coords
        try:
            count = int(args[4])
        except ValueError:
            self.warnings.append("DOTS: count must be an integer")
            return
        count = min(count, 2000)  # cap to avoid DoS
        colour = _parse_colour(args[5])
        radius = 3
        if len(args) > 6:
            try:
                radius = int(args[6])
            except ValueError:
                pass
        rng = random.Random(42)  # deterministic seed for reproducibility
        try:
            for _ in range(count):
                dx = rng.randint(int(x1), int(x2))
                dy = rng.randint(int(y1), int(y2))
                self._d.ellipse(
                    [dx - radius, dy - radius, dx + radius, dy + radius],
                    fill=colour,
                )
        except (ValueError, TypeError) as exc:
            self.warnings.append(f"DOTS error: {exc}")

    # ── turtle commands ──────────────────────────────────────────────

    def _cmd_turtle(self, args: list[str]) -> None:
        """TURTLE x y — teleport turtle (pen up)."""
        coords = _parse_coords(args, 2)
        if not coords:
            self.warnings.append("TURTLE requires x y")
            return
        self._tx, self._ty = coords

    def _cmd_heading(self, args: list[str]) -> None:
        """HEADING degrees — set absolute direction."""
        if not args:
            self.warnings.append("HEADING requires degrees")
            return
        try:
            self._heading = float(args[0]) % 360
        except ValueError:
            self.warnings.append(f"HEADING: invalid angle {args[0]!r}")

    def _cmd_pencolor(self, args: list[str]) -> None:
        if not args:
            self.warnings.append("PENCOLOR requires a colour")
            return
        self._pen_color = _parse_colour(args[0])

    def _cmd_penwidth(self, args: list[str]) -> None:
        if not args:
            self.warnings.append("PENWIDTH requires a number")
            return
        try:
            self._pen_width = max(1, int(args[0]))
        except ValueError:
            self.warnings.append(f"PENWIDTH: invalid value {args[0]!r}")

    def _cmd_penup(self, _args: list[str]) -> None:
        self._pen_down = False

    def _cmd_pendown(self, _args: list[str]) -> None:
        self._pen_down = True

    def _cmd_forward(self, args: list[str]) -> None:
        """FORWARD distance — move turtle forward; draw if pen down."""
        if not args:
            self.warnings.append("FORWARD requires distance")
            return
        try:
            dist = float(args[0])
        except ValueError:
            self.warnings.append(f"FORWARD: invalid distance {args[0]!r}")
            return
        rad = math.radians(self._heading)
        nx = self._tx + dist * math.cos(rad)
        ny = self._ty + dist * math.sin(rad)
        if self._pen_down:
            try:
                self._d.line(
                    [int(self._tx), int(self._ty), int(nx), int(ny)],
                    fill=self._pen_color,
                    width=self._pen_width,
                )
            except (ValueError, TypeError) as exc:
                self.warnings.append(f"FORWARD draw error: {exc}")
        self._tx, self._ty = nx, ny

    def _cmd_backward(self, args: list[str]) -> None:
        """BACKWARD distance — move turtle backward; draw if pen down."""
        if not args:
            self.warnings.append("BACKWARD requires distance")
            return
        try:
            dist = float(args[0])
        except ValueError:
            self.warnings.append(f"BACKWARD: invalid distance {args[0]!r}")
            return
        # Reuse forward with negated distance
        self._cmd_forward([str(-dist)])

    def _cmd_right(self, args: list[str]) -> None:
        """RIGHT degrees — turn turtle clockwise."""
        if not args:
            self.warnings.append("RIGHT requires degrees")
            return
        try:
            self._heading = (self._heading + float(args[0])) % 360
        except ValueError:
            self.warnings.append(f"RIGHT: invalid angle {args[0]!r}")

    def _cmd_left(self, args: list[str]) -> None:
        """LEFT degrees — turn turtle counter-clockwise."""
        if not args:
            self.warnings.append("LEFT requires degrees")
            return
        try:
            self._heading = (self._heading - float(args[0])) % 360
        except ValueError:
            self.warnings.append(f"LEFT: invalid angle {args[0]!r}")

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
        # Expand REPEAT n ... END blocks before dispatch
        expanded = self._expand_repeats(commands)
        for lineno, raw_line in enumerate(expanded.splitlines(), start=1):
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

    # ── REPEAT block expansion ──────────────────────────────────────

    def _expand_repeats(self, commands: str, _depth: int = 0) -> str:
        """Recursively expand ``REPEAT n ... END`` blocks."""
        if _depth > 10:
            self.warnings.append("REPEAT nesting too deep (max 10)")
            return commands
        lines = commands.splitlines()
        out: list[str] = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            upper = stripped.upper()
            if upper.startswith("REPEAT "):
                parts = stripped.split()
                try:
                    count = min(int(parts[1]), 1000)
                except (IndexError, ValueError):
                    self.warnings.append(f"REPEAT: invalid count near line {i + 1}")
                    i += 1
                    continue
                # Collect body until matching END
                body_lines: list[str] = []
                depth = 1
                i += 1
                while i < len(lines) and depth > 0:
                    inner = lines[i].strip().upper()
                    if inner.startswith("REPEAT "):
                        depth += 1
                    elif inner == "END":
                        depth -= 1
                        if depth == 0:
                            break
                    if depth > 0:
                        body_lines.append(lines[i])
                    i += 1
                body = "\n".join(body_lines)
                expanded_body = self._expand_repeats(body, _depth + 1)
                for _ in range(count):
                    out.append(expanded_body)
            else:
                out.append(lines[i])
            i += 1
        return "\n".join(out)


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
