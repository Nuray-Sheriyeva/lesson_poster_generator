"""The single source of mascot/bubble geometry.

``layout_mascot`` is a deterministic, side-effect-free geometry function.  Stage
5 calls it to reserve a clean zone in the image prompt; Stage 8 calls it again
before drawing the overlay.  Everything the two stages need (the mascot square,
the tightest bubble rectangle, the chosen font size, the wrapped lines, and the
combined block) comes from this one function.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import lru_cache
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from ..errors import GenLessonError
from .assets import resolve_asset_path

Corner = Literal["top_left", "top_right", "bottom_left", "bottom_right"]
CORNERS: tuple[Corner, ...] = ("top_left", "top_right", "bottom_left", "bottom_right")

_CORNER_ALIASES = {
    "top_left": "top_left",
    "top-left": "top_left",
    "topleft": "top_left",
    "tl": "top_left",
    "top_right": "top_right",
    "top-right": "top_right",
    "topright": "top_right",
    "tr": "top_right",
    "bottom_left": "bottom_left",
    "bottom-left": "bottom_left",
    "bottomleft": "bottom_left",
    "bl": "bottom_left",
    "bottom_right": "bottom_right",
    "bottom-right": "bottom_right",
    "bottomright": "bottom_right",
    "br": "bottom_right",
}


def normalize_corner(value: str) -> Corner:
    key = str(value or "").strip().casefold().replace(" ", "_")
    try:
        return _CORNER_ALIASES[key]  # type: ignore[return-value]
    except KeyError as exc:
        raise GenLessonError(
            "mascot_corner_invalid",
            f"Unknown mascot corner: {value!r}",
            hint="Use top-left, top-right, bottom-left, or bottom-right.",
        ) from exc


def _is_left(corner: Corner) -> bool:
    return corner.endswith("_left")


def _is_top(corner: Corner) -> bool:
    return corner.startswith("top")


class LineTuple(tuple):
    """A tuple of wrapped lines that also compares equal to a list."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return NotImplemented

    __hash__ = tuple.__hash__


@dataclass(frozen=True, eq=False)
class Rect:
    """An integer pixel rectangle.  Right/bottom edges are exclusive."""

    x: int
    y: int
    width: int
    height: int

    @property
    def left(self) -> int:
        return self.x

    @property
    def top(self) -> int:
        return self.y

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def w(self) -> int:
        return self.width

    @property
    def x1(self) -> int:
        return self.left

    @property
    def y1(self) -> int:
        return self.top

    @property
    def x2(self) -> int:
        return self.right

    @property
    def y2(self) -> int:
        return self.bottom

    @property
    def h(self) -> int:
        return self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Pillow-style half-open box: ``(left, top, right, bottom)``."""
        return self.left, self.top, self.right, self.bottom

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def intersection(self, other: Rect) -> Rect:
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return Rect(left, top, 0, 0)
        return Rect(left, top, right - left, bottom - top)

    def overlap_area(self, other: Rect) -> int:
        return self.intersection(other).area

    def expand(self, amount: int) -> Rect:
        return Rect(
            self.x - amount,
            self.y - amount,
            self.width + amount * 2,
            self.height + amount * 2,
        )

    def union(self, other: Rect) -> Rect:
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        return Rect(left, top, right - left, bottom - top)

    def clamp(self, width: int, height: int) -> Rect:
        x = min(max(0, self.x), max(0, width - self.width))
        y = min(max(0, self.y), max(0, height - self.height))
        return Rect(x, y, self.width, self.height)

    def __iter__(self) -> Iterator[int]:
        yield self.x
        yield self.y
        yield self.width
        yield self.height

    def __getitem__(self, index: int) -> int:
        return self.as_tuple()[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rect):
            return self.as_tuple() == other.as_tuple()
        if isinstance(other, (tuple, list)) and len(other) == 4:
            return self.as_tuple() == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.as_tuple())


def _rect_from_points(points: tuple[tuple[int, int], ...]) -> Rect:
    if not points:
        return Rect(0, 0, 0, 0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return Rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


@dataclass(frozen=True)
class MascotLayout:
    """The complete, resolved geometry for one mascot overlay."""

    mascot: Rect
    bubble: Rect
    font_size: int
    lines: tuple[str, ...]
    block: Rect
    question: str
    corner: Corner
    padding: tuple[int, int]
    line_height: int
    radius: int
    outline_width: int
    tail: tuple[tuple[int, int], ...]
    font_path: str

    # Convenience aliases keep call sites and external tests readable.
    @property
    def mascot_rect(self) -> Rect:
        return self.mascot

    @property
    def bubble_rect(self) -> Rect:
        return self.bubble

    @property
    def block_rect(self) -> Rect:
        return self.block

    @property
    def combined(self) -> Rect:
        return self.block

    @property
    def wrapped_lines(self) -> tuple[str, ...]:
        return self.lines

    @property
    def text_lines(self) -> tuple[str, ...]:
        return self.lines

    @property
    def font(self) -> int:
        return self.font_size

    @property
    def font_px(self) -> int:
        return self.font_size

    @property
    def combined_rect(self) -> Rect:
        return self.block

    def as_dict(self) -> dict:
        return {
            "mascot": self.mascot.as_dict(),
            "bubble": self.bubble.as_dict(),
            "font_size": self.font_size,
            "lines": list(self.lines),
            "block": self.block.as_dict(),
            "question": self.question,
            "corner": self.corner,
            "padding": {"x": self.padding[0], "y": self.padding[1]},
            "line_height": self.line_height,
            "radius": self.radius,
            "outline_width": self.outline_width,
            "tail": [{"x": x, "y": y} for x, y in self.tail],
        }

    def __getitem__(self, key: str):
        if key in {"mascot", "mascot_rect"}:
            return self.mascot
        if key in {"bubble", "bubble_rect"}:
            return self.bubble
        if key in {"font_size", "font"}:
            return self.font_size
        if key in {"lines", "wrapped_lines", "text_lines"}:
            return self.lines
        if key in {"block", "block_rect", "combined"}:
            return self.block
        if key in {
            "question",
            "corner",
            "padding",
            "line_height",
            "radius",
            "outline_width",
            "tail",
        }:
            return getattr(self, key)
        raise KeyError(key)

    def keys(self) -> tuple[str, ...]:
        return (
            "mascot",
            "bubble",
            "font_size",
            "lines",
            "block",
            "question",
            "corner",
            "padding",
            "line_height",
            "radius",
            "outline_width",
            "tail",
        )


@lru_cache(maxsize=256)
def _load_font_cached(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError as exc:
        raise GenLessonError(
            "mascot_font_missing",
            f"Could not load the mascot font at {path!r}: {exc}",
            hint="Bundle a font that covers the lesson language, or fix mascot.font_path.",
        ) from exc


def load_font(font_path: str | object, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font, resolving the configured relative path."""
    path = resolve_asset_path(font_path)  # type: ignore[arg-type]
    if not path.is_file():
        raise GenLessonError(
            "mascot_font_missing",
            f"Mascot font does not exist: {path}",
            hint="Set mascot.font_path to a bundled Open Font License TTF.",
        )
    return _load_font_cached(str(path), int(size))


_MEASURE_IMAGE = Image.new("L", (1, 1))
_MEASURE_DRAW = ImageDraw.Draw(_MEASURE_IMAGE)


def measure_text(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Measure rendered ink, including accents and descenders."""
    if not text:
        return 0, 0
    bbox = _MEASURE_DRAW.textbbox((0, 0), text, font=font)
    return max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1])


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> tuple[str, ...]:
    """Greedy word wrapping with explicit newlines preserved."""
    if max_width < 1:
        return (text.strip(),) if text.strip() else ("",)
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            width, _ = measure_text(candidate, font)
            if width <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return tuple(lines) if lines else ("",)


FONT_SIZE_TRIM_PX = 2


def _fit_text(
    question: str,
    *,
    font_path: str,
    max_width: int,
    max_height: int,
    min_px: int,
    max_px: int,
    pad_x: int,
    pad_y: int,
) -> tuple[int, tuple[str, ...], int, int, int, ImageFont.FreeTypeFont]:
    """Find the largest font size that fits, then trim it by a couple of px.

    Returns ``(font_size, lines, bubble_width, text_width, text_height, font)``.
    Width is the tightest that fits the wrapped lines: the widest rendered line
    plus the horizontal padding, never simply the cap.
    """
    max_content = max(1, max_width - pad_x * 2)

    def record_for(size: int):
        font = load_font(font_path, size)
        lines = wrap_text(question, font, max_content)
        widths = [measure_text(line, font)[0] for line in lines]
        text_width = max(widths, default=0)
        ascent, descent = font.getmetrics()
        line_height = max(round(size * 1.24), ascent + descent)
        text_height = line_height * len(lines)
        bubble_width = text_width + pad_x * 2
        bubble_height = text_height + pad_y * 2
        return (
            size,
            lines,
            bubble_width,
            text_width,
            text_height,
            font,
            bubble_height,
        )

    def fits(record) -> bool:
        return record[3] <= max_content and record[2] <= max_width and record[6] <= max_height

    found = None
    best_overflow = None
    for size in range(max_px, min_px - 1, -1):
        record = record_for(size)
        if fits(record):
            found = record
            break
        if best_overflow is None or record[6] < best_overflow[6]:
            best_overflow = record
    if found is None:
        detail = best_overflow or (min_px, (question,), max_width, max_width, max_height, None)
        raise GenLessonError(
            "mascot_text_overflow",
            "The mascot question does not fit the bubble at the minimum font size.",
            hint=(
                "Shorten the question in review (stage-03-plan/poster_spec.json) or raise "
                "mascot.question_max_chars / bubble_max_width_ratio."
            ),
            details={
                "question": question,
                "min_font_px": min_px,
                "max_width": max_width,
                "max_height": max_height,
                "longest_text_width": detail[3],
                "minimum_text_height": detail[4],
            },
        )

    # The product wants the fitted text a touch smaller than the mathematical
    # maximum.  Re-wrap and re-measure at the trimmed size so the tight bubble
    # still reflects the actual rendered lines.
    trimmed_size = max(min_px, found[0] - FONT_SIZE_TRIM_PX)
    if trimmed_size != found[0]:
        trimmed = record_for(trimmed_size)
        if fits(trimmed):
            found = trimmed
    return found[:6]


def _clamp(value: int, low: int, high: int) -> int:
    if high < low:
        return low
    return min(max(value, low), high)


def _truncate_question(question: str, max_chars: int) -> str:
    clean = " ".join(str(question).split())
    if len(clean) <= max_chars:
        return clean
    cut = clean[: max(0, max_chars - 1)].rstrip()
    while cut and not cut[-1].isalnum() and not cut[-1].isspace():
        cut = cut[:-1].rstrip()
    return f"{cut}…"


def layout_mascot(
    width: int,
    height: int,
    question: str,
    corner: str | object = "bottom_right",
    config: object | None = None,
    *,
    font_path: str | None = None,
    mascot_size: tuple[int, int] | None = None,
    mascot_aspect: float | None = None,
    **options: object,
) -> MascotLayout:
    """Resolve the mascot, bubble, text, and combined block for a canvas.

    Parameters
    ----------
    width, height:
        Canvas size in pixels.
    question:
        The single question shown in the speech bubble.
    corner:
        ``top-left``, ``top-right``, ``bottom-left``, or ``bottom-right``.
        Dashes and underscores are both accepted.
    config:
        A :class:`genlesson.app_config.MascotConfig`.  When omitted the active
        ``config.toml`` ``[mascot]`` section is used.
    font_path:
        Optional override for the configured font.
    mascot_size:
        Source asset size in pixels.  The short side is scaled to
        ``size_ratio * canvas_short_side`` and the long side keeps the source
        aspect ratio.  When omitted, the mascot is treated as square.
    mascot_aspect:
        Width / height override when ``mascot_size`` is not supplied.

    The function is deterministic for the same arguments and configuration.
    """
    # Allow the config as the fourth positional argument.
    if config is None and hasattr(corner, "size_ratio") and not isinstance(corner, str):
        config = corner
        corner = "bottom_right"
    if config is None:
        from ..app_config import load_app_config

        config = load_app_config().mascot
    elif isinstance(config, dict):
        from ..app_config import MascotConfig

        config = MascotConfig(**config)
    if options:
        if not is_dataclass(config):
            raise GenLessonError(
                "mascot_option_invalid",
                "Overrides need a MascotConfig instance.",
            )
        known = {item.name for item in fields(config)}
        unknown = sorted(set(options) - known)
        if unknown:
            raise GenLessonError(
                "mascot_option_invalid",
                f"Unknown layout option(s): {', '.join(unknown)}.",
                details={"supported": sorted(known)},
            )
        config = replace(config, **options)

    if width < 32 or height < 32:
        raise GenLessonError(
            "mascot_canvas_too_small",
            f"Canvas {width}x{height} is too small for a mascot overlay.",
        )
    resolved_corner = normalize_corner(str(corner) if corner is not None else "bottom_right")
    selected_font = font_path or str(
        getattr(config, "font_path", "assets/fonts/NotoSans-Regular.ttf")
    )
    size_ratio = float(getattr(config, "size_ratio", 0.09))
    min_ratio = float(getattr(config, "min_font_ratio", 0.016))
    max_ratio = float(getattr(config, "max_font_ratio", 0.03))
    bubble_ratio = float(getattr(config, "bubble_max_width_ratio", 0.42))
    max_chars = int(getattr(config, "question_max_chars", 140))
    outline_base = int(getattr(config, "outline_width", 4))
    edge_setting = float(getattr(config, "edge_margin", 0.025))

    short = min(width, height)
    text = _truncate_question(question, max_chars)
    if not text:
        raise GenLessonError(
            "mascot_question_empty",
            "The mascot question must not be empty.",
            hint="Pass a question related to the lesson.",
        )

    margin = round(edge_setting if edge_setting > 1 else short * edge_setting)
    margin = max(0, margin)
    mascot_short = max(1, round(short * size_ratio))
    if mascot_size is not None:
        source_w, source_h = mascot_size
        if source_w < 1 or source_h < 1:
            raise GenLessonError(
                "mascot_assets_invalid",
                f"Mascot source size must be positive, got {mascot_size!r}.",
            )
        aspect = source_w / source_h
    else:
        aspect = float(mascot_aspect) if mascot_aspect and mascot_aspect > 0 else 1.0
    if aspect >= 1.0:
        # Width >= height: the short side is the height.
        mascot_h = mascot_short
        mascot_w = max(mascot_h, round(mascot_h * aspect))
    else:
        # Height >= width: the short side is the width.
        mascot_w = mascot_short
        mascot_h = max(mascot_w, round(mascot_w / aspect))
    max_mascot_w = max(1, width - margin * 2)
    max_mascot_h = max(1, height - margin * 2)
    if mascot_w > max_mascot_w:
        scale = max_mascot_w / mascot_w
        mascot_w = max_mascot_w
        mascot_h = max(1, round(mascot_h * scale))
    if mascot_h > max_mascot_h:
        scale = max_mascot_h / mascot_h
        mascot_h = max_mascot_h
        mascot_w = max(1, round(mascot_w * scale))
    mascot_side = min(mascot_w, mascot_h)

    if _is_left(resolved_corner):
        mascot_x = margin
    else:
        mascot_x = width - margin - mascot_w
    if _is_top(resolved_corner):
        mascot_y = margin
    else:
        mascot_y = height - margin - mascot_h
    mascot = Rect(mascot_x, mascot_y, mascot_w, mascot_h)

    gap = max(2, round(mascot_side * 0.24))
    pad_x = max(3, round(short * 0.011))
    pad_y = max(3, round(short * 0.008))
    radius = max(5, round(mascot_side * 0.24))
    line_scale = max(1, round(outline_base * short / 1024))
    outline_width = max(1, line_scale)

    if _is_left(resolved_corner):
        available = (width - margin) - (mascot.right + gap)
    else:
        available = (mascot.left - gap) - margin
    max_bubble_width = min(round(width * bubble_ratio), max(0, available))
    max_bubble_height = height - margin * 2
    if max_bubble_width < 32 or max_bubble_height < 32:
        raise GenLessonError(
            "mascot_canvas_too_small",
            "The canvas leaves no room for a mascot speech bubble.",
            details={
                "width": width,
                "height": height,
                "mascot": mascot.as_dict(),
                "max_bubble_width": max_bubble_width,
            },
        )

    min_px = max(8, round(short * min_ratio))
    max_px = max(min_px, round(short * max_ratio))
    font_size, lines, bubble_width, _text_width, text_height, _font = _fit_text(
        text,
        font_path=selected_font,
        max_width=max_bubble_width,
        max_height=max_bubble_height,
        min_px=min_px,
        max_px=max_px,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    ascent, descent = _font.getmetrics()
    line_height = max(round(font_size * 1.24), ascent + descent)
    bubble_height = text_height + pad_y * 2

    if _is_left(resolved_corner):
        bubble_x = mascot.right + gap
    else:
        bubble_x = mascot.left - gap - bubble_width
    bubble_y = mascot.y + mascot_h // 2 - bubble_height // 2
    bubble_y = _clamp(bubble_y, margin, height - margin - bubble_height)
    bubble = Rect(bubble_x, bubble_y, bubble_width, bubble_height)

    # The tail points from the bubble's inward edge at the mascot.
    tail_half = max(4, round(mascot_side * 0.16))
    center_y = mascot.y + mascot_h // 2
    base_low = bubble_y + radius
    base_high = bubble.bottom - radius
    base_y1 = _clamp(center_y - tail_half, base_low, base_high)
    base_y2 = _clamp(center_y + tail_half, base_low, base_high)
    tip_y = _clamp(center_y, mascot.y + 2, mascot.bottom - 3)
    if _is_left(resolved_corner):
        base_x = bubble.left + max(1, outline_width)
        tip_x = mascot.right
    else:
        base_x = bubble.right - max(1, outline_width)
        tip_x = mascot.left
    tail = ((base_x, base_y1), (base_x, base_y2), (tip_x, tip_y))

    block = mascot.union(bubble).union(_rect_from_points(tail))
    return MascotLayout(
        mascot=mascot,
        bubble=bubble,
        font_size=font_size,
        lines=LineTuple(lines),
        block=block,
        question=text,
        corner=resolved_corner,
        padding=(pad_x, pad_y),
        line_height=line_height,
        radius=radius,
        outline_width=outline_width,
        tail=tail,
        font_path=str(selected_font),
    )


def bubble_text_origin(layout: MascotLayout) -> tuple[int, int]:
    """Top-left pixel where the first rendered line begins."""
    pad_x, pad_y = layout.padding
    return layout.bubble.x + pad_x, layout.bubble.y + pad_y
