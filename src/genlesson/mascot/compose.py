"""Draw the mascot and its question bubble onto a poster."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageColor, ImageDraw

from ..errors import GenLessonError
from .assets import MascotAsset, load_mascot_assets
from .layout import MascotLayout, Rect, layout_mascot, load_font
from .select import (
    Placement,
    choose_variant,
    normalize_requested_corner,
    select_placement,
)
from .zones import TextDetector, ZoneScore, get_text_detector

RGBA = tuple[int, int, int, int]


def _color(value: str, *, label: str) -> RGBA:
    try:
        parsed = ImageColor.getcolor(str(value), "RGBA")
    except ValueError as exc:
        raise GenLessonError(
            "mascot_color_invalid",
            f"Invalid mascot {label} color: {value!r}",
            hint="Use a CSS hex color such as #FFFFFF or #6360FF.",
        ) from exc
    if len(parsed) == 3:  # pragma: no cover - Pillow returns RGBA here
        parsed = (*parsed, 255)
    return parsed  # type: ignore[return-value]


@dataclass(frozen=True)
class Composition:
    image: Image.Image
    variant: str
    corner: str
    layout: MascotLayout
    text_boxes: tuple[Rect, ...]
    overlap_ratio: float
    detector: str

    def save(self, fp, format=None, **params):
        return self.image.save(fp, format=format, **params)

    @property
    def size(self):
        return self.image.size

    @property
    def mode(self):
        return self.image.mode

    def metadata(self) -> dict:
        return {
            "applied": True,
            "variant": self.variant,
            "corner": self.corner,
            "question": self.layout.question,
            "font_size": self.layout.font_size,
            "lines": list(self.layout.lines),
            "mascot": self.layout.mascot.as_dict(),
            "bubble": self.layout.bubble.as_dict(),
            "block": self.layout.block.as_dict(),
            "text_boxes": [box.as_dict() for box in self.text_boxes],
            "overlap_ratio": round(self.overlap_ratio, 4),
            "detector": self.detector,
        }


def _draw_overlay(
    base: Image.Image,
    *,
    layout: MascotLayout,
    asset: MascotAsset,
    config,
) -> Image.Image:
    image = base.convert("RGBA") if base.mode != "RGBA" else base.copy()
    mascot_image = asset.load_for_box(layout.mascot.width, layout.mascot.height)
    image.alpha_composite(mascot_image, (layout.mascot.x, layout.mascot.y))
    draw = ImageDraw.Draw(image)

    fill = _color(getattr(config, "fill", "#FFFFFF"), label="fill")
    outline = _color(getattr(config, "outline", "#6360FF"), label="outline")
    text_color = _color(getattr(config, "text_color", "#3B0764"), label="text")
    outline_width = max(1, int(layout.outline_width))

    bubble = layout.bubble
    radius = min(layout.radius, max(2, min(bubble.width, bubble.height) // 2))

    # Bubble body first, then the tail fill over its border, then the two tail
    # sides.  This leaves a clean join instead of a line across the tail base.
    draw.rounded_rectangle(
        bubble.box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=outline_width,
    )
    if len(layout.tail) >= 3:
        draw.polygon(layout.tail, fill=fill)
        draw.line([layout.tail[0], layout.tail[2]], fill=outline, width=outline_width)
        draw.line([layout.tail[1], layout.tail[2]], fill=outline, width=outline_width)

    font = load_font(layout.font_path, layout.font_size)
    _pad_x, pad_y = layout.padding
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in layout.lines]
    top_offsets = [index * layout.line_height + bbox[1] for index, bbox in enumerate(bboxes)]
    bottom_offsets = [index * layout.line_height + bbox[3] for index, bbox in enumerate(bboxes)]
    ink_top = min(top_offsets, default=0)
    ink_bottom = max(bottom_offsets, default=0)
    ink_height = max(1, ink_bottom - ink_top)
    # Optically center the actual ink block, not the font line boxes.
    top_margin = max(pad_y, (bubble.height - ink_height) // 2)
    base_y = bubble.y + top_margin - ink_top
    for index, line in enumerate(layout.lines):
        bbox = bboxes[index]
        width = bbox[2] - bbox[0]
        x = bubble.x + (bubble.width - width) // 2 - bbox[0]
        y = base_y + index * layout.line_height
        draw.text((x, y), line, font=font, fill=text_color)

    if base.mode != "RGBA":
        return image.convert(base.mode)
    return image


def _placement_for_explicit_corner(
    image: Image.Image,
    question: str,
    *,
    config,
    assets: dict[str, MascotAsset],
    corner: str,
    variant: str | None,
    detector: TextDetector,
) -> Placement:
    resolved = normalize_requested_corner(corner)
    assert resolved is not None
    chosen = variant if variant in assets else choose_variant(question, sorted(assets))
    asset = assets[chosen]
    layout = layout_mascot(
        image.width,
        image.height,
        question,
        resolved,
        config,
        mascot_size=asset.size,
    )
    score = ZoneScore(
        corner=layout.corner,
        layout=layout,
        boxes=(),
        overlap=0,
        overlap_ratio=0.0,
        near_overlap=0,
    )
    return Placement(
        variant=chosen,
        corner=layout.corner,
        layout=layout,
        score=score,
        scores=(score,),
    )


def compose_at_layout(
    base: Image.Image,
    layout: MascotLayout,
    *,
    asset: MascotAsset,
    config,
) -> Image.Image:
    """Draw a pre-resolved layout.  Used by Stage 8's recorded placement."""
    return _draw_overlay(base, layout=layout, asset=asset, config=config)


def check_text_inside_bubble(
    image: Image.Image,
    layout: MascotLayout,
    config,
) -> dict:
    """Pixel-test that rendered bubble text stays inside the padding.

    Text-coloured pixels inside the bubble are collected and their bounding box
    must sit inside ``bubble`` inset by the configured padding.  A failure raises
    instead of silently saving an overflowing bubble.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with the detector
        return {"checked": False, "reason": "numpy_missing"}
    bubble = layout.bubble
    inset = max(1, int(layout.outline_width) + 1)
    crop_rect = Rect(
        bubble.x + inset,
        bubble.y + inset,
        max(1, bubble.width - inset * 2),
        max(1, bubble.height - inset * 2),
    )
    crop_rect = crop_rect.intersection(Rect(0, 0, image.width, image.height))
    if crop_rect.area <= 0:
        raise GenLessonError(
            "mascot_text_outside_bubble",
            "The bubble lies outside the canvas; text placement cannot be checked.",
            details={"bubble": bubble.as_dict()},
        )
    target = _color(getattr(config, "text_color", "#3B0764"), label="text")[:3]
    array = np.asarray(image.convert("RGB").crop(crop_rect.box), dtype=np.int16)
    distance = np.abs(array - np.asarray(target, dtype=np.int16)).sum(axis=2)
    mask = distance < 140
    # Ignore the rounded-corner cut-outs: outside the bubble shape the poster
    # remains visible and can contain dark text that is not bubble text.
    shape = Image.new("L", (crop_rect.width, crop_rect.height), 0)
    ImageDraw.Draw(shape).rounded_rectangle(
        (
            bubble.x + inset - crop_rect.x,
            bubble.y + inset - crop_rect.y,
            bubble.right - inset - 1 - crop_rect.x,
            bubble.bottom - inset - 1 - crop_rect.y,
        ),
        radius=max(2, layout.radius - inset),
        fill=255,
    )
    mask = np.logical_and(mask, np.asarray(shape) > 0)
    count = int(mask.sum())
    if count == 0:
        raise GenLessonError(
            "mascot_text_outside_bubble",
            "No bubble text was found in the composed poster.",
            details={"bubble": bubble.as_dict()},
        )
    ys, xs = np.nonzero(mask)
    min_x = int(xs.min()) + crop_rect.x
    min_y = int(ys.min()) + crop_rect.y
    max_x = int(xs.max()) + crop_rect.x
    max_y = int(ys.max()) + crop_rect.y
    inner = Rect(
        bubble.x + layout.padding[0],
        bubble.y + layout.padding[1],
        max(1, bubble.width - layout.padding[0] * 2),
        max(1, bubble.height - layout.padding[1] * 2),
    )
    if min_x < inner.left or min_y < inner.top or max_x >= inner.right or max_y >= inner.bottom:
        raise GenLessonError(
            "mascot_text_outside_bubble",
            "The composed bubble text is not strictly inside the bubble padding.",
            hint="Check the layout font fitting and bubble padding.",
            details={
                "text_bbox": [min_x, min_y, max_x, max_y],
                "inner": inner.as_dict(),
                "bubble": bubble.as_dict(),
            },
        )
    return {
        "checked": True,
        "text_pixels": count,
        "text_bbox": [min_x, min_y, max_x, max_y],
        "inner": inner.as_dict(),
    }


def compose_recorded(
    base: Image.Image,
    entry,
    *,
    final_corner: str,
    config,
    assets: dict[str, MascotAsset] | None = None,
) -> tuple[Image.Image, MascotLayout, dict]:
    """Compose the recorded final corner and pixel-check the text fit."""
    loaded = assets if assets is not None else load_mascot_assets(config.assets_dir)
    requested_variant = getattr(entry, "variant", None) or (
        entry.get("variant") if isinstance(entry, dict) else None
    )
    if requested_variant in loaded:
        variant = requested_variant
    else:
        variant = choose_variant(_entry_question(entry), sorted(loaded))
    asset = loaded[variant]
    layout = layout_mascot(
        base.width,
        base.height,
        _entry_question(entry),
        final_corner,
        config,
        mascot_size=asset.size,
    )
    image = compose_at_layout(base, layout, asset=asset, config=config)
    self_check = check_text_inside_bubble(image, layout, config)
    return image, layout, self_check


def _entry_question(entry) -> str:
    if isinstance(entry, dict):
        return str(entry.get("question") or "")
    return str(getattr(entry, "question", "") or "")


def compose_overlay(
    base: Image.Image,
    question: str,
    config=None,
    *,
    variant: str | None = None,
    corner: str | None = None,
    detector: TextDetector | None = None,
    preferred_variant: str | None = None,
    preferred_corner: str | None = None,
    assets: dict[str, MascotAsset] | None = None,
) -> Composition:
    """Composite the mascot and question bubble onto ``base``.

    ``corner='auto'`` (the default) asks the text detector for the least
    obstructed corner.  Passing a real corner honors it exactly, which is useful
    for the free preview command.
    """
    if isinstance(base, (bytes, bytearray)):
        try:
            with Image.open(io.BytesIO(bytes(base))) as source:
                base = source.copy()
        except OSError as exc:
            raise GenLessonError(
                "mascot_image_invalid", f"Could not read poster image: {exc}"
            ) from exc
    if not isinstance(base, Image.Image):
        raise GenLessonError(
            "mascot_image_invalid",
            f"Expected a PIL image or PNG bytes, got {type(base).__name__}.",
        )
    if config is None:
        from ..app_config import load_app_config

        config = load_app_config().mascot
    elif isinstance(config, dict):
        from ..app_config import MascotConfig

        config = MascotConfig(**config)
    loaded = assets if assets is not None else load_mascot_assets(config.assets_dir)
    resolved_detector = detector or get_text_detector("auto")
    requested = normalize_requested_corner(corner)
    selected_variant = variant or preferred_variant

    if requested is not None:
        placement = _placement_for_explicit_corner(
            base,
            question,
            config=config,
            assets=loaded,
            corner=requested,
            variant=selected_variant,
            detector=resolved_detector,
        )
    else:
        placement = select_placement(
            base,
            question,
            config=config,
            detector=resolved_detector,
            assets=loaded,
            # Bottom-right is the deterministic tie-break for a blank preview or
            # a poster with no detected text.
            preferred_corner=preferred_corner or "bottom_right",
            preferred_variant=selected_variant,
        )

    asset = loaded[placement.variant]
    image = _draw_overlay(base, layout=placement.layout, asset=asset, config=config)
    boxes = placement.score.boxes if requested is None else ()
    return Composition(
        image=image,
        variant=placement.variant,
        corner=placement.corner,
        layout=placement.layout,
        text_boxes=tuple(boxes),
        overlap_ratio=placement.score.overlap_ratio if requested is None else 0.0,
        detector=resolved_detector.name,
    )


def compose_png_result(
    data: bytes,
    question: str,
    config=None,
    **kwargs,
) -> tuple[bytes, Composition]:
    """Decode PNG/JPEG bytes, composite, and return PNG bytes plus metadata."""
    try:
        with Image.open(io.BytesIO(data)) as source:
            base = source.copy()
            if base.mode not in {"RGB", "RGBA"}:
                base = base.convert("RGB")
    except OSError as exc:
        raise GenLessonError("mascot_image_invalid", f"Could not read poster image: {exc}") from exc
    composition = compose_overlay(base, question, config, **kwargs)
    buffer = io.BytesIO()
    composition.image.save(buffer, format="PNG")
    return buffer.getvalue(), composition


def compose_png(data: bytes, question: str, config=None, **kwargs) -> bytes:
    """Decode PNG/JPEG bytes, composite, and return the encoded PNG bytes."""
    return compose_png_result(data, question, config, **kwargs)[0]


def compose_poster(data: bytes, question: str, config=None, **kwargs) -> bytes:
    """Alias for :func:`compose_png`."""
    return compose_png(data, question, config, **kwargs)


def render_preview(
    width: int,
    height: int,
    question: str,
    config=None,
    *,
    background: str = "#FFFFFF",
    **kwargs,
) -> Composition:
    """Render the overlay on a blank canvas without spending image credits."""
    if width < 1 or height < 1:
        raise GenLessonError("mascot_canvas_too_small", f"Invalid preview size: {width}x{height}.")
    canvas = Image.new("RGBA", (width, height), _color(background, label="background"))
    return compose_overlay(canvas, question, config, **kwargs)
