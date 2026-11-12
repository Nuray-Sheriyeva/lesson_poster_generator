"""Text-aware corner zones.

The detector runs only on the candidate rectangle crop, not on the full poster:
that keeps placement cheap and avoids unrelated illustration texture.  Any
dependency that exposes ``detect(image) -> sequence[Rect]`` can be injected,
which keeps the geometry testable without the OCR model.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

from .layout import CORNERS, Corner, MascotLayout, Rect, layout_mascot, normalize_corner

log = logging.getLogger(__name__)

BUSYNESS_THRESHOLD = 0.08
TEXT_MARGIN_RATIO = 0.012


@runtime_checkable
class TextDetector(Protocol):
    name: str

    def detect(self, image: Image.Image) -> Sequence[Rect]:
        """Return text bounding boxes in the coordinate space of ``image``."""


@dataclass(frozen=True)
class ZoneScore:
    corner: Corner
    layout: MascotLayout
    boxes: tuple[Rect, ...]
    overlap: int
    overlap_ratio: float
    near_overlap: int
    text_boxes_hit: int = 0
    busyness: float = 0.0
    accepted: bool = False

    @property
    def clear(self) -> bool:
        return self.overlap == 0

    @property
    def quiet_key(self) -> tuple[int, float, int]:
        return self.text_boxes_hit, self.busyness, self.near_overlap

    def as_dict(self) -> dict:
        return {
            "corner": self.corner,
            "overlap": self.overlap,
            "overlap_ratio": round(self.overlap_ratio, 4),
            "near_overlap": self.near_overlap,
            "text_boxes_hit": self.text_boxes_hit,
            "busyness": round(self.busyness, 4),
            "accepted": self.accepted,
            "text_boxes": [box.as_dict() for box in self.boxes],
            "block": self.layout.block.as_dict(),
        }


def pixel_busyness(image: Image.Image, rect: Rect, *, max_side: int = 256) -> float:
    """Return mean normalized pixel gradient inside ``rect``.

    Empty margins and flat illustration areas score close to zero; dense text,
    logos, and busy art score higher.  The crop is downsampled first so the
    measure is stable across poster sizes and cheap for 2000 px canvases.
    """
    crop_rect = rect.intersection(Rect(0, 0, image.width, image.height))
    if crop_rect.area <= 0:
        return 0.0
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with the detector
        return 0.0
    crop = image.crop(crop_rect.box).convert("L")
    if max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.BILINEAR,
        )
    array = np.asarray(crop, dtype=np.float32)
    if array.ndim != 2 or array.size < 2:
        return 0.0
    gx = float(np.abs(np.diff(array, axis=1)).mean()) if array.shape[1] > 1 else 0.0
    gy = float(np.abs(np.diff(array, axis=0)).mean()) if array.shape[0] > 1 else 0.0
    return max(0.0, min(1.0, (gx + gy) / 255.0))


class NullTextDetector:
    """Placement fallback when OCR is unavailable: assume no text."""

    name = "none"

    def detect(self, image: Image.Image) -> Sequence[Rect]:
        return ()


class RapidOCRDetector:
    """PP-OCR DB detector through RapidOCR (boxes only; language agnostic)."""

    name = "rapidocr-db"

    def __init__(
        self,
        *,
        box_thresh: float = 0.25,
        min_score: float = 0.5,
        min_side: int = 3,
    ) -> None:
        self.box_thresh = box_thresh
        self.min_score = min_score
        self.min_side = min_side
        self._engine = None

    def _load(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise RuntimeError(
                "rapidocr is not installed; install it or use a null detector"
            ) from exc
        # RapidOCR logs every empty crop; the caller treats "no text" as a
        # normal result, so keep CLI output clean.
        self._engine = RapidOCR(params={"Global.log_level": "error"})
        return self._engine

    @property
    def available(self) -> bool:
        try:
            import rapidocr  # noqa: F401
        except ImportError:
            return False
        return True

    def detect(self, image: Image.Image) -> tuple[Rect, ...]:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy ships with the detector
            return ()
        engine = self._load()
        array = np.asarray(image.convert("RGB"))
        try:
            result = engine(
                array,
                use_det=True,
                use_cls=False,
                use_rec=False,
                box_thresh=self.box_thresh,
            )
        except Exception:  # pragma: no cover - detector failure must not break save
            log.warning(
                "Text detection failed; placement will use the preferred corner.", exc_info=True
            )
            return ()
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return ()
        scores = getattr(result, "scores", None)
        rects: list[Rect] = []
        for index, polygon in enumerate(boxes):
            if scores is not None and index < len(scores):
                try:
                    if float(scores[index]) < self.min_score:
                        continue
                except (TypeError, ValueError):
                    pass
            try:
                points = [(float(point[0]), float(point[1])) for point in polygon]
            except (TypeError, ValueError, IndexError):
                continue
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            left, right = round(min(xs)), round(max(xs))
            top, bottom = round(min(ys)), round(max(ys))
            if right - left < self.min_side or bottom - top < self.min_side:
                continue
            rects.append(Rect(left, top, right - left, bottom - top))
        return tuple(rects)


_DETECTOR: TextDetector | None = None


def get_text_detector(name: str = "auto") -> TextDetector:
    """Return a detector by name, falling back to null when OCR is unavailable."""
    global _DETECTOR
    normalized = str(name or "auto").strip().casefold()
    if normalized in {"none", "null", "off"}:
        return NullTextDetector()
    if normalized not in {"auto", "rapidocr", "db", "ppocr"}:
        raise ValueError(f"Unknown text detector: {name!r}")
    if _DETECTOR is not None:
        return _DETECTOR
    detector = RapidOCRDetector()
    if detector.available:
        _DETECTOR = detector
        return detector
    log.warning("rapidocr is not installed; mascot placement will not avoid text.")
    _DETECTOR = NullTextDetector()
    return _DETECTOR


def reset_text_detector() -> None:
    """Test helper: drop the cached detector instance."""
    global _DETECTOR
    _DETECTOR = None


def _normalize_boxes(boxes: Sequence[Rect], *, width: int, height: int) -> tuple[Rect, ...]:
    rects: list[Rect] = []
    for box in boxes:
        if isinstance(box, Rect):
            rect = box
        elif hasattr(box, "x") and hasattr(box, "width"):
            rect = Rect(
                round(float(box.x)),
                round(float(box.y)),
                round(float(box.width)),
                round(float(box.height)),
            )
        else:
            try:
                values = tuple(box)  # type: ignore[arg-type]
            except TypeError:
                continue
            if len(values) == 4:
                rect = Rect(*(round(float(value)) for value in values))
            else:
                continue
        left = min(max(0, rect.left), width)
        top = min(max(0, rect.top), height)
        right = min(max(0, rect.right), width)
        bottom = min(max(0, rect.bottom), height)
        if right > left and bottom > top:
            rects.append(Rect(left, top, right - left, bottom - top))
    return tuple(rects)


def detect_in_block(
    image: Image.Image,
    layout: MascotLayout,
    detector: TextDetector,
    *,
    halo_ratio: float = 0.02,
) -> tuple[Rect, ...]:
    """Detect text inside the candidate block crop and map it to canvas space."""
    width, height = image.size
    halo = max(2, round(min(width, height) * halo_ratio))
    crop_rect = layout.block.expand(halo).intersection(Rect(0, 0, width, height))
    if crop_rect.area <= 0:
        return ()
    crop = image.crop(crop_rect.box)
    local_boxes = detector.detect(crop)
    boxes = _normalize_boxes(local_boxes, width=crop.width, height=crop.height)
    return tuple(
        Rect(box.x + crop_rect.x, box.y + crop_rect.y, box.width, box.height) for box in boxes
    )


def score_layout(
    image: Image.Image,
    layout: MascotLayout,
    detector: TextDetector,
    *,
    halo_ratio: float = 0.02,
    text_margin: int | None = None,
    busyness_threshold: float = BUSYNESS_THRESHOLD,
) -> ZoneScore:
    """Measure detected text, keep-out hits, and pixel busyness for a block."""
    boxes = detect_in_block(image, layout, detector, halo_ratio=halo_ratio)
    halo = max(2, round(min(image.size) * 0.008))
    if text_margin is None:
        text_margin = max(2, round(min(image.size) * TEXT_MARGIN_RATIO))
    keepout = layout.block.expand(max(0, text_margin))
    overlap = sum(box.overlap_area(layout.block) for box in boxes)
    near_overlap = sum(box.overlap_area(layout.block.expand(halo)) for box in boxes)
    text_boxes_hit = sum(1 for box in boxes if box.overlap_area(keepout) > 0)
    busyness = pixel_busyness(image, layout.block)
    ratio = min(1.0, overlap / layout.block.area) if layout.block.area else 0.0
    return ZoneScore(
        corner=layout.corner,
        layout=layout,
        boxes=boxes,
        overlap=overlap,
        overlap_ratio=ratio,
        near_overlap=near_overlap,
        text_boxes_hit=text_boxes_hit,
        busyness=busyness,
        accepted=text_boxes_hit == 0 and busyness < busyness_threshold,
    )


def rank_corners(
    image: Image.Image,
    question: str,
    *,
    config,
    detector: TextDetector,
    preferred_corner: str | None = None,
    corners: Sequence[str] = CORNERS,
    mascot_size: tuple[int, int] | None = None,
    busyness_threshold: float = BUSYNESS_THRESHOLD,
) -> list[ZoneScore]:
    """Return every fitting corner, least text overlap first.

    The preferred corner wins ties, then the fixed corner order keeps the
    result deterministic when the poster has no detected text.
    """
    width, height = image.size
    preferred = normalize_corner(preferred_corner) if preferred_corner else None
    order = [normalize_corner(corner) for corner in corners]
    if preferred is not None:
        # Deduplicate while moving the preferred corner to the front.
        order = [preferred] + [corner for corner in order if corner != preferred]
    scores: list[ZoneScore] = []
    failures: list[dict] = []
    for corner in order:
        try:
            layout = layout_mascot(width, height, question, corner, config, mascot_size=mascot_size)
        except Exception as exc:  # noqa: BLE001 - report the real failure below
            failures.append({"corner": corner, "error": str(exc)})
            continue
        scores.append(
            score_layout(
                image,
                layout,
                detector,
                busyness_threshold=busyness_threshold,
            )
        )
    if not scores:
        from ..errors import GenLessonError

        raise GenLessonError(
            "mascot_canvas_too_small",
            "No corner has room for the mascot and speech bubble.",
            details={"failures": failures},
        )
    scores.sort(
        key=lambda item: (
            item.text_boxes_hit,
            item.busyness,
            item.near_overlap,
            0 if item.corner == preferred else 1,
            order.index(item.corner),
        )
    )
    return scores
