"""Choose the mascot variant, the lesson question, and the least-obstructed corner."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from PIL import Image

from ..errors import GenLessonError
from .assets import MascotAsset, load_mascot_assets
from .layout import CORNERS, Corner, MascotLayout, normalize_corner
from .zones import TextDetector, ZoneScore, get_text_detector, rank_corners

_FORMAT_CORNERS: dict[str, Corner] = {
    "portrait": "bottom_right",
    "landscape": "bottom_right",
    "square": "bottom_right",
}


def default_corner_for_format(fmt: str) -> Corner:
    return _FORMAT_CORNERS.get(str(fmt).casefold(), "bottom_right")


def choose_variant(
    question: str,
    variants: Sequence[str],
    *,
    avoid: str | None = None,
    salt: str | None = None,
) -> str:
    """Pick a variant deterministically from the question.

    ``avoid`` makes the next variant win, which is used when
    ``mascot.avoid_repeat`` is enabled across poster formats.
    """
    choices = [str(variant) for variant in variants]
    if not choices:
        raise GenLessonError("mascot_assets_missing", "No mascot variants are available.")
    payload = f"{salt or ''}\0{question}".encode()
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(choices)
    chosen = choices[index]
    if avoid and chosen == avoid and len(choices) > 1:
        chosen = choices[(index + 1) % len(choices)]
    return chosen


def _shorten_title(title: str, max_chars: int) -> str:
    clean = " ".join(title.split()).strip(" .!?,;:—-«»“”")
    if len(clean) <= max_chars:
        return clean
    cut = clean[: max(0, max_chars - 1)].rstrip()
    if " " in cut:
        word_cut = cut.rsplit(" ", 1)[0].rstrip()
        # Prefer a whole word unless that would throw away most of the budget.
        if len(word_cut) >= (max_chars - 1) * 0.55:
            cut = word_cut
    while cut and not cut[-1].isalnum():
        cut = cut[:-1].rstrip()
    return f"{cut}…"


QuestionSource = Literal["planner", "central_idea", "fallback"]


@dataclass(frozen=True)
class QuestionPlan:
    """A mascot question plus the provenance recorded in the sidecar."""

    text: str
    source: QuestionSource

    def as_dict(self) -> dict[str, str]:
        return {"question": self.text, "source": self.source}


def _finalize_question(text: str, max_chars: int) -> str:
    question = " ".join(str(text).split())
    if not question.endswith(("?", "!")):
        question = f"{question}?"
    if len(question) > max_chars:
        cut = question[: max(1, max_chars - 1)].rstrip(" ,;:-—")
        question = f"{cut}?"
    return question


def plan_question(spec, config) -> QuestionPlan:
    """Choose the mascot question and record whether the planner supplied it.

    Priority:
    1. ``poster_spec.mascot_question`` from Stage 3 (source ``planner``);
    2. a deterministic question built from the central-idea title
       (source ``central_idea``);
    3. a generic language-appropriate question (source ``fallback``).
    """
    max_chars = int(getattr(config, "question_max_chars", 140))
    planner_question = str(getattr(spec, "mascot_question", "") or "").strip()
    if planner_question:
        return QuestionPlan(_finalize_question(planner_question, max_chars), "planner")

    language = str(getattr(spec, "language", "en") or "en").split("-")[0].casefold()
    central = getattr(spec, "central_idea", None)
    title = str(getattr(central, "title", "") or "").strip()
    if not title:
        fallback_title = {"kk": "бұл сабақ", "ru": "этот урок"}.get(language, "this lesson")
        if language == "kk":
            return QuestionPlan(
                _finalize_question(f"«{fallback_title}» туралы не білесің?", max_chars),
                "fallback",
            )
        if language == "ru":
            return QuestionPlan(
                _finalize_question(f"Что ты знаешь о «{fallback_title}»?", max_chars),
                "fallback",
            )
        return QuestionPlan(
            _finalize_question(f"What do you know about “{fallback_title}”?", max_chars),
            "fallback",
        )

    fixed = {
        "kk": "«» туралы не білесің?",
        "ru": "Что ты знаешь о «»?",
        "en": "What do you know about “”?",
    }.get(language, "What do you know about “”?")
    # A mascot question is a short nudge, not a second title block.  Keep the
    # title short enough for a two- or three-line bubble even at 1024 px.
    title_budget = max(8, min(38, max_chars - len(fixed)))
    title = _shorten_title(title, title_budget)

    if language == "kk":
        question = f"«{title}» туралы не білесің?"
    elif language == "ru":
        question = f"Что ты знаешь о «{title}»?"
    else:
        question = f"What do you know about “{title}”?"
    return QuestionPlan(_finalize_question(question, max_chars), "central_idea")


def build_question(spec, config) -> str:
    """Return only the mascot question text (compatibility wrapper)."""
    return plan_question(spec, config).text


@dataclass(frozen=True)
class Placement:
    variant: str
    corner: Corner
    layout: MascotLayout
    score: ZoneScore
    scores: tuple[ZoneScore, ...]
    method: str = "preferred"
    preferred_corner: Corner | None = None
    accepted: bool = True

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "corner": self.corner,
            "preferred_corner": self.preferred_corner or self.corner,
            "method": self.method,
            "accepted": self.accepted,
            "text_boxes_hit": self.score.text_boxes_hit,
            "busyness": round(self.score.busyness, 4),
            "rect": self.layout.block.as_dict(),
            "layout": self.layout.as_dict(),
            "score": self.score.as_dict(),
            "candidates": [score.as_dict() for score in self.scores],
        }


@dataclass(frozen=True)
class PlacementFailure:
    preferred_corner: Corner
    scores: tuple[ZoneScore, ...]
    text_boxes_hit: int

    def as_dict(self) -> dict:
        return {
            "corner": None,
            "rect": None,
            "method": "none",
            "text_boxes_hit": self.text_boxes_hit,
            "preferred_corner": self.preferred_corner,
            "candidates": [score.as_dict() for score in self.scores],
        }


def select_placement(
    image: Image.Image,
    question: str,
    *,
    config,
    detector: TextDetector | None = None,
    assets: dict[str, MascotAsset] | None = None,
    preferred_corner: str | None = None,
    preferred_variant: str | None = None,
    avoid_variant: str | None = None,
    corners: Sequence[str] = CORNERS,
) -> Placement:
    """Pick the mascot variant and the corner with the least text overlap."""
    loaded = assets if assets is not None else load_mascot_assets(config.assets_dir)
    variants = sorted(loaded)
    if preferred_variant and preferred_variant in loaded:
        variant = preferred_variant
    else:
        variant = choose_variant(
            question,
            variants,
            avoid=avoid_variant if getattr(config, "avoid_repeat", False) else None,
        )
    resolved_detector = detector or get_text_detector("auto")
    scores = rank_corners(
        image,
        question,
        config=config,
        detector=resolved_detector,
        preferred_corner=preferred_corner,
        corners=corners,
        mascot_size=loaded[variant].size,
        busyness_threshold=getattr(config, "busyness_threshold", 0.08),
    )
    best = scores[0]
    preferred = normalize_corner(preferred_corner) if preferred_corner else None
    return Placement(
        variant=variant,
        corner=best.corner,
        layout=best.layout,
        score=best,
        scores=tuple(scores),
        method="preferred" if preferred == best.corner else "detected",
        preferred_corner=preferred or best.corner,
        accepted=best.accepted,
    )


def resolve_placement(
    image: Image.Image,
    question: str,
    *,
    config,
    entry,
    detector: TextDetector | None = None,
    assets: dict[str, MascotAsset] | None = None,
) -> Placement | PlacementFailure:
    """Resolve a Stage 7 placement from a Stage 5 layout entry.

    The preferred corner is tried first, then the remaining corners by
    quietness.  A corner is accepted only when no detected text box intersects
    its block (with a margin) and the block's pixel busyness is under the
    configured threshold.  A non-preferred acceptance is a placement change,
    not a regeneration.
    """
    loaded = assets if assets is not None else load_mascot_assets(config.assets_dir)
    if isinstance(entry, dict):
        preferred_value = entry.get("preferred_corner") or entry.get("corner")
        requested_variant = entry.get("variant")
    else:
        preferred_value = getattr(entry, "preferred_corner", None) or getattr(entry, "corner", None)
        requested_variant = getattr(entry, "variant", None)
    preferred = normalize_corner(preferred_value)
    if requested_variant and requested_variant in loaded:
        variant = requested_variant
    else:
        variant = choose_variant(question, sorted(loaded))
    scores = rank_corners(
        image,
        question,
        config=config,
        detector=detector or get_text_detector("auto"),
        preferred_corner=preferred,
        mascot_size=loaded[variant].size,
        busyness_threshold=getattr(config, "busyness_threshold", 0.08),
    )
    by_corner = {score.corner: score for score in scores}
    ordered = [by_corner[preferred]] if preferred in by_corner else []
    ordered.extend(
        sorted(
            (score for score in scores if score.corner != preferred),
            key=lambda score: score.quiet_key,
        )
    )
    for score in ordered:
        if not score.accepted:
            continue
        method = "preferred" if score.corner == preferred else "detected"
        return Placement(
            variant=variant,
            corner=score.corner,
            layout=score.layout,
            score=score,
            scores=tuple(scores),
            method=method,
            preferred_corner=preferred,
            accepted=True,
        )
    preferred_score = by_corner.get(preferred)
    if preferred_score is None and scores:
        preferred_score = scores[0]
    return PlacementFailure(
        preferred_corner=preferred,
        scores=tuple(scores),
        text_boxes_hit=preferred_score.text_boxes_hit if preferred_score else 0,
    )


def choose_corner(
    image: Image.Image,
    question: str,
    *,
    config,
    detector: TextDetector | None = None,
    preferred_corner: str | None = None,
) -> Corner:
    """Convenience wrapper returning only the selected corner."""
    return select_placement(
        image,
        question,
        config=config,
        detector=detector,
        preferred_corner=preferred_corner,
    ).corner


def normalize_requested_corner(corner: str | None) -> str | None:
    if corner is None or str(corner).strip().casefold() in {"", "auto", "automatic"}:
        return None
    return normalize_corner(str(corner))
