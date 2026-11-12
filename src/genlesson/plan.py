"""Stage 3: turn classified lesson content into a strict poster spec.

Short lessons use one Luna call. Long lessons are split into chunks, each chunk
produces candidate ideas, and a final Luna call merges those candidates into the
poster spec. Every Luna response is validated against a strict JSON schema.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .classify import normalize_bcp47
from .config import PlanOptions
from .errors import GenLessonError
from .models import (
    CentralIdea,
    ClassificationDocument,
    Composition,
    LessonDocument,
    PlanConfig,
    PlanStats,
    PosterDetail,
    PosterSpecDocument,
    RunManifest,
    SourceConflict,
    Style,
    utc_now,
)
from .planner import (
    CandidateCentralIdea,
    CandidateDetail,
    CandidateRequest,
    CandidateSet,
    PageImage,
    PlanRequest,
    PosterPlanner,
    PosterSpecDraft,
    build_planner,
)
from .runs import RunPaths, RunStore
from .storage import read_model, sha256_file, write_json_atomic

log = logging.getLogger(__name__)

MAX_TITLE = 80
MAX_DEFINITION = 180
MAX_LABEL = 60
MAX_EXPLANATION = 240
MAX_EXAMPLE = 180
MAX_VISUAL = 320
MAX_COMPOSITION_DESCRIPTION = 600
MAX_STYLE_FIELD = 200
MAX_PALETTE_ENTRY = 60
MAX_RENDER_STRING = 260
MAX_MASCOT_QUESTION = 90
MAX_CONFLICT_FIELD = 400

_COMPOSITION_FORBIDDEN = re.compile(r"\b(?:grid|table|mind[\s-]?map)\b", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass
class PlanResult:
    document: PosterSpecDocument
    output_path: Path
    resumed: bool


@dataclass
class _Chunk:
    index: int
    text: str
    source_pages: tuple[int, ...]


def _inline(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def _optional_inline(text: str | None) -> str | None:
    value = _inline(text or "")
    return value or None


def _content_segments(classification: ClassificationDocument) -> list:
    return [
        segment
        for segment in classification.segments
        if segment.label == "lesson_content" and segment.text.strip()
    ]


def _split_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts = _SENTENCE_RE.split(text)
    output: list[str] = []
    buffer = ""
    for part in parts:
        if len(part) > limit:
            if buffer:
                output.append(buffer)
                buffer = ""
            output.extend(part[start : start + limit] for start in range(0, len(part), limit))
            continue
        candidate = part if not buffer else f"{buffer} {part}"
        if len(candidate) <= limit:
            buffer = candidate
        else:
            if buffer:
                output.append(buffer)
            buffer = part
    if buffer:
        output.append(buffer)
    return [chunk.strip() for chunk in output if chunk.strip()]


def _chunk_content(classification: ClassificationDocument, limit: int) -> list[_Chunk]:
    raw: list[tuple[str, int]] = []
    for segment in _content_segments(classification):
        for piece in _split_text(segment.text, limit):
            raw.append((piece, segment.page))

    chunks: list[_Chunk] = []
    buffer: list[str] = []
    pages: set[int] = set()
    chars = 0
    for text, page in raw:
        added = len(text) + 2
        if buffer and chars + added > limit:
            chunks.append(
                _Chunk(
                    index=len(chunks) + 1,
                    text="\n\n".join(buffer),
                    source_pages=tuple(sorted(pages)),
                )
            )
            buffer = []
            pages = set()
            chars = 0
        buffer.append(text)
        pages.add(page)
        chars += added
    if buffer:
        chunks.append(
            _Chunk(
                index=len(chunks) + 1,
                text="\n\n".join(buffer),
                source_pages=tuple(sorted(pages)),
            )
        )
    return chunks


def _merge_candidates(
    candidate_sets: list[CandidateSet],
    *,
    max_details: int,
    warnings: list[str],
) -> CandidateSet:
    central: dict[str, CandidateCentralIdea] = {}
    details: dict[str, CandidateDetail] = {}
    terms: dict[str, str] = {}
    conflicts: dict[str, SourceConflict] = {}

    for candidate_set in candidate_sets:
        for idea in candidate_set.central_ideas:
            key = _inline(idea.title).casefold()
            if key and key not in central:
                central[key] = idea
        for detail in candidate_set.details:
            key = _inline(detail.label).casefold()
            if key and key not in details:
                details[key] = detail
        for term in candidate_set.key_terms:
            value = _inline(term)
            if value:
                terms.setdefault(value.casefold(), value)
        for conflict in candidate_set.source_conflicts:
            key = _inline(conflict.issue).casefold()
            if key and key not in conflicts:
                conflicts[key] = conflict

    cap = max(16, min(48, max_details * 3))
    merged_details = list(details.values())
    if len(merged_details) > cap:
        warnings.append(f"Merged {len(merged_details)} candidate details; kept the first {cap}.")
        merged_details = merged_details[:cap]
    merged_central = list(central.values())[:8]
    merged_terms = list(terms.values())[:30]
    merged_conflicts = list(conflicts.values())[:10]
    return CandidateSet(
        central_ideas=merged_central,
        details=merged_details,
        key_terms=merged_terms,
        source_conflicts=merged_conflicts,
    )


def _select_page_images(
    lesson: LessonDocument,
    classification: ClassificationDocument,
    paths: RunPaths,
    limit: int,
    warnings: list[str],
) -> list[PageImage]:
    if limit <= 0:
        return []
    content_pages = {
        segment.page for segment in classification.segments if segment.label == "lesson_content"
    }
    candidates = [page for page in lesson.pages if page.image_path]
    if not candidates:
        return []
    preferred = [page for page in candidates if page.index in content_pages]
    rest = [page for page in candidates if page.index not in content_pages]
    ordered = preferred + rest

    if len(ordered) > limit:
        if limit == 1:
            indexes = [0]
        else:
            indexes = sorted(
                {round(pos * (len(ordered) - 1) / (limit - 1)) for pos in range(limit)}
            )
        warnings.append(f"Used {len(indexes)} of {len(ordered)} page images as planner context.")
    else:
        indexes = list(range(len(ordered)))

    images: list[PageImage] = []
    for index in indexes:
        page = ordered[index]
        path = paths.resolve(page.image_path or "")
        if not path.is_file():
            raise GenLessonError(
                "artifact_missing",
                f"Rendered page image is missing: {page.image_path}",
                hint="Re-run ingestion for a new run.",
            )
        images.append(PageImage(page=page.index, path=path))
    return images


def _check_length(value: str, field: str, limit: int, problems: list[str]) -> None:
    if len(value) > limit:
        problems.append(f"{field} is {len(value)} characters (max {limit}).")


def _normalize_mascot_question(value: str | None, warnings: list[str]) -> str | None:
    question = _optional_inline(value)
    if not question:
        return None
    if not question.endswith(("?", "!")):
        question = f"{question}?"
    if len(question) > MAX_MASCOT_QUESTION:
        cut = question[: MAX_MASCOT_QUESTION - 1].rstrip()
        if " " in cut:
            word_cut = cut.rsplit(" ", 1)[0].rstrip()
            if len(word_cut) >= (MAX_MASCOT_QUESTION - 1) * 0.55:
                cut = word_cut
        question = cut.rstrip(" ,;:-—") + "?"
        warnings.append(
            f"Mascot question shortened to {MAX_MASCOT_QUESTION} characters for the bubble."
        )
    return question


def _normalize_slug(value: str, title: str, warnings: list[str]) -> str:
    original = value
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = slug[:60].rstrip("-")
    if not slug:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60].rstrip("-")
    if not slug:
        slug = "lesson-poster"
    if slug != original:
        warnings.append(f"Normalized file_slug to {slug!r}.")
    return slug


def _build_spec(
    draft: PosterSpecDraft,
    *,
    classification: ClassificationDocument,
    lesson: LessonDocument,
    options: PlanOptions,
    planner: PosterPlanner,
    classification_sha256: str,
    chunks: int,
    candidate_details: int,
    page_images_used: int,
) -> PosterSpecDocument:
    warnings: list[str] = []

    model_language = normalize_bcp47(draft.language)
    if model_language != classification.poster_language:
        warnings.append(
            "Planner language differed from Stage 2; using the Stage 2 poster language "
            f"({classification.poster_language})."
        )

    central_title = _inline(draft.central_idea.title)
    central_definition = _inline(draft.central_idea.definition)
    problems: list[str] = []
    if not central_title:
        problems.append("central_idea.title is empty.")
    if not central_definition:
        problems.append("central_idea.definition is empty.")
    _check_length(central_title, "central_idea.title", MAX_TITLE, problems)
    _check_length(central_definition, "central_idea.definition", MAX_DEFINITION, problems)

    details: list[PosterDetail] = []
    seen_labels: set[str] = set()
    page_count = len(lesson.pages)
    for index, detail in enumerate(draft.details, start=1):
        label = _inline(detail.label)
        explanation = _inline(detail.explanation)
        example = _optional_inline(detail.example)
        visual = _inline(detail.visual)
        key = label.casefold()
        if not label or not explanation or not visual:
            problems.append(f"detail {index} is missing a label, explanation, or visual.")
            continue
        if key in seen_labels:
            warnings.append(f"Removed duplicate detail label {label!r}.")
            continue
        seen_labels.add(key)
        source_page = detail.source_page
        if not 1 <= source_page <= page_count:
            warnings.append(
                f"Detail {label!r} used source page {source_page}; clamped to the document range."
            )
            source_page = min(max(source_page, 1), page_count)
        _check_length(label, f"detail {index} label", MAX_LABEL, problems)
        _check_length(explanation, f"detail {index} explanation", MAX_EXPLANATION, problems)
        _check_length(visual, f"detail {index} visual", MAX_VISUAL, problems)
        if example:
            _check_length(example, f"detail {index} example", MAX_EXAMPLE, problems)
        details.append(
            PosterDetail(
                label=label,
                explanation=explanation,
                example=example,
                visual=visual,
                source_page=source_page,
            )
        )
        if len(details) >= options.max_details:
            break

    if len(draft.details) > options.max_details:
        warnings.append(
            f"Planner returned {len(draft.details)} details; kept {options.max_details}."
        )
    if not details:
        raise GenLessonError(
            "no_plan_details",
            "The planner produced no usable poster details.",
            hint="Retry the stage, or check that the lesson content has enough substance.",
        )

    concept = _inline(draft.composition.concept)
    composition_description = _inline(draft.composition.description)
    if not concept or not composition_description:
        problems.append("composition is missing a concept or description.")
    if _COMPOSITION_FORBIDDEN.search(concept):
        problems.append("composition.concept is a forbidden layout (grid, table, or mind map).")
    _check_length(concept, "composition.concept", 120, problems)
    _check_length(
        composition_description,
        "composition.description",
        MAX_COMPOSITION_DESCRIPTION,
        problems,
    )

    palette = [_inline(color) for color in draft.style.palette if _inline(color)]
    if len(palette) < 2:
        problems.append("style.palette must contain at least two colors.")
    if len(palette) > 8:
        warnings.append("Trimmed style.palette to eight colors.")
        palette = palette[:8]
    if any(len(color) > MAX_PALETTE_ENTRY for color in palette):
        problems.append("A style.palette entry is too long.")
    mood = _inline(draft.style.mood)
    illustration_style = _inline(draft.style.illustration_style)
    if not mood or not illustration_style:
        problems.append("style is missing mood or illustration_style.")
    _check_length(mood, "style.mood", MAX_STYLE_FIELD, problems)
    _check_length(illustration_style, "style.illustration_style", MAX_STYLE_FIELD, problems)

    if problems:
        raise GenLessonError(
            "plan_invalid",
            "The planner returned a poster spec that failed validation.",
            hint="Retry the stage, or switch --planner-model.",
            details={"problems": problems[:20]},
        )

    canonical_strings = [central_title, central_definition]
    for detail in details:
        canonical_strings.append(detail.label)
        canonical_strings.append(detail.explanation)
        if detail.example:
            canonical_strings.append(detail.example)

    mascot_question = _normalize_mascot_question(draft.mascot_question, warnings)
    mascot_question_key = _inline(mascot_question or "").casefold()

    strings: list[str] = []
    seen_strings: set[str] = set()
    for value in canonical_strings:
        if value and value not in seen_strings:
            strings.append(value)
            seen_strings.add(value)
    for raw in draft.strings_to_render:
        value = _inline(raw)
        if not value or value in seen_strings:
            continue
        if mascot_question_key and value.casefold() == mascot_question_key:
            warnings.append("Removed the mascot question from strings_to_render.")
            continue
        if len(value) > MAX_RENDER_STRING:
            warnings.append("Dropped an over-long extra string from strings_to_render.")
            continue
        strings.append(value)
        seen_strings.add(value)

    conflicts: list[SourceConflict] = []
    seen_conflicts: set[str] = set()
    for conflict in draft.source_conflicts:
        issue = _inline(conflict.issue)
        resolution = _inline(conflict.resolution)
        key = issue.casefold()
        if not issue or not resolution or key in seen_conflicts:
            continue
        seen_conflicts.add(key)
        conflicts.append(
            SourceConflict(
                issue=issue[:MAX_CONFLICT_FIELD],
                resolution=resolution[:MAX_CONFLICT_FIELD],
            )
        )
        if len(conflicts) >= 10:
            warnings.append("Trimmed source_conflicts to ten entries.")
            break

    file_slug = _normalize_slug(draft.file_slug, central_title, warnings)
    if not _SLUG_RE.fullmatch(file_slug):
        raise GenLessonError("plan_invalid", f"Could not normalize file_slug: {file_slug!r}")

    config = PlanConfig(
        planner_provider=options.provider,
        planner_model=planner.model,
        max_details=options.max_details,
        context_char_limit=options.context_char_limit,
        chunk_char_limit=options.chunk_char_limit,
        max_page_images=options.max_page_images,
    )
    stats = PlanStats(
        content_chars=classification.stats.content_char_count,
        chunks=chunks,
        candidate_details=candidate_details,
        details=len(details),
        strings=len(strings),
        page_images_used=page_images_used,
    )
    return PosterSpecDocument(
        source_sha256=classification.source_sha256,
        classification_sha256=classification_sha256,
        language=classification.poster_language,
        age_band=draft.age_band,
        central_idea=CentralIdea(title=central_title, definition=central_definition),
        details=details,
        composition=Composition(
            concept=concept,
            description=composition_description,
        ),
        style=Style(
            palette=palette,
            mood=mood,
            illustration_style=illustration_style,
        ),
        strings_to_render=strings,
        mascot_question=mascot_question,
        source_conflicts=conflicts,
        file_slug=file_slug,
        warnings=warnings,
        stats=stats,
        config=config,
        planned_at=utc_now(),
    )


def _plan_document(
    classification: ClassificationDocument,
    lesson: LessonDocument,
    *,
    options: PlanOptions,
    planner: PosterPlanner,
    paths: RunPaths,
    classification_sha256: str,
) -> PosterSpecDocument:
    content_text = classification.lesson_text
    if not content_text.strip():
        raise GenLessonError("no_plan_details", "There is no lesson content to plan from.")

    warnings: list[str] = []
    capabilities = getattr(planner, "capabilities", None)
    supports_vision = bool(getattr(capabilities, "vision", True))
    images = (
        _select_page_images(lesson, classification, paths, options.max_page_images, warnings)
        if supports_vision
        else []
    )
    if not supports_vision:
        warnings.append(
            f"Planner {options.provider}/{planner.model} lacks vision; page images were omitted."
        )
    context_char_limit = options.context_char_limit
    max_context = int(getattr(capabilities, "max_context", 0) or 0)
    if max_context:
        capability_chars = max(1000, max_context * 4)
        if capability_chars < context_char_limit:
            warnings.append(
                f"Planner max_context ({max_context} tokens) lowered the planning "
                f"context to {capability_chars} characters."
            )
            context_char_limit = capability_chars
    chunk_char_limit = min(options.chunk_char_limit, context_char_limit)
    chunks: list[_Chunk] = []
    candidate_details = 0
    candidate_conflicts: list[SourceConflict] = []
    if len(content_text) <= context_char_limit:
        draft = planner.plan(
            PlanRequest(
                language=classification.poster_language,
                title=lesson.title,
                dominant_topic=classification.dominant_topic,
                max_details=options.max_details,
                text=content_text,
                images=tuple(images),
            )
        )
        chunk_count = 1
    else:
        chunks = _chunk_content(classification, chunk_char_limit)
        if not chunks:
            raise GenLessonError("no_plan_details", "Could not split lesson content for planning.")
        candidate_sets: list[CandidateSet] = []
        max_candidates = max(16, min(48, options.max_details * 3))
        for chunk in chunks:
            candidate_set = planner.extract_candidates(
                CandidateRequest(
                    language=classification.poster_language,
                    title=lesson.title,
                    dominant_topic=classification.dominant_topic,
                    text=chunk.text,
                    chunk_index=chunk.index,
                    chunk_count=len(chunks),
                    source_pages=chunk.source_pages,
                    max_candidates=max_candidates,
                )
            )
            candidate_sets.append(candidate_set)
            if not candidate_set.details:
                warnings.append(f"Chunk {chunk.index} produced no candidate details.")
        merged = _merge_candidates(
            candidate_sets,
            max_details=options.max_details,
            warnings=warnings,
        )
        candidate_details = len(merged.details)
        candidate_conflicts = list(merged.source_conflicts)
        if not merged.details:
            raise GenLessonError(
                "no_plan_details",
                "Chunked extraction produced no candidate details.",
                hint="Check the lesson content, or retry with a larger --chunk-chars.",
            )
        draft = planner.plan(
            PlanRequest(
                language=classification.poster_language,
                title=lesson.title,
                dominant_topic=classification.dominant_topic,
                max_details=options.max_details,
                candidates=merged,
                images=tuple(images),
            )
        )
        chunk_count = len(chunks)

    document = _build_spec(
        draft,
        classification=classification,
        lesson=lesson,
        options=options,
        planner=planner,
        classification_sha256=classification_sha256,
        chunks=chunk_count,
        candidate_details=candidate_details,
        page_images_used=len(images),
    )
    existing_conflicts = {conflict.issue.casefold() for conflict in document.source_conflicts}
    for conflict in candidate_conflicts:
        if len(document.source_conflicts) >= 10:
            warnings.append("Trimmed merged source_conflicts to ten entries.")
            break
        if conflict.issue.casefold() not in existing_conflicts:
            document.source_conflicts.append(conflict)
            existing_conflicts.add(conflict.issue.casefold())
    document.warnings = warnings + document.warnings
    return document


def _capabilities_dict(planner: PosterPlanner) -> dict:
    capabilities = getattr(planner, "capabilities", None)
    if capabilities is None:
        return {}
    return {
        "structured_output": bool(getattr(capabilities, "structured_output", False)),
        "vision": bool(getattr(capabilities, "vision", False)),
        "max_context": int(getattr(capabilities, "max_context", 0) or 0),
    }


def run_plan(
    paths: RunPaths,
    manifest: RunManifest,
    store: RunStore,
    options: PlanOptions,
    *,
    planner: PosterPlanner | None = None,
) -> PlanResult:
    source = manifest.source
    if source is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing source metadata.")
    classify = manifest.stages.get("classify")
    if classify is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the classify stage.")
    if classify.status != "complete":
        raise GenLessonError(
            "stage_incomplete",
            "Classification must complete before planning can run.",
            hint="Run `genlesson classify <run_id>` first.",
        )

    lesson_path = paths.stage_dir("ingest") / "lesson.json"
    classification_path = paths.stage_dir("classify") / "classification.json"
    lesson = read_model(lesson_path, LessonDocument)
    classification = read_model(classification_path, ClassificationDocument)
    if lesson.source.sha256 != source.sha256:
        raise GenLessonError("artifact_invalid", "lesson.json does not match the manifest source.")
    lesson_sha256 = sha256_file(lesson_path)
    if classification.source_sha256 != source.sha256:
        raise GenLessonError(
            "artifact_invalid",
            "classification.json does not match the manifest source.",
        )
    if classification.lesson_sha256 != lesson_sha256:
        raise GenLessonError(
            "artifact_invalid",
            "classification.json was produced from a different ingest artifact.",
        )
    classification_sha256 = sha256_file(classification_path)

    stage = manifest.stages.get("plan")
    if stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the plan stage.")
    output_path = paths.stage_dir("plan") / "poster_spec.json"
    if stage.status == "complete":
        document = read_model(output_path, PosterSpecDocument)
        if (
            document.source_sha256 != source.sha256
            or document.classification_sha256 != classification_sha256
        ):
            raise GenLessonError(
                "artifact_invalid",
                "poster_spec.json does not match the current classification artifact.",
            )
        return PlanResult(document=document, output_path=output_path, resumed=True)

    stage_dir = paths.stage_dir("plan")
    stage_dir.mkdir(parents=True, exist_ok=True)
    store.set_stage(paths, manifest, "plan", "running")
    try:
        if planner is None:
            built = build_planner(options.planner_options())
            if not isinstance(built, PosterPlanner):
                raise GenLessonError(
                    "planner_unavailable",
                    "The configured planner cannot produce poster specs.",
                )
            planner = built
        document = _plan_document(
            classification,
            lesson,
            options=options,
            planner=planner,
            paths=paths,
            classification_sha256=classification_sha256,
        )
        write_json_atomic(output_path, document.model_dump(mode="json"))
        manifest.config["planner"] = {
            "provider": options.provider,
            "model": planner.model,
            "max_context": options.max_context,
            "capabilities": _capabilities_dict(planner),
        }
        manifest.config["plan"] = {
            "planner_provider": options.provider,
            "planner_model": planner.model,
            "max_details": options.max_details,
            "context_char_limit": options.context_char_limit,
            "chunk_char_limit": options.chunk_char_limit,
            "max_page_images": options.max_page_images,
            "planner_capabilities": _capabilities_dict(planner),
        }
    except GenLessonError as exc:
        store.set_stage(paths, manifest, "plan", "failed", error=exc.to_dict())
        raise
    except Exception as exc:
        log.exception("Unexpected planning failure")
        wrapped = GenLessonError(
            "internal_error",
            f"Planning failed unexpectedly: {exc}",
            details={"exception": type(exc).__name__},
        )
        store.set_stage(paths, manifest, "plan", "failed", error=wrapped.to_dict())
        raise wrapped from exc

    store.set_stage(
        paths,
        manifest,
        "plan",
        "complete",
        output=paths.relative(output_path),
    )
    return PlanResult(document=document, output_path=output_path, resumed=False)
