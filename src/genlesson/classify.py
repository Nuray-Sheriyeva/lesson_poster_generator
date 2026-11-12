"""Stage 2: classify lesson segments and decide the poster language."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .config import PlannerOptions
from .errors import GenLessonError
from .ingest.pdf import count_chars, normalize_text
from .models import (
    ClassificationDocument,
    ClassifyConfig,
    ClassifyStats,
    LessonDocument,
    RunManifest,
    SegmentArtifact,
    TopicArtifact,
    utc_now,
)
from .planner import (
    ClassificationRequest,
    LanguageRequest,
    PageImage,
    Planner,
    SegmentInput,
    build_planner,
)
from .runs import RunPaths, RunStore
from .storage import read_model, sha256_file, write_json_atomic

log = logging.getLogger(__name__)

MAX_SEGMENT_CHARS = 1800
MAX_LANGUAGE_CHARS = 8000
LOW_CONFIDENCE = 0.6

_BLOCK_RE = re.compile(r"\n[ \t]*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_LANGUAGE_NAMES = {
    "arabic": "ar",
    "armenian": "hy",
    "azerbaijani": "az",
    "belarusian": "be",
    "chinese": "zh",
    "dutch": "nl",
    "english": "en",
    "farsi": "fa",
    "french": "fr",
    "georgian": "ka",
    "german": "de",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "kazakh": "kk",
    "korean": "ko",
    "kyrgyz": "ky",
    "mongolian": "mn",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "qazaq": "kk",
    "russian": "ru",
    "spanish": "es",
    "tajik": "tg",
    "turkish": "tr",
    "turkmen": "tk",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
}


@dataclass
class ClassifyResult:
    document: ClassificationDocument
    output_path: Path
    resumed: bool


def segment_document(document: LessonDocument) -> list[SegmentInput]:
    segments: list[SegmentInput] = []
    order = 0
    for page in document.pages:
        for block in _split_text(page.text):
            order += 1
            segments.append(
                SegmentInput(
                    id=f"s{order:04d}",
                    page=page.index,
                    page_label=page.label,
                    order=order,
                    text=block,
                    text_source=page.text_source,
                )
            )
    return segments


def _split_text(text: str) -> list[str]:
    blocks: list[str] = []
    for raw_block in _BLOCK_RE.split(text):
        block = normalize_text(raw_block)
        if not block:
            continue
        blocks.extend(_split_to_limit(block, MAX_SEGMENT_CHARS))
    return blocks


def _split_to_limit(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    output: list[str] = []
    buffer = ""
    for line in text.split("\n"):
        if len(line) > limit:
            if buffer:
                output.append(buffer)
                buffer = ""
            output.extend(_split_sentences(line, limit))
            continue
        candidate = line if not buffer else f"{buffer}\n{line}"
        if len(candidate) <= limit:
            buffer = candidate
        else:
            if buffer:
                output.append(buffer)
            buffer = line
    if buffer:
        output.append(buffer)
    return [chunk.strip() for chunk in output if chunk.strip()]


def _split_sentences(text: str, limit: int) -> list[str]:
    output: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_RE.split(text):
        if len(sentence) > limit:
            if buffer:
                output.append(buffer)
                buffer = ""
            output.extend(
                sentence[start : start + limit]
                for start in range(0, len(sentence), limit)
            )
            continue
        candidate = sentence if not buffer else f"{buffer} {sentence}"
        if len(candidate) <= limit:
            buffer = candidate
        else:
            if buffer:
                output.append(buffer)
            buffer = sentence
    if buffer:
        output.append(buffer)
    return output


def normalize_bcp47(value: str) -> str | None:
    raw = value.strip()
    if not raw:
        return None
    key = re.split(r"[\s(]", raw.lower(), maxsplit=1)[0].replace("_", "-")
    key = _LANGUAGE_NAMES.get(key, key)
    parts = key.split("-")
    primary = parts[0]
    if not (2 <= len(primary) <= 3 and primary.isalpha()):
        return None
    normalized = [primary]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            normalized.append(part.title())
        elif (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            normalized.append(part.upper())
        elif len(part) == 4 and part.isdigit():
            normalized.append(part)
        elif 5 <= len(part) <= 8 and part.isalnum():
            normalized.append(part.lower())
        else:
            return None
    return "-".join(normalized)


def _normalize_confidence(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GenLessonError(
            "language_undetermined", "Luna returned an invalid confidence."
        ) from exc
    if math.isnan(number) or math.isinf(number):
        raise GenLessonError("language_undetermined", "Luna returned an invalid confidence.")
    if 0 <= number <= 1:
        return number
    if 1 < number <= 100:
        return number / 100.0
    raise GenLessonError(
        "language_undetermined",
        f"Luna returned confidence outside 0..1: {value!r}",
    )


def _select_page_images(
    document: LessonDocument,
    paths: RunPaths,
    limit: int,
    warnings: list[str],
) -> list[PageImage]:
    candidates = [page for page in document.pages if page.image_path]
    if not candidates or limit <= 0:
        return []
    if len(candidates) > limit:
        if limit == 1:
            indexes = [0]
        else:
            indexes = sorted(
                {round(pos * (len(candidates) - 1) / (limit - 1)) for pos in range(limit)}
            )
        warnings.append(
            f"Used {len(indexes)} of {len(candidates)} page images as planner context."
        )
    else:
        indexes = list(range(len(candidates)))

    images: list[PageImage] = []
    for index in indexes:
        page = candidates[index]
        path = paths.resolve(page.image_path or "")
        if not path.is_file():
            raise GenLessonError(
                "artifact_missing",
                f"Rendered page image is missing: {page.image_path}",
                hint="Re-run ingestion for a new run.",
            )
        images.append(PageImage(page=page.index, path=path))
    return images


def _decision_map(
    draft_segments,
    segments: list[SegmentInput],
) -> dict:
    expected = {segment.id: segment for segment in segments}
    by_id: dict = {}
    duplicates: set[str] = set()
    for decision in draft_segments:
        if decision.id in by_id:
            duplicates.add(decision.id)
        by_id[decision.id] = decision
    missing = sorted(set(expected) - set(by_id))
    unknown = sorted(set(by_id) - set(expected))
    if duplicates or missing or unknown:
        raise GenLessonError(
            "planner_response_invalid",
            "Luna did not classify every segment exactly once.",
            hint="Retry the stage, or switch --planner-model.",
            details={
                "missing": missing[:20],
                "unknown": unknown[:20],
                "duplicates": sorted(duplicates)[:20],
            },
        )
    return by_id


def _topic_artifacts(
    draft_topics,
    content: list[SegmentArtifact],
    warnings: list[str],
) -> list[TopicArtifact]:
    content_ids = {segment.id for segment in content}
    char_by_id = {segment.id: segment.char_count for segment in content}
    order_by_id = {segment.id: segment.order for segment in content}
    merged: dict[str, set[str]] = {}
    unknown_ids: set[str] = set()
    for topic in draft_topics:
        title = topic.title.strip()
        if not title:
            continue
        ids = merged.setdefault(title, set())
        for segment_id in topic.segment_ids:
            if segment_id in content_ids:
                ids.add(segment_id)
            else:
                unknown_ids.add(segment_id)

    topics: list[TopicArtifact] = []
    for title, ids in merged.items():
        if not ids:
            continue
        ordered = sorted(ids, key=lambda value: order_by_id[value])
        topics.append(
            TopicArtifact(
                title=title,
                segment_ids=ordered,
                char_count=sum(char_by_id[value] for value in ordered),
            )
        )
    if unknown_ids:
        warnings.append(
            "Luna assigned non-content or unknown segment ids to topics; those ids were ignored."
        )
    topics.sort(key=lambda topic: (-topic.char_count, topic.title.casefold()))
    return topics


def _language_evidence(
    document: LessonDocument,
    content: list[SegmentArtifact],
    warnings: list[str],
) -> tuple[list[SegmentArtifact], str]:
    text_layer = [segment for segment in content if segment.text_source == "text_layer"]
    if document.stats.scanned_document:
        return content, "vision_transcript"
    if text_layer:
        if len(text_layer) < len(content):
            warnings.append(
                "Vision-transcribed text was excluded from the language decision because "
                "a text layer was available."
            )
        return text_layer, "text_layer"
    warnings.append(
        "No lesson content came from the PDF text layer; the language decision uses vision "
        "transcripts."
    )
    return content, "vision_transcript"


def _sample_language_text(text: str, warnings: list[str]) -> str:
    if len(text) <= MAX_LANGUAGE_CHARS:
        return text
    warnings.append(
        f"Language evidence was truncated to the first {MAX_LANGUAGE_CHARS} characters."
    )
    return text[:MAX_LANGUAGE_CHARS]


def _classify_document(
    document: LessonDocument,
    *,
    lesson_sha256: str,
    options: PlannerOptions,
    planner: Planner,
    paths: RunPaths,
) -> ClassificationDocument:
    segments = segment_document(document)
    if not segments:
        raise GenLessonError(
            "no_lesson_content",
            "No text segments were available to classify.",
            hint="Check the ingested lesson text and re-run ingestion.",
        )
    warnings: list[str] = []
    capabilities = getattr(planner, "capabilities", None)
    supports_vision = bool(getattr(capabilities, "vision", True))
    images = (
        _select_page_images(document, paths, options.max_page_images, warnings)
        if supports_vision
        else []
    )
    if not supports_vision:
        warnings.append(
            f"Planner {options.provider}/{planner.model} lacks vision; page images were omitted."
        )
    max_context = int(getattr(capabilities, "max_context", 0) or 0)
    if max_context and len(document.title or "") + sum(
        len(segment.text) for segment in segments
    ) > max_context * 4:
        warnings.append(
            "Lesson text may exceed the planner's max_context; "
            "classification was sent as one request."
        )
    draft = planner.classify(
        ClassificationRequest(
            title=document.title,
            segments=tuple(segments),
            images=tuple(images),
            scanned_document=document.stats.scanned_document,
        )
    )
    decisions = _decision_map(draft.segments, segments)

    artifacts: list[SegmentArtifact] = []
    for segment in segments:
        decision = decisions[segment.id]
        if decision.label == "lesson_content":
            category = "lesson_content"
        elif decision.category == "lesson_content":
            category = "other"
        else:
            category = decision.category
        artifacts.append(
            SegmentArtifact(
                id=segment.id,
                page=segment.page,
                page_label=segment.page_label,
                order=segment.order,
                text=segment.text,
                char_count=count_chars(segment.text),
                text_source=segment.text_source,
                label=decision.label,
                category=category,
                reason=decision.reason.strip()[:500],
            )
        )

    content = [segment for segment in artifacts if segment.label == "lesson_content"]
    if not content:
        raise GenLessonError(
            "no_lesson_content",
            "Luna marked every segment as non-content, so there is no lesson to poster.",
            hint="Check that the file contains lesson body text, not only instructions or tasks.",
            details={"segments": len(artifacts)},
        )

    topics = _topic_artifacts(draft.topics, content, warnings)
    dominant_topic = topics[0].title if topics else document.title
    if len(topics) > 1:
        names = ", ".join(topic.title for topic in topics[:3])
        warnings.append(
            f"Multiple topics detected ({names}); using dominant topic "
            f"'{dominant_topic}'."
        )
        log.warning(
            "Multiple topics detected (%s); using dominant topic %r",
            names,
            dominant_topic,
        )
    elif not topics:
        warnings.append(
            "No usable topic grouping was returned; using the lesson title as the topic."
        )

    language_segments, language_source = _language_evidence(document, content, warnings)
    language_text = _sample_language_text(
        "\n\n".join(segment.text for segment in language_segments), warnings
    )
    if not language_text.strip():
        raise GenLessonError("language_undetermined", "No lesson-content text was available.")

    decision = planner.detect_language(
        LanguageRequest(text=language_text, title=document.title)
    )
    poster_language = normalize_bcp47(decision.poster_language)
    if poster_language is None:
        raise GenLessonError(
            "language_undetermined",
            f"Luna returned an invalid BCP-47 language tag: {decision.poster_language!r}",
            hint="Retry the stage, or switch --planner-model.",
        )
    confidence = _normalize_confidence(decision.confidence)
    evidence = decision.evidence.strip()[:500]
    if not evidence:
        warnings.append("Luna returned no evidence note for the language decision.")
    if confidence < LOW_CONFIDENCE:
        warnings.append(
            f"Language confidence is low ({confidence:.2f}); "
            "review poster_language before generating."
        )

    lesson_text = "\n\n".join(segment.text for segment in content)
    stats = ClassifyStats(
        segment_count=len(artifacts),
        content_segments=len(content),
        non_content_segments=len(artifacts) - len(content),
        content_char_count=count_chars(lesson_text),
        language_char_count=count_chars(language_text),
        page_images_used=len(images),
    )
    return ClassificationDocument(
        source_sha256=document.source.sha256,
        lesson_sha256=lesson_sha256,
        poster_language=poster_language,
        confidence=confidence,
        evidence=evidence,
        language_source=language_source,
        dominant_topic=dominant_topic,
        topics=topics,
        warnings=warnings,
        segments=artifacts,
        lesson_text=lesson_text,
        stats=stats,
        config=ClassifyConfig(
            planner_provider=options.provider,
            planner_model=planner.model,
            max_page_images=options.max_page_images,
        ),
        classified_at=utc_now(),
    )


def _capabilities_dict(planner: Planner) -> dict:
    capabilities = getattr(planner, "capabilities", None)
    if capabilities is None:
        return {}
    return {
        "structured_output": bool(getattr(capabilities, "structured_output", False)),
        "vision": bool(getattr(capabilities, "vision", False)),
        "max_context": int(getattr(capabilities, "max_context", 0) or 0),
    }


def run_classify(
    paths: RunPaths,
    manifest: RunManifest,
    store: RunStore,
    options: PlannerOptions,
    *,
    planner: Planner | None = None,
) -> ClassifyResult:
    source = manifest.source
    if source is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing source metadata.")
    ingest = manifest.stages.get("ingest")
    if ingest is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the ingest stage.")
    if ingest.status != "complete":
        raise GenLessonError(
            "stage_incomplete",
            "Ingestion must complete before classification can run.",
            hint="Run `genlesson ingest ... --resume` first.",
        )

    lesson_path = paths.stage_dir("ingest") / "lesson.json"
    document = read_model(lesson_path, LessonDocument)
    if document.source.sha256 != source.sha256:
        raise GenLessonError(
            "artifact_invalid",
            "lesson.json does not match the source recorded in the manifest.",
        )
    lesson_sha256 = sha256_file(lesson_path)

    stage = manifest.stages.get("classify")
    if stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the classify stage.")
    output_path = paths.stage_dir("classify") / "classification.json"
    if stage.status == "complete":
        classification = read_model(output_path, ClassificationDocument)
        if (
            classification.source_sha256 != source.sha256
            or classification.lesson_sha256 != lesson_sha256
        ):
            raise GenLessonError(
                "artifact_invalid",
                "classification.json does not match the current ingest artifact.",
            )
        return ClassifyResult(document=classification, output_path=output_path, resumed=True)

    stage_dir = paths.stage_dir("classify")
    stage_dir.mkdir(parents=True, exist_ok=True)
    store.set_stage(paths, manifest, "classify", "running")
    try:
        if planner is None:
            planner = build_planner(options)
        classification = _classify_document(
            document,
            lesson_sha256=lesson_sha256,
            options=options,
            planner=planner,
            paths=paths,
        )
        write_json_atomic(output_path, classification.model_dump(mode="json"))
        manifest.config["planner"] = {
            "provider": options.provider,
            "model": planner.model,
            "max_context": options.max_context,
            "capabilities": _capabilities_dict(planner),
        }
        manifest.config["classify"] = {
            "planner_provider": options.provider,
            "planner_model": planner.model,
            "max_page_images": options.max_page_images,
        }
    except GenLessonError as exc:
        store.set_stage(paths, manifest, "classify", "failed", error=exc.to_dict())
        raise
    except Exception as exc:
        log.exception("Unexpected classification failure")
        wrapped = GenLessonError(
            "internal_error",
            f"Classification failed unexpectedly: {exc}",
            details={"exception": type(exc).__name__},
        )
        store.set_stage(paths, manifest, "classify", "failed", error=wrapped.to_dict())
        raise wrapped from exc

    store.set_stage(
        paths,
        manifest,
        "classify",
        "complete",
        output=paths.relative(output_path),
    )
    return ClassifyResult(
        document=classification, output_path=output_path, resumed=False
    )
