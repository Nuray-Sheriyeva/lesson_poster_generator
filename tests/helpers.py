from __future__ import annotations

from pathlib import Path

import pymupdf
from docx import Document

from genlesson.config import IngestOptions, PlanOptions, PlannerOptions, manifest_config
from genlesson.image import ImageProvider, ImageProviderError, ModerationRefusal
from genlesson.models import (
    CentralIdea,
    Composition,
    PosterDetail,
    SourceConflict,
    Style,
)
from genlesson.planner import (
    PlannerCapabilities,
    CandidateCentralIdea,
    CandidateDetail,
    CandidateRequest,
    CandidateSet,
    ClassificationDraft,
    ClassificationRequest,
    LanguageDecision,
    LanguageRequest,
    PlanRequest,
    Planner,
    PosterPlanner,
    PosterSpecDraft,
    SegmentDecision,
    TopicDraft,
)
from genlesson.runs import RunStore, describe_source
from genlesson.vision.base import Transcriber


def make_text_pdf(path: Path, pages: list[str], *, title: str | None = None) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            pymupdf.Rect(60, 60, 535, 780),
            text,
            fontsize=14,
            fontname="helv",
            lineheight=1.4,
        )
    if title:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()
    return path


def _render_text_image(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        pymupdf.Rect(60, 60, 535, 780),
        text,
        fontsize=20,
        fontname="helv",
        lineheight=1.5,
    )
    pixmap = page.get_pixmap(dpi=150, colorspace=pymupdf.csRGB, alpha=False)
    data = pixmap.tobytes("png")
    doc.close()
    return data


def make_scanned_pdf(path: Path, pages: list[str]) -> Path:
    """Build a PDF whose pages are images and therefore have no text layer."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_image(
            pymupdf.Rect(0, 0, 595, 842),
            stream=_render_text_image(text),
        )
    doc.save(path)
    doc.close()
    return path


def make_mixed_pdf(path: Path, *, text_page: str, scanned_page: str) -> Path:
    doc = pymupdf.open()
    first = doc.new_page(width=595, height=842)
    first.insert_textbox(
        pymupdf.Rect(60, 60, 535, 780),
        text_page,
        fontsize=14,
        fontname="helv",
    )
    second = doc.new_page(width=595, height=842)
    second.insert_image(
        pymupdf.Rect(0, 0, 595, 842),
        stream=_render_text_image(scanned_page),
    )
    doc.save(path)
    doc.close()
    return path


def make_docx(path: Path, *, title: str = "Photosynthesis") -> Path:
    document = Document()
    document.core_properties.title = title
    document.add_heading("Photosynthesis", level=1)
    document.add_paragraph("Plants turn light, water, and carbon dioxide into sugar.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Input"
    table.cell(0, 1).text = "Output"
    table.cell(1, 0).text = "Sunlight"
    table.cell(1, 1).text = "Glucose"
    document.add_paragraph("Terms such as фотосинтез stay in their original language.")
    document.save(path)
    return path


def make_encrypted_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Secret lesson", fontsize=18)
    doc.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    doc.close()
    return path


class FakeTranscriber(Transcriber):
    def __init__(self, texts: dict[int, str] | None = None, *, model: str = "fake-vision"):
        self.model = model
        self.texts = texts or {}
        self.calls: list[int] = []
        self.fail_on: set[int] = set()

    def transcribe(self, image_path: Path, page_number: int) -> str:
        self.calls.append(page_number)
        if page_number in self.fail_on:
            from genlesson.errors import GenLessonError

            raise GenLessonError("vision_failed", f"fake failure on page {page_number}")
        return self.texts.get(page_number, f"Transcribed page {page_number}")


def start_run(
    tmp_path: Path,
    source_path: Path,
    options: IngestOptions,
    *,
    run_id: str = "test-run",
):
    source = describe_source(source_path)
    store = RunStore(tmp_path / "runs")
    paths, manifest = store.create(source, manifest_config(options), run_id=run_id)
    return store, paths, manifest


def default_options(**overrides) -> IngestOptions:
    values = {
        "render_dpi": 72,
        "vision_mode": "none",
        "vision_model": "fake-vision",
        "openai_api_key": None,
    }
    values.update(overrides)
    return IngestOptions(**values)


def make_lesson_docx(path: Path, paragraphs: list[str], *, title: str = "Lesson") -> Path:
    document = Document()
    document.core_properties.title = title
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)
    return path


def planner_options(**overrides) -> PlannerOptions:
    values = {
        "model": "fake-luna",
        "api_key": "test-key",
        "max_page_images": 6,
    }
    values.update(overrides)
    return PlannerOptions(**values)


class FakePlanner(Planner):
    """Deterministic Stage 2 planner for tests. No network."""

    def __init__(
        self,
        *,
        non_content_markers: tuple[str, ...] = (),
        all_non_content: bool = False,
        language: str = "kk",
        confidence: float = 0.92,
        evidence: str = "BODY TEXT IS KAZAKH",
        topic_count: int = 1,
        missing_ids: tuple[str, ...] = (),
        model: str = "fake-luna",
        vision: bool = True,
        structured_output: bool = True,
        max_context: int = 1050000,
    ) -> None:
        self.model = model
        self.capabilities = PlannerCapabilities(
            structured_output=structured_output,
            vision=vision,
            max_context=max_context,
        )
        self.non_content_markers = tuple(marker.casefold() for marker in non_content_markers)
        self.all_non_content = all_non_content
        self.language = language
        self.confidence = confidence
        self.evidence = evidence
        self.topic_count = topic_count
        self.missing_ids = set(missing_ids)
        self.classify_calls: list[ClassificationRequest] = []
        self.language_calls: list[LanguageRequest] = []

    def _category(self, text: str) -> str:
        lowered = text.casefold()
        if "objective" in lowered:
            return "learning_objective"
        if "criteria" in lowered:
            return "assessment_criteria"
        if "exam" in lowered or "instruction" in lowered:
            return "exam_instruction"
        if (
            "анықта" in lowered
            or "мысал келтір" in lowered
            or "exercise" in lowered
        ):
            return "exercise_prompt"
        return "other"

    def classify(self, request: ClassificationRequest) -> ClassificationDraft:
        self.classify_calls.append(request)
        decisions: list[SegmentDecision] = []
        content_ids: list[str] = []
        for segment in request.segments:
            lowered = segment.text.casefold()
            is_non = self.all_non_content or any(
                marker in lowered for marker in self.non_content_markers
            )
            if self.missing_ids and segment.id in self.missing_ids:
                continue
            if is_non:
                decisions.append(
                    SegmentDecision(
                        id=segment.id,
                        label="non_content",
                        category=self._category(segment.text),
                        reason="not lesson body",
                    )
                )
            else:
                content_ids.append(segment.id)
                decisions.append(
                    SegmentDecision(
                        id=segment.id,
                        label="lesson_content",
                        category="lesson_content",
                        reason="lesson body",
                    )
                )

        if not content_ids:
            topics = []
        elif self.topic_count >= 2 and len(content_ids) >= 2:
            cut = max(1, len(content_ids) // 2)
            topics = [
                TopicDraft(title="Topic A", segment_ids=content_ids[:cut]),
                TopicDraft(title="Topic B", segment_ids=content_ids[cut:]),
            ]
        else:
            topics = [TopicDraft(title="Lesson topic", segment_ids=content_ids)]
        return ClassificationDraft(segments=decisions, topics=topics)

    def detect_language(self, request: LanguageRequest) -> LanguageDecision:
        self.language_calls.append(request)
        return LanguageDecision(
            poster_language=self.language,
            confidence=self.confidence,
            evidence=self.evidence,
        )


def plan_options(**overrides) -> PlanOptions:
    values = {
        "model": "fake-luna",
        "api_key": "test-key",
        "max_details": 8,
        "context_char_limit": 24000,
        "chunk_char_limit": 12000,
        "max_page_images": 6,
    }
    values.update(overrides)
    return PlanOptions(**values)


class FakePosterPlanner(PosterPlanner):
    """Deterministic Stage 3 planner for tests. No network."""

    def __init__(
        self,
        *,
        language: str = "kk",
        age_band: str = "middle",
        detail_count: int = 2,
        composition_concept: str = "hero scene with callouts",
        palette: list[str] | None = None,
        file_slug: str = "communication-types",
        source_conflicts: list[SourceConflict] | None = None,
        mascot_question: str | None = "Байланыс түрлері туралы не білесің?",
        model: str = "fake-luna",
        vision: bool = True,
        structured_output: bool = True,
        max_context: int = 1050000,
    ) -> None:
        self.model = model
        self.capabilities = PlannerCapabilities(
            structured_output=structured_output,
            vision=vision,
            max_context=max_context,
        )
        self.language = language
        self.age_band = age_band
        self.detail_count = detail_count
        self.composition_concept = composition_concept
        self.palette = palette or ["deep teal", "warm ochre", "cream"]
        self.file_slug = file_slug
        self.source_conflicts = source_conflicts or []
        self.mascot_question = mascot_question
        self.plan_calls: list[PlanRequest] = []
        self.candidate_calls: list[CandidateRequest] = []

    def _detail(self, index: int, *, source_page: int = 1) -> PosterDetail:
        return PosterDetail(
            label=f"Түр {index}",
            explanation=f"Қысқа түсіндірме {index}.",
            example=f"Мысал {index}.",
            visual=f"An icon representing type {index}.",
            source_page=source_page,
        )

    def plan(self, request: PlanRequest) -> PosterSpecDraft:
        self.plan_calls.append(request)
        if request.candidates is not None:
            details = [
                self._detail(index, source_page=source_page)
                for index, source_page in enumerate(
                    (detail.source_page for detail in request.candidates.details[: self.detail_count]),
                    start=1,
                )
            ]
        else:
            details = [self._detail(index) for index in range(1, self.detail_count + 1)]
        return PosterSpecDraft(
            language=self.language,
            age_band=self.age_band,
            central_idea=CentralIdea(
                title="Байланыс түрлері",
                definition="Байланыс — ақпарат алмасудың түрлері.",
            ),
            details=details,
            composition=Composition(
                concept=self.composition_concept,
                description="A hero scene with callouts for each communication type.",
            ),
            style=Style(
                palette=self.palette,
                mood="bright and encouraging",
                illustration_style="friendly flat vector illustration, age-appropriate",
            ),
            strings_to_render=["Байланыс түрлері", "EXTRA_CALLOUT"],
            mascot_question=self.mascot_question,
            source_conflicts=self.source_conflicts,
            file_slug=self.file_slug,
        )

    def extract_candidates(self, request: CandidateRequest) -> CandidateSet:
        self.candidate_calls.append(request)
        page = request.source_pages[0] if request.source_pages else 1
        return CandidateSet(
            central_ideas=[
                CandidateCentralIdea(
                    title="Байланыс түрлері",
                    definition="Байланыс — ақпарат алмасудың түрлері.",
                )
            ],
            details=[
                CandidateDetail(
                    label=f"Chunk {request.chunk_index} detail {index}",
                    explanation=f"Chunk {request.chunk_index} explanation {index}.",
                    example=None,
                    visual=f"An illustration for chunk {request.chunk_index} type {index}.",
                    source_page=page,
                )
                for index in range(1, min(request.max_candidates, 3) + 1)
            ],
            key_terms=[f"term-{request.chunk_index}"],
            source_conflicts=[],
        )


def _png_bytes(width: int = 8, height: int = 12) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    pixmap = page.get_pixmap(dpi=72, colorspace=pymupdf.csRGB, alpha=False)
    data = pixmap.tobytes("png")
    doc.close()
    return data


def _format_for_size(size: str) -> str:
    return {
        "1024x1536": "portrait",
        "1536x1024": "landscape",
        "1024x1024": "square",
    }.get(size, "unknown")


class FakeImageProvider(ImageProvider):
    name = "fake"
    model = "gpt-image-1"

    def __init__(
        self,
        *,
        fail_formats: set[str] | None = None,
        moderate_formats: set[str] | None = None,
        moderate_twice_formats: set[str] | None = None,
        invalid_formats: set[str] | None = None,
    ) -> None:
        self.fail_formats = fail_formats or set()
        self.moderate_formats = moderate_formats or set()
        self.moderate_twice_formats = moderate_twice_formats or set()
        self.invalid_formats = invalid_formats or set()
        self.calls: list[tuple[str, str, str]] = []
        self._moderation_seen: dict[str, int] = {}

    def generate(self, prompt: str, size: str) -> bytes:
        fmt = _format_for_size(size)
        self.calls.append((fmt, prompt, size))
        if fmt in self.invalid_formats:
            return b"not an image"
        if fmt in self.moderate_twice_formats:
            raise ModerationRefusal("content policy")
        if fmt in self.moderate_formats:
            count = self._moderation_seen.get(fmt, 0)
            self._moderation_seen[fmt] = count + 1
            if count == 0:
                raise ModerationRefusal("content policy")
        if fmt in self.fail_formats:
            raise ImageProviderError("image_failed", "fake provider failure", retryable=True)
        return _png_bytes()


class FakeVisionTranscriber:
    model = "fake-vision"

    def __init__(self, transcripts: list[str] | str) -> None:
        self.transcripts = transcripts if isinstance(transcripts, list) else [transcripts]
        self.calls: list[tuple[Path, int]] = []

    def transcribe(self, image_path: Path, page_number: int) -> str:
        self.calls.append((image_path, page_number))
        index = min(len(self.calls) - 1, len(self.transcripts) - 1)
        return self.transcripts[index]
