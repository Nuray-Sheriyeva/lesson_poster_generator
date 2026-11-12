"""Serializable artifacts written under ``runs/<run_id>/``.

These schemas are the contract between stages. Keep them explicit: the review
gate and resume logic depend on being able to read a previous run's output.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MediaType = Literal["pdf", "docx"]
TextSource = Literal["text_layer", "vision", "blank"]
SegmentLabel = Literal["lesson_content", "non_content"]
SegmentCategory = Literal[
    "lesson_content",
    "learning_objective",
    "assessment_criteria",
    "exam_instruction",
    "exercise_prompt",
    "other",
]
LanguageSource = Literal["text_layer", "vision_transcript"]
AgeBand = Literal["primary", "middle", "high"]
ReviewMode = Literal["off", "on", "on_conflict"]
ReviewStatus = Literal["auto_approved", "awaiting_review", "approved"]
PosterFormat = Literal["portrait", "landscape", "square"]
MascotCorner = Literal["top_left", "top_right", "bottom_left", "bottom_right"]
MascotQuestionSource = Literal["planner", "central_idea", "fallback"]
GenerationStatus = Literal["generated", "failed"]
PromptVariant = Literal["original", "toned_down"]
VerificationFormatStatus = Literal["passed", "needs_attention", "generation_failed"]
CheckKind = Literal["title", "label", "term", "sentence"]
CheckStatus = Literal["exact", "similar", "missing"]
StageStatus = Literal["pending", "running", "complete", "failed", "skipped", "awaiting_review"]
RunStatus = Literal["created", "in_progress", "awaiting_review", "complete", "failed"]


def utc_now() -> str:
    """Return a stable UTC timestamp for artifacts."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceInfo(StrictModel):
    path: str
    filename: str
    media_type: MediaType
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    modified_at: str


class IngestConfig(StrictModel):
    render_dpi: int = Field(ge=72, le=600)
    vision_mode: Literal["openai", "none"]
    vision_model: str | None = None
    min_page_chars: int = Field(ge=0)
    min_document_chars: int = Field(ge=0)


class IngestStats(StrictModel):
    page_count: int = Field(ge=1)
    text_layer_pages: int = Field(ge=0)
    vision_pages: int = Field(ge=0)
    blank_pages: int = Field(ge=0)
    char_count: int = Field(ge=0)
    scanned_document: bool


class PageArtifact(StrictModel):
    index: int = Field(ge=1, description="1-based page number in the source.")
    label: str
    text: str
    text_source: TextSource
    char_count: int = Field(ge=0)
    image_path: str | None = Field(
        default=None,
        description="PNG path relative to the run root, or null for DOCX pseudo-pages.",
    )
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    transcription_model: str | None = None


class LessonDocument(StrictModel):
    schema_version: Literal[1] = 1
    source: SourceInfo
    title: str | None = None
    pages: list[PageArtifact]
    text: str
    char_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    stats: IngestStats
    config: IngestConfig
    extracted_at: str


class SegmentArtifact(StrictModel):
    id: str = Field(pattern=r"^s[0-9]{4,}$")
    page: int = Field(ge=1)
    page_label: str
    order: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)
    text_source: TextSource
    label: SegmentLabel
    category: SegmentCategory
    reason: str = ""


class TopicArtifact(StrictModel):
    title: str
    segment_ids: list[str]
    char_count: int = Field(ge=0)


class ClassifyConfig(StrictModel):
    planner_provider: Literal["openai", "anthropic", "deepseek"] = "openai"
    planner_model: str
    max_page_images: int = Field(ge=0)


class ClassifyStats(StrictModel):
    segment_count: int = Field(ge=1)
    content_segments: int = Field(ge=0)
    non_content_segments: int = Field(ge=0)
    content_char_count: int = Field(ge=0)
    language_char_count: int = Field(ge=0)
    page_images_used: int = Field(ge=0)


class ClassificationDocument(StrictModel):
    schema_version: Literal[1] = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lesson_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    poster_language: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
    confidence: float = Field(ge=0, le=1)
    evidence: str
    language_source: LanguageSource
    dominant_topic: str | None = None
    topics: list[TopicArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    segments: list[SegmentArtifact]
    lesson_text: str
    stats: ClassifyStats
    config: ClassifyConfig
    classified_at: str


class CentralIdea(StrictModel):
    title: str
    definition: str


class PosterDetail(StrictModel):
    label: str
    explanation: str
    example: str | None
    visual: str
    source_page: int


class Composition(StrictModel):
    concept: str
    description: str


class Style(StrictModel):
    palette: list[str]
    mood: str
    illustration_style: str


class SourceConflict(StrictModel):
    issue: str
    resolution: str


class PlanConfig(StrictModel):
    planner_provider: Literal["openai", "anthropic", "deepseek"] = "openai"
    planner_model: str
    max_details: int
    context_char_limit: int
    chunk_char_limit: int
    max_page_images: int


class PlanStats(StrictModel):
    content_chars: int = Field(ge=0)
    chunks: int = Field(ge=1)
    candidate_details: int = Field(ge=0)
    details: int = Field(ge=1)
    strings: int = Field(ge=1)
    page_images_used: int = Field(ge=0)


class PosterSpecDocument(StrictModel):
    schema_version: Literal[1] = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    classification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    language: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
    age_band: AgeBand
    central_idea: CentralIdea
    details: list[PosterDetail]
    composition: Composition
    style: Style
    strings_to_render: list[str]
    mascot_question: str | None = None
    source_conflicts: list[SourceConflict]
    file_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    warnings: list[str] = Field(default_factory=list)
    stats: PlanStats
    config: PlanConfig
    planned_at: str


class ReviewDocument(StrictModel):
    schema_version: Literal[1] = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    classification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: ReviewMode
    status: ReviewStatus
    source_conflicts: list[SourceConflict] = Field(default_factory=list)
    summary_path: str | None = None
    approved_spec_path: str | None = None
    approved_spec_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    edited: bool = False
    requested_at: str
    reviewed_at: str | None = None
    notes: str | None = None


class MascotRect(StrictModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class MascotPoint(StrictModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class MascotPlan(StrictModel):
    """The layout Stage 5 reserves and Stage 8 draws.

    Keeping this in ``prompts.json`` and ``mascot_layout.json`` means the stages
    use exactly the same question, variant, corner, and geometry.
    """

    enabled: bool = True
    question: str
    question_source: MascotQuestionSource = "central_idea"
    corner: MascotCorner
    preferred_corner: MascotCorner | None = None
    variant: str | None = None
    font_size: int = Field(ge=1)
    lines: list[str]
    mascot: MascotRect
    bubble: MascotRect
    block: MascotRect
    padding_x: int = Field(default=0, ge=0)
    padding_y: int = Field(default=0, ge=0)
    line_height: int = Field(default=0, ge=0)
    radius: int = Field(default=0, ge=0)
    outline_width: int = Field(default=0, ge=0)
    tail: list[MascotPoint] = Field(default_factory=list)


class MascotLayoutFormat(MascotPlan):
    format: PosterFormat
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class MascotLayoutDocument(StrictModel):
    schema_version: Literal[1] = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    formats: list[MascotLayoutFormat]
    generated_at: str


class PromptFormatSpec(StrictModel):
    format: PosterFormat
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    filename: str
    prompt: str
    mascot: MascotPlan | None = None


class PromptConfig(StrictModel):
    image_provider: Literal["openai"] = "openai"
    image_model: str
    formats: list[PosterFormat]


class PromptBundle(StrictModel):
    schema_version: Literal[1] = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_model: str
    formats: list[PromptFormatSpec]
    config: PromptConfig
    built_at: str


class GenerationErrorInfo(StrictModel):
    code: str
    message: str
    moderation: bool = False
    retryable: bool = False


class GeneratedFormat(StrictModel):
    format: PosterFormat
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    size: str
    status: GenerationStatus
    image_path: str | None = None
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    bytes: int | None = Field(default=None, ge=1)
    media_type: str | None = None
    attempts: int = Field(ge=1)
    moderation_retry: bool = False
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_variant: PromptVariant | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = Field(default=None, ge=0)
    last_call: dict[str, Any] | None = None
    error: GenerationErrorInfo | None = None
    generated_at: str | None = None


class GenerationConfig(StrictModel):
    provider: Literal["openai"] = "openai"
    image_model: str
    timeout_s: float
    max_retries: int


class GenerationStats(StrictModel):
    requested: int = Field(ge=1)
    generated: int = Field(ge=0)
    failed: int = Field(ge=0)


class GenerationDocument(StrictModel):
    schema_version: Literal[1] = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_model: str
    formats: list[GeneratedFormat]
    warnings: list[str] = Field(default_factory=list)
    stats: GenerationStats
    config: GenerationConfig
    generated_at: str


class StringCheck(StrictModel):
    kind: CheckKind
    expected: str
    normalized: str
    best_match: str | None = None
    similarity: float = Field(ge=0, le=1)
    status: CheckStatus
    threshold: float | None = None


class ExtraString(StrictModel):
    text: str
    normalized: str


class VerificationAttempt(StrictModel):
    attempt: int = Field(ge=1)
    mascot: dict | None = None
    mascot_required_corner: MascotCorner | None = None
    image_path: str | None = None
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    transcript: str = ""
    checks: list[StringCheck] = Field(default_factory=list)
    extra_strings: list[ExtraString] = Field(default_factory=list)
    exact_mismatches: list[str] = Field(default_factory=list)
    below_threshold: list[str] = Field(default_factory=list)
    score: float = 0.0
    passed: bool = False
    text_passed: bool = False
    error: GenerationErrorInfo | None = None
    verified_at: str = ""


class FormatVerification(StrictModel):
    format: PosterFormat
    status: VerificationFormatStatus
    mascot: dict | None = None
    mascot_reason: str | None = None
    reason: str | None = None
    attempts: list[VerificationAttempt] = Field(default_factory=list)
    best_attempt: int | None = None
    best_image_path: str | None = None
    best_image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    best_score: float | None = None
    warnings: list[str] = Field(default_factory=list)


class VerificationConfig(StrictModel):
    verifier_provider: Literal["openai"] = "openai"
    verifier_model: str
    planner_provider: str
    planner_model: str
    fallback_used: bool
    similarity_threshold: float
    max_retries: int
    image_model: str


class VerificationStats(StrictModel):
    formats: int = Field(ge=1)
    passed: int = Field(ge=0)
    needs_attention: int = Field(ge=0)
    generation_failed: int = Field(ge=0)
    retries: int = Field(ge=0)


class VerificationDocument(StrictModel):
    schema_version: Literal[1] = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    formats: list[FormatVerification]
    warnings: list[str] = Field(default_factory=list)
    stats: VerificationStats
    config: VerificationConfig
    verified_at: str


class CostInfo(StrictModel):
    currency: Literal["USD"] = "USD"
    amount_usd: float | None = Field(default=None, ge=0)
    estimated: bool = False
    note: str
    image_calls: int = Field(ge=0)
    vision_calls: int = Field(ge=0)


class SavedPoster(StrictModel):
    format: PosterFormat
    filename: str
    sidecar_filename: str
    image_path: str
    sidecar_path: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    verification_status: Literal["passed", "needs_attention"]
    best_attempt: int | None = None
    mascot: dict | None = None
    saved_at: str


class SkippedFormat(StrictModel):
    format: PosterFormat
    reason: str


class SaveStats(StrictModel):
    formats: int = Field(ge=1)
    saved: int = Field(ge=0)
    needs_attention: int = Field(ge=0)
    skipped: int = Field(ge=0)
    mascots_applied: int = Field(default=0, ge=0)


class SaveDocument(StrictModel):
    schema_version: Literal[1] = 1
    out_dir: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[SavedPoster]
    skipped: list[SkippedFormat] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: SaveStats
    saved_at: str


class StageRecord(StrictModel):
    status: StageStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    output: str | None = None
    error: dict | None = None


class RunManifest(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    created_at: str
    updated_at: str
    status: RunStatus = "created"
    source: SourceInfo | None = None
    config: dict = Field(default_factory=dict)
    stages: dict[str, StageRecord] = Field(default_factory=dict)
