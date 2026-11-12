"""Runtime configuration and the fixed stage order."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .app_config import load_app_config
from .errors import GenLessonError, invalid_arguments
from .models import ClassifyConfig, GenerationConfig, IngestConfig, PlanConfig, PosterFormat

DEFAULT_RENDER_DPI = 180
DEFAULT_MIN_PAGE_CHARS = 20
DEFAULT_MIN_DOCUMENT_CHARS = 40
DEFAULT_MAX_PAGE_IMAGES = 6
DEFAULT_CONTEXT_CHARS = 24000
DEFAULT_CHUNK_CHARS = 12000
DEFAULT_IMAGE_TIMEOUT = 180.0
DEFAULT_IMAGE_RETRIES = 3
DEFAULT_VERIFY_SIMILARITY = 0.9
DEFAULT_VERIFY_TIMEOUT = 120.0
DEFAULT_VERIFY_VISION_RETRIES = 3

VisionMode = Literal["openai", "none"]


@dataclass(frozen=True)
class StageSpec:
    name: str
    directory: str
    title: str


# The stage order is the pipeline order. Later stages are scaffolded in the
# manifest now so a resumed run has a stable shape.
STAGES: tuple[StageSpec, ...] = (
    StageSpec("ingest", "stage-01-ingest", "Ingest and render"),
    StageSpec("classify", "stage-02-classify", "Classify and detect language"),
    StageSpec("plan", "stage-03-plan", "Plan the poster"),
    StageSpec("review", "stage-04-review", "Review gate"),
    StageSpec("prompts", "stage-05-prompts", "Build image prompts"),
    StageSpec("generate", "stage-06-generate", "Generate posters"),
    StageSpec("verify", "stage-07-verify", "Verify and retry"),
    StageSpec("save", "stage-08-save", "Save posters"),
)
STAGE_BY_NAME = {stage.name: stage for stage in STAGES}


def default_vision_model() -> str:
    return os.environ.get("GENLESSON_VISION_MODEL") or load_app_config().vision.model


def default_planner_model() -> str:
    return os.environ.get("GENLESSON_PLANNER_MODEL") or load_app_config().planner.model


def default_planner_provider() -> str:
    return os.environ.get("POSTER_PLANNER_PROVIDER") or load_app_config().planner.provider


def default_planner_max_context() -> int:
    return _env_int(
        "POSTER_PLANNER_MAX_CONTEXT", load_app_config().planner_max_context
    )


def default_image_provider() -> str:
    return os.environ.get("POSTER_IMAGE_PROVIDER") or load_app_config().image.provider


def default_verifier_model() -> str | None:
    configured = load_app_config().verification
    return os.environ.get("GENLESSON_VERIFY_MODEL") or (configured.model if configured else None)


def default_review_mode() -> str:
    if os.environ.get("POSTER_REVIEW"):
        from .app_config import _normalize_review

        return _normalize_review(os.environ["POSTER_REVIEW"])
    return load_app_config().review


def default_review_on_conflict() -> bool:
    raw = os.environ.get("POSTER_REVIEW_ON_CONFLICT")
    if raw is not None:
        return raw.strip().casefold() in {"1", "true", "yes", "on"}
    return load_app_config().review_on_conflict


def default_out_dir() -> Path:
    value = os.environ.get("POSTER_OUT_DIR")
    return Path(value).expanduser() if value else load_app_config().out_dir


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise invalid_arguments(f"{name} must be an integer, got {raw!r}.") from exc


def default_max_details() -> int:
    if "GENLESSON_MAX_DETAILS" in os.environ:
        return _env_int("GENLESSON_MAX_DETAILS", 0)
    if "POSTER_MAX_DETAILS" in os.environ:
        return _env_int("POSTER_MAX_DETAILS", 0)
    return load_app_config().max_details


def default_context_chars() -> int:
    return _env_int("GENLESSON_CONTEXT_CHARS", DEFAULT_CONTEXT_CHARS)


def default_chunk_chars() -> int:
    return _env_int("GENLESSON_CHUNK_CHARS", DEFAULT_CHUNK_CHARS)


def default_image_model() -> str:
    return os.environ.get("GENLESSON_IMAGE_MODEL") or load_app_config().image.model


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise invalid_arguments(f"{name} must be a number, got {raw!r}.") from exc


def default_image_timeout() -> float:
    return _env_float("GENLESSON_IMAGE_TIMEOUT", DEFAULT_IMAGE_TIMEOUT)


def default_image_retries() -> int:
    return _env_int("GENLESSON_IMAGE_RETRIES", DEFAULT_IMAGE_RETRIES)


def default_verify_similarity() -> float:
    return _env_float("GENLESSON_VERIFY_SIMILARITY", DEFAULT_VERIFY_SIMILARITY)


def default_verify_retries() -> int:
    if "GENLESSON_VERIFY_RETRIES" in os.environ:
        return _env_int("GENLESSON_VERIFY_RETRIES", 0)
    if "POSTER_MAX_RETRIES" in os.environ:
        return _env_int("POSTER_MAX_RETRIES", 0)
    return load_app_config().max_retries




def _planner_provider_defaults(provider: str) -> dict[str, object]:
    prefix = provider.upper()
    app = load_app_config()
    model = os.environ.get("POSTER_PLANNER_MODEL")
    if not model and provider == app.planner.provider:
        model = app.planner.model
    if not model:
        model = os.environ.get(f"POSTER_{prefix}_MODEL", "")
    return {
        "provider": provider,
        "api_key": os.environ.get("POSTER_PLANNER_API_KEY")
        or os.environ.get(f"{prefix}_API_KEY")
        or None,
        "base_url": os.environ.get(f"{prefix}_BASE_URL") or None,
        "model": model,
    }

def default_verify_timeout() -> float:
    return _env_float("GENLESSON_VERIFY_TIMEOUT", DEFAULT_VERIFY_TIMEOUT)


def default_verify_vision_retries() -> int:
    return _env_int("GENLESSON_VERIFY_VISION_RETRIES", DEFAULT_VERIFY_VISION_RETRIES)


def default_formats() -> tuple[PosterFormat, ...]:
    raw = os.environ.get("GENLESSON_FORMATS") or os.environ.get("POSTER_FORMATS")
    if raw:
        values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    else:
        values = tuple(load_app_config().formats)
    allowed = {"portrait", "landscape", "square"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise invalid_arguments(
            f"Unknown poster format(s): {', '.join(unknown)}. "
            "Use portrait, landscape, or square."
        )
    return values  # type: ignore[return-value]


def default_max_page_images() -> int:
    raw = os.environ.get("GENLESSON_MAX_PAGE_IMAGES", str(DEFAULT_MAX_PAGE_IMAGES))
    try:
        return int(raw)
    except ValueError as exc:
        raise invalid_arguments(
            f"GENLESSON_MAX_PAGE_IMAGES must be an integer, got {raw!r}."
        ) from exc


def default_render_dpi() -> int:
    raw = os.environ.get("GENLESSON_RENDER_DPI", str(DEFAULT_RENDER_DPI))
    try:
        return int(raw)
    except ValueError as exc:
        raise invalid_arguments(
            f"GENLESSON_RENDER_DPI must be an integer, got {raw!r}."
        ) from exc


@dataclass(frozen=True)
class IngestOptions:
    """Everything Stage 1 needs, with secrets excluded from serialized config."""

    render_dpi: int = DEFAULT_RENDER_DPI
    vision_mode: VisionMode = "openai"
    vision_model: str = field(default_factory=default_vision_model)
    min_page_chars: int = DEFAULT_MIN_PAGE_CHARS
    min_document_chars: int = DEFAULT_MIN_DOCUMENT_CHARS
    openai_api_key: str | None = field(default=None, repr=False)
    openai_base_url: str | None = None
    vision_timeout_s: float = 120.0
    vision_max_retries: int = 3

    def validate(self) -> None:
        if not 72 <= self.render_dpi <= 600:
            raise invalid_arguments("--dpi must be between 72 and 600.")
        if self.vision_mode not in ("openai", "none"):
            raise invalid_arguments("--vision must be 'openai' or 'none'.")
        if self.vision_mode == "openai" and not self.vision_model.strip():
            raise invalid_arguments("--vision-model must not be empty.")
        if self.min_page_chars < 0 or self.min_document_chars < 0:
            raise invalid_arguments("Character thresholds must be non-negative.")
        if self.vision_timeout_s <= 0:
            raise invalid_arguments("--vision-timeout must be positive.")
        if not 0 <= self.vision_max_retries <= 10:
            raise invalid_arguments("--vision-retries must be between 0 and 10.")

    @classmethod
    def from_env(cls, **overrides: object) -> IngestOptions:
        values: dict[str, object] = {
            "vision_mode": "openai",
            "min_page_chars": DEFAULT_MIN_PAGE_CHARS,
            "min_document_chars": DEFAULT_MIN_DOCUMENT_CHARS,
            "openai_api_key": os.environ.get("OPENAI_API_KEY") or None,
            "openai_base_url": os.environ.get("OPENAI_BASE_URL") or None,
            "vision_timeout_s": 120.0,
            "vision_max_retries": 3,
        }
        if "render_dpi" not in overrides:
            values["render_dpi"] = default_render_dpi()
        if "vision_model" not in overrides:
            values["vision_model"] = default_vision_model()
        values.update(overrides)
        options = cls(**values)  # type: ignore[arg-type]
        options.validate()
        return options

    def public_config(self) -> IngestConfig:
        return IngestConfig(
            render_dpi=self.render_dpi,
            vision_mode=self.vision_mode,
            vision_model=self.vision_model if self.vision_mode == "openai" else None,
            min_page_chars=self.min_page_chars,
            min_document_chars=self.min_document_chars,
        )


@dataclass(frozen=True)
class PlannerOptions:
    """Configuration for the Stage 2 content classifier and language decision."""

    provider: str = field(default_factory=default_planner_provider)
    model: str = field(default_factory=default_planner_model)
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    timeout_s: float = 120.0
    max_retries: int = 3
    max_context: int = field(default_factory=default_planner_max_context)
    max_page_images: int = DEFAULT_MAX_PAGE_IMAGES

    def validate(self) -> None:
        if self.provider not in {"openai", "anthropic", "deepseek"}:
            raise invalid_arguments(
                f"Unsupported planner provider: {self.provider}"
            )
        if not self.model.strip():
            raise invalid_arguments("--planner-model must not be empty.")
        if self.timeout_s <= 0:
            raise invalid_arguments("--planner-timeout must be positive.")
        if not 0 <= self.max_retries <= 10:
            raise invalid_arguments("--planner-retries must be between 0 and 10.")
        if not 0 <= self.max_page_images <= 20:
            raise invalid_arguments("--max-page-images must be between 0 and 20.")
        if not 1000 <= self.max_context <= 10_000_000:
            raise invalid_arguments("--planner-context must be between 1000 and 10000000.")

    @classmethod
    def from_env(cls, **overrides: object) -> PlannerOptions:
        provider = str(
            overrides.get("provider")
            or os.environ.get("POSTER_PLANNER_PROVIDER")
            or load_app_config().planner.provider
        )
        values: dict[str, object] = _planner_provider_defaults(provider)
        values.update({"timeout_s": 120.0, "max_retries": 3})
        if "max_page_images" not in overrides:
            values["max_page_images"] = default_max_page_images()
        values.update(overrides)
        options = cls(**values)  # type: ignore[arg-type]
        options.validate()
        return options

    def public_config(self) -> ClassifyConfig:
        return ClassifyConfig(
            planner_provider=self.provider,
            planner_model=self.model,
            max_page_images=self.max_page_images,
        )


@dataclass(frozen=True)
class PlanOptions:
    """Configuration for the Stage 3 poster planner."""

    provider: str = field(default_factory=default_planner_provider)
    model: str = field(default_factory=default_planner_model)
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    timeout_s: float = 120.0
    max_retries: int = 3
    max_context: int = field(default_factory=default_planner_max_context)
    max_page_images: int = DEFAULT_MAX_PAGE_IMAGES
    max_details: int = field(default_factory=default_max_details)
    context_char_limit: int = DEFAULT_CONTEXT_CHARS
    chunk_char_limit: int = DEFAULT_CHUNK_CHARS

    def validate(self) -> None:
        if self.provider not in {"openai", "anthropic", "deepseek"}:
            raise invalid_arguments(
                f"Unsupported planner provider: {self.provider}"
            )
        if not self.model.strip():
            raise invalid_arguments("--planner-model must not be empty.")
        if self.timeout_s <= 0:
            raise invalid_arguments("--planner-timeout must be positive.")
        if not 0 <= self.max_retries <= 10:
            raise invalid_arguments("--planner-retries must be between 0 and 10.")
        if not 0 <= self.max_page_images <= 20:
            raise invalid_arguments("--max-page-images must be between 0 and 20.")
        if not 1000 <= self.max_context <= 10_000_000:
            raise invalid_arguments("--planner-context must be between 1000 and 10000000.")
        if not 1 <= self.max_details <= 20:
            raise invalid_arguments("--max-details must be between 1 and 20.")
        if not 1000 <= self.context_char_limit <= 200000:
            raise invalid_arguments("--context-chars must be between 1000 and 200000.")
        if not 500 <= self.chunk_char_limit <= self.context_char_limit:
            raise invalid_arguments(
                "--chunk-chars must be between 500 and --context-chars."
            )

    @classmethod
    def from_env(cls, **overrides: object) -> PlanOptions:
        provider = str(
            overrides.get("provider")
            or os.environ.get("POSTER_PLANNER_PROVIDER")
            or load_app_config().planner.provider
        )
        values: dict[str, object] = _planner_provider_defaults(provider)
        values.update({"timeout_s": 120.0, "max_retries": 3})
        if "max_page_images" not in overrides:
            values["max_page_images"] = default_max_page_images()
        if "max_details" not in overrides:
            values["max_details"] = default_max_details()
        if "context_char_limit" not in overrides:
            values["context_char_limit"] = default_context_chars()
        if "chunk_char_limit" not in overrides:
            values["chunk_char_limit"] = default_chunk_chars()
        values.update(overrides)
        if (
            "chunk_char_limit" not in overrides
            and isinstance(values["chunk_char_limit"], int)
            and isinstance(values["context_char_limit"], int)
            and values["chunk_char_limit"] > values["context_char_limit"]
        ):
            values["chunk_char_limit"] = values["context_char_limit"]
        options = cls(**values)  # type: ignore[arg-type]
        options.validate()
        return options

    def public_config(self) -> PlanConfig:
        return PlanConfig(
            planner_provider=self.provider,
            planner_model=self.model,
            max_details=self.max_details,
            context_char_limit=self.context_char_limit,
            chunk_char_limit=self.chunk_char_limit,
            max_page_images=self.max_page_images,
        )

    def planner_options(self) -> PlannerOptions:
        return PlannerOptions(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            max_context=self.max_context,
            max_page_images=self.max_page_images,
        )


@dataclass(frozen=True)
class ReviewOptions:
    mode: str = field(default_factory=default_review_mode)

    def validate(self) -> None:
        if self.mode not in {"off", "on", "on_conflict"}:
            raise invalid_arguments("--review must be off, on, or on-conflict.")


@dataclass(frozen=True)
class PromptOptions:
    formats: tuple[PosterFormat, ...] = field(default_factory=default_formats)
    image_model: str = field(default_factory=default_image_model)
    provider: str = "openai"

    def validate(self) -> None:
        if not self.formats:
            raise invalid_arguments("At least one poster format is required.")
        if len(set(self.formats)) != len(self.formats):
            raise invalid_arguments("Duplicate poster formats are not allowed.")
        for fmt in self.formats:
            if fmt not in {"portrait", "landscape", "square"}:
                raise invalid_arguments(f"Unsupported poster format: {fmt}")
        if not self.image_model.strip():
            raise invalid_arguments("--image-model must not be empty.")
        if self.provider not in {"openai"}:
            raise invalid_arguments("--image-provider currently supports only openai.")

    @classmethod
    def from_env(
        cls,
        *,
        formats: tuple[PosterFormat, ...] | None = None,
        image_model: str | None = None,
        provider: str | None = None,
    ) -> PromptOptions:
        options = cls(
            formats=formats if formats is not None else default_formats(),
            image_model=image_model if image_model is not None else default_image_model(),
            provider=provider or default_image_provider(),
        )
        options.validate()
        return options


@dataclass(frozen=True)
class GenerationOptions:
    """Configuration for Stage 6 image generation."""

    provider: Literal["openai"] = field(  # type: ignore[assignment]
        default_factory=default_image_provider
    )
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    timeout_s: float = DEFAULT_IMAGE_TIMEOUT
    max_retries: int = DEFAULT_IMAGE_RETRIES

    def validate(self) -> None:
        if self.provider != "openai":
            raise invalid_arguments("Stage 6 currently supports only the openai provider.")
        if not 10 <= self.timeout_s <= 600:
            raise invalid_arguments("--image-timeout must be between 10 and 600 seconds.")
        if not 0 <= self.max_retries <= 10:
            raise invalid_arguments("--image-retries must be between 0 and 10.")

    @classmethod
    def from_env(cls, **overrides: object) -> GenerationOptions:
        values: dict[str, object] = {
            "provider": os.environ.get("POSTER_IMAGE_PROVIDER") or default_image_provider(),
            "api_key": os.environ.get("OPENAI_API_KEY") or None,
            "base_url": os.environ.get("OPENAI_BASE_URL") or None,
        }
        if "timeout_s" not in overrides:
            values["timeout_s"] = default_image_timeout()
        if "max_retries" not in overrides:
            values["max_retries"] = default_image_retries()
        values.update(overrides)
        options = cls(**values)  # type: ignore[arg-type]
        options.validate()
        return options

    def public_config(self, image_model: str) -> GenerationConfig:
        return GenerationConfig(
            provider=self.provider,
            image_model=image_model,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
        )


@dataclass(frozen=True)
class VerificationOptions:
    """Configuration for Stage 7 verification and correction retries."""

    provider: Literal["openai"] = "openai"
    verifier_model: str | None = field(default_factory=default_verifier_model)
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    vision_timeout_s: float = DEFAULT_VERIFY_TIMEOUT
    vision_max_retries: int = DEFAULT_VERIFY_VISION_RETRIES
    similarity_threshold: float = DEFAULT_VERIFY_SIMILARITY
    max_retries: int = field(default_factory=default_verify_retries)
    image_timeout_s: float = DEFAULT_IMAGE_TIMEOUT
    image_max_retries: int = DEFAULT_IMAGE_RETRIES

    def validate(self) -> None:
        if self.provider != "openai":
            raise invalid_arguments("Stage 7 currently supports only the openai verifier.")
        if self.verifier_model is not None and not self.verifier_model.strip():
            raise invalid_arguments("--verifier-model must not be empty.")
        if not 0.5 <= self.similarity_threshold <= 1.0:
            raise invalid_arguments("--similarity must be between 0.5 and 1.0.")
        if not 0 <= self.max_retries <= 10:
            raise invalid_arguments("--max-retries must be between 0 and 10.")
        if not 10 <= self.vision_timeout_s <= 600:
            raise invalid_arguments("--vision-timeout must be between 10 and 600.")
        if not 0 <= self.vision_max_retries <= 10:
            raise invalid_arguments("--vision-retries must be between 0 and 10.")
        if not 10 <= self.image_timeout_s <= 600:
            raise invalid_arguments("--image-timeout must be between 10 and 600.")
        if not 0 <= self.image_max_retries <= 10:
            raise invalid_arguments("--image-retries must be between 0 and 10.")

    @classmethod
    def from_env(cls, **overrides: object) -> VerificationOptions:
        values: dict[str, object] = {
            "provider": os.environ.get("POSTER_VERIFY_PROVIDER") or "openai",
            "verifier_model": default_verifier_model(),
            "api_key": os.environ.get("OPENAI_API_KEY") or None,
            "base_url": os.environ.get("OPENAI_BASE_URL") or None,
        }
        if "vision_timeout_s" not in overrides:
            values["vision_timeout_s"] = default_verify_timeout()
        if "vision_max_retries" not in overrides:
            values["vision_max_retries"] = default_verify_vision_retries()
        if "similarity_threshold" not in overrides:
            values["similarity_threshold"] = default_verify_similarity()
        if "max_retries" not in overrides:
            values["max_retries"] = default_verify_retries()
        if "image_timeout_s" not in overrides:
            values["image_timeout_s"] = default_image_timeout()
        if "image_max_retries" not in overrides:
            values["image_max_retries"] = default_image_retries()
        values.update(overrides)
        options = cls(**values)  # type: ignore[arg-type]
        options.validate()
        return options

    def image_options(self) -> GenerationOptions:
        return GenerationOptions(
            provider=default_image_provider(),
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_s=self.image_timeout_s,
            max_retries=self.image_max_retries,
        )


def manifest_config(options: IngestOptions) -> dict:
    """Config safe to persist. Never include the API key."""
    review_mode = default_review_mode()
    return {
        "ingest": options.public_config().model_dump(mode="json"),
        "planner": {
            "provider": default_planner_provider(),
            "model": default_planner_model(),
            "max_context": default_planner_max_context(),
        },
        "image_model": {
            "provider": default_image_provider(),
            "model": default_image_model(),
        },
        "formats": list(default_formats()),
        "review_gate": review_mode != "off",
        "review_mode": review_mode,
        "review_on_conflict": default_review_on_conflict(),
        "out_dir": str(default_out_dir()),
    }


def stage_dir_name(stage: str) -> str:
    try:
        return STAGE_BY_NAME[stage].directory
    except KeyError as exc:
        raise GenLessonError("unknown_stage", f"Unknown stage: {stage!r}") from exc
