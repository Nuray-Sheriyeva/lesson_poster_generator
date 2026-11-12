"""Application configuration loaded from TOML.

Model names and operational defaults live here (and in ``config.toml``), not in
pipeline code. CLI flags and POSTER_* environment variables override the file.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .errors import GenLessonError
from .models import PosterFormat, ReviewMode


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    model: str


TracingMode = Literal["auto", "on", "off"]


@dataclass(frozen=True)
class TracingConfig:
    """Optional Langfuse tracing configuration.

    ``enabled = "auto"`` turns tracing on only when both Langfuse keys are
    present. CLI flags override this mode without changing the file.
    """

    enabled: TracingMode = "auto"
    environment: str = "local"
    max_input_chars: int = 8000
    mask_paths: bool = True
    pricing_image: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MascotConfig:
    """Configuration for the post-generation mascot and speech bubble.

    Geometry is expressed as fractions of the canvas short side so a 1024 px
    poster and a 2000 px poster keep the same visual proportion.  ``edge_margin``
    is a fraction by default; an integer greater than 1 is treated as pixels.
    """

    enabled: bool = True
    assets_dir: Path = Path("assets/mascot")
    size_ratio: float = 0.09
    fill: str = "#FFFFFF"
    outline: str = "#6360FF"
    text_color: str = "#3B0764"
    outline_width: int = 4
    font_path: Path = Path("assets/fonts/NotoSans-Regular.ttf")
    min_font_ratio: float = 0.016
    max_font_ratio: float = 0.03
    bubble_max_width_ratio: float = 0.42
    edge_margin: float = 0.025
    question_max_chars: int = 140
    avoid_repeat: bool = False
    busyness_threshold: float = 0.08


@dataclass(frozen=True)
class AppConfig:
    planner: ModelChoice
    planner_max_context: int
    image: ModelChoice
    image_sizes: dict[str, dict[PosterFormat, tuple[int, int]]]
    vision: ModelChoice
    verification: ModelChoice | None
    formats: tuple[PosterFormat, ...]
    review: ReviewMode
    review_on_conflict: bool
    max_details: int
    max_retries: int
    tracing: TracingConfig
    mascot: MascotConfig
    out_dir: Path
    path: Path


_ACTIVE_CONFIG: Path | None = None


def set_active_config_path(path: Path | None) -> None:
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = path


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _find_config_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if _ACTIVE_CONFIG is not None:
        return _ACTIVE_CONFIG.expanduser().resolve()
    env_path = os.environ.get("POSTER_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    cwd_path = Path.cwd() / "config.toml"
    if cwd_path.is_file():
        return cwd_path
    repo_path = Path(__file__).resolve().parents[2] / "config.toml"
    if repo_path.is_file():
        return repo_path
    raise GenLessonError(
        "config_missing",
        "Could not find config.toml.",
        hint="Pass --config PATH or set POSTER_CONFIG.",
    )


def _normalize_review(value: str) -> ReviewMode:
    normalized = value.strip().casefold().replace("-", "_")
    if normalized in {"on_conflict", "onconflict", "conflict"}:
        return "on_conflict"
    if normalized in {"off", "on"}:
        return normalized
    raise GenLessonError(
        "config_invalid",
        f"Unknown review mode: {value!r}",
        hint="Use off, on, or on-conflict.",
    )


def _load_tracing(raw: dict) -> TracingConfig:
    """Parse the optional ``[tracing]`` table."""
    enabled_raw = raw.get("enabled", "auto")
    if isinstance(enabled_raw, bool):
        enabled = "on" if enabled_raw else "off"
    else:
        enabled = str(enabled_raw).strip().casefold()
    if enabled not in {"auto", "on", "off"}:
        raise GenLessonError(
            "config_invalid",
            f"Unknown tracing.enabled value: {enabled_raw!r}",
            hint='Use "auto", "on", or "off".',
        )

    environment = str(raw.get("environment") or "local").strip()
    if not environment:
        raise GenLessonError("config_invalid", "tracing.environment must not be empty.")

    try:
        max_input_chars = int(raw.get("max_input_chars", 8000))
    except (TypeError, ValueError) as exc:
        raise GenLessonError(
            "config_invalid", "tracing.max_input_chars must be an integer."
        ) from exc
    if not 0 <= max_input_chars <= 1_000_000:
        raise GenLessonError(
            "config_invalid",
            "tracing.max_input_chars must be between 0 and 1000000.",
        )

    mask_paths = _truthy(str(raw.get("mask_paths", True)))

    pricing_raw = raw.get("pricing") or {}
    if not isinstance(pricing_raw, dict):
        raise GenLessonError("config_invalid", "tracing.pricing must be a table.")
    image_raw = pricing_raw.get("image")
    if image_raw is None:
        image_raw = {}
    if not isinstance(image_raw, dict):
        raise GenLessonError("config_invalid", "tracing.pricing.image must be a table.")
    pricing_image: dict[str, float] = {}
    for model_name, value in image_raw.items():
        try:
            price = float(value)
        except (TypeError, ValueError) as exc:
            raise GenLessonError(
                "config_invalid",
                f"tracing.pricing.image.{model_name} must be a number.",
            ) from exc
        if price < 0:
            raise GenLessonError(
                "config_invalid",
                f"tracing.pricing.image.{model_name} must not be negative.",
            )
        pricing_image[str(model_name)] = price

    return TracingConfig(
        enabled=enabled,  # type: ignore[arg-type]
        environment=environment,
        max_input_chars=max_input_chars,
        mask_paths=mask_paths,
        pricing_image=pricing_image,
    )


def _config_number(raw: dict, name: str, default: float) -> float:
    value = raw.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise GenLessonError("config_invalid", f"mascot.{name} must be a number.") from exc


def _config_bool(raw: dict, name: str, default: bool) -> bool:
    value = raw.get(name, default)
    if isinstance(value, bool):
        return value
    return _truthy(str(value))


def _load_mascot(raw: dict) -> MascotConfig:
    """Parse the optional ``[mascot]`` table.

    A missing table keeps the bundled defaults, so existing config files keep
    working unchanged.
    """
    if not isinstance(raw, dict):
        raise GenLessonError("config_invalid", "Config [mascot] must be a table.")

    assets_raw = raw.get("assets_dir", "assets/mascot")
    font_raw = raw.get("font_path", raw.get("font", "assets/fonts/NotoSans-Regular.ttf"))
    assets_dir = Path(str(assets_raw)).expanduser()
    font_path = Path(str(font_raw)).expanduser()
    if not str(assets_raw).strip():
        raise GenLessonError("config_invalid", "mascot.assets_dir must not be empty.")
    if not str(font_raw).strip():
        raise GenLessonError("config_invalid", "mascot.font_path must not be empty.")

    fill = str(raw.get("fill", raw.get("fill_color", "#FFFFFF"))).strip()
    outline = str(raw.get("outline", raw.get("outline_color", "#6360FF"))).strip()
    text_color = str(raw.get("text_color", "#3B0764")).strip()
    for label, value in (("fill", fill), ("outline", outline), ("text_color", text_color)):
        if not value:
            raise GenLessonError("config_invalid", f"mascot.{label} must not be empty.")
        try:
            from PIL import ImageColor

            ImageColor.getcolor(value, "RGBA")
        except (ImportError, ValueError) as exc:
            raise GenLessonError(
                "config_invalid",
                f"mascot.{label} is not a valid color: {value!r}",
                hint="Use a CSS hex color such as #FFFFFF or #6360FF.",
            ) from exc

    try:
        outline_width = int(raw.get("outline_width", 4))
    except (TypeError, ValueError) as exc:
        raise GenLessonError("config_invalid", "mascot.outline_width must be an integer.") from exc
    try:
        question_max_chars = int(raw.get("question_max_chars", 140))
    except (TypeError, ValueError) as exc:
        raise GenLessonError(
            "config_invalid", "mascot.question_max_chars must be an integer."
        ) from exc

    config = MascotConfig(
        enabled=_config_bool(raw, "enabled", True),
        assets_dir=assets_dir,
        size_ratio=_config_number(raw, "size_ratio", 0.09),
        fill=fill,
        outline=outline,
        text_color=text_color,
        outline_width=outline_width,
        font_path=font_path,
        min_font_ratio=_config_number(raw, "min_font_ratio", 0.016),
        max_font_ratio=_config_number(raw, "max_font_ratio", 0.03),
        bubble_max_width_ratio=_config_number(raw, "bubble_max_width_ratio", 0.42),
        edge_margin=_config_number(raw, "edge_margin", 0.025),
        question_max_chars=question_max_chars,
        avoid_repeat=_config_bool(raw, "avoid_repeat", False),
        busyness_threshold=_config_number(raw, "busyness_threshold", 0.08),
    )
    _validate_mascot(config)
    return config


def _validate_mascot(config: MascotConfig) -> None:
    if not 0.01 <= config.size_ratio <= 0.4:
        raise GenLessonError("config_invalid", "mascot.size_ratio must be between 0.01 and 0.4.")
    if not 0 <= config.outline_width <= 64:
        raise GenLessonError("config_invalid", "mascot.outline_width must be between 0 and 64.")
    if not 0.001 <= config.min_font_ratio < config.max_font_ratio <= 0.2:
        raise GenLessonError(
            "config_invalid",
            "mascot.min_font_ratio must be positive and smaller than max_font_ratio "
            "(both at most 0.2).",
        )
    if not 0.05 <= config.bubble_max_width_ratio <= 0.9:
        raise GenLessonError(
            "config_invalid",
            "mascot.bubble_max_width_ratio must be between 0.05 and 0.9.",
        )
    if config.edge_margin < 0 or config.edge_margin > 1000:
        raise GenLessonError("config_invalid", "mascot.edge_margin must be non-negative.")
    if not 1 <= config.question_max_chars <= 2000:
        raise GenLessonError(
            "config_invalid", "mascot.question_max_chars must be between 1 and 2000."
        )
    if not 0 <= config.busyness_threshold <= 1:
        raise GenLessonError("config_invalid", "mascot.busyness_threshold must be between 0 and 1.")


def _load(path_text: str) -> AppConfig:
    path = Path(path_text)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GenLessonError("config_missing", f"Config file does not exist: {path}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise GenLessonError(
            "config_invalid", f"Could not read config file: {path}", details={"reason": str(exc)}
        ) from exc

    planner_raw = raw.get("planner") or {}
    image_raw = raw.get("image") or {}
    vision_raw = raw.get("vision") or {}
    verification_raw = raw.get("verification") or {}
    defaults = raw.get("defaults") or {}
    tracing_raw = raw.get("tracing") or {}
    if not isinstance(tracing_raw, dict):
        raise GenLessonError("config_invalid", "Config [tracing] must be a table.")
    tracing = _load_tracing(tracing_raw)
    mascot_raw = raw.get("mascot") or {}
    mascot = _load_mascot(mascot_raw)

    raw_planner_provider = str(planner_raw.get("provider") or "")
    env_planner_provider = os.environ.get("POSTER_PLANNER_PROVIDER")
    planner_provider = str(env_planner_provider or raw_planner_provider)
    if os.environ.get("POSTER_PLANNER_MODEL"):
        planner_model = str(os.environ["POSTER_PLANNER_MODEL"])
    elif env_planner_provider and env_planner_provider != raw_planner_provider:
        planner_model = os.environ.get(f"POSTER_{planner_provider.upper()}_MODEL", "")
    else:
        planner_model = str(planner_raw.get("model") or "")
    planner = ModelChoice(planner_provider, planner_model)
    try:
        planner_max_context = int(
            os.environ.get("POSTER_PLANNER_MAX_CONTEXT") or planner_raw.get("max_context") or 200000
        )
    except ValueError as exc:
        raise GenLessonError("config_invalid", "planner.max_context must be an integer.") from exc
    image = ModelChoice(
        str(os.environ.get("POSTER_IMAGE_PROVIDER") or image_raw.get("provider") or ""),
        str(os.environ.get("POSTER_IMAGE_MODEL") or image_raw.get("model") or ""),
    )
    image_sizes: dict[str, dict[PosterFormat, tuple[int, int]]] = {}
    for model_name, entries in (image_raw.get("sizes") or {}).items():
        parsed: dict[PosterFormat, tuple[int, int]] = {}
        for fmt, value in (entries or {}).items():
            if fmt not in {"portrait", "landscape", "square"}:
                raise GenLessonError(
                    "config_invalid", f"Unknown size format {fmt!r} for {model_name!r}."
                )
            try:
                width_text, height_text = str(value).lower().split("x", 1)
                parsed[fmt] = (int(width_text), int(height_text))
            except ValueError as exc:
                raise GenLessonError(
                    "config_invalid",
                    f"Invalid size {value!r} for {model_name!r}/{fmt}; use WIDTHxHEIGHT.",
                ) from exc
        image_sizes[str(model_name)] = parsed
    vision = ModelChoice(
        str(os.environ.get("POSTER_VISION_PROVIDER") or vision_raw.get("provider") or "openai"),
        str(os.environ.get("POSTER_VISION_MODEL") or vision_raw.get("model") or ""),
    )
    verification_model = str(
        os.environ.get("POSTER_VERIFY_MODEL") or verification_raw.get("model") or ""
    )
    verification = (
        ModelChoice(
            str(
                os.environ.get("POSTER_VERIFY_PROVIDER")
                or verification_raw.get("provider")
                or "openai"
            ),
            verification_model,
        )
        if verification_model
        else None
    )

    formats_raw = defaults.get("formats") or ["portrait"]
    if isinstance(formats_raw, str):
        formats = tuple(part.strip() for part in formats_raw.split(",") if part.strip())
    else:
        formats = tuple(str(part) for part in formats_raw)
    allowed_formats = {"portrait", "landscape", "square"}
    if not formats or any(fmt not in allowed_formats for fmt in formats):
        raise GenLessonError(
            "config_invalid",
            f"Invalid formats in config: {formats!r}",
            hint="Use portrait, landscape, and/or square.",
        )
    if len(set(formats)) != len(formats):
        raise GenLessonError("config_invalid", "Config formats contain duplicates.")

    review_env = os.environ.get("POSTER_REVIEW")
    review = _normalize_review(str(review_env or defaults.get("review") or "off"))
    review_on_conflict = _truthy(os.environ.get("POSTER_REVIEW_ON_CONFLICT")) or bool(
        defaults.get("review_on_conflict", False)
    )

    try:
        max_details_raw = (
            os.environ["POSTER_MAX_DETAILS"]
            if "POSTER_MAX_DETAILS" in os.environ
            else defaults.get("max_details", 8)
        )
        max_retries_raw = (
            os.environ["POSTER_MAX_RETRIES"]
            if "POSTER_MAX_RETRIES" in os.environ
            else defaults.get("max_retries", 2)
        )
        max_details = int(max_details_raw)
        max_retries = int(max_retries_raw)
    except (TypeError, ValueError) as exc:
        raise GenLessonError("config_invalid", "max_details/max_retries must be integers.") from exc
    out_dir = Path(
        str(os.environ.get("POSTER_OUT_DIR") or defaults.get("out_dir") or "outputs")
    ).expanduser()

    for label, choice in (("planner", planner), ("image", image), ("vision", vision)):
        if not choice.provider or not choice.model:
            raise GenLessonError(
                "config_invalid",
                f"Config [{label}] needs both provider and model.",
            )
    if not 1 <= max_details <= 20:
        raise GenLessonError("config_invalid", "max_details must be between 1 and 20.")
    if not 0 <= max_retries <= 10:
        raise GenLessonError("config_invalid", "max_retries must be between 0 and 10.")
    if not 1000 <= planner_max_context <= 10_000_000:
        raise GenLessonError(
            "config_invalid", "planner.max_context must be between 1000 and 10000000."
        )
    if not image_sizes:
        raise GenLessonError(
            "config_invalid",
            "Config [image.sizes] must define confirmed output sizes.",
        )

    return AppConfig(
        planner=planner,
        planner_max_context=planner_max_context,
        image=image,
        image_sizes=image_sizes,
        vision=vision,
        verification=verification,
        formats=formats,  # type: ignore[arg-type]
        review=review,
        review_on_conflict=review_on_conflict,
        max_details=max_details,
        max_retries=max_retries,
        tracing=tracing,
        mascot=mascot,
        out_dir=out_dir,
        path=path,
    )


def load_app_config(path: Path | None = None) -> AppConfig:
    return _load(str(_find_config_path(path)))


def clear_config_cache() -> None:
    """Compatibility no-op; config is re-read so env overrides stay current."""
