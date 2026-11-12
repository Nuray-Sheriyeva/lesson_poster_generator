"""Stage 5: build one image-model prompt per requested poster format."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from . import prompt_management
from .app_config import load_app_config
from .config import PromptOptions
from .errors import GenLessonError
from .mascot.assets import load_mascot_assets
from .mascot.layout import layout_mascot
from .mascot.select import choose_variant, default_corner_for_format, plan_question
from .models import (
    MascotLayoutDocument,
    MascotLayoutFormat,
    MascotPlan,
    MascotPoint,
    MascotRect,
    PosterFormat,
    PosterSpecDocument,
    PromptBundle,
    PromptConfig,
    PromptFormatSpec,
    ReviewDocument,
    RunManifest,
    utc_now,
)
from .runs import RunPaths, RunStore
from .storage import read_model, sha256_file, write_json_atomic, write_text_atomic

log = logging.getLogger(__name__)

# Model names and their confirmed sizes live in config.toml.

FORMAT_LABELS: dict[PosterFormat, str] = {
    "portrait": "Portrait",
    "landscape": "Landscape",
    "square": "Square",
}

FORMAT_LAYOUTS: dict[PosterFormat, str] = {
    "portrait": (
        "Vertical poster. Put the hero scene across the upper third, the central idea as the "
        "largest text block directly below it, and the detail callouts flowing top-to-bottom in "
        "a clear visual path. Use generous side margins and keep all text at least 6% away from "
        "the canvas edges."
    ),
    "landscape": (
        "Horizontal poster. Put the hero scene on the left 40% of the canvas, the central idea "
        "at the top right, and the detail callouts in two readable columns on the right. Build a "
        "clear left-to-right reading path. Keep all text at least 6% away from the canvas edges."
    ),
    "square": (
        "Square poster. Center the hero scene slightly above the middle, place the central idea "
        "as the strongest text block, and arrange detail callouts in a balanced radial or "
        "orbit-like ring around the hero. Keep all text at least 7% away from the canvas edges."
    ),
}

LAYOUT_PROMPT_NAMES: dict[PosterFormat, str] = {
    "portrait": "genlesson/poster/layout-portrait",
    "landscape": "genlesson/poster/layout-landscape",
    "square": "genlesson/poster/layout-square",
}

_KAZAKH_LETTERS = "ә, ғ, қ, ң, ө, ұ, ү, һ, і"
_KAZAKH_UPPER = "Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І"


@dataclass
class PromptResult:
    document: PromptBundle
    output_path: Path
    prompt_paths: list[Path]
    resumed: bool
    mascot_layout_path: Path | None = None


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _sizes_for_model(image_model: str) -> dict[PosterFormat, tuple[int, int]]:
    sizes = load_app_config().image_sizes.get(image_model)
    if not sizes:
        raise GenLessonError(
            "unsupported_image_model",
            f"No confirmed poster sizes for image model {image_model!r}.",
            hint=(
                "Add a [image.sizes.<model>] table with portrait, landscape, and square "
                "sizes to config.toml."
            ),
            details={"supported": sorted(load_app_config().image_sizes)},
        )
    return sizes


def _age_label(age_band: str) -> str:
    return {
        "primary": "primary school students (roughly ages 6-10)",
        "middle": "middle school students (roughly ages 11-14)",
        "high": "high school students (roughly ages 15-18)",
    }.get(age_band, "school students")


def layout_for(fmt: PosterFormat) -> str:
    return prompt_management.get_prompt(LAYOUT_PROMPT_NAMES[fmt], FORMAT_LAYOUTS[fmt])


def _corner_label(corner: str) -> str:
    return corner.replace("_", " ")


def mascot_plan_for(
    spec: PosterSpecDocument,
    fmt: PosterFormat,
    *,
    width: int,
    height: int,
    config=None,
    avoid_variant: str | None = None,
    assets: dict | None = None,
) -> MascotLayoutFormat:
    """Compute the Stage 5 reservation and the exact Stage 8 overlay plan."""
    mascot_config = config or load_app_config().mascot
    loaded = assets or load_mascot_assets(mascot_config.assets_dir)
    planned_question = plan_question(spec, mascot_config)
    question = planned_question.text
    corner = default_corner_for_format(fmt)
    variant = choose_variant(
        question,
        sorted(loaded),
        avoid=avoid_variant if mascot_config.avoid_repeat else None,
        salt=fmt,
    )
    asset = loaded[variant]
    layout = layout_mascot(
        width,
        height,
        question,
        corner,
        mascot_config,
        mascot_size=asset.size,
    )
    return MascotLayoutFormat(
        format=fmt,
        width=width,
        height=height,
        enabled=True,
        question=layout.question,
        question_source=planned_question.source,
        corner=layout.corner,
        preferred_corner=layout.corner,
        variant=variant,
        font_size=layout.font_size,
        lines=list(layout.lines),
        mascot=MascotRect(**layout.mascot.as_dict()),
        bubble=MascotRect(**layout.bubble.as_dict()),
        block=MascotRect(**layout.block.as_dict()),
        padding_x=layout.padding[0],
        padding_y=layout.padding[1],
        line_height=layout.line_height,
        radius=layout.radius,
        outline_width=layout.outline_width,
        tail=[MascotPoint(x=x, y=y) for x, y in layout.tail],
    )


def sync_prompts(*, labels: tuple[str, ...] = ("production",)) -> dict[str, bool]:
    return prompt_management.sync_prompts(
        {LAYOUT_PROMPT_NAMES[fmt]: text for fmt, text in FORMAT_LAYOUTS.items()},
        labels=labels,
    )


def build_prompt(
    spec: PosterSpecDocument,
    fmt: PosterFormat,
    *,
    width: int,
    height: int,
    mascot: MascotPlan | None = None,
) -> str:
    layout = layout_for(fmt)
    palette = ", ".join(spec.style.palette)
    lines: list[str] = [
        "Create a single, finished lesson poster as a flat image. "
        "Do not make a mockup, a photograph of a poster, a multi-panel sheet, or a UI screen.",
        "",
        f"CANVAS: {width} x {height} px, {FORMAT_LABELS[fmt].lower()} orientation.",
        f"LAYOUT: {layout}",
        "",
        f"AUDIENCE: {_age_label(spec.age_band)}. Keep all imagery age-appropriate, friendly, "
        "non-graphic, and free of violence, weapons, or adult content.",
        "",
        "COMPOSITION",
        f"Concept: {spec.composition.concept}",
        spec.composition.description,
        "",
        "STYLE",
        f"Palette: {palette}",
        f"Mood: {spec.style.mood}",
        f"Illustration: {spec.style.illustration_style}",
        "",
        "VISUAL ELEMENTS",
        "Draw one clear visual for each numbered detail. The numbers are ordering markers only; "
        "do not render them. The visual descriptions are guidance for the imagery only; do not "
        "render them as text.",
    ]
    for index, detail in enumerate(spec.details, start=1):
        lines.append(f"{index}. {detail.visual}")
    lines.extend(
        [
            "",
            "TEXT TO RENDER — EXACT, VERBATIM",
            "The double quotes below are delimiters, not part of the text. Render every quoted "
            "string exactly once, in this order. Do not render the quote marks, the list numbers, "
            "or the periods after those numbers. The section labels and instructions in this "
            "prompt are not poster text; do not render them.",
        ]
    )
    for index, value in enumerate(spec.strings_to_render, start=1):
        lines.append(f"{index}. {_quoted(value)}")
    if mascot is not None:
        block = mascot.block
        width_pct = block.width / width * 100
        height_pct = block.height / height * 100
        left_pct = block.x / width * 100
        top_pct = block.y / height * 100
        lines.extend(
            [
                "",
                "MASCOT RESERVED AREA — LEAVE THIS AREA EMPTY",
                f"A mascot and a white speech bubble with a purple outline will be added "
                f"after generation in the {_corner_label(mascot.corner)} area.",
                f"Keep the {_corner_label(mascot.corner)} corner empty of text, labels, "
                "numbers, and key illustrations. The reserved block is "
                f"{width_pct:.2f}% of the canvas width and {height_pct:.2f}% of the canvas "
                f"height, anchored at {left_pct:.2f}% from the left and {top_pct:.2f}% from "
                f"the top ({block.width} x {block.height} px in a {width} x {height} canvas).",
                "Do not draw a mascot, a speech bubble, a question mark, or a thought bubble "
                "anywhere in the image; the overlay is applied later by code.",
            ]
        )
    lines.extend(
        [
            "",
            "TYPOGRAPHY",
            "Use a clean, highly legible poster typeface with full Kazakh Cyrillic support. "
            f"Render the Kazakh letters {_KAZAKH_LETTERS} (uppercase: {_KAZAKH_UPPER}) as "
            "distinct letters, never as Russian lookalikes. If the lesson uses another language, "
            "still use a font that covers these letters so mixed-language terms render correctly.",
            "Make the title the largest text, the definition second, and detail labels stronger "
            "than body text. Keep all text horizontal, high-contrast, and comfortably readable. "
            "Do not use all-caps paragraphs, faux handwriting, or distorted lettering.",
            "",
            "HARD RULES",
            f"- Every visible word must be in {spec.language}. Keep source terms in their original "
            "language and spelling; never translate or transliterate them.",
            "- Do not add any text that is not in the quoted list above: no extra headings, "
            "captions, labels, numbers, slogans, signatures, URLs, QR codes, or decorative "
            "letters.",
            "- The numbered lines in this prompt are an ordering device. Never render 1., 2., 3., "
            "or any other list marker on the poster.",
            "- Do not add watermarks, logos, stock-photo marks, or attribution text.",
            "- Do not render wrong-language text, machine-translated text, placeholder text, or "
            "lorem ipsum.",
            "- Do not fall back to a plain grid, table, or mind map. Follow the composition and "
            "the format-specific layout.",
            "- Do not crop, overlap, or hide any quoted string. The poster must be complete and "
            "readable at a glance.",
            "- Keep imagery non-graphic for school students: no blood, gore, wounds, "
            "exposed bones, carcasses, dead animals, or parasites shown feeding. Imply "
            "predation with distance, tracks, alert posture, or silhouettes.",
            "- Prefer text accuracy over decorative detail.",
            "",
        ]
    )
    return "\n".join(lines)


def _spec_filename(spec: PosterSpecDocument, fmt: PosterFormat) -> str:
    return f"{spec.file_slug}-{fmt}.png"


def run_prompts(
    paths: RunPaths,
    manifest: RunManifest,
    store: RunStore,
    options: PromptOptions,
) -> PromptResult:
    options.validate()
    source = manifest.source
    if source is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing source metadata.")
    review_stage = manifest.stages.get("review")
    if review_stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the review stage.")
    if review_stage.status == "awaiting_review":
        raise GenLessonError(
            "review_pending",
            "The run is awaiting review; prompts cannot be built yet.",
            hint=f"Run `genlesson review {paths.run_id} --resume` after reviewing the spec.",
        )
    if review_stage.status not in {"complete", "skipped"}:
        raise GenLessonError(
            "stage_incomplete",
            "Review must complete before prompts can be built.",
            hint=f"Run `genlesson review {paths.run_id}` first.",
        )

    review_path = paths.stage_dir("review") / "review.json"
    review = read_model(review_path, ReviewDocument)
    if review.status not in {"auto_approved", "approved"}:
        raise GenLessonError(
            "review_pending",
            "The run has not been approved.",
            hint=f"Run `genlesson review {paths.run_id} --resume` after reviewing the spec.",
        )
    if not review.approved_spec_path or not review.approved_spec_sha256:
        raise GenLessonError(
            "artifact_missing", "Review artifact does not point to an approved spec."
        )
    approved_path = paths.resolve(review.approved_spec_path)
    approved_sha = sha256_file(approved_path)
    if approved_sha != review.approved_spec_sha256:
        raise GenLessonError(
            "artifact_invalid",
            "approved_spec.json does not match the hash recorded in review.json.",
        )
    if review.source_sha256 != source.sha256:
        raise GenLessonError(
            "artifact_invalid",
            "review.json does not match the manifest source.",
        )
    classification_sha = sha256_file(paths.stage_dir("classify") / "classification.json")
    if review.classification_sha256 != classification_sha:
        raise GenLessonError(
            "artifact_invalid",
            "review.json does not match the classification artifact.",
        )
    spec = read_model(approved_path, PosterSpecDocument)
    if spec.source_sha256 != source.sha256 or spec.classification_sha256 != classification_sha:
        raise GenLessonError(
            "artifact_invalid",
            "approved_spec.json does not match the source or classification.",
        )
    review_sha = sha256_file(review_path)
    mascot_config = load_app_config().mascot

    stage = manifest.stages.get("prompts")
    if stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the prompts stage.")
    output_path = paths.stage_dir("prompts") / "prompts.json"
    layout_path = paths.stage_dir("prompts") / "mascot_layout.json"
    if stage.status == "complete":
        bundle = read_model(output_path, PromptBundle)
        if bundle.approved_spec_sha256 != approved_sha or bundle.review_sha256 != review_sha:
            raise GenLessonError(
                "artifact_invalid",
                "prompts.json does not match the current approved spec or review.",
            )
        if (
            bundle.image_model == options.image_model
            and [item.format for item in bundle.formats] == list(options.formats)
            and (
                not mascot_config.enabled
                or (
                    all(item.mascot is not None for item in bundle.formats)
                    and layout_path.is_file()
                )
            )
        ):
            prompt_paths = [
                paths.stage_dir("prompts") / f"prompt-{item.format}.txt" for item in bundle.formats
            ]
            return PromptResult(
                document=bundle,
                output_path=output_path,
                prompt_paths=prompt_paths,
                resumed=True,
                mascot_layout_path=layout_path if mascot_config.enabled else None,
            )

    stage_dir = paths.stage_dir("prompts")
    stage_dir.mkdir(parents=True, exist_ok=True)
    store.set_stage(paths, manifest, "prompts", "running")
    try:
        sizes = _sizes_for_model(options.image_model)
        mascot_plans: dict[PosterFormat, MascotLayoutFormat] = {}
        previous_variant: str | None = None
        if mascot_config.enabled:
            mascot_assets = load_mascot_assets(mascot_config.assets_dir)
            for fmt in options.formats:
                width, height = sizes[fmt]
                plan = mascot_plan_for(
                    spec,
                    fmt,
                    width=width,
                    height=height,
                    config=mascot_config,
                    avoid_variant=previous_variant,
                    assets=mascot_assets,
                )
                mascot_plans[fmt] = plan
                previous_variant = plan.variant

        formats: list[PromptFormatSpec] = []
        prompt_texts: dict[str, str] = {}
        for fmt in options.formats:
            width, height = sizes[fmt]
            prompt = build_prompt(
                spec,
                fmt,
                width=width,
                height=height,
                mascot=mascot_plans.get(fmt),
            )
            plan = mascot_plans.get(fmt)
            formats.append(
                PromptFormatSpec(
                    format=fmt,
                    width=width,
                    height=height,
                    filename=_spec_filename(spec, fmt),
                    prompt=prompt,
                    mascot=(
                        MascotPlan.model_validate(
                            plan.model_dump(exclude={"format", "width", "height"})
                        )
                        if plan is not None
                        else None
                    ),
                )
            )
            prompt_texts[fmt] = prompt

        prompt_paths: list[Path] = []
        for fmt, prompt in prompt_texts.items():
            path = stage_dir / f"prompt-{fmt}.txt"
            write_text_atomic(path, prompt)
            prompt_paths.append(path)
        bundle = PromptBundle(
            source_sha256=source.sha256,
            approved_spec_sha256=approved_sha,
            review_sha256=review_sha,
            image_model=options.image_model,
            formats=formats,
            config=PromptConfig(
                image_provider=options.provider,  # type: ignore[arg-type]
                image_model=options.image_model,
                formats=list(options.formats),
            ),
            built_at=utc_now(),
        )
        write_json_atomic(output_path, bundle.model_dump(mode="json"))
        if mascot_plans:
            layout_document = MascotLayoutDocument(
                source_sha256=source.sha256,
                approved_spec_sha256=approved_sha,
                formats=[mascot_plans[fmt] for fmt in options.formats],
                generated_at=bundle.built_at,
            )
            write_json_atomic(layout_path, layout_document.model_dump(mode="json"))
        manifest.config["image_model"] = {
            "provider": options.provider,
            "model": options.image_model,
        }
        manifest.config["formats"] = list(options.formats)
        manifest.config["mascot"] = {
            "enabled": mascot_config.enabled,
            "assets_dir": str(mascot_config.assets_dir),
            "question": (next(iter(mascot_plans.values())).question if mascot_plans else None),
            "variants": {fmt: plan.variant for fmt, plan in mascot_plans.items()},
        }
        manifest.config["prompts"] = {
            "image_provider": options.provider,
            "image_model": options.image_model,
            "formats": list(options.formats),
            "sizes": {
                fmt: {"width": sizes[fmt][0], "height": sizes[fmt][1]} for fmt in options.formats
            },
        }
    except GenLessonError as exc:
        store.set_stage(paths, manifest, "prompts", "failed", error=exc.to_dict())
        raise
    except Exception as exc:
        log.exception("Unexpected prompt-building failure")
        wrapped = GenLessonError(
            "internal_error",
            f"Prompt building failed unexpectedly: {exc}",
            details={"exception": type(exc).__name__},
        )
        store.set_stage(paths, manifest, "prompts", "failed", error=wrapped.to_dict())
        raise wrapped from exc

    store.set_stage(
        paths,
        manifest,
        "prompts",
        "complete",
        output=paths.relative(output_path),
    )
    return PromptResult(
        document=bundle,
        output_path=output_path,
        prompt_paths=prompt_paths,
        resumed=False,
        mascot_layout_path=layout_path if mascot_plans else None,
    )
