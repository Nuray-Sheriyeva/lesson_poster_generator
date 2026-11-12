"""Stage 6: generate one image per requested format with partial-failure handling."""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from . import tracing
from .config import GenerationOptions
from .errors import GenLessonError
from .image import (
    ImageProvider,
    ImageProviderError,
    ModerationRefusal,
    build_image_provider,
)
from .models import (
    GeneratedFormat,
    GenerationConfig,
    GenerationDocument,
    GenerationErrorInfo,
    GenerationStats,
    PosterFormat,
    PromptBundle,
    PromptFormatSpec,
    ReviewDocument,
    RunManifest,
    utc_now,
)
from .runs import RunPaths, RunStore
from .storage import (
    read_model,
    sha256_bytes,
    sha256_file,
    write_bytes_exclusive,
    write_json_atomic,
)

log = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}


@dataclass
class GenerateResult:
    document: GenerationDocument
    output_path: Path
    resumed: bool
    generated: int
    failed: int


@dataclass
class _DecodedImage:
    data: bytes
    width: int
    height: int
    media_type: str
    extension: str


def soften_prompt(prompt: str) -> str:
    """A gentler variant used once after a moderation refusal.

    It removes explicit safety vocabulary that can itself be flagged and
    replaces the audience line, while leaving the composition, style, and exact
    render strings untouched.
    """
    lines: list[str] = []
    for line in prompt.splitlines():
        if line.startswith("AUDIENCE:"):
            lines.append(
                "AUDIENCE: school students. Keep all artwork gentle, calm, friendly, and "
                "suitable for a classroom."
            )
        else:
            lines.append(line)
    prefix = (
        "Create a gentle, friendly, age-appropriate educational illustration. "
        "Avoid frightening imagery and any realistic depiction of harm.\n\n"
    )
    return prefix + "\n".join(lines)


def _detect_media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def decode_image(data: bytes) -> _DecodedImage:
    media_type = _detect_media_type(data)
    if media_type is None:
        raise ImageProviderError("invalid_image", "Provider returned an unsupported image format.")
    try:
        pixmap = pymupdf.Pixmap(data)
        width, height = pixmap.width, pixmap.height
    except Exception as exc:
        raise ImageProviderError(
            "invalid_image", "Provider returned undecodable image bytes."
        ) from exc
    if width < 1 or height < 1:
        raise ImageProviderError("invalid_image", "Provider returned a zero-sized image.")
    return _DecodedImage(
        data=data,
        width=width,
        height=height,
        media_type=media_type,
        extension=_IMAGE_EXTENSIONS[media_type],
    )


def save_image(
    paths: RunPaths,
    stage_dir: Path,
    *,
    base_stem: str,
    image: _DecodedImage,
    subdir: str = "images",
) -> Path:
    images_dir = stage_dir / subdir
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", base_stem).strip("-") or "poster"
    for suffix in range(1, 100):
        name = f"{stem}.{image.extension}" if suffix == 1 else f"{stem}-{suffix}.{image.extension}"
        path = images_dir / name
        try:
            write_bytes_exclusive(path, image.data)
            return path
        except GenLessonError as exc:
            if exc.code != "output_exists":
                raise
            continue
    raise ImageProviderError("image_save_failed", "Could not allocate a unique image filename.")


def _prompt_sha(prompt: PromptFormatSpec) -> str:
    return sha256_bytes(prompt.prompt.encode("utf-8"))


def _result_for_failure(
    prompt: PromptFormatSpec,
    *,
    error: ImageProviderError,
    attempts: int,
    moderation_retry: bool,
    variant: str,
    last_call: dict[str, Any] | None = None,
) -> GeneratedFormat:
    return GeneratedFormat(
        format=prompt.format,
        width=prompt.width,
        height=prompt.height,
        size=f"{prompt.width}x{prompt.height}",
        status="failed",
        attempts=attempts,
        moderation_retry=moderation_retry,
        prompt_sha256=_prompt_sha(prompt),
        prompt_variant=variant,  # type: ignore[arg-type]
        usage=last_call.get("usage") if last_call else None,
        last_call=last_call,
        error=GenerationErrorInfo(**error.to_dict()),
        generated_at=None,
    )


def _result_for_success(
    prompt: PromptFormatSpec,
    *,
    image: _DecodedImage,
    image_path: str,
    attempts: int,
    moderation_retry: bool,
    variant: str,
    last_call: dict[str, Any] | None = None,
) -> GeneratedFormat:
    return GeneratedFormat(
        format=prompt.format,
        width=image.width,
        height=image.height,
        size=f"{prompt.width}x{prompt.height}",
        status="generated",
        image_path=image_path,
        image_sha256=sha256_bytes(image.data),
        bytes=len(image.data),
        media_type=image.media_type,
        attempts=attempts,
        moderation_retry=moderation_retry,
        prompt_sha256=_prompt_sha(prompt),
        prompt_variant=variant,  # type: ignore[arg-type]
        usage=last_call.get("usage") if last_call else None,
        last_call=last_call,
        error=None,
        generated_at=utc_now(),
    )


def _reusable(
    existing: GeneratedFormat | None,
    prompt: PromptFormatSpec,
    paths: RunPaths,
) -> bool:
    if existing is None or existing.status != "generated":
        return False
    if existing.prompt_sha256 != _prompt_sha(prompt):
        return False
    if not existing.image_path or not existing.image_sha256:
        return False
    try:
        image_path = paths.resolve(existing.image_path)
    except GenLessonError:
        return False
    if not image_path.is_file():
        return False
    return sha256_file(image_path) == existing.image_sha256


def _provider_last_call(provider: ImageProvider) -> dict[str, Any] | None:
    value = getattr(provider, "last_call", None)
    return copy.deepcopy(value) if isinstance(value, dict) else None


def _generate_one(
    provider: ImageProvider,
    prompt: PromptFormatSpec,
    *,
    paths: RunPaths,
    stage_dir: Path,
) -> GeneratedFormat:
    with tracing.stage_span(
        f"generate:{prompt.format}",
        run_id=paths.run_id,
        input={"format": prompt.format, "size": f"{prompt.width}x{prompt.height}"},
        metadata={
            "stage": "generate",
            "format": prompt.format,
            "image_model": getattr(provider, "model", None),
        },
    ) as span:
        result = _generate_one_inner(
            provider, prompt, paths=paths, stage_dir=stage_dir
        )
        span.update(
            output={
                "status": result.status,
                "attempts": result.attempts,
                "usage": result.usage,
                "last_call": result.last_call,
            }
        )
        return result


def _generate_one_inner(
    provider: ImageProvider,
    prompt: PromptFormatSpec,
    *,
    paths: RunPaths,
    stage_dir: Path,
) -> GeneratedFormat:
    size = f"{prompt.width}x{prompt.height}"
    attempts = 0
    moderation_retry = False
    variant = "original"
    try:
        attempts = 1
        data = provider.generate(prompt.prompt, size)
        image = decode_image(data)
    except ModerationRefusal:
        moderation_retry = True
        variant = "toned_down"
        try:
            attempts = 2
            data = provider.generate(soften_prompt(prompt.prompt), size)
            image = decode_image(data)
        except ModerationRefusal as second:
            return _result_for_failure(
                prompt,
                error=second,
                attempts=attempts,
                moderation_retry=True,
                variant=variant,
                last_call=_provider_last_call(provider),
            )
        except ImageProviderError as second:
            return _result_for_failure(
                prompt,
                error=second,
                attempts=attempts,
                moderation_retry=True,
                variant=variant,
                last_call=_provider_last_call(provider),
            )
        except Exception as second:
            return _result_for_failure(
                prompt,
                error=ImageProviderError("image_failed", str(second)),
                attempts=attempts,
                moderation_retry=True,
                variant=variant,
                last_call=_provider_last_call(provider),
            )
    except ImageProviderError as exc:
        return _result_for_failure(
            prompt,
            error=exc,
            attempts=attempts or 1,
            moderation_retry=moderation_retry,
            variant=variant,
            last_call=_provider_last_call(provider),
        )
    except Exception as exc:
        return _result_for_failure(
            prompt,
            error=ImageProviderError("image_failed", str(exc)),
            attempts=attempts or 1,
            moderation_retry=moderation_retry,
            variant=variant,
            last_call=_provider_last_call(provider),
        )

    try:
        image_path = save_image(
            paths,
            stage_dir,
            base_stem=Path(prompt.filename).stem,
            image=image,
        )
    except Exception as exc:
        return _result_for_failure(
            prompt,
            error=ImageProviderError(
                "image_save_failed", f"Could not save generated image: {exc}"
            ),
            attempts=attempts,
            moderation_retry=moderation_retry,
            variant=variant,
            last_call=_provider_last_call(provider),
        )
    return _result_for_success(
        prompt,
        image=image,
        image_path=paths.relative(image_path),
        attempts=attempts,
        moderation_retry=moderation_retry,
        variant=variant,
        last_call=_provider_last_call(provider),
    )


def _document(
    *,
    prompts: PromptBundle,
    prompts_sha256: str,
    results: list[GeneratedFormat],
    options: GenerationOptions,
) -> GenerationDocument:
    by_format = {item.format: item for item in results}
    ordered = [by_format[item.format] for item in prompts.formats if item.format in by_format]
    warnings: list[str] = []
    targets = {item.format: item for item in prompts.formats}
    for item in ordered:
        target = targets[item.format]
        if item.status == "failed":
            message = item.error.message if item.error else "generation failed"
            warnings.append(f"{item.format}: {message}")
        elif item.moderation_retry:
            warnings.append(
                f"{item.format}: moderation refusal; retried with a toned-down prompt."
            )
        if item.status == "generated" and (
            item.width != target.width or item.height != target.height
        ):
            warnings.append(
                f"{item.format}: expected {target.width}x{target.height}, got "
                f"{item.width}x{item.height}."
            )
    generated = sum(1 for item in ordered if item.status == "generated")
    failed = sum(1 for item in ordered if item.status == "failed")
    return GenerationDocument(
        source_sha256=prompts.source_sha256,
        approved_spec_sha256=prompts.approved_spec_sha256,
        review_sha256=prompts.review_sha256,
        prompts_sha256=prompts_sha256,
        image_model=prompts.image_model,
        formats=ordered,
        warnings=warnings,
        stats=GenerationStats(
            requested=len(prompts.formats), generated=generated, failed=failed
        ),
        config=GenerationConfig(
            provider=options.provider,
            image_model=prompts.image_model,
            timeout_s=options.timeout_s,
            max_retries=options.max_retries,
        ),
        generated_at=utc_now(),
    )


def run_generate(
    paths: RunPaths,
    manifest: RunManifest,
    store: RunStore,
    options: GenerationOptions,
    *,
    provider: ImageProvider | None = None,
) -> GenerateResult:
    options.validate()
    source = manifest.source
    if source is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing source metadata.")
    prompt_stage = manifest.stages.get("prompts")
    if prompt_stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the prompts stage.")
    if prompt_stage.status != "complete":
        raise GenLessonError(
            "stage_incomplete",
            "Prompts must be complete before generation can run.",
            hint=f"Run `genlesson prompts {paths.run_id}` first.",
        )

    prompt_path = paths.stage_dir("prompts") / "prompts.json"
    prompts = read_model(prompt_path, PromptBundle)
    if prompts.source_sha256 != source.sha256:
        raise GenLessonError("artifact_invalid", "prompts.json does not match the source.")
    if prompts.config.image_provider != options.provider:
        raise GenLessonError(
            "image_provider_mismatch",
            "The generation provider does not match the provider used to build prompts.",
            hint="Rebuild prompts with the same --image-provider.",
        )
    review_path = paths.stage_dir("review") / "review.json"
    review = read_model(review_path, ReviewDocument)
    if review.status not in {"auto_approved", "approved"}:
        raise GenLessonError("review_pending", "The run has not been approved.")
    if (
        prompts.review_sha256 != sha256_file(review_path)
        or prompts.approved_spec_sha256 != review.approved_spec_sha256
    ):
        raise GenLessonError(
            "artifact_invalid", "prompts.json does not match the current review."
        )
    prompts_sha256 = sha256_file(prompt_path)

    stage = manifest.stages.get("generate")
    if stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the generate stage.")
    stage_dir = paths.stage_dir("generate")
    stage_dir.mkdir(parents=True, exist_ok=True)
    output_path = stage_dir / "generation.json"
    history: GenerationDocument | None = None
    if output_path.is_file():
        try:
            candidate = read_model(output_path, GenerationDocument)
            if candidate.prompts_sha256 == prompts_sha256:
                history = candidate
        except GenLessonError:
            history = None

    existing = {item.format: item for item in history.formats} if history else {}
    prompts_by_format: dict[PosterFormat, PromptFormatSpec] = {
        item.format: item for item in prompts.formats
    }
    results: dict[PosterFormat, GeneratedFormat] = {}
    for fmt, item in prompts_by_format.items():
        old = existing.get(fmt)
        if _reusable(old, item, paths):
            results[fmt] = old  # type: ignore[assignment]
    stale = [
        fmt
        for fmt in prompts_by_format
        if fmt not in results
    ]
    if not stale:
        document = _document(
            prompts=prompts,
            prompts_sha256=prompts_sha256,
            results=list(results.values()),
            options=options,
        )
        if stage.status != "complete":
            store.set_stage(
                paths,
                manifest,
                "generate",
                "complete",
                output=paths.relative(output_path),
            )
        return GenerateResult(
            document=document,
            output_path=output_path,
            resumed=True,
            generated=document.stats.generated,
            failed=document.stats.failed,
        )

    store.set_stage(paths, manifest, "generate", "running")
    try:
        if provider is None:
            provider = build_image_provider(options, model=prompts.image_model)
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, GenLessonError)
            else GenLessonError("image_provider_required", str(exc))
        )
        store.set_stage(paths, manifest, "generate", "failed", error=error.to_dict())
        raise error from exc

    for fmt in stale:
        result = _generate_one(
            provider,
            prompts_by_format[fmt],
            paths=paths,
            stage_dir=stage_dir,
        )
        results[fmt] = result
        partial = _document(
            prompts=prompts,
            prompts_sha256=prompts_sha256,
            results=list(results.values()),
            options=options,
        )
        write_json_atomic(output_path, partial.model_dump(mode="json"))

    document = _document(
        prompts=prompts,
        prompts_sha256=prompts_sha256,
        results=list(results.values()),
        options=options,
    )
    write_json_atomic(output_path, document.model_dump(mode="json"))
    manifest.config["generation"] = {
        "provider": options.provider,
        "image_model": prompts.image_model,
        "formats": [item.format for item in prompts.formats],
        "generated": document.stats.generated,
        "failed": document.stats.failed,
    }

    if document.stats.generated == 0:
        error = GenLessonError(
            "all_formats_failed",
            "Every requested poster format failed to generate.",
            hint="Inspect generation.json, then retry.",
            details={"failures": [
                {
                    "format": item.format,
                    "code": item.error.code if item.error else "unknown",
                }
                for item in document.formats
            ]},
        )
        store.set_stage(paths, manifest, "generate", "failed", error=error.to_dict())
        raise error

    store.set_stage(
        paths,
        manifest,
        "generate",
        "complete",
        output=paths.relative(output_path),
    )
    return GenerateResult(
        document=document,
        output_path=output_path,
        resumed=False,
        generated=document.stats.generated,
        failed=document.stats.failed,
    )
