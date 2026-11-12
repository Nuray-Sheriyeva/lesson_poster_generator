"""Stage 8: save final posters and same-stem JSON sidecars without overwriting."""

from __future__ import annotations

import copy
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image

from . import tracing
from .app_config import load_app_config
from .config import default_out_dir
from .errors import GenLessonError
from .generate import decode_image
from .mascot.assets import load_mascot_assets
from .mascot.compose import compose_recorded
from .models import (
    CostInfo,
    FormatVerification,
    GeneratedFormat,
    GenerationDocument,
    MascotLayoutDocument,
    PosterSpecDocument,
    PromptBundle,
    PromptFormatSpec,
    ReviewDocument,
    RunManifest,
    SaveDocument,
    SavedPoster,
    SaveStats,
    SkippedFormat,
    VerificationDocument,
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

_SAVE_DIR = "stage-08-save"
_SIDECAR_SUFFIX = ".json"


@dataclass
class SaveResult:
    document: SaveDocument
    output_path: Path
    files: list[Path]
    resumed: bool


def _input_stem(filename: str) -> str:
    stem = Path(filename).stem.strip()
    stem = re.sub(r'[\\/:*?"<>|]', "-", stem).strip(". ")
    return stem or "poster"


def _base_filename(
    input_stem: str,
    prompt: PromptFormatSpec,
    *,
    multi_format: bool,
) -> str:
    if multi_format:
        return f"{input_stem}-{prompt.format}"
    return input_stem


def _pixel_png(data: bytes) -> tuple[bytes, int, int]:
    image = decode_image(data)
    if image.media_type == "image/png":
        return data, image.width, image.height
    try:
        pixmap = pymupdf.Pixmap(data)
        return pixmap.tobytes("png"), pixmap.width, pixmap.height
    except Exception as exc:
        raise GenLessonError(
            "save_failed", f"Could not convert generated image to PNG: {exc}"
        ) from exc


_MIN_MASCOT_CANVAS = 256


def _mascot_metadata(
    *,
    applied: bool,
    reason: str | None,
    raw_sha: str,
    final_sha: str,
    raw_image_path: str | None,
    entry=None,
    placement: dict | None = None,
    layout=None,
    self_check: dict | None = None,
    method: str | None = None,
    text_boxes_hit: int | None = None,
) -> dict:
    preferred_corner = None
    question = None
    question_source = None
    variant = None
    if entry is not None:
        preferred_corner = getattr(entry, "preferred_corner", None) or getattr(
            entry, "corner", None
        )
        question = getattr(entry, "question", None)
        question_source = getattr(entry, "question_source", None)
        variant = getattr(entry, "variant", None)
    final_corner = (placement or {}).get("corner")
    if layout is not None:
        rect = layout.block.as_dict()
        font_size = layout.font_size
        variant = variant or None
    else:
        rect = (placement or {}).get("rect")
        entry_block = getattr(entry, "block", None) if entry is not None else None
        if rect is None and entry_block is not None:
            rect = entry_block.model_dump() if hasattr(entry_block, "model_dump") else entry_block
        font_size = getattr(entry, "font_size", None) if entry is not None else None
    return {
        "applied": applied,
        "reason": reason,
        "variant": variant,
        "preferred_corner": preferred_corner,
        "final_corner": final_corner,
        "corner": final_corner,
        "method": method if method is not None else (placement or {}).get("method"),
        "text_boxes_hit": (
            text_boxes_hit
            if text_boxes_hit is not None
            else (placement or {}).get("text_boxes_hit")
        ),
        "question": question,
        "question_source": question_source,
        "font_size": font_size,
        "rectangle": rect,
        "raw_image_path": raw_image_path,
        "raw_sha256": raw_sha,
        "final_sha256": final_sha,
        "self_check": self_check,
    }


def _apply_mascot(
    png: bytes,
    *,
    width: int,
    height: int,
    fmt: str,
    config,
    entry=None,
    placement: dict | None = None,
    mascot_reason: str | None = None,
    raw_image_path: str | None = None,
    assets: dict | None = None,
    run_id: str | None = None,
) -> tuple[bytes, dict, list[str]]:
    """Compose the mascot at the Stage 7 placement, before the exclusive write.

    A no-clear-corner result is saved without a mascot and flagged.  A self-check
    failure raises so the poster is never silently saved with text outside the
    bubble.
    """
    raw_sha = sha256_bytes(png)
    warnings: list[str] = []
    if not config.enabled:
        return (
            png,
            _mascot_metadata(
                applied=False,
                reason="disabled",
                raw_sha=raw_sha,
                final_sha=raw_sha,
                raw_image_path=raw_image_path,
            ),
            warnings,
        )
    if min(width, height) < _MIN_MASCOT_CANVAS:
        return (
            png,
            _mascot_metadata(
                applied=False,
                reason="canvas_too_small",
                raw_sha=raw_sha,
                final_sha=raw_sha,
                raw_image_path=raw_image_path,
                entry=entry,
                placement=placement,
                method="none",
                text_boxes_hit=0,
            ),
            [f"{fmt}: mascot skipped (canvas {width}x{height} is too small)."],
        )
    if mascot_reason == "mascot_no_clear_corner":
        return (
            png,
            _mascot_metadata(
                applied=False,
                reason="mascot_no_clear_corner",
                raw_sha=raw_sha,
                final_sha=raw_sha,
                raw_image_path=raw_image_path,
                entry=entry,
                placement=placement,
                method="none",
            ),
            [f"{fmt}: no clear corner for the mascot; saved without it and flagged."],
        )
    if entry is None:
        return (
            png,
            _mascot_metadata(
                applied=False,
                reason="missing_layout",
                raw_sha=raw_sha,
                final_sha=raw_sha,
                raw_image_path=raw_image_path,
                placement=placement,
            ),
            [f"{fmt}: mascot layout is missing; saved without it."],
        )
    final_corner = (placement or {}).get("corner")
    if not final_corner:
        return (
            png,
            _mascot_metadata(
                applied=False,
                reason="no_placement",
                raw_sha=raw_sha,
                final_sha=raw_sha,
                raw_image_path=raw_image_path,
                entry=entry,
                placement=placement,
            ),
            [f"{fmt}: no accepted mascot placement; saved without it."],
        )

    try:
        with Image.open(io.BytesIO(png)) as opened:
            base = opened.convert("RGB")
    except OSError as exc:
        raise GenLessonError("mascot_image_invalid", f"Could not read poster image: {exc}") from exc
    if assets is None:
        assets = load_mascot_assets(config.assets_dir)
    preferred_corner = getattr(entry, "preferred_corner", None) or getattr(entry, "corner", None)
    with tracing.stage_span(
        "mascot.compose",
        run_id=run_id,
        input={
            "format": fmt,
            "variant": getattr(entry, "variant", None),
            "final_corner": final_corner,
            "preferred_corner": preferred_corner,
        },
        metadata={"stage": "save", "format": fmt},
    ) as span:
        composed, layout, self_check = compose_recorded(
            base,
            entry,
            final_corner=str(final_corner),
            config=config,
            assets=assets,
        )
        span.score("font_size", float(layout.font_size))
        span.score(
            "corner_changed",
            0.0 if layout.corner == preferred_corner else 1.0,
        )
        span.update(
            output={
                "status": "composed",
                "self_check": self_check,
                "font_size": layout.font_size,
                "corner": layout.corner,
            }
        )
    buffer = io.BytesIO()
    composed.save(buffer, format="PNG")
    composed_png = buffer.getvalue()
    final_sha = sha256_bytes(composed_png)
    metadata = _mascot_metadata(
        applied=True,
        reason=None,
        raw_sha=raw_sha,
        final_sha=final_sha,
        raw_image_path=raw_image_path,
        entry=entry,
        placement=placement,
        layout=layout,
        self_check=self_check,
    )
    return composed_png, metadata, warnings


def _write_pair(
    paths: RunPaths,
    directory: Path,
    *,
    spec: PosterSpecDocument,
    prompt: PromptFormatSpec,
    png: bytes,
    width: int,
    height: int,
    sidecar: dict,
    input_stem: str | None = None,
    multi_format: bool = False,
    timestamp: str | None = None,
    shortid: str | None = None,
) -> tuple[Path, Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(png)
    stem = input_stem or _input_stem(spec.file_slug)
    base = _base_filename(stem, prompt, multi_format=multi_format)
    for counter in range(1, 1000):
        name = base if counter == 1 else f"{base}{counter}"
        image_path = directory / f"{name}.png"
        sidecar_path = directory / f"{name}{_SIDECAR_SUFFIX}"
        if image_path.exists() or sidecar_path.exists():
            continue
        data = copy.deepcopy(sidecar)
        data["saved"]["filename"] = image_path.name
        data["saved"]["sha256"] = digest
        data["saved"]["bytes"] = len(png)
        data["saved"]["width"] = width
        data["saved"]["height"] = height
        sidecar_bytes = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        try:
            write_bytes_exclusive(image_path, png)
        except GenLessonError as exc:
            if exc.code == "output_exists":
                continue
            raise
        try:
            write_bytes_exclusive(sidecar_path, sidecar_bytes)
        except GenLessonError as exc:
            image_path.unlink(missing_ok=True)
            if exc.code == "output_exists":
                continue
            raise
        except Exception as exc:
            image_path.unlink(missing_ok=True)
            raise GenLessonError("save_failed", f"Could not write sidecar: {exc}") from exc
        return image_path, sidecar_path, digest
    raise GenLessonError(
        "save_failed",
        "Could not allocate a unique poster filename after 999 collisions.",
    )


def _cost(format_verification: FormatVerification, generated: GeneratedFormat) -> CostInfo:
    retry_images = sum(1 for attempt in format_verification.attempts if attempt.attempt > 1)
    vision_calls = len(format_verification.attempts)
    amount = generated.cost_usd
    if amount is not None:
        note = "Estimated from recorded image usage; provider billing is authoritative."
    elif generated.usage or generated.last_call:
        note = (
            "Provider usage and latency were recorded, but no image price is configured; "
            "amount is unknown."
        )
    else:
        note = (
            "The provider does not return per-image pricing here; call counts are recorded "
            "for billing reconciliation."
        )
    return CostInfo(
        currency="USD",
        amount_usd=amount,
        estimated=amount is not None,
        note=note,
        image_calls=generated.attempts + retry_images,
        vision_calls=vision_calls,
    )


def _tracing_data(run_id: str) -> dict[str, str]:
    trace_id = tracing.trace_id_for_run(run_id)
    host = (os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")
    return {"trace_id": trace_id, "url": f"{host}/trace/{trace_id}"}


def _sidecar_data(
    *,
    run_id: str,
    spec: PosterSpecDocument,
    prompt: PromptFormatSpec,
    generated: GeneratedFormat,
    verification: FormatVerification,
    generation: GenerationDocument,
    verification_doc: VerificationDocument,
    mascot: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "format": prompt.format,
        "saved": {
            "filename": "",
            "sha256": "",
            "bytes": 0,
            "width": 0,
            "height": 0,
        },
        "source_sha256": verification_doc.source_sha256,
        "approved_spec_sha256": verification_doc.approved_spec_sha256,
        "verification_sha256": "",
        "spec": spec.model_dump(mode="json"),
        "prompt": prompt.prompt,
        "models": {
            "planner": {
                "provider": spec.config.planner_provider,
                "model": spec.config.planner_model,
            },
            "image": {
                "provider": generation.config.provider,
                "model": generation.image_model,
            },
            "verification": {
                "provider": verification_doc.config.verifier_provider,
                "model": verification_doc.config.verifier_model,
                "fallback_used": verification_doc.config.fallback_used,
            },
        },
        "attempts": {
            "generation": generated.model_dump(mode="json"),
            "verification": [attempt.model_dump(mode="json") for attempt in verification.attempts],
        },
        "verification": verification.model_dump(mode="json"),
        "mascot": mascot,
        "cost": _cost(verification, generated).model_dump(mode="json"),
        "tracing": _tracing_data(run_id),
        "saved_at": utc_now(),
    }


def _reusable(record: SavedPoster, *, require_mascot: bool = False) -> bool:
    image = Path(record.image_path)
    sidecar = Path(record.sidecar_path)
    if not (image.is_file() and sidecar.is_file() and sha256_file(image) == record.image_sha256):
        return False
    metadata = record.mascot
    if not require_mascot:
        # If mascots were just disabled, do not reuse a poster that already has
        # an overlay; recomposing from the verified source gives a clean poster.
        return not (isinstance(metadata, dict) and metadata.get("applied"))
    if not isinstance(metadata, dict):
        # A pre-mascot save record must be recomposed so older runs gain the
        # overlay too, instead of being silently reused.
        return False
    if metadata.get("applied"):
        return True
    # These are terminal skips: a tiny fixture/output or a verified no-clear-corner
    # poster would produce the same result if retried.
    return metadata.get("reason") in {"canvas_too_small", "mascot_no_clear_corner"}


def _document(
    *,
    out_dir: Path,
    source_sha256: str,
    approved_spec_sha256: str,
    verification_sha256: str,
    files: list[SavedPoster],
    skipped: list[SkippedFormat],
    warnings: list[str],
) -> SaveDocument:
    ordered = sorted(files, key=lambda item: item.format)
    return SaveDocument(
        out_dir=str(out_dir),
        source_sha256=source_sha256,
        approved_spec_sha256=approved_spec_sha256,
        verification_sha256=verification_sha256,
        files=ordered,
        skipped=skipped,
        warnings=warnings,
        stats=SaveStats(
            formats=len(ordered) + len(skipped),
            saved=len(ordered),
            needs_attention=sum(
                1 for item in ordered if item.verification_status == "needs_attention"
            ),
            skipped=len(skipped),
            mascots_applied=sum(
                1
                for item in ordered
                if isinstance(item.mascot, dict) and item.mascot.get("applied")
            ),
        ),
        saved_at=utc_now(),
    )


def run_save(
    paths: RunPaths,
    manifest: RunManifest,
    store: RunStore,
    *,
    out_dir: Path | None = None,
) -> SaveResult:
    source = manifest.source
    if source is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing source metadata.")
    verify_stage = manifest.stages.get("verify")
    if verify_stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the verify stage.")
    if verify_stage.status != "complete":
        raise GenLessonError(
            "stage_incomplete",
            "Verification must complete before posters can be saved.",
            hint=f"Run `genlesson verify {paths.run_id}` first.",
        )

    verification_path = paths.stage_dir("verify") / "verification.json"
    generation_path = paths.stage_dir("generate") / "generation.json"
    prompt_path = paths.stage_dir("prompts") / "prompts.json"
    review_path = paths.stage_dir("review") / "review.json"
    verification = read_model(verification_path, VerificationDocument)
    generation = read_model(generation_path, GenerationDocument)
    prompts = read_model(prompt_path, PromptBundle)
    review = read_model(review_path, ReviewDocument)
    layout_path = paths.stage_dir("prompts") / "mascot_layout.json"
    mascot_layout = read_model(layout_path, MascotLayoutDocument) if layout_path.is_file() else None
    if mascot_layout is not None and (
        mascot_layout.source_sha256 != source.sha256
        or (
            review.approved_spec_sha256
            and mascot_layout.approved_spec_sha256 != review.approved_spec_sha256
        )
    ):
        raise GenLessonError("artifact_invalid", "mascot_layout.json does not match the run spec.")
    mascot_by_format = (
        {entry.format: entry for entry in mascot_layout.formats}
        if mascot_layout is not None
        else {}
    )
    if verification.source_sha256 != source.sha256:
        raise GenLessonError("artifact_invalid", "verification.json does not match the source.")
    if generation.prompts_sha256 != sha256_file(prompt_path):
        raise GenLessonError("artifact_invalid", "generation.json does not match prompts.json.")
    if prompts.review_sha256 != sha256_file(review_path):
        raise GenLessonError("artifact_invalid", "prompts.json does not match review.json.")
    if not review.approved_spec_path or not review.approved_spec_sha256:
        raise GenLessonError("artifact_invalid", "Review artifact has no approved spec.")
    approved_path = paths.resolve(review.approved_spec_path)
    if sha256_file(approved_path) != review.approved_spec_sha256:
        raise GenLessonError("artifact_invalid", "approved_spec.json does not match review.json.")
    spec = read_model(approved_path, PosterSpecDocument)

    verification_sha = sha256_file(verification_path)
    stage = manifest.stages.get("save")
    if stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the save stage.")
    save_dir = paths.run_dir / _SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    input_stem = _input_stem(source.filename)
    output_root = (out_dir or default_out_dir()).expanduser().resolve()
    output_dir = output_root / input_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "save.json"
    mascot_config = load_app_config().mascot
    history: SaveDocument | None = None
    if output_path.is_file():
        try:
            candidate = read_model(output_path, SaveDocument)
            if candidate.verification_sha256 == verification_sha and candidate.out_dir == str(
                output_dir
            ):
                history = candidate
        except GenLessonError:
            history = None

    if stage.status == "complete" and history is not None:
        accounted = len(history.files) + len(history.skipped) == len(verification.formats)
        if accounted and all(
            _reusable(record, require_mascot=mascot_config.enabled) for record in history.files
        ):
            return SaveResult(
                document=history,
                output_path=output_path,
                files=[Path(record.image_path) for record in history.files],
                resumed=True,
            )

    store.set_stage(paths, manifest, "save", "running")
    prompts_by_format = {item.format: item for item in prompts.formats}
    generated_by_format = {item.format: item for item in generation.formats}
    existing = {record.format: record for record in history.files} if history else {}
    files: list[SavedPoster] = []
    skipped: list[SkippedFormat] = []
    warnings: list[str] = []
    multi_format = len(prompts.formats) > 1
    mascot_assets = None
    if mascot_config.enabled and mascot_by_format:
        try:
            mascot_assets = load_mascot_assets(mascot_config.assets_dir)
        except GenLessonError as exc:
            warnings.append(f"mascot assets are unavailable: {exc.message}")
            mascot_assets = None

    for fmt_verification in verification.formats:
        if fmt_verification.status == "generation_failed" or not fmt_verification.best_image_path:
            reason = (
                "generation failed"
                if fmt_verification.status == "generation_failed"
                else "no best image"
            )
            skipped.append(SkippedFormat(format=fmt_verification.format, reason=reason))
            warnings.append(f"{fmt_verification.format}: skipped ({reason}).")
            continue
        prompt = prompts_by_format.get(fmt_verification.format)
        generated = generated_by_format.get(fmt_verification.format)
        if prompt is None or generated is None:
            skipped.append(
                SkippedFormat(
                    format=fmt_verification.format,
                    reason="missing prompt or generation record",
                )
            )
            warnings.append(f"{fmt_verification.format}: missing prompt/generation record.")
            continue
        old = existing.get(fmt_verification.format)
        if old is not None and _reusable(old, require_mascot=mascot_config.enabled):
            files.append(old)
            continue
        try:
            source_image = paths.resolve(fmt_verification.best_image_path)
            png, width, height = _pixel_png(source_image.read_bytes())
        except Exception as exc:  # noqa: BLE001 - per-format skip
            skipped.append(
                SkippedFormat(format=fmt_verification.format, reason=f"image unreadable: {exc}")
            )
            warnings.append(f"{fmt_verification.format}: could not read best image.")
            continue
        try:
            png, mascot_meta, mascot_warnings = _apply_mascot(
                png,
                width=width,
                height=height,
                fmt=fmt_verification.format,
                config=mascot_config,
                entry=mascot_by_format.get(fmt_verification.format),
                placement=fmt_verification.mascot,
                mascot_reason=fmt_verification.mascot_reason,
                raw_image_path=fmt_verification.best_image_path,
                assets=mascot_assets,
                run_id=paths.run_id,
            )
        except GenLessonError as exc:
            store.set_stage(paths, manifest, "save", "failed", error=exc.to_dict())
            raise
        warnings.extend(mascot_warnings)
        sidecar = _sidecar_data(
            run_id=paths.run_id,
            spec=spec,
            prompt=prompt,
            generated=generated,
            verification=fmt_verification,
            generation=generation,
            verification_doc=verification,
            mascot=mascot_meta,
        )
        sidecar["verification_sha256"] = verification_sha
        try:
            image_path, sidecar_path, digest = _write_pair(
                paths,
                output_dir,
                spec=spec,
                prompt=prompt,
                png=png,
                width=width,
                height=height,
                sidecar=sidecar,
                input_stem=input_stem,
                multi_format=multi_format,
            )
        except GenLessonError as exc:
            skipped.append(SkippedFormat(format=fmt_verification.format, reason=exc.message))
            warnings.append(f"{fmt_verification.format}: save failed: {exc.message}")
            continue
        files.append(
            SavedPoster(
                format=fmt_verification.format,
                filename=image_path.name,
                sidecar_filename=sidecar_path.name,
                image_path=str(image_path),
                sidecar_path=str(sidecar_path),
                image_sha256=digest,
                bytes=len(png),
                width=width,
                height=height,
                verification_status=fmt_verification.status,  # type: ignore[arg-type]
                best_attempt=fmt_verification.best_attempt,
                mascot=mascot_meta,
                saved_at=utc_now(),
            )
        )
        partial = _document(
            out_dir=output_dir,
            source_sha256=source.sha256,
            approved_spec_sha256=review.approved_spec_sha256,
            verification_sha256=verification_sha,
            files=files,
            skipped=skipped,
            warnings=warnings,
        )
        write_json_atomic(output_path, partial.model_dump(mode="json"))

    document = _document(
        out_dir=output_dir,
        source_sha256=source.sha256,
        approved_spec_sha256=review.approved_spec_sha256,
        verification_sha256=verification_sha,
        files=files,
        skipped=skipped,
        warnings=warnings,
    )
    write_json_atomic(output_path, document.model_dump(mode="json"))
    if not files:
        error = GenLessonError(
            "no_posters_saved",
            "No poster images could be saved.",
            hint="Inspect verification.json and the save warnings.",
            details={"skipped": [item.model_dump(mode="json") for item in skipped]},
        )
        store.set_stage(paths, manifest, "save", "failed", error=error.to_dict())
        raise error

    store.set_stage(paths, manifest, "save", "complete", output=paths.relative(output_path))
    return SaveResult(
        document=document,
        output_path=output_path,
        files=[Path(record.image_path) for record in document.files],
        resumed=False,
    )
