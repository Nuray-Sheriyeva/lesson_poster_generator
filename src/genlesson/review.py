"""Stage 4: optional human review gate with a resumable edited spec."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import ReviewOptions
from .errors import GenLessonError
from .models import (
    PosterSpecDocument,
    ReviewDocument,
    RunManifest,
    SourceConflict,
    utc_now,
)
from .runs import RunPaths, RunStore
from .storage import (
    read_model,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
    write_text_atomic,
)

log = logging.getLogger(__name__)

SUMMARY_NAME = "review_summary.md"
APPROVED_NAME = "approved_spec.json"
REVIEW_NAME = "review.json"


@dataclass
class ReviewResult:
    document: ReviewDocument
    output_path: Path
    summary_path: Path | None
    approved_spec_path: Path | None
    paused: bool
    resumed: bool


def _canonical_strings(spec: PosterSpecDocument) -> list[str]:
    values = [spec.central_idea.title, spec.central_idea.definition]
    for detail in spec.details:
        values.append(detail.label)
        values.append(detail.explanation)
        if detail.example:
            values.append(detail.example)
    return [value for value in values if value]


def _validate_spec(
    spec: PosterSpecDocument,
    *,
    source_sha256: str,
    classification_sha256: str,
) -> None:
    problems: list[str] = []
    if spec.source_sha256 != source_sha256:
        problems.append("source_sha256 does not match the run source.")
    if spec.classification_sha256 != classification_sha256:
        problems.append("classification_sha256 does not match the classification artifact.")
    if not spec.details:
        problems.append("details must not be empty.")
    if not spec.strings_to_render:
        problems.append("strings_to_render must not be empty.")
    if len(set(spec.strings_to_render)) != len(spec.strings_to_render):
        problems.append("strings_to_render contains duplicate strings.")
    missing = [
        value
        for value in _canonical_strings(spec)
        if value not in spec.strings_to_render
    ]
    if missing:
        problems.append(
            "strings_to_render is missing canonical text: "
            + "; ".join(repr(value) for value in missing[:10])
        )
    if problems:
        raise GenLessonError(
            "spec_invalid",
            "The poster spec did not pass review validation.",
            hint="Fix the edited spec and resume again.",
            details={"problems": problems},
        )


def _read_spec(path: Path) -> tuple[PosterSpecDocument, bytes]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise GenLessonError(
            "spec_invalid",
            f"Could not read poster spec: {path}",
            details={"reason": str(exc)},
        ) from exc
    try:
        spec = PosterSpecDocument.model_validate(json.loads(data))
    except Exception as exc:
        raise GenLessonError(
            "spec_invalid",
            f"Poster spec is not valid JSON for the schema: {path}",
            hint="Fix the edited spec and resume again.",
            details={"reason": str(exc)},
        ) from exc
    return spec, data


def _write_summary(
    path: Path,
    spec: PosterSpecDocument,
    review: ReviewDocument,
    *,
    run_id: str,
) -> None:
    status_label = {
        "awaiting_review": "Awaiting review",
        "approved": "Approved",
        "auto_approved": "Auto-approved",
    }.get(review.status, review.status)
    mode_label = {
        "off": "off",
        "on": "on",
        "on_conflict": "on conflict",
    }.get(review.mode, review.mode)
    lines = [
        f"# Review: {spec.central_idea.title}",
        "",
        f"- Run: `{run_id}`",
        f"- Status: **{status_label}**",
        f"- Mode: `{mode_label}`",
        f"- Language: `{spec.language}`",
        f"- Age band: `{spec.age_band}`",
        f"- File slug: `{spec.file_slug}`",
        "",
        "## Central idea",
        "",
        f"**{spec.central_idea.title}** — {spec.central_idea.definition}",
        "",
        "## Details",
        "",
    ]
    for index, detail in enumerate(spec.details, start=1):
        example = f" Example: {detail.example}" if detail.example else ""
        lines.append(
            f"{index}. **{detail.label}** — {detail.explanation}{example} "
            f"(source page {detail.source_page})"
        )
    lines.extend(
        [
            "",
            "## Composition",
            "",
            f"**{spec.composition.concept}** — {spec.composition.description}",
            "",
            "## Style",
            "",
            f"- Palette: {', '.join(spec.style.palette)}",
            f"- Mood: {spec.style.mood}",
            f"- Illustration: {spec.style.illustration_style}",
            "",
            "## Source conflicts",
            "",
        ]
    )
    if spec.source_conflicts:
        for conflict in spec.source_conflicts:
            lines.append(f"- {conflict.issue} **Resolution:** {conflict.resolution}")
    else:
        lines.append("None flagged by the planner.")
    lines.extend(["", "## Strings to render", ""])
    for value in spec.strings_to_render:
        lines.append(f"- {json.dumps(value, ensure_ascii=False)}")
    lines.extend(
        [
            "",
            "## How to review",
            "",
            "1. Edit `stage-03-plan/poster_spec.json` or prepare a separate spec file.",
            f"2. Run `genlesson review {run_id} --resume` to approve the current spec.",
            f"3. Or run `genlesson review {run_id} --resume --spec path/to/edited.json`.",
            "4. The edited spec is revalidated before the run continues.",
            "",
        ]
    )
    write_text_atomic(path, "\n".join(lines))


def _review_payload(
    *,
    source_sha256: str,
    classification_sha256: str,
    mode: str,
    status: str,
    source_conflicts: list[SourceConflict],
    plan_sha256: str,
    requested_at: str,
    reviewed_at: str | None,
    summary_path: str | None,
    approved_spec_path: str | None,
    approved_spec_sha256: str | None,
    edited: bool,
) -> ReviewDocument:
    return ReviewDocument(
        schema_version=1,
        source_sha256=source_sha256,
        classification_sha256=classification_sha256,
        plan_sha256=plan_sha256,
        mode=mode,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        source_conflicts=source_conflicts,
        summary_path=summary_path,
        approved_spec_path=approved_spec_path,
        approved_spec_sha256=approved_spec_sha256,
        edited=edited,
        requested_at=requested_at,
        reviewed_at=reviewed_at,
    )


def run_review(
    paths: RunPaths,
    manifest: RunManifest,
    store: RunStore,
    options: ReviewOptions,
    *,
    resume: bool = False,
    spec_path: Path | None = None,
) -> ReviewResult:
    options.validate()
    source = manifest.source
    if source is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing source metadata.")
    plan_stage = manifest.stages.get("plan")
    if plan_stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the plan stage.")
    if plan_stage.status != "complete":
        raise GenLessonError(
            "stage_incomplete",
            "Planning must complete before review can run.",
            hint="Run `genlesson plan <run_id>` first.",
        )

    plan_path = paths.stage_dir("plan") / "poster_spec.json"
    classification_path = paths.stage_dir("classify") / "classification.json"
    plan_spec, plan_bytes = _read_spec(plan_path)
    classification_sha256 = sha256_file(classification_path)
    _validate_spec(
        plan_spec,
        source_sha256=source.sha256,
        classification_sha256=classification_sha256,
    )
    plan_sha256 = sha256_file(plan_path)

    review_stage = manifest.stages.get("review")
    if review_stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the review stage.")
    review_dir = paths.stage_dir("review")
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / REVIEW_NAME
    summary_path = review_dir / SUMMARY_NAME
    approved_path = review_dir / APPROVED_NAME

    if review_stage.status in {"complete", "skipped"}:
        document = read_model(review_path, ReviewDocument)
        approved = (
            paths.resolve(document.approved_spec_path)
            if document.approved_spec_path
            else None
        )
        if approved is None or not approved.is_file():
            raise GenLessonError(
                "artifact_missing",
                "Approved poster spec is missing for a completed review.",
                hint="Re-run review without --resume to rebuild it.",
            )
        if spec_path is not None:
            raise GenLessonError(
                "review_not_pending",
                "This run is already approved; an edited spec cannot be applied.",
            )
        reopen = (
            review_stage.status == "skipped"
            and not resume
            and options.mode in {"on", "on_conflict"}
            and (options.mode == "on" or bool(plan_spec.source_conflicts))
        )
        if not reopen:
            return ReviewResult(
                document=document,
                output_path=review_path,
                summary_path=(
                    paths.resolve(document.summary_path) if document.summary_path else None
                ),
                approved_spec_path=approved,
                paused=False,
                resumed=True,
            )

    if review_stage.status == "awaiting_review":
        if not resume:
            raise GenLessonError(
                "review_pending",
                "The run is awaiting review.",
                hint=(
                    f"Edit the spec, then run `genlesson review {paths.run_id} --resume` "
                    "or pass --spec <edited.json>."
                ),
            )
        document = read_model(review_path, ReviewDocument)
        source_path = spec_path if spec_path is not None else plan_path
        spec, raw = _read_spec(source_path)
        _validate_spec(
            spec,
            source_sha256=source.sha256,
            classification_sha256=classification_sha256,
        )
        write_bytes_atomic(approved_path, raw)
        approved_sha = sha256_file(approved_path)
        document.status = "approved"
        document.reviewed_at = utc_now()
        document.source_conflicts = spec.source_conflicts
        document.approved_spec_path = paths.relative(approved_path)
        document.approved_spec_sha256 = approved_sha
        document.edited = approved_sha != document.plan_sha256
        document.summary_path = paths.relative(summary_path)
        _write_summary(summary_path, spec, document, run_id=paths.run_id)
        write_json_atomic(review_path, document.model_dump(mode="json"))
        manifest.config["review_gate"] = document.mode != "off"
        manifest.config["review_mode"] = document.mode
        store.set_stage(paths, manifest, "review", "complete", output=paths.relative(review_path))
        return ReviewResult(
            document=document,
            output_path=review_path,
            summary_path=summary_path,
            approved_spec_path=approved_path,
            paused=False,
            resumed=True,
        )

    if spec_path is not None:
        raise GenLessonError(
            "review_not_pending",
            "--spec can only be used when a review is awaiting approval.",
        )

    mode = options.mode
    conflicts = list(plan_spec.source_conflicts)
    should_pause = mode == "on" or (mode == "on_conflict" and bool(conflicts))
    now = utc_now()

    if should_pause:
        document = _review_payload(
            source_sha256=source.sha256,
            classification_sha256=classification_sha256,
            mode=mode,
            status="awaiting_review",
            source_conflicts=conflicts,
            plan_sha256=plan_sha256,
            requested_at=now,
            reviewed_at=None,
            summary_path=paths.relative(summary_path),
            approved_spec_path=None,
            approved_spec_sha256=None,
            edited=False,
        )
        _write_summary(summary_path, plan_spec, document, run_id=paths.run_id)
        write_json_atomic(review_path, document.model_dump(mode="json"))
        manifest.config["review_gate"] = True
        manifest.config["review_mode"] = mode
        store.set_stage(
            paths,
            manifest,
            "review",
            "awaiting_review",
            output=paths.relative(review_path),
        )
        log.info("Run %s paused for review; summary at %s", paths.run_id, summary_path)
        return ReviewResult(
            document=document,
            output_path=review_path,
            summary_path=summary_path,
            approved_spec_path=None,
            paused=True,
            resumed=False,
        )

    write_bytes_atomic(approved_path, plan_bytes)
    document = _review_payload(
        source_sha256=source.sha256,
        classification_sha256=classification_sha256,
        mode=mode,
        status="auto_approved",
        source_conflicts=conflicts,
        plan_sha256=plan_sha256,
        requested_at=now,
        reviewed_at=now,
        summary_path=None,
        approved_spec_path=paths.relative(approved_path),
        approved_spec_sha256=sha256_file(approved_path),
        edited=False,
    )
    write_json_atomic(review_path, document.model_dump(mode="json"))
    manifest.config["review_gate"] = mode != "off"
    manifest.config["review_mode"] = mode
    store.set_stage(paths, manifest, "review", "skipped", output=paths.relative(review_path))
    if conflicts:
        for conflict in conflicts:
            log.warning("Source conflict in run %s: %s", paths.run_id, conflict.issue)
    return ReviewResult(
        document=document,
        output_path=review_path,
        summary_path=None,
        approved_spec_path=approved_path,
        paused=False,
        resumed=False,
    )
