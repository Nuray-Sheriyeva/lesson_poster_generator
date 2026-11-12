"""Command-line interface for Stage 1."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__, tracing
from .app_config import load_app_config, set_active_config_path
from .classify import ClassifyResult, run_classify
from .config import (
    STAGES,
    GenerationOptions,
    IngestOptions,
    PlannerOptions,
    PlanOptions,
    PromptOptions,
    ReviewOptions,
    VerificationOptions,
    default_formats,
    default_image_model,
    default_image_provider,
    default_review_mode,
    default_review_on_conflict,
    manifest_config,
)
from .dotenv import load_dotenv
from .errors import GenLessonError, invalid_arguments, run_exists
from .generate import GenerateResult, run_generate
from .ingest import IngestResult, run_ingest
from .mascot.assets import load_mascot_assets
from .mascot.compose import compose_overlay, check_text_inside_bubble, render_preview
from .mascot.select import (
    PlacementFailure,
    normalize_requested_corner,
    resolve_placement,
)
from .mascot.zones import get_text_detector, score_layout
from .models import RunManifest, SourceInfo
from .plan import PlanResult, run_plan
from .prompts import PromptResult, run_prompts
from .review import ReviewResult, run_review
from .runs import RunPaths, RunStore, describe_source, validate_run_id
from .save import SaveResult, run_save
from .storage import read_model
from .verify import VerifyResult, run_verify

_JSON = "json"
_DEBUG = "debug"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genlesson",
        description="Turn a lesson file into a resumable poster-generation run.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="Config TOML path")
    parser.add_argument(
        "--tracing",
        dest="tracing",
        action="store_true",
        default=None,
        help="Force Langfuse tracing on (requires LANGFUSE_* keys)",
    )
    parser.add_argument(
        "--no-tracing",
        dest="tracing",
        action="store_false",
        default=None,
        help="Disable Langfuse tracing for this command",
    )
    parser.add_argument(
        "--trace-tag",
        dest="trace_tag",
        action="append",
        default=None,
        metavar="TAG",
        help="Tag Langfuse traces; repeat for multiple tags",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        metavar="NAME",
        help="Experiment name recorded on Langfuse traces",
    )
    parser.set_defaults(tracing=None)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser(
        "ingest",
        help="Ingest a PDF or DOCX (pipeline Stage 1)",
        description="Extract lesson text, render PDF pages, and transcribe scanned pages.",
    )
    ingest.add_argument("source", type=Path, help="Lesson file (.pdf or .docx)")
    ingest.add_argument("--run-id", help="Explicit run id; required for --resume")
    ingest.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ingest.add_argument("--resume", action="store_true", help="Continue an existing run")
    ingest.add_argument("--dpi", type=int, default=None, help="PDF render DPI (default: 180)")
    ingest.add_argument(
        "--vision",
        choices=("openai", "none"),
        default=None,
        help="Vision provider for pages without a text layer (default: openai)",
    )
    ingest.add_argument("--vision-model", default=None, help="Vision model name")
    ingest.add_argument("--vision-timeout", type=float, default=None)
    ingest.add_argument("--vision-retries", type=int, default=None)
    ingest.add_argument("--json", action="store_true", help="Machine-readable output")
    ingest.add_argument("--debug", action="store_true", help="Show tracebacks on failure")

    classify = sub.add_parser(
        "classify",
        help="Classify lesson content and decide the poster language (Stage 2)",
        description=(
            "Use Luna to separate lesson content from objectives, criteria, instructions, "
            "and exercise prompts, then decide the poster language from lesson content only."
        ),
    )
    classify.add_argument("run_id")
    classify.add_argument("--runs-dir", type=Path, default=Path("runs"))
    classify.add_argument("--resume", action="store_true", help="Reuse a completed classification")
    classify.add_argument("--planner-provider", default=None, help="Planner provider name")
    classify.add_argument("--planner-model", default=None, help="Planner model name")
    classify.add_argument("--planner-context", type=int, default=None)
    classify.add_argument("--planner-timeout", type=float, default=None)
    classify.add_argument("--planner-retries", type=int, default=None)
    classify.add_argument(
        "--max-page-images",
        type=int,
        default=None,
        help="Maximum page images sent as planner context (default: 6)",
    )
    classify.add_argument("--json", action="store_true", help="Machine-readable output")
    classify.add_argument("--debug", action="store_true", help="Show tracebacks on failure")

    plan = sub.add_parser(
        "plan",
        help="Plan the poster from classified content (Stage 3)",
        description=(
            "Use Luna to produce a strict poster spec. Long lessons are chunked, candidate "
            "ideas are extracted per chunk, and a final call merges them."
        ),
    )
    plan.add_argument("run_id")
    plan.add_argument("--runs-dir", type=Path, default=Path("runs"))
    plan.add_argument("--resume", action="store_true", help="Reuse a completed plan")
    plan.add_argument("--planner-provider", default=None, help="Planner provider name")
    plan.add_argument("--planner-model", default=None, help="Planner model name")
    plan.add_argument("--planner-context", type=int, default=None)
    plan.add_argument("--planner-timeout", type=float, default=None)
    plan.add_argument("--planner-retries", type=int, default=None)
    plan.add_argument("--max-page-images", type=int, default=None)
    plan.add_argument("--max-details", type=int, default=None)
    plan.add_argument("--context-chars", type=int, default=None)
    plan.add_argument("--chunk-chars", type=int, default=None)
    plan.add_argument("--json", action="store_true", help="Machine-readable output")
    plan.add_argument("--debug", action="store_true", help="Show tracebacks on failure")

    review = sub.add_parser(
        "review",
        help="Review the planner output (Stage 4)",
        description=(
            "Pass through by default, or pause the run for human review. An edited spec "
            "can be supplied with --spec and is revalidated before approval."
        ),
    )
    review.add_argument("run_id")
    review.add_argument("--runs-dir", type=Path, default=Path("runs"))
    review.add_argument("--resume", action="store_true", help="Approve an awaiting review")
    review.add_argument("--spec", type=Path, default=None, help="Edited poster spec to approve")
    review.add_argument("--review", choices=("off", "on"), default=None)
    review.add_argument(
        "--review-on-conflict",
        action="store_true",
        help="Pause only when the planner flagged source conflicts",
    )
    review.add_argument("--json", action="store_true", help="Machine-readable output")
    review.add_argument("--debug", action="store_true", help="Show tracebacks on failure")

    prompts = sub.add_parser(
        "prompts",
        help="Build image-model prompts (Stage 5)",
        description="Build one quoted, format-specific prompt from the approved poster spec.",
    )
    prompts.add_argument("run_id")
    prompts.add_argument("--runs-dir", type=Path, default=Path("runs"))
    prompts.add_argument("--resume", action="store_true", help="Reuse completed prompts")
    prompts.add_argument(
        "--formats",
        default=None,
        help="Comma-separated: portrait,landscape,square (default: manifest or portrait)",
    )
    prompts.add_argument("--image-provider", default=None, help="Image provider name")
    prompts.add_argument("--image-model", default=None, help="Image model name")
    prompts.add_argument(
        "--format",
        dest="format",
        action="append",
        choices=("portrait", "landscape", "square"),
        default=None,
        help="Poster format; repeat for multiple formats",
    )
    prompts.add_argument("--json", action="store_true", help="Machine-readable output")
    prompts.add_argument("--debug", action="store_true", help="Show tracebacks on failure")

    generate = sub.add_parser(
        "generate",
        help="Generate poster images per format (Stage 6)",
        description=(
            "Generate one image per prompt format. Failed formats are recorded individually "
            "and do not stop successful formats."
        ),
    )
    generate.add_argument("run_id")
    generate.add_argument("--runs-dir", type=Path, default=Path("runs"))
    generate.add_argument("--resume", action="store_true", help="Reuse successful images")
    generate.add_argument("--image-provider", default=None, help="Image provider name")
    generate.add_argument("--image-timeout", type=float, default=None)
    generate.add_argument("--image-retries", type=int, default=None)
    generate.add_argument("--json", action="store_true", help="Machine-readable output")
    generate.add_argument("--debug", action="store_true", help="Show tracebacks on failure")

    verify = sub.add_parser(
        "verify",
        help="Verify rendered text and retry mismatches (Stage 7)",
        description=(
            "Read each poster back with a vision model, compare its text with the spec, "
            "and retry mismatches with correction notes."
        ),
    )
    verify.add_argument("run_id")
    verify.add_argument("--runs-dir", type=Path, default=Path("runs"))
    verify.add_argument("--resume", action="store_true", help="Reuse completed verification")
    verify.add_argument("--verifier-model", default=None, help="Vision model for verification")
    verify.add_argument("--similarity", type=float, default=None)
    verify.add_argument("--max-retries", type=int, default=None)
    verify.add_argument("--vision-timeout", type=float, default=None)
    verify.add_argument("--vision-retries", type=int, default=None)
    verify.add_argument("--image-timeout", type=float, default=None)
    verify.add_argument("--image-retries", type=int, default=None)
    verify.add_argument("--json", action="store_true", help="Machine-readable output")
    verify.add_argument("--debug", action="store_true", help="Show tracebacks on failure")

    save = sub.add_parser(
        "save",
        help="Save final posters and sidecars (Stage 8)",
        description=(
            "Save each verified poster with an exclusive-create filename and a JSON sidecar."
        ),
    )
    save.add_argument("run_id")
    save.add_argument("--runs-dir", type=Path, default=Path("runs"))
    save.add_argument("--resume", action="store_true", help="Reuse already saved posters")
    save.add_argument("--out-dir", type=Path, default=None, help="Final poster directory")
    save.add_argument("--json", action="store_true", help="Machine-readable output")
    save.add_argument("--debug", action="store_true", help="Show tracebacks on failure")

    run = sub.add_parser(
        "run",
        help="Run the whole pipeline with LangGraph",
        description="Run ingest through save, stopping at the review gate when required.",
    )
    run.add_argument("source", nargs="?", type=Path, default=None, help="Lesson file")
    run.add_argument("--run-id", help="Run id; required with --resume")
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    run.add_argument("--resume", action="store_true", help="Resume an existing run")
    run.add_argument("--planner-provider", default=None)
    run.add_argument("--planner-model", default=None)
    run.add_argument("--image-provider", default=None)
    run.add_argument("--image-model", default=None)
    run.add_argument(
        "--format",
        dest="format",
        action="append",
        choices=("portrait", "landscape", "square"),
        default=None,
        help="Poster format; repeat for multiple formats",
    )
    run.add_argument("--formats", default=None, help="Comma-separated poster formats")
    run.add_argument(
        "--review",
        choices=("off", "on", "on-conflict"),
        default=None,
        help="Review gate mode",
    )
    run.add_argument(
        "--review-on-conflict",
        action="store_true",
        help="Pause only when the planner flagged source conflicts",
    )
    run.add_argument("--similarity", type=float, default=None)
    run.add_argument("--max-retries", type=int, default=None)
    run.add_argument("--out-dir", type=Path, default=None)
    run.add_argument("--dpi", type=int, default=None, help="PDF render DPI")
    run.add_argument(
        "--vision",
        choices=("openai", "none"),
        default=None,
        help="Vision provider for pages without a text layer",
    )
    run.add_argument("--json", action="store_true", help="Only print the final status")
    run.add_argument("--debug", action="store_true", help="Show tracebacks on failure")

    preview = sub.add_parser(
        "mascot-preview",
        help="Render the mascot overlay on a blank canvas (no image credits)",
        description=(
            "Preview the mascot, its question bubble, text fitting, and corner placement "
            "on a blank canvas. This command never calls an image provider."
        ),
    )
    preview.add_argument(
        "--size",
        default="1024x1536",
        metavar="WxH",
        help="Blank-canvas size in pixels (default: 1024x1536)",
    )
    preview.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Overlay on an existing poster instead of a blank canvas (no credits)",
    )
    preview.add_argument(
        "--corner",
        default="auto",
        metavar="CORNER",
        help=("auto, top-left, top-right, bottom-left, or bottom-right (default: auto)"),
    )
    preview.add_argument("--variant", default=None, help="Mascot PNG variant id")
    preview.add_argument("--question", required=True, help="Question in the speech bubble")
    preview.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/mascot-preview.png"),
        help="Output PNG path (default: outputs/mascot-preview.png)",
    )
    preview.add_argument(
        "--background",
        default="#FFFFFF",
        help="Blank canvas color (default: #FFFFFF)",
    )
    preview.add_argument(
        "--detector",
        choices=("auto", "rapidocr", "none"),
        default="auto",
        help="Text detector used for auto corner selection",
    )
    preview.add_argument("--json", action="store_true", help="Machine-readable output")
    preview.add_argument("--debug", action="store_true", help="Show tracebacks on failure")

    status = sub.add_parser("status", help="Show a run's stage status")
    status.add_argument("run_id")
    status.add_argument("--runs-dir", type=Path, default=Path("runs"))
    status.add_argument("--json", action="store_true", help="Machine-readable output")
    status.add_argument("--debug", action="store_true", help="Show tracebacks on failure")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    load_dotenv()
    args = parser.parse_args(argv)
    if args.config is not None:
        set_active_config_path(args.config)
    try:
        tracing.init(
            enabled=getattr(args, "tracing", None),
            tags=tuple(getattr(args, "trace_tag", None) or ()),
            experiment=getattr(args, "experiment", None),
        )
        return _dispatch(args)
    except GenLessonError as exc:
        _emit_error(args, exc)
        if getattr(args, _DEBUG, False):
            traceback.print_exc()
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - last-resort guard
        if getattr(args, _DEBUG, False):
            traceback.print_exc()
        else:
            print(f"error [internal_error]: {exc}", file=sys.stderr)
        return 1
    finally:
        tracing.shutdown()


def _dispatch(args: argparse.Namespace) -> int:
    """Run the command selected by argparse.

    Every pipeline stage runs inside a stage span so per-stage CLI commands and
    future graph nodes share the same tracing shape.  Ingest is special only
    when no run id exists yet: the command creates the run and opens its own
    span after the run id is known.
    """
    if args.command == "status":
        return _cmd_status(args)
    if args.command == "mascot-preview":
        return _cmd_mascot_preview(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "ingest":
        if getattr(args, "run_id", None):
            return _dispatch_stage(args, "ingest", _cmd_ingest)
        return _cmd_ingest(args)

    handlers = {
        "classify": ("classify", _cmd_classify),
        "plan": ("plan", _cmd_plan),
        "review": ("review", _cmd_review),
        "prompts": ("prompts", _cmd_prompts),
        "generate": ("generate", _cmd_generate),
        "verify": ("verify", _cmd_verify),
        "save": ("save", _cmd_save),
    }
    entry = handlers.get(args.command)
    if entry is None:
        raise invalid_arguments(f"unknown command: {args.command}")
    return _dispatch_stage(args, entry[0], entry[1])


def _dispatch_stage(
    args: argparse.Namespace,
    stage: str,
    handler: Any,
) -> int:
    run_id = getattr(args, "run_id", None)
    trace_id = tracing.trace_id_for_run(run_id) if run_id else None
    metadata: dict[str, Any] = {"stage": stage, "command": args.command}
    if run_id and tracing.is_enabled():
        try:
            _, manifest = RunStore(args.runs_dir).load(run_id)
            trace_meta = manifest.config.get("tracing") or {}
            trace_id = trace_meta.get("trace_id") or trace_id
            record = manifest.stages.get(stage)
            if record is not None:
                metadata["resumed"] = record.status == "complete"
        except Exception:  # noqa: S110 - tracing must not change command behavior
            # The command itself reports missing/invalid runs with the usual
            # error shape; tracing must not change that behavior.
            pass
    with tracing.stage_span(stage, run_id=run_id, trace_id=trace_id, metadata=metadata) as span:
        code = handler(args)
        span.update(output={"exit_code": code})
        return code


def _cmd_ingest(args: argparse.Namespace) -> int:
    options = _options_from_args(args)
    source = describe_source(args.source)
    store = RunStore(args.runs_dir)
    paths, manifest, created = _open_or_create_run(
        args, store=store, source=source, options=options
    )

    if manifest.source is not None and manifest.source.sha256 != source.sha256:
        raise GenLessonError(
            "input_changed",
            "The input file changed since this run was created; refusing to mix artifacts.",
            hint="Start a new run for the current file.",
        )
    if manifest.source is not None and manifest.source.path != source.path:
        # The same bytes at a new path are safe to resume; keep the current path.
        manifest.source = source
        store.save(paths, manifest)

    # Tracing identity is run metadata, not stage config: keep it outside the
    # per-stage public_config() compared on resume.
    _record_tracing(store, paths, manifest)

    stage = manifest.stages.get("ingest")
    if stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the ingest stage.")
    if stage.status != "complete":
        desired = options.public_config().model_dump(mode="json")
        stored = manifest.config.get("ingest")
        if stored is not None and stored != desired:
            raise GenLessonError(
                "config_changed",
                "Ingest options differ from the options recorded for this run.",
                hint="Resume with the original options, or start a new run.",
                details={"stored": stored, "requested": desired},
            )

    started = time.monotonic()
    result = _run_ingest_stage(manifest, paths, store, options, source, stage)
    elapsed = time.monotonic() - started
    manifest = _reload_manifest(paths)
    payload = _ingest_payload(paths, manifest, result, created=created, elapsed_s=elapsed)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_ingest(payload)
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    paths, manifest = store.load(args.run_id)
    options = _planner_options_from_args(args, manifest)
    started = time.monotonic()
    result = run_classify(paths, manifest, store, options)
    elapsed = time.monotonic() - started
    manifest = _reload_manifest(paths)
    payload = _classify_payload(paths, manifest, result, elapsed_s=elapsed)
    tracing.annotate(
        "classify",
        metadata={
            "language": payload["poster_language"],
            "confidence": payload["confidence"],
            "segments": payload["stats"]["segment_count"],
            "resumed": payload["resumed"],
        },
    )
    tracing.score("language_confidence", float(payload["confidence"]))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_classify(payload)
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    paths, manifest = store.load(args.run_id)
    options = _plan_options_from_args(args, manifest)
    started = time.monotonic()
    result = run_plan(paths, manifest, store, options)
    elapsed = time.monotonic() - started
    manifest = _reload_manifest(paths)
    payload = _plan_payload(paths, manifest, result, elapsed_s=elapsed)
    tracing.annotate(
        "plan",
        metadata={
            "details": len(payload["details"]),
            "chunks": payload["stats"]["chunks"],
            "conflicts": len(payload["source_conflicts"]),
            "resumed": payload["resumed"],
        },
    )
    tracing.score("details", float(len(payload["details"])))
    tracing.score("conflicts", float(len(payload["source_conflicts"])))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_plan(payload)
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    paths, manifest = store.load(args.run_id)
    options = _review_options_from_args(args, manifest)
    result = run_review(
        paths,
        manifest,
        store,
        options,
        resume=args.resume,
        spec_path=args.spec,
    )
    manifest = _reload_manifest(paths)
    payload = _review_payload(paths, manifest, result)
    tracing.annotate(
        "review",
        metadata={
            "status": payload["status"],
            "paused": payload["paused"],
            "edited": payload["edited"],
        },
    )
    tracing.score(
        "review_edited",
        1.0 if payload["edited"] else 0.0,
        data_type="BOOLEAN",
    )
    tracing.score("human_wait_s", _review_wait_seconds(result.document))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_review(payload)
    return 0


def _cmd_prompts(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    paths, manifest = store.load(args.run_id)
    options = _prompt_options_from_args(args, manifest)
    started = time.monotonic()
    result = run_prompts(paths, manifest, store, options)
    elapsed = time.monotonic() - started
    manifest = _reload_manifest(paths)
    payload = _prompts_payload(paths, manifest, result, elapsed_s=elapsed)
    tracing.annotate(
        "prompts",
        metadata={
            "formats": [item["format"] for item in payload["formats"]],
            "image_model": payload["image_model"],
            "resumed": payload["resumed"],
        },
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_prompts(payload)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    paths, manifest = store.load(args.run_id)
    options = _generation_options_from_args(args)
    started = time.monotonic()
    result = run_generate(paths, manifest, store, options)
    elapsed = time.monotonic() - started
    manifest = _reload_manifest(paths)
    payload = _generate_payload(paths, manifest, result, elapsed_s=elapsed)
    tracing.annotate(
        "generate",
        metadata={
            "generated": payload["generated"],
            "failed": payload["failed"],
            "formats": len(payload["formats"]),
            "resumed": payload["resumed"],
        },
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_generated(payload)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    paths, manifest = store.load(args.run_id)
    options = _verification_options_from_args(args)
    started = time.monotonic()
    result = run_verify(paths, manifest, store, options)
    elapsed = time.monotonic() - started
    manifest = _reload_manifest(paths)
    payload = _verify_payload(paths, manifest, result, elapsed_s=elapsed)
    tracing.annotate(
        "verify",
        metadata={
            **payload["stats"],
            "resumed": payload["resumed"],
        },
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_verified(payload)
    return 0


def _cmd_save(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    paths, manifest = store.load(args.run_id)
    result = run_save(paths, manifest, store, out_dir=args.out_dir)
    manifest = _reload_manifest(paths)
    payload = _save_payload(paths, manifest, result)
    tracing.annotate(
        "save",
        metadata={
            **payload["stats"],
            "resumed": payload["resumed"],
        },
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_saved(payload)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    paths, manifest = store.load(args.run_id)
    payload = {
        "ok": True,
        "run_id": manifest.run_id,
        "run_dir": str(paths.run_dir),
        "status": manifest.status,
        "source": manifest.source.model_dump(mode="json") if manifest.source else None,
        "stages": {
            name: record.model_dump(mode="json") for name, record in manifest.stages.items()
        },
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"Run:    {manifest.run_id}")
    print(f"Dir:    {paths.run_dir}")
    print(f"Status: {manifest.status}")
    if manifest.source:
        print(f"Source: {manifest.source.filename} ({manifest.source.media_type})")
    print()
    for spec in STAGES:
        record = manifest.stages.get(spec.name)
        if record is None:
            continue
        line = f"[{record.status:15}] {spec.name:9} {spec.title}"
        if record.output:
            line += f"  -> {record.output}"
        print(line)
        if record.error:
            print(f"           {record.error.get('code')}: {record.error.get('message')}")
    return 0


def _parse_canvas_size(value: str) -> tuple[int, int]:
    text = str(value).strip().lower().replace("×", "x")
    try:
        width_text, height_text = text.split("x", 1)
        width, height = int(width_text), int(height_text)
    except (ValueError, AttributeError) as exc:
        raise invalid_arguments(f"--size must look like WIDTHxHEIGHT, got {value!r}.") from exc
    if width < 64 or height < 64 or width > 10000 or height > 10000:
        raise invalid_arguments("--size dimensions must be between 64 and 10000 px.")
    return width, height


def _cmd_mascot_preview(args: argparse.Namespace) -> int:
    config = load_app_config().mascot
    if not config.enabled:
        raise GenLessonError(
            "mascot_disabled",
            "The mascot overlay is disabled in config.toml.",
            hint="Set [mascot] enabled = true to preview it.",
        )
    assets = load_mascot_assets(config.assets_dir)
    if args.variant and args.variant not in assets:
        raise invalid_arguments(
            f"Unknown mascot variant {args.variant!r}.",
            hint=f"Available variants: {', '.join(sorted(assets))}.",
        )
    corner = normalize_requested_corner(args.corner)
    detector = get_text_detector(args.detector)
    source_image = None
    if args.image is not None:
        if not args.image.is_file():
            raise GenLessonError(
                "mascot_preview_failed",
                f"Poster image does not exist: {args.image}",
            )
        try:
            from PIL import Image

            with Image.open(args.image) as opened:
                source_image = opened.convert("RGB")
        except OSError as exc:
            raise GenLessonError(
                "mascot_preview_failed", f"Could not read poster image: {exc}"
            ) from exc
        width, height = source_image.size
        if corner is None:
            from types import SimpleNamespace

            entry = SimpleNamespace(
                question=args.question,
                preferred_corner="bottom_right",
                corner="bottom_right",
                variant=args.variant,
            )
            placement = resolve_placement(
                source_image,
                args.question,
                config=config,
                entry=entry,
                detector=detector,
                assets=assets,
            )
            if isinstance(placement, PlacementFailure):
                raise GenLessonError(
                    "mascot_no_clear_corner",
                    "No clear corner was found on this poster.",
                    hint=(
                        "Pick a corner with --corner, or generate a poster that "
                        "reserves the mascot block."
                    ),
                    details=placement.as_dict(),
                )
            composition = compose_overlay(
                source_image,
                args.question,
                config,
                variant=placement.variant,
                corner=placement.corner,
                detector=detector,
                assets=assets,
            )
            preview_placement = placement.as_dict()
        else:
            composition = compose_overlay(
                source_image,
                args.question,
                config,
                variant=args.variant,
                corner=corner,
                detector=detector,
                assets=assets,
            )
            score = score_layout(source_image, composition.layout, detector)
            preview_placement = score.as_dict()
        preview_self_check = check_text_inside_bubble(composition.image, composition.layout, config)
    else:
        width, height = _parse_canvas_size(args.size)
        composition = render_preview(
            width,
            height,
            args.question,
            config,
            background=args.background,
            variant=args.variant,
            corner=corner,
            detector=detector,
            assets=assets,
        )
    output = args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        composition.image.save(output, format="PNG")
    except OSError as exc:
        raise GenLessonError(
            "mascot_preview_failed", f"Could not write preview PNG: {exc}"
        ) from exc
    payload = {
        "ok": True,
        "stage": "mascot-preview",
        "width": width,
        "height": height,
        "output": str(output),
        "source_image": str(args.image) if args.image is not None else None,
        "self_check": preview_self_check if args.image is not None else None,
        "placement": preview_placement if args.image is not None else None,
        **composition.metadata(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Canvas:   {width}x{height}")
        print(f"Variant:  {composition.variant}")
        print(f"Corner:   {composition.corner}")
        print(
            f"Bubble:   {composition.layout.bubble.width}x"
            f"{composition.layout.bubble.height} at "
            f"({composition.layout.bubble.x}, {composition.layout.bubble.y})"
        )
        print(f"Font:     {composition.layout.font_size}px")
        for line in composition.layout.lines:
            print(f"  {line}")
        print(f"Output:   {output}")
    return 0


def _run_ingest_stage(
    manifest: RunManifest,
    paths: RunPaths,
    store: RunStore,
    options: IngestOptions,
    source: SourceInfo,
    stage: Any,
) -> IngestResult:
    def _annotate(result: IngestResult) -> None:
        tracing.annotate(
            "ingest",
            metadata={
                "source": source.filename,
                "media_type": source.media_type,
                "size_bytes": source.size_bytes,
                "pages": len(result.document.pages),
                "chars": result.document.char_count,
                "vision_pages": result.document.stats.vision_pages,
                "resumed": result.resumed,
            },
        )

    # With an explicit --run-id, _dispatch already opened the stage span.
    if tracing.current_trace_id() is not None:
        result = run_ingest(paths, manifest, store, options)
        _annotate(result)
        return result

    trace_meta = manifest.config.get("tracing") or {}
    with tracing.stage_span(
        "ingest",
        run_id=manifest.run_id,
        trace_id=trace_meta.get("trace_id"),
        input={
            "source": source.filename,
            "media_type": source.media_type,
            "size_bytes": source.size_bytes,
        },
        metadata={"stage": "ingest", "resumed": stage.status == "complete"},
    ) as span:
        result = run_ingest(paths, manifest, store, options)
        span.update(
            output={
                "status": "complete",
                "resumed": result.resumed,
                "pages": len(result.document.pages),
                "chars": result.document.char_count,
            }
        )
        _annotate(result)
        return result


def _review_wait_seconds(document: Any) -> float:
    if not document.reviewed_at:
        return 0.0
    try:
        requested = datetime.fromisoformat(document.requested_at)
        reviewed = datetime.fromisoformat(document.reviewed_at)
    except ValueError:
        return 0.0
    return max(0.0, (reviewed - requested).total_seconds())


def _record_tracing(store: RunStore, paths: RunPaths, manifest: RunManifest) -> None:
    """Persist the deterministic trace/session ids for a run."""
    desired = {
        "trace_id": tracing.trace_id_for_run(manifest.run_id),
        "session_id": manifest.run_id,
    }
    if manifest.config.get("tracing") != desired:
        manifest.config["tracing"] = desired
        store.save(paths, manifest)


def _new_graph_run_id(source: Path, store: RunStore) -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    stem = re.sub(r"[^a-z0-9]+", "-", source.stem.lower()).strip("-")[:40] or "lesson"
    base = f"{stamp}-{stem}"
    candidate = base
    counter = 2
    while store.exists(candidate):
        candidate = f"{base}-{counter:02d}"
        counter += 1
    return validate_run_id(candidate)


def _graph_flags(args: argparse.Namespace) -> dict[str, Any]:
    if args.review is not None and args.review_on_conflict:
        raise invalid_arguments("Use either --review or --review-on-conflict, not both.")
    if args.review_on_conflict or args.review == "on-conflict":
        review = "on_conflict"
    else:
        review = args.review
    if args.format:
        formats = list(args.format)
    elif args.formats:
        formats = [part.strip().lower() for part in args.formats.split(",") if part.strip()]
    else:
        formats = []
    return {
        "planner_provider": args.planner_provider,
        "planner_model": args.planner_model,
        "image_provider": args.image_provider,
        "image_model": args.image_model,
        "formats": formats,
        "review": review,
        "similarity": args.similarity,
        "max_retries": args.max_retries,
        "out_dir": str(args.out_dir) if args.out_dir is not None else None,
        "dpi": args.dpi,
        "vision": args.vision,
        "json": args.json,
        "resume": args.resume,
    }


def _cmd_run(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir)
    if args.resume:
        if not args.run_id:
            raise invalid_arguments(
                "--resume needs an explicit --run-id.",
                hint="Find the id with the run directory name.",
            )
        run_id = validate_run_id(args.run_id)
        if not store.exists(run_id):
            raise GenLessonError(
                "run_not_found",
                f"Run does not exist: {store.runs_root / run_id}",
                hint="Omit --resume to create it.",
            )
        _, manifest = store.load(run_id)
        if manifest.source is None:
            raise GenLessonError("manifest_invalid", "Run manifest is missing source metadata.")
        source = Path(manifest.source.path)
    else:
        if args.source is None:
            raise invalid_arguments(
                "A lesson file is required unless --resume is used.",
                hint="Pass a PDF or DOCX path.",
            )
        source = args.source
        if args.run_id:
            run_id = validate_run_id(args.run_id)
            if store.exists(run_id):
                raise run_exists(str(store.runs_root / run_id))
        else:
            run_id = _new_graph_run_id(source, store)

    try:
        from .graph import build_graph, initial_state
    except ImportError as exc:
        raise GenLessonError(
            "graph_unavailable",
            "LangGraph is not installed, so the run command is unavailable.",
            hint="Install it with: uv pip install 'genlesson[graph]'",
        ) from exc

    flags = _graph_flags(args)
    state = initial_state(
        run_id=run_id,
        runs_dir=args.runs_dir,
        source=source,
        flags=flags,
    )
    trace_id = tracing.trace_id_for_run(run_id)
    with tracing.stage_span(
        f"run:{run_id}",
        run_id=run_id,
        trace_id=trace_id,
        input={"source": str(source), "resume": args.resume},
        metadata={"stage": "run", "resume": args.resume},
    ) as span:
        graph = build_graph()
        stages_run: list[str] = []
        buffer = io.StringIO() if args.json else None
        redirect = (
            contextlib.redirect_stdout(buffer) if buffer is not None else contextlib.nullcontext()
        )
        with redirect:
            for update in graph.stream(state, stream_mode="updates"):
                for node, payload in update.items():
                    if isinstance(payload, dict) and "stages_run" in payload:
                        stages_run = list(payload["stages_run"])
                    if not args.json:
                        print(f"[{node}]")
        _, manifest = store.load(run_id)
        span.update(
            output={
                "status": manifest.status,
                "stages_run": stages_run,
                "resume": args.resume,
            }
        )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "run_id": run_id,
                    "run_dir": str(store.runs_root / run_id),
                    "status": manifest.status,
                    "paused": manifest.status == "awaiting_review",
                    "resumed": args.resume,
                    "stages_run": stages_run,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Run:    {run_id}")
        print(f"Status: {manifest.status}")
        if manifest.status == "awaiting_review":
            print(f"Next:   approve with `genlesson review {run_id} --resume`")
    return 0


def _open_or_create_run(
    args: argparse.Namespace,
    *,
    store: RunStore,
    source: SourceInfo,
    options: IngestOptions,
) -> tuple[RunPaths, RunManifest, bool]:
    config = manifest_config(options)
    if args.run_id:
        run_id = validate_run_id(args.run_id)
        if store.exists(run_id):
            if not args.resume:
                raise run_exists(str(store.runs_root / run_id))
            paths, manifest = store.load(run_id)
            return paths, manifest, False
        if args.resume:
            raise GenLessonError(
                "run_not_found",
                f"Run does not exist: {store.runs_root / run_id}",
                hint="Omit --resume to create it.",
            )
        paths, manifest = store.create(source, config, run_id=run_id)
        return paths, manifest, True

    if args.resume:
        raise invalid_arguments(
            "--resume needs an explicit --run-id.",
            hint="Find the id with the run directory name.",
        )
    paths, manifest = store.create(source, config)
    return paths, manifest, True


def _reload_manifest(paths: RunPaths) -> RunManifest:
    return read_model(paths.manifest_path, RunManifest)


def _apply_manifest_planner(
    overrides: dict[str, object],
    args: argparse.Namespace,
    manifest: RunManifest | None,
) -> None:
    if manifest is None:
        return
    configured = manifest.config.get("planner") or {}
    provider = (
        args.planner_provider
        or os.environ.get("POSTER_PLANNER_PROVIDER")
        or configured.get("provider")
    )
    if args.planner_provider is None and not os.environ.get("POSTER_PLANNER_PROVIDER"):
        if configured.get("provider"):
            overrides["provider"] = configured["provider"]
    if (
        args.planner_model is None
        and not os.environ.get("POSTER_PLANNER_MODEL")
        and not os.environ.get(f"POSTER_{str(provider).upper()}_MODEL")
        and provider == configured.get("provider")
        and configured.get("model")
    ):
        overrides["model"] = configured["model"]
    if (
        args.planner_context is None
        and not os.environ.get("POSTER_PLANNER_MAX_CONTEXT")
        and provider == configured.get("provider")
        and configured.get("max_context")
    ):
        overrides["max_context"] = configured["max_context"]


def _planner_options_from_args(
    args: argparse.Namespace, manifest: RunManifest | None = None
) -> PlannerOptions:
    overrides: dict[str, object] = {}
    if args.planner_provider is not None:
        overrides["provider"] = args.planner_provider
    if args.planner_model is not None:
        overrides["model"] = args.planner_model
    if args.planner_context is not None:
        overrides["max_context"] = args.planner_context
    _apply_manifest_planner(overrides, args, manifest)
    if args.planner_timeout is not None:
        overrides["timeout_s"] = args.planner_timeout
    if args.planner_retries is not None:
        overrides["max_retries"] = args.planner_retries
    if args.max_page_images is not None:
        overrides["max_page_images"] = args.max_page_images
    elif manifest is not None and not os.environ.get("GENLESSON_MAX_PAGE_IMAGES"):
        configured_images = (manifest.config.get("classify") or {}).get("max_page_images")
        if isinstance(configured_images, int):
            overrides["max_page_images"] = configured_images
    return PlannerOptions.from_env(**overrides)


def _plan_options_from_args(
    args: argparse.Namespace, manifest: RunManifest | None = None
) -> PlanOptions:
    overrides: dict[str, object] = {}
    if args.planner_provider is not None:
        overrides["provider"] = args.planner_provider
    if args.planner_model is not None:
        overrides["model"] = args.planner_model
    if args.planner_context is not None:
        overrides["max_context"] = args.planner_context
    _apply_manifest_planner(overrides, args, manifest)
    if args.planner_timeout is not None:
        overrides["timeout_s"] = args.planner_timeout
    if args.planner_retries is not None:
        overrides["max_retries"] = args.planner_retries
    if args.max_page_images is not None:
        overrides["max_page_images"] = args.max_page_images
    elif manifest is not None and not os.environ.get("GENLESSON_MAX_PAGE_IMAGES"):
        configured_images = (manifest.config.get("plan") or {}).get("max_page_images")
        if not isinstance(configured_images, int):
            configured_images = (manifest.config.get("classify") or {}).get("max_page_images")
        if isinstance(configured_images, int):
            overrides["max_page_images"] = configured_images
    if args.max_details is not None:
        overrides["max_details"] = args.max_details
    elif manifest is not None and not os.environ.get("GENLESSON_MAX_DETAILS"):
        configured = (manifest.config.get("plan") or {}).get("max_details")
        if isinstance(configured, int):
            overrides["max_details"] = configured
    if args.context_chars is not None:
        overrides["context_char_limit"] = args.context_chars
    elif manifest is not None and not os.environ.get("GENLESSON_CONTEXT_CHARS"):
        configured = (manifest.config.get("plan") or {}).get("context_char_limit")
        if isinstance(configured, int):
            overrides["context_char_limit"] = configured
    if args.chunk_chars is not None:
        overrides["chunk_char_limit"] = args.chunk_chars
    elif manifest is not None and not os.environ.get("GENLESSON_CHUNK_CHARS"):
        configured = (manifest.config.get("plan") or {}).get("chunk_char_limit")
        if isinstance(configured, int):
            overrides["chunk_char_limit"] = configured
    return PlanOptions.from_env(**overrides)


def _review_options_from_args(
    args: argparse.Namespace, manifest: RunManifest | None = None
) -> ReviewOptions:
    if args.review is not None and args.review_on_conflict:
        raise invalid_arguments("Use either --review or --review-on-conflict, not both.")
    if args.review_on_conflict:
        mode = "on_conflict"
    elif args.review is not None:
        mode = args.review
    elif os.environ.get("POSTER_REVIEW"):
        mode = default_review_mode()
    elif default_review_on_conflict():
        mode = "on_conflict"
    elif manifest is not None and manifest.config.get("review_mode"):
        mode = manifest.config["review_mode"]
    else:
        mode = default_review_mode()
    options = ReviewOptions(mode=mode)
    options.validate()
    return options


def _prompt_options_from_args(
    args: argparse.Namespace, manifest: RunManifest | None = None
) -> PromptOptions:
    if args.format:
        formats = tuple(args.format)
    elif args.formats is not None:
        formats = tuple(part.strip().lower() for part in args.formats.split(",") if part.strip())
    elif os.environ.get("GENLESSON_FORMATS") or os.environ.get("POSTER_FORMATS"):
        formats = default_formats()
    else:
        configured = (manifest.config.get("formats") or []) if manifest else []
        formats = tuple(configured) if configured else default_formats()
    image_model = args.image_model
    if image_model is None:
        image_model = os.environ.get("GENLESSON_IMAGE_MODEL") or os.environ.get(
            "POSTER_IMAGE_MODEL"
        )
    if image_model is None and manifest is not None:
        image_model = (manifest.config.get("image_model") or {}).get("model")
    if image_model is None:
        image_model = default_image_model()
    provider = args.image_provider
    if provider is None:
        provider = os.environ.get("POSTER_IMAGE_PROVIDER")
    if provider is None:
        provider = default_image_provider()
    options = PromptOptions.from_env(  # type: ignore[arg-type]
        formats=formats,
        image_model=image_model,
        provider=provider,
    )
    return options


def _generation_options_from_args(args: argparse.Namespace) -> GenerationOptions:
    overrides: dict[str, object] = {}
    if args.image_provider is not None:
        overrides["provider"] = args.image_provider
    if args.image_timeout is not None:
        overrides["timeout_s"] = args.image_timeout
    if args.image_retries is not None:
        overrides["max_retries"] = args.image_retries
    return GenerationOptions.from_env(**overrides)


def _verification_options_from_args(args: argparse.Namespace) -> VerificationOptions:
    overrides: dict[str, object] = {}
    if args.verifier_model is not None:
        overrides["verifier_model"] = args.verifier_model
    if args.similarity is not None:
        overrides["similarity_threshold"] = args.similarity
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries
    if args.vision_timeout is not None:
        overrides["vision_timeout_s"] = args.vision_timeout
    if args.vision_retries is not None:
        overrides["vision_max_retries"] = args.vision_retries
    if args.image_timeout is not None:
        overrides["image_timeout_s"] = args.image_timeout
    if args.image_retries is not None:
        overrides["image_max_retries"] = args.image_retries
    return VerificationOptions.from_env(**overrides)


def _options_from_args(args: argparse.Namespace) -> IngestOptions:
    overrides: dict[str, object] = {}
    if args.dpi is not None:
        overrides["render_dpi"] = args.dpi
    if args.vision is not None:
        overrides["vision_mode"] = args.vision
    if args.vision_model is not None:
        overrides["vision_model"] = args.vision_model
    if args.vision_timeout is not None:
        overrides["vision_timeout_s"] = args.vision_timeout
    if args.vision_retries is not None:
        overrides["vision_max_retries"] = args.vision_retries
    return IngestOptions.from_env(**overrides)


def _ingest_payload(
    paths: RunPaths,
    manifest: RunManifest,
    result: IngestResult,
    *,
    created: bool,
    elapsed_s: float,
) -> dict:
    document = result.document
    return {
        "ok": True,
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_status": manifest.status,
        "stage": "ingest",
        "status": "complete",
        "created": created,
        "resumed": result.resumed,
        "output": str(result.output_path),
        "elapsed_s": round(elapsed_s, 3),
        "source": document.source.model_dump(mode="json"),
        "title": document.title,
        "pages": len(document.pages),
        "stats": document.stats.model_dump(mode="json"),
        "warnings": document.warnings,
    }


def _classify_payload(
    paths: RunPaths,
    manifest: RunManifest,
    result: ClassifyResult,
    *,
    elapsed_s: float,
) -> dict:
    document = result.document
    return {
        "ok": True,
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_status": manifest.status,
        "stage": "classify",
        "status": "complete",
        "resumed": result.resumed,
        "output": str(result.output_path),
        "elapsed_s": round(elapsed_s, 3),
        "poster_language": document.poster_language,
        "confidence": document.confidence,
        "evidence": document.evidence,
        "language_source": document.language_source,
        "dominant_topic": document.dominant_topic,
        "topics": [topic.model_dump(mode="json") for topic in document.topics],
        "stats": document.stats.model_dump(mode="json"),
        "warnings": document.warnings,
    }


def _plan_payload(
    paths: RunPaths,
    manifest: RunManifest,
    result: PlanResult,
    *,
    elapsed_s: float,
) -> dict:
    document = result.document
    return {
        "ok": True,
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_status": manifest.status,
        "stage": "plan",
        "status": "complete",
        "resumed": result.resumed,
        "output": str(result.output_path),
        "elapsed_s": round(elapsed_s, 3),
        "language": document.language,
        "age_band": document.age_band,
        "central_idea": document.central_idea.model_dump(mode="json"),
        "details": [detail.model_dump(mode="json") for detail in document.details],
        "composition": document.composition.model_dump(mode="json"),
        "style": document.style.model_dump(mode="json"),
        "strings_to_render": document.strings_to_render,
        "source_conflicts": [
            conflict.model_dump(mode="json") for conflict in document.source_conflicts
        ],
        "file_slug": document.file_slug,
        "stats": document.stats.model_dump(mode="json"),
        "warnings": document.warnings,
    }


def _review_payload(paths: RunPaths, manifest: RunManifest, result: ReviewResult) -> dict:
    document = result.document
    return {
        "ok": True,
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_status": manifest.status,
        "stage": "review",
        "status": document.status,
        "mode": document.mode,
        "paused": result.paused,
        "resumed": result.resumed,
        "output": str(result.output_path),
        "summary": str(result.summary_path) if result.summary_path else None,
        "approved_spec": (str(result.approved_spec_path) if result.approved_spec_path else None),
        "edited": document.edited,
        "source_conflicts": [
            conflict.model_dump(mode="json") for conflict in document.source_conflicts
        ],
    }


def _print_review(payload: dict) -> None:
    print(f"Run:    {payload['run_id']}")
    print(f"Status: {payload['status']} (mode: {payload['mode']})")
    if payload["paused"]:
        print(f"Summary: {payload['summary']}")
        print("Review:  awaiting review; the run is stopped")
        print(
            f"Next:    edit stage-03-plan/poster_spec.json, then run "
            f"`genlesson review {payload['run_id']} --resume`"
        )
    else:
        if payload["edited"]:
            print("Spec:    edited and revalidated")
        print(f"Spec:    {payload['approved_spec']}")
        print(f"Review:  {payload['status']}")
        print("Next:    build prompts with `genlesson prompts`")
    if payload["source_conflicts"]:
        print(f"Conflicts: {len(payload['source_conflicts'])}")
        for conflict in payload["source_conflicts"]:
            print(f"  - {conflict['issue']} -> {conflict['resolution']}")
    print(f"Report:  {payload['output']}")


def _prompts_payload(
    paths: RunPaths,
    manifest: RunManifest,
    result: PromptResult,
    *,
    elapsed_s: float,
) -> dict:
    document = result.document
    return {
        "ok": True,
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_status": manifest.status,
        "stage": "prompts",
        "status": "complete",
        "resumed": result.resumed,
        "output": str(result.output_path),
        "elapsed_s": round(elapsed_s, 3),
        "image_provider": document.config.image_provider,
        "image_model": document.image_model,
        "formats": [item.model_dump(mode="json") for item in document.formats],
        "prompt_files": [str(path) for path in result.prompt_paths],
        "mascot_layout": (str(result.mascot_layout_path) if result.mascot_layout_path else None),
    }


def _print_prompts(payload: dict) -> None:
    print(f"Run:       {payload['run_id']}")
    print(f"Image:     {payload['image_model']}")
    for item in payload["formats"]:
        print(f"Format:    {item['format']} {item['width']}x{item['height']} -> {item['filename']}")
    if payload["resumed"]:
        print("Prompts:  already complete; reused existing prompts")
    else:
        print(f"Prompts:  {len(payload['formats'])} built in {payload['elapsed_s']:.2f}s")
    print(f"Output:   {payload['output']}")
    for path in payload["prompt_files"]:
        print(f"Prompt:   {path}")
    if payload.get("mascot_layout"):
        print(f"Mascot:   {payload['mascot_layout']}")
    print("Next:     generate images")


def _generate_payload(
    paths: RunPaths,
    manifest: RunManifest,
    result: GenerateResult,
    *,
    elapsed_s: float,
) -> dict:
    document = result.document
    return {
        "ok": True,
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_status": manifest.status,
        "stage": "generate",
        "status": "complete",
        "resumed": result.resumed,
        "output": str(result.output_path),
        "elapsed_s": round(elapsed_s, 3),
        "image_provider": document.config.provider,
        "image_model": document.image_model,
        "generated": result.generated,
        "failed": result.failed,
        "formats": [item.model_dump(mode="json") for item in document.formats],
        "warnings": document.warnings,
    }


def _print_generated(payload: dict) -> None:
    print(f"Run:       {payload['run_id']}")
    print(f"Image:     {payload['image_model']}")
    print(
        f"Generated: {payload['generated']} of {len(payload['formats'])} "
        f"({payload['failed']} failed)"
    )
    for item in payload["formats"]:
        if item["status"] == "generated":
            print(f"  {item['format']:9} OK      -> {item['image_path']}")
        else:
            error = item.get("error") or {}
            print(
                f"  {item['format']:9} FAILED  {error.get('code', 'unknown')}: "
                f"{error.get('message', '')}"
            )
    if payload["resumed"]:
        print("Resume:    reused previously generated formats")
    for warning in payload["warnings"]:
        print(f"Warning:   {warning}")
    print(f"Report:    {payload['output']}")
    print("Next:      verify each generated poster (not implemented yet)")


def _verify_payload(
    paths: RunPaths,
    manifest: RunManifest,
    result: VerifyResult,
    *,
    elapsed_s: float,
) -> dict:
    document = result.document
    return {
        "ok": True,
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_status": manifest.status,
        "stage": "verify",
        "status": "complete",
        "resumed": result.resumed,
        "output": str(result.output_path),
        "elapsed_s": round(elapsed_s, 3),
        "formats": [item.model_dump(mode="json") for item in document.formats],
        "stats": document.stats.model_dump(mode="json"),
        "config": document.config.model_dump(mode="json"),
        "warnings": document.warnings,
    }


def _print_verified(payload: dict) -> None:
    print(f"Run:          {payload['run_id']}")
    stats = payload["stats"]
    print(
        f"Verification: {stats['passed']} passed, {stats['needs_attention']} needs attention, "
        f"{stats['generation_failed']} generation failed"
    )
    for item in payload["formats"]:
        status = item["status"].upper().replace("_", " ")
        print(
            f"  {item['format']:9} {status:17} "
            f"best attempt {item['best_attempt']} score {item['best_score']}"
        )
        if item.get("best_image_path"):
            print(f"             best -> {item['best_image_path']}")
    if payload["resumed"]:
        print("Resume:       reused completed verification")
    for warning in payload["warnings"]:
        print(f"Warning:      {warning}")
    print(f"Report:       {payload['output']}")
    print("Next:         save posters (not implemented yet)")


def _save_payload(paths: RunPaths, manifest: RunManifest, result: SaveResult) -> dict:
    document = result.document
    return {
        "ok": True,
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "run_status": manifest.status,
        "stage": "save",
        "status": "complete",
        "resumed": result.resumed,
        "output": str(result.output_path),
        "out_dir": document.out_dir,
        "files": [item.model_dump(mode="json") for item in document.files],
        "skipped": [item.model_dump(mode="json") for item in document.skipped],
        "stats": document.stats.model_dump(mode="json"),
        "warnings": document.warnings,
    }


def _print_saved(payload: dict) -> None:
    stats = payload["stats"]
    print(f"Run:     {payload['run_id']}")
    print(f"Out dir: {payload['out_dir']}")
    print(
        f"Saved:   {stats['saved']} poster(s), {stats['needs_attention']} needs attention, "
        f"{stats['skipped']} skipped"
    )
    for item in payload["files"]:
        print(
            f"  {item['filename']}  ({item['width']}x{item['height']}, "
            f"{item['verification_status']})"
        )
        print(f"    sidecar: {item['sidecar_filename']}")
    for skipped in payload["skipped"]:
        print(f"  {skipped['format']}: skipped ({skipped['reason']})")
    if payload["resumed"]:
        print("Resume:  reused already saved posters")
    for warning in payload["warnings"]:
        print(f"Warning: {warning}")
    print(f"Report:  {payload['output']}")
    print("Done.")


def _print_plan(payload: dict) -> None:
    print(f"Run:       {payload['run_id']}")
    print(f"Run dir:   {payload['run_dir']}")
    print(f"Language:  {payload['language']} ({payload['age_band']})")
    print(
        f"Central:   {payload['central_idea']['title']} — {payload['central_idea']['definition']}"
    )
    print(f"Details:   {len(payload['details'])}")
    print(f"Composition: {payload['composition']['concept']}")
    print(f"Strings:   {len(payload['strings_to_render'])}")
    if payload["source_conflicts"]:
        print(f"Conflicts: {len(payload['source_conflicts'])}")
    if payload["resumed"]:
        print("Plan:      already complete; reused existing artifacts")
    else:
        print(f"Plan:      complete in {payload['elapsed_s']:.2f}s")
    for warning in payload["warnings"]:
        print(f"Warning:   {warning}")
    print(f"Output:    {payload['output']}")
    print("Next:      review gate and prompt building (not implemented yet)")


def _print_classify(payload: dict) -> None:
    print(f"Run:      {payload['run_id']}")
    print(f"Run dir:  {payload['run_dir']}")
    print(
        f"Language: {payload['poster_language']} "
        f"(confidence {payload['confidence']:.2f}, {payload['language_source']})"
    )
    if payload["evidence"]:
        print(f"Evidence: {payload['evidence']}")
    if payload["dominant_topic"]:
        print(f"Topic:    {payload['dominant_topic']}")
    stats = payload["stats"]
    print(
        f"Segments: {stats['segment_count']} "
        f"({stats['content_segments']} content, {stats['non_content_segments']} non-content)"
    )
    if payload["resumed"]:
        print("Classify: already complete; reused existing artifacts")
    else:
        print(f"Classify: complete in {payload['elapsed_s']:.2f}s")
    for warning in payload["warnings"]:
        print(f"Warning:  {warning}")
    print(f"Output:   {payload['output']}")
    print("Next:     plan the poster (not implemented yet)")


def _print_ingest(payload: dict) -> None:
    stats = payload["stats"]
    source = payload["source"]
    print(f"Run:      {payload['run_id']}")
    print(f"Run dir:  {payload['run_dir']}")
    print(
        f"Source:   {source['filename']} ({source['media_type']}, "
        f"{_human_size(source['size_bytes'])})"
    )
    if source["media_type"] == "pdf":
        print(
            f"Pages:    {stats['page_count']} rendered; "
            f"{stats['text_layer_pages']} text-layer, {stats['vision_pages']} vision, "
            f"{stats['blank_pages']} blank"
        )
    else:
        print("Segment:  1 text-layer")
    print(f"Text:     {stats['char_count']:,} characters")
    if payload["resumed"]:
        print("Ingest:   already complete; reused existing artifacts")
    else:
        print(f"Ingest:   complete in {payload['elapsed_s']:.2f}s")
    for warning in payload["warnings"]:
        print(f"Warning:  {warning}")
    print(f"Output:   {payload['output']}")
    print("Next:     classify and detect language (not implemented yet)")


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _emit_error(args: argparse.Namespace, exc: GenLessonError) -> None:
    if getattr(args, _JSON, False):
        print(json.dumps({"ok": False, "error": exc.to_dict()}, ensure_ascii=False, indent=2))
        return
    print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
    if exc.hint:
        print(f"hint: {exc.hint}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
