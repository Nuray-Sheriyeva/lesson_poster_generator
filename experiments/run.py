# ruff: noqa: E402
"""Run Langfuse dataset experiments over the fixture lessons.

Usage:
    python experiments/run.py --mode plan
    python experiments/run.py --mode image --item relationtype --sync-prompts
    python experiments/run.py --mode full --item relationtype
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from langgraph.graph import END, START, StateGraph

from genlesson import prompt_management, tracing
from genlesson import prompts as poster_prompts
from genlesson.dotenv import load_dotenv
from genlesson.graph import NODE_FUNCTIONS, RunState, build_graph, initial_state
from genlesson.models import (
    ClassificationDocument,
    PosterSpecDocument,
    VerificationDocument,
)
from genlesson.planner import prompts as planner_prompts
from genlesson.runs import RunStore
from genlesson.storage import read_model

DATASET_NAME = "genlesson-fixtures"
GOLDEN_FIXTURE = "relationtype.pdf"
GOLDEN = {
    "language": "kk",
    "details": 7,
    "file_slug": "biotic-interactions",
    "conflicts_present": True,
}
FROZEN_SPEC = ROOT / "fixtures" / "expected" / "relationtype_approved_spec.json"
FIXTURES = ROOT / "fixtures" / "input"


def _slug(value: str, limit: int = 60) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return text[:limit].rstrip("-") or "item"


def _fixture_files() -> list[Path]:
    return sorted(
        path
        for path in FIXTURES.iterdir()
        if path.is_file() and path.suffix.casefold() in {".pdf", ".docx"}
    )


def _dataset_items(client: Any, dataset_name: str) -> Any:
    try:
        client.get_dataset(dataset_name)
    except Exception:
        client.create_dataset(
            name=dataset_name,
            description="Five genlesson fixture lessons; relationtype has golden expectations.",
        )
    for path in _fixture_files():
        stem = path.stem
        expected = GOLDEN if path.name == GOLDEN_FIXTURE else None
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=f"genlesson-fixture-{_slug(stem)}",
            input={
                "filename": path.name,
                "stem": stem,
                "path": str(path),
            },
            expected_output=expected,
            metadata={"fixture": path.name, "golden": expected is not None},
        )
    return client.get_dataset(dataset_name)


def _plan_graph():
    builder = StateGraph(RunState)
    for name in ("ingest", "classify", "plan"):
        builder.add_node(name, NODE_FUNCTIONS[name])
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "classify")
    builder.add_edge("classify", "plan")
    builder.add_edge("plan", END)
    return builder.compile()


def _graph_flags(args: argparse.Namespace, *, review: str) -> dict[str, Any]:
    return {
        "formats": ["portrait"],
        "review": review,
        "planner_provider": args.planner_provider,
        "planner_model": args.planner_model,
        "image_provider": args.image_provider,
        "image_model": args.image_model,
        "similarity": args.similarity,
        "max_retries": args.max_retries,
        "out_dir": None,
    }


def _run_cli(args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "genlesson", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"genlesson {' '.join(args)} failed ({completed.returncode}): {detail[-1500:]}"
        )


def _read_plan(paths) -> PosterSpecDocument:
    return read_model(paths.stage_dir("plan") / "poster_spec.json", PosterSpecDocument)


def _read_classification(paths) -> ClassificationDocument:
    return read_model(
        paths.stage_dir("classify") / "classification.json",
        ClassificationDocument,
    )


def _read_verification(paths) -> VerificationDocument:
    return read_model(
        paths.stage_dir("verify") / "verification.json",
        VerificationDocument,
    )


def _output_from_plan(run_id: str, runs_dir: Path, mode: str) -> dict[str, Any]:
    store = RunStore(runs_dir)
    paths, manifest = store.load(run_id)
    classification = _read_classification(paths)
    plan = _read_plan(paths)
    return {
        "status": "ok",
        "mode": mode,
        "run_id": run_id,
        "run_dir": str(paths.run_dir),
        "trace_id": tracing.trace_id_for_run(run_id),
        "planner_model": classification.config.planner_model,
        "prompt_label": prompt_management.prompt_label(),
        "language": classification.poster_language,
        "details": len(plan.details),
        "file_slug": plan.file_slug,
        "conflicts_present": bool(plan.source_conflicts),
        "plan_status": manifest.stages["plan"].status,
    }


def _output_from_verification(run_id: str, runs_dir: Path, mode: str) -> dict[str, Any]:
    store = RunStore(runs_dir)
    paths, manifest = store.load(run_id)
    plan = _read_plan(paths)
    verification = _read_verification(paths)
    return {
        "status": "ok",
        "mode": mode,
        "run_id": run_id,
        "run_dir": str(paths.run_dir),
        "trace_id": tracing.trace_id_for_run(run_id),
        "planner_model": plan.config.planner_model,
        "prompt_label": prompt_management.prompt_label(),
        "language": plan.language,
        "details": len(plan.details),
        "file_slug": plan.file_slug,
        "conflicts_present": bool(plan.source_conflicts),
        "verify_status": [item.status for item in verification.formats],
        "best_scores": [item.best_score for item in verification.formats],
        "run_status": manifest.status,
    }


def _plan_item(item: Any, *, args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    data = item.input
    filename = data["filename"]
    runs_dir = run_root / "runs"
    run_id = f"plan-{_slug(Path(filename).stem)}"
    state = initial_state(
        run_id=run_id,
        runs_dir=runs_dir,
        source=data["path"],
        flags=_graph_flags(args, review="off"),
    )
    _plan_graph().invoke(state)
    return _output_from_plan(run_id, runs_dir, "plan")


def _image_item(item: Any, *, args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    data = item.input
    filename = data["filename"]
    if filename != GOLDEN_FIXTURE:
        return {
            "status": "skipped",
            "mode": "image",
            "reason": "image mode uses the frozen relationtype approved spec only",
            "filename": filename,
        }
    runs_dir = run_root / "runs"
    run_id = f"image-{_slug(Path(filename).stem)}"
    state = initial_state(
        run_id=run_id,
        runs_dir=runs_dir,
        source=data["path"],
        flags=_graph_flags(args, review="on"),
    )
    build_graph().invoke(state)

    _run_cli(
        [
            "review",
            run_id,
            "--runs-dir",
            str(runs_dir),
            "--resume",
            "--spec",
            str(FROZEN_SPEC),
            "--json",
        ]
    )
    prompt_args = [
        "prompts",
        run_id,
        "--runs-dir",
        str(runs_dir),
        "--formats",
        "portrait",
        "--json",
    ]
    if args.image_provider:
        prompt_args.extend(["--image-provider", args.image_provider])
    if args.image_model:
        prompt_args.extend(["--image-model", args.image_model])
    _run_cli(prompt_args)

    generate_args = ["generate", run_id, "--runs-dir", str(runs_dir), "--json"]
    if args.image_provider:
        generate_args.extend(["--image-provider", args.image_provider])
    _run_cli(generate_args)

    verify_args = ["verify", run_id, "--runs-dir", str(runs_dir), "--json"]
    if args.similarity is not None:
        verify_args.extend(["--similarity", str(args.similarity)])
    if args.max_retries is not None:
        verify_args.extend(["--max-retries", str(args.max_retries)])
    _run_cli(verify_args)
    return _output_from_verification(run_id, runs_dir, "image")


def _full_item(item: Any, *, args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    data = item.input
    filename = data["filename"]
    runs_dir = run_root / "runs"
    run_id = f"full-{_slug(Path(filename).stem)}"
    state = initial_state(
        run_id=run_id,
        runs_dir=runs_dir,
        source=data["path"],
        flags=_graph_flags(args, review="off"),
    )
    build_graph().invoke(state)
    return _output_from_verification(run_id, runs_dir, "full")


def _evaluate(*, input: Any, output: Any, expected_output: Any, metadata=None, **kwargs):
    if not expected_output or not isinstance(output, dict):
        return []
    from langfuse import Evaluation

    if output.get("status") != "ok":
        return [
            Evaluation(
                name="run_ok",
                value=0.0,
                comment=str(output.get("reason") or output.get("status")),
            )
        ]
    checks = {
        "language_match": output.get("language") == expected_output["language"],
        "details_match": output.get("details") == expected_output["details"],
        "slug_match": output.get("file_slug") == expected_output["file_slug"],
        "conflicts_present": bool(output.get("conflicts_present"))
        == bool(expected_output["conflicts_present"]),
    }
    fields = {
        "language_match": "language",
        "details_match": "details",
        "slug_match": "file_slug",
        "conflicts_present": "conflicts_present",
    }
    return [
        Evaluation(
            name=name,
            value=1.0 if passed else 0.0,
            comment=(
                "match"
                if passed
                else f"got {output.get(fields.get(name, name))!r}"
            ),
        )
        for name, passed in checks.items()
    ]


def _selected_items(items: list[Any], wanted: list[str] | None) -> list[Any]:
    if not wanted:
        return items
    wanted_names = {value.casefold() for value in wanted}
    return [
        item
        for item in items
        if str(item.input.get("filename", "")).casefold() in wanted_names
        or str(item.input.get("stem", "")).casefold() in wanted_names
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "image", "full"), default="plan")
    parser.add_argument(
        "--item",
        action="append",
        default=None,
        help="Fixture filename or stem; repeat to select multiple",
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "experiments" / "runs")
    parser.add_argument("--planner-provider", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--image-provider", default=None)
    parser.add_argument("--image-model", default=None)
    parser.add_argument("--prompt-label", default=None)
    parser.add_argument("--similarity", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sync-prompts", action="store_true")
    parser.add_argument(
        "--sync-prompts-only",
        action="store_true",
        help="Upload prompt versions and exit without building a dataset run",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv()
    if args.prompt_label:
        os.environ[prompt_management.PROMPT_LABEL_ENV] = args.prompt_label
    prompt_management.clear_prompt_cache()
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    model = args.planner_model or "default"
    label = args.prompt_label or prompt_management.prompt_label()
    run_name = args.run_name or _slug(f"{args.mode}-{model}-{label}-{stamp}")
    runs_root = (args.runs_root / run_name).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    tracing.init(tags=("experiment", args.mode), experiment=run_name)
    try:
        client = tracing.get_client()
        if client is None:
            print(
                "error: Langfuse is not enabled; set LANGFUSE_* keys to run experiments.",
                file=sys.stderr,
            )
            return 2
        if args.sync_prompts or args.sync_prompts_only:
            synced = {
                **planner_prompts.sync_prompts(labels=(label,)),
                **poster_prompts.sync_prompts(labels=(label,)),
            }
            if args.sync_prompts_only:
                if args.json:
                    print(json.dumps({"ok": True, "prompts": synced}, indent=2))
                else:
                    print(f"Synced {sum(synced.values())}/{len(synced)} prompts.")
                return 0 if all(synced.values()) else 1
            if not args.json:
                print(f"Synced {sum(synced.values())}/{len(synced)} prompts.")
        dataset = _dataset_items(client, args.dataset)
        items = _selected_items(list(dataset.items), args.item)
        if not items:
            print("error: no dataset items selected.", file=sys.stderr)
            return 2

        def task(*, item, **kwargs):
            if args.mode == "plan":
                return _plan_item(item, args=args, run_root=runs_root)
            if args.mode == "image":
                return _image_item(item, args=args, run_root=runs_root)
            return _full_item(item, args=args, run_root=runs_root)

        result = client.run_experiment(
            name=f"genlesson-{args.mode}",
            run_name=run_name,
            description=(
                "Langfuse dataset experiment over fixtures/input. "
                "Only relationtype has golden expectations."
            ),
            data=items,
            task=task,
            evaluators=[_evaluate],
            max_concurrency=max(1, args.concurrency),
            metadata={
                "mode": args.mode,
                "planner_model": str(args.planner_model or ""),
                "prompt_label": str(label),
            },
        )
        failed_items = max(0, len(items) - len(result.item_results))
        payload = {
            "ok": failed_items == 0,
            "mode": args.mode,
            "run_name": run_name,
            "dataset": args.dataset,
            "items": len(result.item_results),
            "failed_items": failed_items,
            "experiment_url": getattr(result, "experiment_url", None),
            "runs_root": str(runs_root),
            "results": [
                {
                    "trace_id": getattr(item, "trace_id", None),
                    "output": getattr(item, "output", None),
                    "evaluations": [
                        {
                            "name": evaluation.name,
                            "value": evaluation.value,
                            "comment": evaluation.comment,
                        }
                        for evaluation in getattr(item, "evaluations", [])
                    ],
                }
                for item in result.item_results
            ],
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Experiment: {run_name}")
            print(f"Items:      {payload['items']}")
            if payload["experiment_url"]:
                print(f"URL:        {payload['experiment_url']}")
            if payload["failed_items"]:
                print(f"Failed:     {payload['failed_items']} item(s)")
            print(f"Runs:       {runs_root}")
        return 0 if payload["ok"] else 1
    finally:
        tracing.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
