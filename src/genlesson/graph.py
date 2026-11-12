"""Optional LangGraph orchestration for the eight-stage pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .errors import GenLessonError
from .runs import RunStore

NODES = (
    "ingest",
    "classify",
    "plan",
    "review",
    "prompts",
    "generate",
    "verify",
    "save",
)

EDGES = (
    ("START", "ingest"),
    ("ingest", "classify"),
    ("classify", "plan"),
    ("plan", "review"),
    ("review", "awaiting_review"),
    ("review", "prompts"),
    ("prompts", "generate"),
    ("generate", "verify"),
    ("verify", "save"),
    ("save", "END"),
)


class RunState(TypedDict, total=False):
    run_id: str
    runs_dir: str
    source: str | None
    flags: dict[str, Any]
    stages_run: list[str]


State = RunState


def initial_state(
    *,
    run_id: str,
    runs_dir: Path | str,
    source: Path | str | None,
    flags: dict[str, Any] | None = None,
    stages_run: list[str] | tuple[str, ...] = (),
) -> RunState:
    return {
        "run_id": run_id,
        "runs_dir": str(runs_dir),
        "source": str(source) if source is not None else None,
        "flags": dict(flags or {}),
        "stages_run": list(stages_run),
    }


def _append_flag(argv: list[str], flag: str, value: Any) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _stage_argv(stage: str, state: RunState) -> list[str]:
    run_id = state["run_id"]
    runs_dir = state["runs_dir"]
    flags = state.get("flags") or {}
    argv: list[str] = []

    if stage == "ingest":
        if not state.get("source"):
            raise GenLessonError(
                "graph_source_missing",
                "The run has no source file; pass one or resume an existing run.",
            )
        argv = ["ingest", str(state["source"]), "--run-id", run_id, "--runs-dir", runs_dir]
        if RunStore(Path(runs_dir)).exists(run_id):
            argv.append("--resume")
        _append_flag(argv, "--dpi", flags.get("dpi"))
        _append_flag(argv, "--vision", flags.get("vision"))
        return argv

    argv = [stage, run_id, "--runs-dir", runs_dir]
    if stage in {"classify", "plan"}:
        _append_flag(argv, "--planner-provider", flags.get("planner_provider"))
        _append_flag(argv, "--planner-model", flags.get("planner_model"))
    elif stage == "review":
        mode = flags.get("review")
        if mode == "on_conflict":
            argv.append("--review-on-conflict")
        elif mode:
            argv.extend(["--review", str(mode)])
    elif stage == "prompts":
        _append_flag(argv, "--image-provider", flags.get("image_provider"))
        _append_flag(argv, "--image-model", flags.get("image_model"))
        formats = flags.get("formats") or []
        if formats:
            argv.extend(["--formats", ",".join(str(item) for item in formats)])
    elif stage == "generate":
        _append_flag(argv, "--image-provider", flags.get("image_provider"))
    elif stage == "verify":
        _append_flag(argv, "--similarity", flags.get("similarity"))
        _append_flag(argv, "--max-retries", flags.get("max_retries"))
    elif stage == "save":
        _append_flag(argv, "--out-dir", flags.get("out_dir"))
    return argv


def _run_stage(stage: str, state: RunState) -> dict[str, list[str]]:
    from .cli import _dispatch, build_parser

    args = build_parser().parse_args(_stage_argv(stage, state))
    code = _dispatch(args)
    if code != 0:
        raise GenLessonError(
            "graph_stage_failed",
            f"Stage {stage!r} exited with status {code}.",
        )
    stages = list(state.get("stages_run") or [])
    if stage not in stages:
        stages.append(stage)
    return {"stages_run": stages}


def ingest_node(state: RunState) -> dict[str, list[str]]:
    return _run_stage("ingest", state)


def classify_node(state: RunState) -> dict[str, list[str]]:
    return _run_stage("classify", state)


def plan_node(state: RunState) -> dict[str, list[str]]:
    return _run_stage("plan", state)


def review_node(state: RunState) -> dict[str, list[str]]:
    return _run_stage("review", state)


def prompts_node(state: RunState) -> dict[str, list[str]]:
    return _run_stage("prompts", state)


def generate_node(state: RunState) -> dict[str, list[str]]:
    return _run_stage("generate", state)


def verify_node(state: RunState) -> dict[str, list[str]]:
    return _run_stage("verify", state)


def save_node(state: RunState) -> dict[str, list[str]]:
    return _run_stage("save", state)


NODE_FUNCTIONS = {
    "ingest": ingest_node,
    "classify": classify_node,
    "plan": plan_node,
    "review": review_node,
    "prompts": prompts_node,
    "generate": generate_node,
    "verify": verify_node,
    "save": save_node,
}


def route_after_review(state: RunState) -> str:
    _, manifest = RunStore(Path(state["runs_dir"])).load(state["run_id"])
    record = manifest.stages.get("review")
    if record is not None and record.status == "awaiting_review":
        return "awaiting_review"
    return "continue"


def build_graph():
    builder = StateGraph(RunState)
    for name in NODES:
        builder.add_node(name, NODE_FUNCTIONS[name])
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "classify")
    builder.add_edge("classify", "plan")
    builder.add_edge("plan", "review")
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"awaiting_review": END, "continue": "prompts"},
    )
    builder.add_edge("prompts", "generate")
    builder.add_edge("generate", "verify")
    builder.add_edge("verify", "save")
    builder.add_edge("save", END)
    return builder.compile()
