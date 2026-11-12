from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from genlesson import tracing
from genlesson.cli import main
from genlesson.models import (
    ClassificationDocument,
    GenerationDocument,
    PosterSpecDocument,
    SaveDocument,
    VerificationDocument,
)
from genlesson.storage import read_model

from .helpers import (
    FakeImageProvider,
    FakePlanner,
    FakePosterPlanner,
    FakeVisionTranscriber,
    make_lesson_docx,
)
from .test_tracing import _enable

TRANSCRIPT = "\n".join(
    [
        "Байланыс түрлері",
        "Байланыс — ақпарат алмасудың түрлері.",
        "Түр 1",
        "Қысқа түсіндірме 1.",
        "Мысал 1.",
        "Түр 2",
        "Қысқа түсіндірме 2.",
        "Мысал 2.",
        "EXTRA_CALLOUT",
    ]
)


@pytest.fixture(autouse=True)
def _openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _patch_providers(monkeypatch) -> dict[str, list]:
    import genlesson.classify as classify_module
    import genlesson.generate as generate_module
    import genlesson.plan as plan_module
    import genlesson.verify as verify_module

    calls: dict[str, list] = {
        "classify": [],
        "plan": [],
        "image": [],
        "verify": [],
    }

    def build_classifier(options):
        planner = FakePlanner(model=options.model)
        calls["classify"].append(planner)
        return planner

    def build_poster_planner(options):
        planner = FakePosterPlanner(detail_count=2)
        calls["plan"].append(planner)
        return planner

    def build_image(options, *, model):
        provider = FakeImageProvider()
        calls["image"].append(provider)
        return provider

    def build_vision(**kwargs):
        transcriber = FakeVisionTranscriber(TRANSCRIPT)
        calls["verify"].append(transcriber)
        return transcriber

    monkeypatch.setattr(classify_module, "build_planner", build_classifier)
    monkeypatch.setattr(plan_module, "build_planner", build_poster_planner)
    monkeypatch.setattr(generate_module, "build_image_provider", build_image)
    monkeypatch.setattr(verify_module, "OpenAIVisionTranscriber", build_vision)
    return calls


def _docx(tmp_path: Path) -> Path:
    return make_lesson_docx(
        tmp_path / "lesson.docx",
        [
            "Байланыс дегеніміз — адамдар арасындағы ақпарат алмасу.",
            "Байланыстың ауызша, жазбаша және визуалды түрлері бар.",
        ],
    )


def _sequential(docx: Path, runs: Path, out_dir: Path) -> None:
    base = ["--runs-dir", str(runs), "--json"]
    assert main(["ingest", str(docx), "--run-id", "seq-run", *base]) == 0
    assert main(["classify", "seq-run", *base]) == 0
    assert main(["plan", "seq-run", *base]) == 0
    assert main(["review", "seq-run", *base]) == 0
    assert main(["prompts", "seq-run", *base]) == 0
    assert main(["generate", "seq-run", *base]) == 0
    assert main(["verify", "seq-run", *base]) == 0
    assert main(["save", "seq-run", *base, "--out-dir", str(out_dir)]) == 0


def _graph_run(
    docx: Path | None,
    runs: Path,
    *,
    run_id: str,
    out_dir: Path | None = None,
    review: str = "off",
    resume: bool = False,
    json_output: bool = True,
) -> tuple[int, str]:
    argv = ["run"]
    if docx is not None and not resume:
        argv.append(str(docx))
    argv.extend(["--run-id", run_id, "--runs-dir", str(runs)])
    if resume:
        argv.append("--resume")
    if out_dir is not None:
        argv.extend(["--out-dir", str(out_dir)])
    argv.extend(["--review", review])
    if json_output:
        argv.append("--json")
    return main(argv), ""


def _stage_paths(runs: Path, run_id: str) -> dict[str, Path]:
    root = runs / run_id
    return {
        "classify": root / "stage-02-classify" / "classification.json",
        "plan": root / "stage-03-plan" / "poster_spec.json",
        "generation": root / "stage-06-generate" / "generation.json",
        "verification": root / "stage-07-verify" / "verification.json",
        "save": root / "stage-08-save" / "save.json",
    }


def test_graph_matches_sequential_commands(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    calls = _patch_providers(monkeypatch)
    docx = _docx(tmp_path)
    seq_runs = tmp_path / "runs-seq"
    graph_runs = tmp_path / "runs-graph"
    seq_out = tmp_path / "out-seq"
    graph_out = tmp_path / "out-graph"

    _sequential(docx, seq_runs, seq_out)
    capsys.readouterr()
    code, _ = _graph_run(
        docx,
        graph_runs,
        run_id="graph-run",
        out_dir=graph_out,
        review="off",
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "complete"

    seq = _stage_paths(seq_runs, "seq-run")
    graph = _stage_paths(graph_runs, "graph-run")
    seq_classify = read_model(seq["classify"], ClassificationDocument)
    graph_classify = read_model(graph["classify"], ClassificationDocument)
    assert seq_classify.poster_language == graph_classify.poster_language
    assert [item.id for item in seq_classify.segments] == [
        item.id for item in graph_classify.segments
    ]
    assert seq_classify.stats.model_dump() == graph_classify.stats.model_dump()

    seq_plan = read_model(seq["plan"], PosterSpecDocument)
    graph_plan = read_model(graph["plan"], PosterSpecDocument)
    assert [item.label for item in seq_plan.details] == [
        item.label for item in graph_plan.details
    ]
    assert seq_plan.file_slug == graph_plan.file_slug

    seq_generation = read_model(seq["generation"], GenerationDocument)
    graph_generation = read_model(graph["generation"], GenerationDocument)
    assert [item.status for item in seq_generation.formats] == [
        item.status for item in graph_generation.formats
    ]
    assert [item.image_sha256 for item in seq_generation.formats] == [
        item.image_sha256 for item in graph_generation.formats
    ]

    seq_verification = read_model(seq["verification"], VerificationDocument)
    graph_verification = read_model(graph["verification"], VerificationDocument)
    assert [item.status for item in seq_verification.formats] == [
        item.status for item in graph_verification.formats
    ]
    assert [item.best_score for item in seq_verification.formats] == [
        item.best_score for item in graph_verification.formats
    ]

    seq_save = read_model(seq["save"], SaveDocument)
    graph_save = read_model(graph["save"], SaveDocument)
    assert seq_save.stats.model_dump() == graph_save.stats.model_dump()

    assert len(calls["classify"]) == 2
    assert len(calls["plan"]) == 2
    assert len(calls["image"]) == 2
    assert len(calls["verify"]) == 2
    assert all(len(provider.calls) == 1 for provider in calls["image"])


def test_graph_review_gate_halts_then_completes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    calls = _patch_providers(monkeypatch)
    docx = _docx(tmp_path)
    runs = tmp_path / "runs"
    out_dir = tmp_path / "out"

    code, _ = _graph_run(
        docx,
        runs,
        run_id="gate-run",
        out_dir=out_dir,
        review="on",
    )
    first = json.loads(capsys.readouterr().out)
    assert code == 0
    assert first["status"] == "awaiting_review"
    assert calls["image"] == []
    assert calls["verify"] == []

    assert main(["review", "gate-run", "--runs-dir", str(runs), "--resume", "--json"]) == 0
    capsys.readouterr()

    code, _ = _graph_run(
        None,
        runs,
        run_id="gate-run",
        out_dir=out_dir,
        review="on",
        resume=True,
    )
    final = json.loads(capsys.readouterr().out)
    assert code == 0
    assert final["status"] == "complete"
    assert final["stages_run"] == [
        "ingest",
        "classify",
        "plan",
        "review",
        "prompts",
        "generate",
        "verify",
        "save",
    ]
    assert len(calls["image"]) == 1
    assert len(calls["verify"]) == 1


def test_graph_resume_makes_no_new_provider_calls(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    calls = _patch_providers(monkeypatch)
    docx = _docx(tmp_path)
    runs = tmp_path / "runs"
    out_dir = tmp_path / "out"

    code, _ = _graph_run(docx, runs, run_id="resume-run", out_dir=out_dir)
    assert code == 0
    capsys.readouterr()
    before = {key: len(value) for key, value in calls.items()}

    code, _ = _graph_run(
        None,
        runs,
        run_id="resume-run",
        out_dir=out_dir,
        resume=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "complete"
    assert {key: len(value) for key, value in calls.items()} == before


def test_graph_topology() -> None:
    from genlesson.graph import EDGES, NODES, build_graph

    assert NODES == (
        "ingest",
        "classify",
        "plan",
        "review",
        "prompts",
        "generate",
        "verify",
        "save",
    )
    assert ("review", "awaiting_review") in EDGES
    assert ("review", "prompts") in EDGES
    compiled = build_graph().get_graph()
    node_names = {name for name in compiled.nodes if not name.startswith("__")}
    assert node_names == set(NODES)
    graph_edges = {(edge.source, edge.target) for edge in compiled.edges}
    assert ("__start__", "ingest") in graph_edges
    assert ("review", "prompts") in graph_edges
    assert ("review", "__end__") in graph_edges
    assert ("save", "__end__") in graph_edges


def test_graph_prints_stage_progress(tmp_path: Path, monkeypatch, capsys) -> None:
    _patch_providers(monkeypatch)
    docx = _docx(tmp_path)
    code, _ = _graph_run(
        docx,
        tmp_path / "runs",
        run_id="progress-run",
        out_dir=tmp_path / "out",
        json_output=False,
    )
    captured = capsys.readouterr()
    assert code == 0
    for stage in (
        "ingest",
        "classify",
        "plan",
        "review",
        "prompts",
        "generate",
        "verify",
        "save",
    ):
        assert f"[{stage}]" in captured.out


def test_graph_root_span_wraps_stage_spans(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_providers(monkeypatch)
    backend = _enable(monkeypatch)
    docx = _docx(tmp_path)
    try:
        code, _ = _graph_run(
            docx,
            tmp_path / "runs",
            run_id="graph-trace",
            out_dir=tmp_path / "out",
        )
        capsys.readouterr()
        assert code == 0
        entries = list(zip(backend.created, backend.observations))
        by_name = {
            created["name"]: (created, observation)
            for created, observation in entries
        }
        root_entry, root_observation = by_name["run:graph-trace"]
        ingest_entry, _ = by_name["ingest"]
        classify_entry, _ = by_name["classify"]
        assert root_entry["_parent"] is None
        assert root_entry["trace_context"]["trace_id"] == tracing.trace_id_for_run(
            "graph-trace"
        )
        assert ingest_entry["_parent"] == root_observation.id
        assert classify_entry["_parent"] == root_observation.id
        assert "trace_context" not in ingest_entry
    finally:
        tracing.shutdown()
