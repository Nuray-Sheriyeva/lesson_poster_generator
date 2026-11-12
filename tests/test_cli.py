from __future__ import annotations

import json
from pathlib import Path

from genlesson.cli import main

from .helpers import make_scanned_pdf, make_text_pdf


def test_cli_ingest_json_and_status(tmp_path: Path, capsys) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["A complete lesson about the water cycle."])
    runs = tmp_path / "runs"

    code = main(
        [
            "ingest",
            str(pdf),
            "--run-id",
            "cli-run",
            "--runs-dir",
            str(runs),
            "--vision",
            "none",
            "--dpi",
            "72",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["run_id"] == "cli-run"
    assert payload["stage"] == "ingest"
    assert payload["status"] == "complete"
    assert payload["resumed"] is False
    assert Path(payload["output"]).is_file()

    code = main(["status", "cli-run", "--runs-dir", str(runs), "--json"])
    captured = capsys.readouterr()
    assert code == 0
    status = json.loads(captured.out)
    assert status["stages"]["ingest"]["status"] == "complete"
    assert status["stages"]["plan"]["status"] == "pending"


def test_cli_refuses_existing_run_without_resume(tmp_path: Path, capsys) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["A lesson about the water cycle."])
    runs = tmp_path / "runs"
    args = [
        "ingest",
        str(pdf),
        "--run-id",
        "same-run",
        "--runs-dir",
        str(runs),
        "--vision",
        "none",
    ]
    assert main(args) == 0
    capsys.readouterr()

    code = main(args)
    captured = capsys.readouterr()
    assert code == 2
    assert "run_exists" in captured.err

    code = main(args + ["--resume", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["resumed"] is True


def test_cli_rejects_unsupported_type_as_json(tmp_path: Path, capsys) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    code = main(
        ["ingest", str(path), "--runs-dir", str(tmp_path / "runs"), "--vision", "none", "--json"]
    )
    captured = capsys.readouterr()
    assert code == 2
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_file_type"


def test_cli_detects_changed_input_on_resume(tmp_path: Path, capsys) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["Original lesson text."])
    runs = tmp_path / "runs"
    args = [
        "ingest",
        str(pdf),
        "--run-id",
        "changed",
        "--runs-dir",
        str(runs),
        "--vision",
        "none",
    ]
    assert main(args) == 0
    capsys.readouterr()

    make_text_pdf(pdf, ["Completely different lesson text."])
    code = main(args + ["--resume"])
    captured = capsys.readouterr()
    assert code == 2
    assert "input_changed" in captured.err


def test_cli_rejects_changed_config_on_failed_resume(tmp_path: Path, capsys) -> None:
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", ["Scanned text."])
    runs = tmp_path / "runs"
    base = [
        "ingest",
        str(pdf),
        "--run-id",
        "config-run",
        "--runs-dir",
        str(runs),
        "--vision",
        "none",
    ]
    assert main(base) == 2  # no key / disabled vision
    capsys.readouterr()

    code = main(base + ["--resume", "--vision", "openai", "--vision-model", "different"])
    captured = capsys.readouterr()
    assert code == 2
    assert "config_changed" in captured.err


def test_cli_invalid_run_id(tmp_path: Path, capsys) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["Lesson text."])
    code = main(
        [
            "ingest",
            str(pdf),
            "--run-id",
            "../escape",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--vision",
            "none",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "run_invalid_id" in captured.err


def test_cli_classify_requires_key_then_reuses_completion(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import genlesson.classify as classify_module

    monkeypatch.setenv("POSTER_DISABLE_DOTENV", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pdf = make_text_pdf(
        tmp_path / "lesson.pdf",
        ["Photosynthesis is the process plants use to turn light into sugar."],
    )
    runs = tmp_path / "runs"
    assert (
        main(
            [
                "ingest",
                str(pdf),
                "--run-id",
                "stage2",
                "--runs-dir",
                str(runs),
                "--vision",
                "none",
            ]
        )
        == 0
    )
    capsys.readouterr()

    code = main(["classify", "stage2", "--runs-dir", str(runs), "--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert "planner_required" in captured.out

    from .helpers import FakePlanner

    monkeypatch.setattr(
        classify_module,
        "build_planner",
        lambda options: FakePlanner(model=options.model),
    )
    code = main(
        [
            "classify",
            "stage2",
            "--runs-dir",
            str(runs),
            "--planner-model",
            "fake-luna",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["stage"] == "classify"
    assert payload["poster_language"] == "kk"
    assert payload["resumed"] is False
    assert Path(payload["output"]).is_file()

    # A completed classification is reusable without a key or a network call.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(["classify", "stage2", "--runs-dir", str(runs), "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["resumed"] is True


def test_cli_classify_uses_manifest_planner_model(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import genlesson.classify as classify_module

    from .helpers import FakePlanner

    pdf = make_text_pdf(
        tmp_path / "lesson.pdf",
        ["Photosynthesis is the process plants use to turn light into sugar."],
    )
    runs = tmp_path / "runs"
    monkeypatch.setenv("GENLESSON_PLANNER_MODEL", "manifest-luna")
    assert (
        main(
            [
                "ingest",
                str(pdf),
                "--run-id",
                "manifest-model",
                "--runs-dir",
                str(runs),
                "--vision",
                "none",
            ]
        )
        == 0
    )
    capsys.readouterr()

    monkeypatch.delenv("GENLESSON_PLANNER_MODEL", raising=False)
    seen: dict[str, str] = {}

    def _build(options):
        seen["model"] = options.model
        return FakePlanner(model=options.model)

    monkeypatch.setattr(classify_module, "build_planner", _build)
    code = main(["classify", "manifest-model", "--runs-dir", str(runs), "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert seen["model"] == "manifest-luna"
    assert json.loads(captured.out)["stage"] == "classify"


def _classified_run(tmp_path: Path, monkeypatch):
    from genlesson.classify import run_classify
    from genlesson.ingest import run_ingest
    from genlesson.runs import RunStore, describe_source
    from genlesson.config import manifest_config

    from .helpers import FakePlanner, default_options, planner_options

    pdf = make_text_pdf(
        tmp_path / "lesson.pdf",
        ["Байланыс түрлері туралы қысқа сабақ мәтіні."],
    )
    options = default_options()
    store = RunStore(tmp_path / "runs")
    paths, manifest = store.create(
        describe_source(pdf), manifest_config(options), run_id="plan-cli"
    )
    run_ingest(paths, manifest, store, options)
    run_classify(paths, manifest, store, planner_options(), planner=FakePlanner())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return paths


def test_cli_plan_requires_key_then_reuses_completion(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import genlesson.plan as plan_module

    from .helpers import FakePosterPlanner

    paths = _classified_run(tmp_path, monkeypatch)
    monkeypatch.setenv("POSTER_DISABLE_DOTENV", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = main(["plan", paths.run_id, "--runs-dir", str(tmp_path / "runs"), "--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert "planner_required" in captured.out

    monkeypatch.setattr(
        plan_module,
        "build_planner",
        lambda options: FakePosterPlanner(model=options.model),
    )
    code = main(
        [
            "plan",
            paths.run_id,
            "--runs-dir",
            str(tmp_path / "runs"),
            "--planner-model",
            "fake-luna",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["stage"] == "plan"
    assert payload["language"] == "kk"
    assert payload["details"]
    assert payload["file_slug"] == "communication-types"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(["plan", paths.run_id, "--runs-dir", str(tmp_path / "runs"), "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["resumed"] is True


def _planned_cli_run(tmp_path: Path, monkeypatch):
    from genlesson.plan import run_plan
    from genlesson.runs import RunStore

    from .helpers import FakePosterPlanner, plan_options

    paths = _classified_run(tmp_path, monkeypatch)
    store = RunStore(tmp_path / "runs")
    _, manifest = store.load(paths.run_id)
    run_plan(
        paths,
        manifest,
        store,
        plan_options(),
        planner=FakePosterPlanner(detail_count=2),
    )
    return paths


def test_cli_review_pause_resume_and_prompts(tmp_path: Path, capsys, monkeypatch) -> None:
    paths = _planned_cli_run(tmp_path, monkeypatch)
    runs = tmp_path / "runs"

    code = main(
        [
            "review",
            paths.run_id,
            "--runs-dir",
            str(runs),
            "--review",
            "on",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    paused = json.loads(captured.out)
    assert paused["status"] == "awaiting_review"
    assert paused["paused"] is True
    assert paused["run_status"] == "awaiting_review"
    assert paused["summary"]

    code = main(["status", paths.run_id, "--runs-dir", str(runs), "--json"])
    captured = capsys.readouterr()
    assert code == 0
    run_status = json.loads(captured.out)
    assert run_status["status"] == "awaiting_review"
    assert run_status["stages"]["review"]["status"] == "awaiting_review"

    code = main(["review", paths.run_id, "--runs-dir", str(runs), "--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert "review_pending" in captured.out

    # Exercise the --resume <run_id> ordering.
    code = main(["review", "--resume", paths.run_id, "--runs-dir", str(runs), "--json"])
    captured = capsys.readouterr()
    assert code == 0
    approved = json.loads(captured.out)
    assert approved["status"] == "approved"
    assert approved["paused"] is False

    code = main(
        [
            "prompts",
            paths.run_id,
            "--runs-dir",
            str(runs),
            "--formats",
            "portrait,square",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    bundle = json.loads(captured.out)
    assert [item["format"] for item in bundle["formats"]] == ["portrait", "square"]
    assert Path(bundle["output"]).is_file()
    assert len(bundle["prompt_files"]) == 2


def _prompted_cli_run(tmp_path: Path, monkeypatch):
    from genlesson.config import PromptOptions, ReviewOptions
    from genlesson.prompts import run_prompts
    from genlesson.review import run_review
    from genlesson.runs import RunStore

    paths = _planned_cli_run(tmp_path, monkeypatch)
    store = RunStore(tmp_path / "runs")
    _, manifest = store.load(paths.run_id)
    run_review(paths, manifest, store, ReviewOptions(mode="off"))
    run_prompts(
        paths,
        manifest,
        store,
        PromptOptions(formats=("portrait", "landscape"), image_model="gpt-image-1"),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return paths


def test_cli_generate_partial_failure_is_success(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import genlesson.generate as generate_module

    from .helpers import FakeImageProvider

    paths = _prompted_cli_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        generate_module,
        "build_image_provider",
        lambda options, model: FakeImageProvider(fail_formats={"landscape"}),
    )
    code = main(
        [
            "generate",
            paths.run_id,
            "--runs-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["stage"] == "generate"
    assert payload["generated"] == 1
    assert payload["failed"] == 1
    assert payload["run_status"] == "in_progress"
    assert any(item["status"] == "failed" for item in payload["formats"])


def test_cli_generate_requires_key(tmp_path: Path, capsys, monkeypatch) -> None:
    paths = _prompted_cli_run(tmp_path, monkeypatch)
    monkeypatch.setenv("POSTER_DISABLE_DOTENV", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = main(
        [
            "generate",
            paths.run_id,
            "--runs-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "image_provider_required" in captured.out


def _generated_cli_run(tmp_path: Path, monkeypatch):
    from genlesson.config import GenerationOptions
    from genlesson.generate import run_generate
    from genlesson.models import PosterSpecDocument, ReviewDocument
    from genlesson.runs import RunStore
    from genlesson.storage import read_model

    from .helpers import FakeImageProvider

    paths = _prompted_cli_run(tmp_path, monkeypatch)
    store = RunStore(tmp_path / "runs")
    _, manifest = store.load(paths.run_id)
    run_generate(
        paths,
        manifest,
        store,
        GenerationOptions(api_key="test-key", timeout_s=30, max_retries=1),
        provider=FakeImageProvider(),
    )
    review = read_model(paths.stage_dir("review") / "review.json", ReviewDocument)
    spec = read_model(paths.resolve(review.approved_spec_path), PosterSpecDocument)
    return paths, spec


def test_cli_verify_passes_with_mocked_vision(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import genlesson.verify as verify_module

    from .helpers import FakeVisionTranscriber

    paths, spec = _generated_cli_run(tmp_path, monkeypatch)
    transcript = "\n".join(spec.strings_to_render)
    monkeypatch.setattr(
        verify_module,
        "OpenAIVisionTranscriber",
        lambda **kwargs: FakeVisionTranscriber(transcript),
    )

    code = main(
        [
            "verify",
            paths.run_id,
            "--runs-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["stage"] == "verify"
    assert payload["stats"]["passed"] == 2
    assert payload["stats"]["needs_attention"] == 0
    assert Path(payload["output"]).is_file()


def test_cli_verify_requires_vision_key(tmp_path: Path, capsys, monkeypatch) -> None:
    paths, spec = _generated_cli_run(tmp_path, monkeypatch)
    monkeypatch.setenv("POSTER_DISABLE_DOTENV", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Only remove the generated verification report if a prior helper created one.
    code = main(
        [
            "verify",
            paths.run_id,
            "--runs-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "vision_provider_required" in captured.out


def test_cli_save_writes_final_files(tmp_path: Path, capsys, monkeypatch) -> None:
    from genlesson.config import VerificationOptions
    from genlesson.models import PosterSpecDocument, ReviewDocument
    from genlesson.runs import RunStore
    from genlesson.storage import read_model
    from genlesson.verify import run_verify

    from .helpers import FakeVisionTranscriber

    paths, spec = _generated_cli_run(tmp_path, monkeypatch)
    store = RunStore(tmp_path / "runs")
    _, manifest = store.load(paths.run_id)
    run_verify(
        paths,
        manifest,
        store,
        VerificationOptions(api_key="test-key", max_retries=0),
        transcriber=FakeVisionTranscriber("\n".join(spec.strings_to_render)),
    )

    code = main(
        [
            "save",
            paths.run_id,
            "--runs-dir",
            str(tmp_path / "runs"),
            "--out-dir",
            str(tmp_path / "outputs"),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["stage"] == "save"
    assert payload["stats"]["saved"] == 2
    assert all(Path(item["image_path"]).is_file() for item in payload["files"])
    assert all(Path(item["sidecar_path"]).is_file() for item in payload["files"])


def test_cli_prompts_repeatable_format_flags(tmp_path: Path, capsys, monkeypatch) -> None:
    paths = _planned_cli_run(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    assert main(["review", paths.run_id, "--runs-dir", str(runs)]) == 0
    capsys.readouterr()

    code = main(
        [
            "prompts",
            paths.run_id,
            "--runs-dir",
            str(runs),
            "--format",
            "portrait",
            "--format",
            "square",
            "--image-provider",
            "openai",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert [item["format"] for item in payload["formats"]] == ["portrait", "square"]
    assert payload["image_provider"] == "openai"


def test_poster_review_env_pauses_without_flag(tmp_path: Path, capsys, monkeypatch) -> None:
    paths = _planned_cli_run(tmp_path, monkeypatch)
    monkeypatch.setenv("POSTER_REVIEW", "on")

    code = main(
        ["review", paths.run_id, "--runs-dir", str(tmp_path / "runs"), "--json"]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "awaiting_review"
    assert payload["paused"] is True
