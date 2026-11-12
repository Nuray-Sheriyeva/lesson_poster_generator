from __future__ import annotations

import json
from pathlib import Path

import pytest

from genlesson.classify import run_classify
from genlesson.config import PromptOptions, ReviewOptions
from genlesson.errors import GenLessonError
from genlesson.ingest import run_ingest
from genlesson.models import (
    PosterSpecDocument,
    PromptBundle,
    ReviewDocument,
    RunManifest,
    SourceConflict,
)
from genlesson.plan import run_plan
from genlesson.prompts import run_prompts
from genlesson.review import run_review
from genlesson.storage import read_model

from .helpers import (
    FakePlanner,
    FakePosterPlanner,
    default_options,
    make_lesson_docx,
    plan_options,
    planner_options,
    start_run,
)

CONFLICT = SourceConflict(
    issue="The source gives two different definitions of feedback.",
    resolution="Used the fuller definition from page 2.",
)


def _planned_run(tmp_path: Path, planner: FakePosterPlanner | None = None):
    path = make_lesson_docx(
        tmp_path / "lesson.docx",
        [
            "Байланыс дегеніміз — адамдар арасындағы ақпарат алмасу.",
            "Байланыстың ауызша, жазбаша және визуалды түрлері бар.",
        ],
    )
    ingest_options = default_options()
    store, paths, manifest = start_run(tmp_path, path, ingest_options)
    run_ingest(paths, manifest, store, ingest_options)
    run_classify(paths, manifest, store, planner_options(), planner=FakePlanner())
    run_plan(
        paths,
        manifest,
        store,
        plan_options(),
        planner=planner or FakePosterPlanner(detail_count=2),
    )
    return store, paths, manifest


def test_review_off_auto_approves_and_logs_conflicts(tmp_path: Path) -> None:
    store, paths, manifest = _planned_run(
        tmp_path,
        FakePosterPlanner(detail_count=2, source_conflicts=[CONFLICT]),
    )

    result = run_review(paths, manifest, store, ReviewOptions(mode="off"))

    assert result.paused is False
    assert result.document.status == "auto_approved"
    assert result.document.mode == "off"
    assert result.document.source_conflicts == [CONFLICT]
    assert result.document.summary_path is None
    assert result.approved_spec_path is not None
    assert result.approved_spec_path.is_file()
    assert result.document.approved_spec_sha256 is not None
    assert read_model(paths.manifest_path, RunManifest).stages["review"].status == "skipped"
    saved = read_model(paths.stage_dir("review") / "review.json", ReviewDocument)
    assert saved.source_conflicts[0].issue == CONFLICT.issue


def test_review_on_pauses_and_writes_summary(tmp_path: Path) -> None:
    store, paths, manifest = _planned_run(tmp_path)

    result = run_review(paths, manifest, store, ReviewOptions(mode="on"))

    assert result.paused is True
    assert result.document.status == "awaiting_review"
    assert result.document.mode == "on"
    assert result.summary_path is not None
    assert result.summary_path.is_file()
    assert result.approved_spec_path is None
    assert "Review:" in result.summary_path.read_text(encoding="utf-8")
    saved = read_model(paths.manifest_path, RunManifest)
    assert saved.status == "awaiting_review"
    assert saved.stages["review"].status == "awaiting_review"

    with pytest.raises(GenLessonError) as excinfo:
        run_review(paths, saved, store, ReviewOptions(mode="on"))
    assert excinfo.value.code == "review_pending"


def test_review_resume_approves_current_spec(tmp_path: Path) -> None:
    store, paths, manifest = _planned_run(tmp_path)
    run_review(paths, manifest, store, ReviewOptions(mode="on"))
    saved = read_model(paths.manifest_path, RunManifest)

    result = run_review(paths, saved, store, ReviewOptions(mode="on"), resume=True)

    assert result.resumed is True
    assert result.paused is False
    assert result.document.status == "approved"
    assert result.document.edited is False
    assert result.approved_spec_path is not None
    assert result.approved_spec_path.is_file()
    final = read_model(paths.manifest_path, RunManifest)
    assert final.stages["review"].status == "complete"
    assert final.status == "in_progress"


def test_review_resume_accepts_edited_spec(tmp_path: Path) -> None:
    store, paths, manifest = _planned_run(tmp_path)
    run_review(paths, manifest, store, ReviewOptions(mode="on"))
    saved = read_model(paths.manifest_path, RunManifest)

    plan_path = paths.stage_dir("plan") / "poster_spec.json"
    original = json.loads(plan_path.read_text(encoding="utf-8"))
    original["central_idea"]["title"] = "Жаңа тақырып"
    original["central_idea"]["definition"] = "Жаңа бір жолдық анықтама."
    original["strings_to_render"] = [
        "Жаңа тақырып",
        "Жаңа бір жолдық анықтама.",
        *original["strings_to_render"][2:],
    ]
    edited_path = tmp_path / "edited.json"
    edited_path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")

    result = run_review(
        paths,
        saved,
        store,
        ReviewOptions(mode="on"),
        resume=True,
        spec_path=edited_path,
    )

    assert result.document.edited is True
    approved = read_model(result.approved_spec_path, PosterSpecDocument)
    assert approved.central_idea.title == "Жаңа тақырып"
    assert result.document.status == "approved"


def test_review_resume_rejects_invalid_spec(tmp_path: Path) -> None:
    store, paths, manifest = _planned_run(tmp_path)
    run_review(paths, manifest, store, ReviewOptions(mode="on"))
    saved = read_model(paths.manifest_path, RunManifest)

    plan_path = paths.stage_dir("plan") / "poster_spec.json"
    original = json.loads(plan_path.read_text(encoding="utf-8"))
    original["file_slug"] = "Not Valid"
    edited_path = tmp_path / "bad.json"
    edited_path.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(GenLessonError) as excinfo:
        run_review(
            paths,
            saved,
            store,
            ReviewOptions(mode="on"),
            resume=True,
            spec_path=edited_path,
        )

    assert excinfo.value.code == "spec_invalid"
    after = read_model(paths.manifest_path, RunManifest)
    assert after.stages["review"].status == "awaiting_review"
    assert read_model(
        paths.stage_dir("review") / "review.json", ReviewDocument
    ).status == "awaiting_review"


def test_review_on_conflict_pauses_only_with_conflicts(tmp_path: Path) -> None:
    store, paths, manifest = _planned_run(tmp_path)
    result = run_review(paths, manifest, store, ReviewOptions(mode="on_conflict"))
    assert result.paused is False
    assert result.document.status == "auto_approved"

    conflict_dir = tmp_path / "conflict"
    conflict_dir.mkdir()
    store2, paths2, manifest2 = _planned_run(
        conflict_dir,
        FakePosterPlanner(detail_count=1, source_conflicts=[CONFLICT]),
    )
    result2 = run_review(paths2, manifest2, store2, ReviewOptions(mode="on_conflict"))
    assert result2.paused is True
    assert result2.document.status == "awaiting_review"


def _reviewed_run(tmp_path: Path):
    store, paths, manifest = _planned_run(tmp_path)
    run_review(paths, manifest, store, ReviewOptions(mode="off"))
    return store, paths, manifest


def test_prompts_builds_one_prompt_per_format(tmp_path: Path) -> None:
    store, paths, manifest = _reviewed_run(tmp_path)
    options = PromptOptions(
        formats=("portrait", "landscape", "square"),
        image_model="gpt-image-1",
    )

    result = run_prompts(paths, manifest, store, options)

    assert [item.format for item in result.document.formats] == [
        "portrait",
        "landscape",
        "square",
    ]
    assert [(item.width, item.height) for item in result.document.formats] == [
        (1024, 1536),
        (1536, 1024),
        (1024, 1024),
    ]
    assert len(result.prompt_paths) == 3
    spec = read_model(
        paths.resolve(read_model(
            paths.stage_dir("review") / "review.json", ReviewDocument
        ).approved_spec_path),
        PosterSpecDocument,
    )
    for fmt, item in zip(("portrait", "landscape", "square"), result.document.formats):
        prompt = item.prompt
        assert item.filename.endswith(f"-{fmt}.png")
        for value in spec.strings_to_render:
            assert json.dumps(value, ensure_ascii=False) in prompt
        assert all(letter in prompt for letter in "әғқңөұүһі")
        assert "watermark" in prompt.lower()
        assert "wrong-language" in prompt.lower()
        assert "Do not add any text" in prompt
        assert "Never render 1., 2., 3." in prompt
        assert "no blood, gore, wounds" in prompt
        assert "the list numbers" in prompt.lower()
        assert spec.composition.concept in prompt
        assert spec.style.mood in prompt
    assert "Vertical poster" in result.document.formats[0].prompt
    assert "Horizontal poster" in result.document.formats[1].prompt
    assert "Square poster" in result.document.formats[2].prompt
    assert read_model(paths.manifest_path, RunManifest).stages["prompts"].status == "complete"
    saved = read_model(paths.stage_dir("prompts") / "prompts.json", PromptBundle)
    assert [item.format for item in saved.formats] == ["portrait", "landscape", "square"]


def test_prompts_defaults_to_portrait(tmp_path: Path) -> None:
    store, paths, manifest = _reviewed_run(tmp_path)
    result = run_prompts(paths, manifest, store, PromptOptions())
    assert len(result.document.formats) == 1
    assert result.document.formats[0].format == "portrait"
    assert result.document.formats[0].width == 1024
    assert result.document.formats[0].height == 1536


def test_prompts_require_approved_review(tmp_path: Path) -> None:
    store, paths, manifest = _planned_run(tmp_path)
    run_review(paths, manifest, store, ReviewOptions(mode="on"))
    saved = read_model(paths.manifest_path, RunManifest)

    with pytest.raises(GenLessonError) as excinfo:
        run_prompts(paths, saved, store, PromptOptions())
    assert excinfo.value.code == "review_pending"


def test_prompts_resume_reuses_bundle(tmp_path: Path) -> None:
    store, paths, manifest = _reviewed_run(tmp_path)
    options = PromptOptions(formats=("portrait",), image_model="gpt-image-1")
    first = run_prompts(paths, manifest, store, options)
    second = run_prompts(paths, manifest, store, options)
    assert second.resumed is True
    assert second.document == first.document


def test_prompts_unknown_image_model_fails(tmp_path: Path) -> None:
    store, paths, manifest = _reviewed_run(tmp_path)
    with pytest.raises(GenLessonError) as excinfo:
        run_prompts(
            paths,
            manifest,
            store,
            PromptOptions(formats=("portrait",), image_model="unknown-image-model"),
        )
    assert excinfo.value.code == "unsupported_image_model"


def test_prompts_rebuild_when_formats_change(tmp_path: Path) -> None:
    store, paths, manifest = _reviewed_run(tmp_path)
    first = run_prompts(
        paths,
        manifest,
        store,
        PromptOptions(formats=("portrait",), image_model="gpt-image-1"),
    )
    assert [item.format for item in first.document.formats] == ["portrait"]

    second = run_prompts(
        paths,
        manifest,
        store,
        PromptOptions(formats=("portrait", "landscape"), image_model="gpt-image-1"),
    )
    assert second.resumed is False
    assert [item.format for item in second.document.formats] == ["portrait", "landscape"]
    assert len(second.prompt_paths) == 2


def test_review_can_be_reopened_after_auto_approval(tmp_path: Path) -> None:
    store, paths, manifest = _planned_run(tmp_path)
    first = run_review(paths, manifest, store, ReviewOptions(mode="off"))
    assert first.document.status == "auto_approved"

    saved = read_model(paths.manifest_path, RunManifest)
    reopened = run_review(paths, saved, store, ReviewOptions(mode="on"))
    assert reopened.paused is True
    assert reopened.document.status == "awaiting_review"
    assert read_model(paths.manifest_path, RunManifest).status == "awaiting_review"
