from __future__ import annotations

from pathlib import Path

import pytest

from genlesson.errors import GenLessonError
from genlesson.ingest import run_ingest
from genlesson.models import PosterSpecDocument, RunManifest
from genlesson.plan import run_plan
from genlesson.storage import read_model

from .helpers import (
    FakePlanner,
    FakePosterPlanner,
    default_options,
    make_lesson_docx,
    make_text_pdf,
    plan_options,
    planner_options,
    start_run,
)


def _prepared_classified(tmp_path: Path, paragraphs: list[str]):
    path = make_lesson_docx(tmp_path / "lesson.docx", paragraphs)
    ingest_options = default_options()
    store, paths, manifest = start_run(tmp_path, path, ingest_options)
    run_ingest(paths, manifest, store, ingest_options)
    planner = FakePlanner(language="kk", topic_count=1)
    from genlesson.classify import run_classify

    run_classify(paths, manifest, store, planner_options(), planner=planner)
    return store, paths, manifest


def _run_plan(paths, manifest, store, planner, options=None):
    return run_plan(
        paths,
        manifest,
        store,
        options or plan_options(),
        planner=planner,
    )


def test_short_lesson_uses_one_structured_plan_call(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_classified(
        tmp_path,
        [
            "Байланыс дегеніміз — адамдар арасындағы ақпарат алмасу.",
            "Байланыстың ауызша, жазбаша және визуалды түрлері бар.",
        ],
    )
    planner = FakePosterPlanner(detail_count=2)

    result = _run_plan(paths, manifest, store, planner)

    document = result.document
    assert result.resumed is False
    assert len(planner.plan_calls) == 1
    assert planner.candidate_calls == []
    assert planner.plan_calls[0].text is not None
    assert planner.plan_calls[0].candidates is None
    assert document.language == "kk"
    assert document.age_band == "middle"
    assert document.central_idea.title == "Байланыс түрлері"
    assert len(document.details) == 2
    assert document.file_slug == "communication-types"
    assert document.stats.chunks == 1
    assert document.stats.candidate_details == 0
    assert document.stats.page_images_used == 0
    assert document.config.planner_model == "fake-luna"
    assert "Байланыс түрлері" in document.strings_to_render
    assert "Түр 1" in document.strings_to_render
    assert "Мысал 1." in document.strings_to_render
    assert "EXTRA_CALLOUT" in document.strings_to_render

    saved_manifest = read_model(paths.manifest_path, RunManifest)
    assert saved_manifest.stages["plan"].status == "complete"
    assert saved_manifest.config["plan"]["max_details"] == 8
    saved = read_model(
        paths.stage_dir("plan") / "poster_spec.json", PosterSpecDocument
    )
    assert saved.classification_sha256 != "0" * 64


def test_long_lesson_chunks_extracts_candidates_and_merges(tmp_path: Path) -> None:
    paragraph = (
        "Байланыс түрлері туралы мәтін. Ақпаратты беру жолдары мен "
        "қарым-қатынас дағдылары туралы қысқа түсіндірме. " * 10
    )
    store, paths, manifest = _prepared_classified(tmp_path, [paragraph] * 12)
    planner = FakePosterPlanner(detail_count=2)
    options = plan_options(context_char_limit=4000, chunk_char_limit=2000)

    result = _run_plan(paths, manifest, store, planner, options)

    assert len(planner.candidate_calls) > 1
    assert len(planner.plan_calls) == 1
    final_request = planner.plan_calls[0]
    assert final_request.text is None
    assert final_request.candidates is not None
    assert result.document.stats.chunks == len(planner.candidate_calls)
    assert result.document.stats.candidate_details == len(final_request.candidates.details)
    for request in planner.candidate_calls:
        assert len(request.text) <= 2000
    assert result.document.details


def test_max_details_is_enforced(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_classified(
        tmp_path,
        ["Байланыс түрлері туралы қысқа сабақ мәтіні."],
    )
    planner = FakePosterPlanner(detail_count=10)

    result = _run_plan(
        paths, manifest, store, planner, plan_options(max_details=3)
    )

    assert len(result.document.details) == 3
    assert any("kept 3" in warning for warning in result.document.warnings)


def test_no_details_fails(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_classified(
        tmp_path,
        ["Байланыс түрлері туралы қысқа сабақ мәтіні."],
    )
    planner = FakePosterPlanner(detail_count=0)

    with pytest.raises(GenLessonError) as excinfo:
        _run_plan(paths, manifest, store, planner)

    assert excinfo.value.code == "no_plan_details"
    assert read_model(paths.manifest_path, RunManifest).stages["plan"].status == "failed"


def test_language_mismatch_and_slug_are_normalized(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_classified(
        tmp_path,
        ["Байланыс түрлері туралы қысқа сабақ мәтіні."],
    )
    planner = FakePosterPlanner(
        language="ru",
        file_slug="Қатынас түрлері",
        detail_count=1,
    )

    result = _run_plan(paths, manifest, store, planner)

    assert result.document.language == "kk"
    assert result.document.file_slug == "lesson-poster"
    assert result.document.file_slug.isascii()
    assert any("Planner language" in warning for warning in result.document.warnings)
    assert any("file_slug" in warning for warning in result.document.warnings)


def test_forbidden_composition_fails(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_classified(
        tmp_path,
        ["Байланыс түрлері туралы қысқа сабақ мәтіні."],
    )
    planner = FakePosterPlanner(
        composition_concept="plain grid with boxes",
        detail_count=1,
    )

    with pytest.raises(GenLessonError) as excinfo:
        _run_plan(paths, manifest, store, planner)

    assert excinfo.value.code == "plan_invalid"
    assert "forbidden layout" in str(excinfo.value.details)


def test_plan_requires_completed_classification(tmp_path: Path) -> None:
    path = make_lesson_docx(tmp_path / "lesson.docx", ["Байланыс туралы мәтін."])
    options = default_options()
    store, paths, manifest = start_run(tmp_path, path, options)
    run_ingest(paths, manifest, store, options)

    with pytest.raises(GenLessonError) as excinfo:
        _run_plan(paths, manifest, store, FakePosterPlanner())

    assert excinfo.value.code == "stage_incomplete"


def test_plan_resume_is_idempotent(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_classified(
        tmp_path,
        ["Байланыс түрлері туралы қысқа сабақ мәтіні."],
    )
    first = FakePosterPlanner()
    result = _run_plan(paths, manifest, store, first)
    assert result.resumed is False

    second = FakePosterPlanner()
    again = _run_plan(paths, manifest, store, second)

    assert again.resumed is True
    assert again.document.file_slug == result.document.file_slug
    assert second.plan_calls == []
    assert second.candidate_calls == []


def test_pdf_page_images_are_passed_to_planner(tmp_path: Path) -> None:
    pdf = make_text_pdf(
        tmp_path / "lesson.pdf",
        [
            "Page one explains communication types in enough detail.",
            "Page two explains written communication in enough detail.",
            "Page three explains visual communication in enough detail.",
        ],
    )
    ingest_options = default_options()
    store, paths, manifest = start_run(tmp_path, pdf, ingest_options)
    run_ingest(paths, manifest, store, ingest_options)
    from genlesson.classify import run_classify

    run_classify(paths, manifest, store, planner_options(), planner=FakePlanner())
    planner = FakePosterPlanner()

    result = _run_plan(
        paths, manifest, store, planner, plan_options(max_page_images=1)
    )

    assert result.document.stats.page_images_used == 1
    assert len(planner.plan_calls[0].images) == 1
    assert any("page images" in warning for warning in result.document.warnings)
