from __future__ import annotations

from pathlib import Path

import pytest

from genlesson.classify import normalize_bcp47, run_classify, segment_document
from genlesson.errors import GenLessonError
from genlesson.ingest import run_ingest
from genlesson.models import ClassificationDocument, LessonDocument, RunManifest
from genlesson.storage import read_model

from .helpers import (
    FakePlanner,
    FakeTranscriber,
    default_options,
    make_lesson_docx,
    make_mixed_pdf,
    make_scanned_pdf,
    planner_options,
    start_run,
)


def _prepared_docx(tmp_path: Path, paragraphs: list[str]):
    path = make_lesson_docx(tmp_path / "lesson.docx", paragraphs)
    options = default_options()
    store, paths, manifest = start_run(tmp_path, path, options)
    run_ingest(paths, manifest, store, options)
    return store, paths, manifest


def _run_classify(paths, manifest, store, planner, options=None):
    return run_classify(
        paths,
        manifest,
        store,
        options or planner_options(),
        planner=planner,
    )


def test_classify_separates_content_and_uses_only_content_for_language(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_docx(
        tmp_path,
        [
            "Learning objective: explain how plants turn light into sugar.",
            "Assessment criteria: name the inputs and outputs.",
            "Exam instruction: answer questions 1-3.",
            "Анықта: фотосинтез үдерісін анықта.",
            "Photosynthesis is the process plants use to turn light, water, and "
            "carbon dioxide into sugar.",
            "Chlorophyll absorbs light and gives leaves their green color.",
            "Мысал келтір: фотосинтезге мысал келтір.",
        ],
    )
    planner = FakePlanner(
        non_content_markers=(
            "learning objective",
            "assessment criteria",
            "exam instruction",
            "анықта",
            "мысал келтір",
        ),
        language="Kazakh",
        confidence=0.93,
        evidence="Photosynthesis is the process...",
        topic_count=2,
    )

    result = _run_classify(paths, manifest, store, planner)

    document = result.document
    assert result.resumed is False
    assert document.poster_language == "kk"
    assert document.confidence == pytest.approx(0.93)
    assert "Photosynthesis" in document.lesson_text
    assert "Chlorophyll" in document.lesson_text
    assert "Learning objective" not in document.lesson_text
    assert "Assessment criteria" not in document.lesson_text
    assert "Exam instruction" not in document.lesson_text
    assert "Анықта" not in document.lesson_text
    assert "Мысал келтір" not in document.lesson_text

    assert len(planner.classify_calls) == 1
    assert len(planner.language_calls) == 1
    language_text = planner.language_calls[0].text
    assert "Photosynthesis" in language_text
    assert "Learning objective" not in language_text
    assert "Анықта" not in language_text

    categories = {
        segment.text.split(":", 1)[0]: segment.category for segment in document.segments
    }
    assert categories["Learning objective"] == "learning_objective"
    assert categories["Assessment criteria"] == "assessment_criteria"
    assert categories["Exam instruction"] == "exam_instruction"
    assert categories["Анықта"] == "exercise_prompt"
    assert categories["Мысал келтір"] == "exercise_prompt"
    assert document.language_source == "text_layer"
    assert document.config.planner_model == "fake-luna"
    assert document.stats.segment_count == 7
    assert document.stats.content_segments == 2
    assert document.stats.non_content_segments == 5
    assert document.stats.page_images_used == 0
    expected_dominant = max(document.topics, key=lambda topic: topic.char_count).title
    assert document.dominant_topic == expected_dominant
    assert any("Multiple topics" in warning for warning in document.warnings)
    saved_manifest = read_model(paths.manifest_path, RunManifest)
    assert saved_manifest.stages["classify"].status == "complete"
    assert saved_manifest.config["planner"]["model"] == "fake-luna"
    saved = read_model(
        paths.stage_dir("classify") / "classification.json", ClassificationDocument
    )
    assert saved.poster_language == "kk"
    assert saved.lesson_text == document.lesson_text


def test_no_lesson_content_fails_instead_of_guessing(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_docx(
        tmp_path,
        [
            "Learning objective: explain photosynthesis.",
            "Assessment criteria: answer correctly.",
            "Анықта: негізгі ұғымды анықта.",
        ],
    )
    planner = FakePlanner(all_non_content=True)

    with pytest.raises(GenLessonError) as excinfo:
        _run_classify(paths, manifest, store, planner)

    assert excinfo.value.code == "no_lesson_content"
    assert planner.language_calls == []
    saved = read_model(paths.manifest_path, RunManifest)
    assert saved.stages["classify"].status == "failed"
    assert not (paths.stage_dir("classify") / "classification.json").exists()


def test_classify_resume_is_idempotent(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_docx(
        tmp_path,
        ["Photosynthesis is the process plants use to make sugar."],
    )
    first = FakePlanner()
    result = _run_classify(paths, manifest, store, first)
    assert result.resumed is False

    second = FakePlanner()
    again = _run_classify(paths, manifest, store, second)

    assert again.resumed is True
    assert again.document.poster_language == result.document.poster_language
    assert second.classify_calls == []
    assert second.language_calls == []


def test_classify_requires_completed_ingest(tmp_path: Path) -> None:
    path = make_lesson_docx(tmp_path / "lesson.docx", ["Photosynthesis is a process."])
    options = default_options()
    store, paths, manifest = start_run(tmp_path, path, options)

    with pytest.raises(GenLessonError) as excinfo:
        _run_classify(paths, manifest, store, FakePlanner())

    assert excinfo.value.code == "stage_incomplete"


def test_mixed_document_excludes_vision_text_from_language(tmp_path: Path) -> None:
    pdf = make_mixed_pdf(
        tmp_path / "mixed.pdf",
        text_page=(
            "Photosynthesis is the process plants use to turn light into "
            "chemical energy."
        ),
        scanned_page=(
            "Это русский текст с картинки про фотосинтез."
        ),
    )
    ingest_options = default_options(
        vision_mode="openai", openai_api_key="test-key"
    )
    store, paths, manifest = start_run(tmp_path, pdf, ingest_options)
    run_ingest(
        paths,
        manifest,
        store,
        ingest_options,
        transcriber=FakeTranscriber(
            {2: "Это русский текст с картинки про фотосинтез."}
        ),
    )
    planner = FakePlanner()

    result = _run_classify(paths, manifest, store, planner)

    assert result.document.language_source == "text_layer"
    assert "Photosynthesis" in planner.language_calls[0].text
    assert "русский" not in planner.language_calls[0].text
    assert any("Vision-transcribed" in warning for warning in result.document.warnings)


def test_scanned_document_uses_ocr_as_body_text(tmp_path: Path) -> None:
    pdf = make_scanned_pdf(
        tmp_path / "scan.pdf", ["Фотосинтез туралы сабақ мәтіні."]
    )
    ingest_options = default_options(
        vision_mode="openai", openai_api_key="test-key"
    )
    store, paths, manifest = start_run(tmp_path, pdf, ingest_options)
    run_ingest(
        paths,
        manifest,
        store,
        ingest_options,
        transcriber=FakeTranscriber(
            {1: "Фотосинтез туралы сабақ мәтіні."}
        ),
    )
    planner = FakePlanner()

    result = _run_classify(paths, manifest, store, planner)

    assert result.document.language_source == "vision_transcript"
    assert "Фотосинтез" in planner.language_calls[0].text


def test_missing_segment_decisions_are_rejected(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_docx(
        tmp_path,
        ["Photosynthesis is the process plants use to make sugar."],
    )
    planner = FakePlanner(missing_ids=("s0001",))

    with pytest.raises(GenLessonError) as excinfo:
        _run_classify(paths, manifest, store, planner)

    assert excinfo.value.code == "planner_response_invalid"
    assert read_model(paths.manifest_path, RunManifest).stages["classify"].status == "failed"


def test_invalid_language_tag_fails(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_docx(
        tmp_path,
        ["Photosynthesis is the process plants use to make sugar."],
    )
    planner = FakePlanner(language="!!!")

    with pytest.raises(GenLessonError) as excinfo:
        _run_classify(paths, manifest, store, planner)

    assert excinfo.value.code == "language_undetermined"


def test_docx_has_no_planner_page_images(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_docx(
        tmp_path,
        ["Photosynthesis is the process plants use to make sugar."],
    )
    planner = FakePlanner()
    result = _run_classify(
        paths, manifest, store, planner, planner_options(max_page_images=2)
    )
    assert result.document.stats.page_images_used == 0
    assert planner.classify_calls[0].images == ()


def test_pdf_page_images_are_capped_for_planner_context(tmp_path: Path) -> None:
    from .helpers import make_text_pdf

    pdf = make_text_pdf(
        tmp_path / "three-pages.pdf",
        [
            "Page one explains photosynthesis in enough detail for classification.",
            "Page two explains chlorophyll and light absorption in plant leaves.",
            "Page three explains the outputs of photosynthesis for the lesson.",
        ],
    )
    ingest_options = default_options()
    store, paths, manifest = start_run(tmp_path, pdf, ingest_options)
    run_ingest(paths, manifest, store, ingest_options)
    planner = FakePlanner()

    result = _run_classify(
        paths, manifest, store, planner, planner_options(max_page_images=1)
    )

    assert result.document.stats.page_images_used == 1
    assert len(planner.classify_calls[0].images) == 1
    assert any("page images" in warning for warning in result.document.warnings)


def test_normalize_bcp47() -> None:
    assert normalize_bcp47("Kazakh") == "kk"
    assert normalize_bcp47("kk_KZ") == "kk-KZ"
    assert normalize_bcp47("zh-hans") == "zh-Hans"
    assert normalize_bcp47("es-419") == "es-419"
    assert normalize_bcp47("Kazakh language") == "kk"
    assert normalize_bcp47("!!!") is None
    assert normalize_bcp47("") is None
    assert normalize_bcp47("english") == "en"


def test_segments_split_on_paragraphs(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_docx(
        tmp_path,
        ["First paragraph with enough text.", "Second paragraph with enough text."],
    )
    document = read_model(paths.stage_dir("ingest") / "lesson.json", LessonDocument)
    segments = segment_document(document)
    assert [segment.text for segment in segments] == [
        "First paragraph with enough text.",
        "Second paragraph with enough text.",
    ]
    assert [segment.id for segment in segments] == ["s0001", "s0002"]


def test_planner_without_vision_receives_no_page_images(tmp_path: Path) -> None:
    pdf = make_scanned_pdf(
        tmp_path / "scanned.pdf",
        ["Photosynthesis turns light into chemical energy."],
    )
    ingest_options = default_options(
        vision_mode="openai", openai_api_key="test-key"
    )
    store, paths, manifest = start_run(tmp_path, pdf, ingest_options)
    run_ingest(
        paths,
        manifest,
        store,
        ingest_options,
        transcriber=FakeTranscriber(
            {1: "Photosynthesis turns light into chemical energy."}
        ),
    )
    planner = FakePlanner(vision=False)

    result = _run_classify(paths, manifest, store, planner)

    assert result.document.stats.page_images_used == 0
    assert planner.classify_calls[0].images == ()
    assert any("lacks vision" in warning for warning in result.document.warnings)
    saved = read_model(
        paths.stage_dir("classify") / "classification.json", ClassificationDocument
    )
    assert saved.stats.page_images_used == 0
    saved_manifest = read_model(paths.manifest_path, RunManifest)
    assert saved_manifest.config["planner"]["capabilities"]["vision"] is False


def test_russian_lesson_selects_russian(tmp_path: Path) -> None:
    store, paths, manifest = _prepared_docx(
        tmp_path,
        [
            "Фотосинтез — это процесс, при котором растения превращают свет в сахар.",
            "Хлорофилл поглощает свет и придаёт листьям зелёный цвет.",
        ],
    )
    planner = FakePlanner(
        language="ru",
        confidence=0.97,
        evidence="Фотосинтез — это процесс...",
    )

    result = _run_classify(paths, manifest, store, planner)

    assert result.document.poster_language == "ru"
    assert result.document.stats.page_images_used == 0
    assert "Фотосинтез" in result.document.lesson_text
