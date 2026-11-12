from __future__ import annotations

from pathlib import Path

from genlesson.ingest import run_ingest
from genlesson.models import LessonDocument
from genlesson.storage import read_model

from .helpers import FakeTranscriber, default_options, start_run

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "input"


def test_russian_sample_docx(tmp_path: Path) -> None:
    path = FIXTURES / "russian_sample.docx"
    options = default_options(vision_mode="none")
    store, paths, manifest = start_run(tmp_path, path, options)

    run_ingest(paths, manifest, store, options)

    lesson = read_model(paths.stage_dir("ingest") / "lesson.json", LessonDocument)
    text = "\n".join(page.text for page in lesson.pages)
    assert "Фотосинтез" in text
    assert "Хлорофилл" in text


def test_scanned_sample_pdf_uses_vision(tmp_path: Path) -> None:
    path = FIXTURES / "scanned_sample.pdf"
    options = default_options(vision_mode="openai", openai_api_key="test-key")
    store, paths, manifest = start_run(tmp_path, path, options)
    transcribed = "Фотосинтез туралы қысқа сабақ мәтіні. Өсімдіктер жарықтан энергия алады."

    run_ingest(
        paths,
        manifest,
        store,
        options,
        transcriber=FakeTranscriber({1: transcribed}),
    )

    lesson = read_model(paths.stage_dir("ingest") / "lesson.json", LessonDocument)
    assert lesson.stats.scanned_document is True
    assert lesson.pages[0].text_source == "vision"
    assert transcribed in lesson.pages[0].text
