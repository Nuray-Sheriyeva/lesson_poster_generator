from __future__ import annotations

from pathlib import Path

import pytest

from genlesson.errors import GenLessonError
from genlesson.ingest import run_ingest
from genlesson.models import RunManifest
from genlesson.runs import RunPaths
from genlesson.storage import read_model

from .helpers import (
    FakeTranscriber,
    default_options,
    make_encrypted_pdf,
    make_mixed_pdf,
    make_scanned_pdf,
    make_text_pdf,
    start_run,
)


def _manifest(paths: RunPaths) -> RunManifest:
    return read_model(paths.manifest_path, RunManifest)


def test_text_pdf_extracts_text_and_renders_every_page(tmp_path: Path) -> None:
    pdf = make_text_pdf(
        tmp_path / "lesson.pdf",
        [
            "Photosynthesis is the process plants use to turn light into chemical energy.",
            "Chlorophyll absorbs red and blue light and reflects green light.",
        ],
        title="Photosynthesis",
    )
    options = default_options()
    store, paths, manifest = start_run(tmp_path, pdf, options)

    result = run_ingest(paths, manifest, store, options)

    document = result.document
    assert result.resumed is False
    assert document.title == "Photosynthesis"
    assert document.stats.page_count == 2
    assert document.stats.text_layer_pages == 2
    assert document.stats.vision_pages == 0
    assert document.stats.scanned_document is False
    assert "Chlorophyll" in document.text
    assert len(document.pages) == 2
    for page in document.pages:
        assert page.text_source == "text_layer"
        assert page.image_path is not None
        image_path = paths.resolve(page.image_path)
        assert image_path.is_file()
        assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert page.width and page.width > 100

    saved = _manifest(paths)
    assert saved.stages["ingest"].status == "complete"
    assert saved.status == "in_progress"  # later stages are still pending


def test_scanned_pdf_uses_vision_transcript(tmp_path: Path) -> None:
    pdf = make_scanned_pdf(
        tmp_path / "scan.pdf",
        [
            "САБАҚ: Фотосинтез",
            "Өсімдіктер жарықтан энергия алады.",
        ],
    )
    fake = FakeTranscriber(
        {
            1: "САБАҚ: Фотосинтез",
            2: "Өсімдіктер жарықтан энергия алады.",
        }
    )
    options = default_options(vision_mode="openai", openai_api_key="test-key")
    store, paths, manifest = start_run(tmp_path, pdf, options)

    result = run_ingest(paths, manifest, store, options, transcriber=fake)

    document = result.document
    assert fake.calls == [1, 2]
    assert document.stats.scanned_document is True
    assert document.stats.vision_pages == 2
    assert document.stats.text_layer_pages == 0
    assert document.text == (
        "САБАҚ: Фотосинтез\n\n"
        "Өсімдіктер жарықтан энергия алады."
    )
    assert all(page.text_source == "vision" for page in document.pages)
    assert all(page.transcription_model == "fake-vision" for page in document.pages)
    assert any("vision transcription" in warning for warning in document.warnings)
    for page in document.pages:
        assert page.image_path is not None
        assert paths.resolve(page.image_path).is_file()

    cache_files = sorted((paths.stage_dir("ingest") / "transcripts").glob("page-*.txt"))
    assert len(cache_files) == 2


def test_mixed_pdf_transcribes_only_pages_without_text(tmp_path: Path) -> None:
    pdf = make_mixed_pdf(
        tmp_path / "mixed.pdf",
        text_page=(
            "This page has a real text layer with enough characters to avoid OCR. "
            "It describes the wider lesson context."
        ),
        scanned_page="Scanned diagram: light reactions happen in the thylakoid membrane.",
    )
    fake = FakeTranscriber(
        {2: "Scanned diagram: light reactions happen in the thylakoid membrane."}
    )
    options = default_options(vision_mode="openai", openai_api_key="test-key")
    store, paths, manifest = start_run(tmp_path, pdf, options)

    result = run_ingest(paths, manifest, store, options, transcriber=fake)

    document = result.document
    assert fake.calls == [2]
    assert document.stats.scanned_document is False
    assert document.stats.text_layer_pages == 1
    assert document.stats.vision_pages == 1
    assert document.pages[0].text_source == "text_layer"
    assert document.pages[1].text_source == "vision"
    assert "thylakoid" in document.text


def test_scanned_pdf_with_vision_disabled_fails_clearly(tmp_path: Path) -> None:
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", ["No text layer here."])
    options = default_options(vision_mode="none")
    store, paths, manifest = start_run(tmp_path, pdf, options)

    with pytest.raises(GenLessonError) as excinfo:
        run_ingest(paths, manifest, store, options)

    assert excinfo.value.code == "vision_required"
    assert "OPENAI_API_KEY" in (excinfo.value.hint or "")
    saved = _manifest(paths)
    assert saved.stages["ingest"].status == "failed"
    assert saved.stages["ingest"].error is not None
    assert not (paths.stage_dir("ingest") / "lesson.json").exists()


def test_openai_mode_without_key_is_fine_for_text_pdf(tmp_path: Path) -> None:
    pdf = make_text_pdf(
        tmp_path / "lesson.pdf",
        ["A lesson with a real text layer does not need a vision call."],
    )
    options = default_options(vision_mode="openai", openai_api_key=None)
    store, paths, manifest = start_run(tmp_path, pdf, options)

    result = run_ingest(paths, manifest, store, options)

    assert result.document.stats.vision_pages == 0


def test_resume_reuses_cached_transcriptions(tmp_path: Path) -> None:
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", ["Page one text.", "Page two text."])
    options = default_options(vision_mode="openai", openai_api_key="test-key")
    store, paths, manifest = start_run(tmp_path, pdf, options)

    first = FakeTranscriber({1: "Page one text.", 2: "Page two text."}, model="fake-vision")
    first.fail_on = {2}
    with pytest.raises(GenLessonError):
        run_ingest(paths, manifest, store, options, transcriber=first)
    assert first.calls == [1, 2]
    assert _manifest(paths).stages["ingest"].status == "failed"
    assert (paths.stage_dir("ingest") / "transcripts" / "page-0001.txt").is_file()

    saved = _manifest(paths)
    second = FakeTranscriber({1: "Page one text.", 2: "Page two text."}, model="fake-vision")
    result = run_ingest(paths, saved, store, options, transcriber=second)

    assert second.calls == [2]  # page 1 came from the cache
    assert result.document.text == "Page one text.\n\nPage two text."
    assert _manifest(paths).stages["ingest"].status == "complete"


def test_resuming_a_complete_ingest_is_idempotent(tmp_path: Path) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["Already ingested lesson text."])
    options = default_options()
    store, paths, manifest = start_run(tmp_path, pdf, options)
    first = run_ingest(paths, manifest, store, options)

    second = run_ingest(paths, manifest, store, options)

    assert second.resumed is True
    assert second.document.text == first.document.text
    assert second.output_path == first.output_path


def test_encrypted_pdf_reports_encrypted(tmp_path: Path) -> None:
    pdf = make_encrypted_pdf(tmp_path / "secret.pdf")
    options = default_options()
    store, paths, manifest = start_run(tmp_path, pdf, options)

    with pytest.raises(GenLessonError) as excinfo:
        run_ingest(paths, manifest, store, options)

    assert excinfo.value.code == "encrypted_pdf"
    assert _manifest(paths).stages["ingest"].status == "failed"


def test_corrupt_pdf_reports_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.7\nthis is not really a pdf")
    options = default_options()
    store, paths, manifest = start_run(tmp_path, bad, options)

    with pytest.raises(GenLessonError) as excinfo:
        run_ingest(paths, manifest, store, options)

    assert excinfo.value.code in {"corrupt_pdf", "empty_pdf"}
    assert "PDF" in excinfo.value.message


def test_blank_pdf_reports_no_text(tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    import pymupdf

    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    doc.save(pdf)
    doc.close()

    options = default_options()
    store, paths, manifest = start_run(tmp_path, pdf, options)

    with pytest.raises(GenLessonError) as excinfo:
        run_ingest(paths, manifest, store, options)

    assert excinfo.value.code == "no_text"
