from __future__ import annotations

from pathlib import Path

import pytest

from genlesson.errors import GenLessonError
from genlesson.ingest import run_ingest
from genlesson.models import RunManifest
from genlesson.storage import read_model

from .helpers import default_options, make_docx, start_run


def test_docx_extracts_paragraphs_and_tables_in_order(tmp_path: Path) -> None:
    path = make_docx(tmp_path / "lesson.docx", title="Photosynthesis")
    options = default_options()
    store, paths, manifest = start_run(tmp_path, path, options)

    result = run_ingest(paths, manifest, store, options)

    document = result.document
    assert document.title == "Photosynthesis"
    assert document.stats.page_count == 1
    assert document.stats.text_layer_pages == 1
    assert document.stats.scanned_document is False
    assert document.pages[0].image_path is None
    assert document.pages[0].label == "Document"
    assert "Photosynthesis" in document.text
    assert "Input | Output" in document.text
    assert "Sunlight | Glucose" in document.text
    assert "фотосинтез" in document.text
    # Paragraph order: heading, explanation, table, closing paragraph.
    assert document.text.index("Plants turn light") < document.text.index("Input | Output")
    assert document.text.index("Sunlight | Glucose") < document.text.index("фотосинтез")

    saved = read_model(paths.manifest_path, RunManifest)
    assert saved.stages["ingest"].status == "complete"


def test_docx_without_text_fails(tmp_path: Path) -> None:
    path = tmp_path / "empty.docx"
    from docx import Document

    Document().save(path)
    options = default_options()
    store, paths, manifest = start_run(tmp_path, path, options)

    with pytest.raises(GenLessonError) as excinfo:
        run_ingest(paths, manifest, store, options)

    assert excinfo.value.code == "no_text"
    assert read_model(paths.manifest_path, RunManifest).stages["ingest"].status == "failed"


def test_corrupt_docx_reports_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip archive")
    options = default_options()
    store, paths, manifest = start_run(tmp_path, path, options)

    with pytest.raises(GenLessonError) as excinfo:
        run_ingest(paths, manifest, store, options)

    assert excinfo.value.code == "corrupt_docx"


def test_legacy_or_encrypted_word_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "secret.docx"
    path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64)
    options = default_options()
    store, paths, manifest = start_run(tmp_path, path, options)

    with pytest.raises(GenLessonError) as excinfo:
        run_ingest(paths, manifest, store, options)

    assert excinfo.value.code == "encrypted_docx"
