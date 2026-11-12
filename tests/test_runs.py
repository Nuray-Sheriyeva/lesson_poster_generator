from __future__ import annotations

import json
from pathlib import Path

import pytest

from genlesson.config import manifest_config
from genlesson.errors import GenLessonError
from genlesson.models import RunManifest
from genlesson.runs import RunStore, describe_source
from genlesson.storage import read_model, write_bytes_exclusive

from .helpers import default_options, make_text_pdf


def test_manifest_has_stable_stage_order_and_never_contains_key(tmp_path: Path) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["Lesson text."])
    options = default_options(openai_api_key="super-secret")
    config = manifest_config(options)
    serialized = json.dumps(config)
    assert "super-secret" not in serialized
    assert config["review_gate"] is False
    assert config["formats"] == ["portrait"]

    store = RunStore(tmp_path / "runs")
    paths, manifest = store.create(describe_source(pdf), config, run_id="stable")
    assert list(manifest.stages) == [
        "ingest",
        "classify",
        "plan",
        "review",
        "prompts",
        "generate",
        "verify",
        "save",
    ]
    assert all(record.status == "pending" for record in manifest.stages.values())


def test_explicit_run_is_never_reused(tmp_path: Path) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["Lesson text."])
    store = RunStore(tmp_path / "runs")
    source = describe_source(pdf)
    config = manifest_config(default_options())
    store.create(source, config, run_id="one")

    with pytest.raises(GenLessonError) as excinfo:
        store.create(source, config, run_id="one")

    assert excinfo.value.code == "run_exists"


def test_generated_run_ids_are_unique(tmp_path: Path) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["Lesson text."])
    store = RunStore(tmp_path / "runs")
    source = describe_source(pdf)
    config = manifest_config(default_options())

    first, _ = store.create(source, config)
    second, _ = store.create(source, config)

    assert first.run_id != second.run_id
    assert first.run_dir.is_dir() and second.run_dir.is_dir()


def test_artifacts_cannot_escape_the_run_directory(tmp_path: Path) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["Lesson text."])
    store = RunStore(tmp_path / "runs")
    paths, _ = store.create(describe_source(pdf), manifest_config(default_options()), run_id="safe")

    with pytest.raises(GenLessonError) as excinfo:
        paths.resolve("../outside.txt")
    assert excinfo.value.code == "path_outside_run"


def test_exclusive_write_never_replaces(tmp_path: Path) -> None:
    target = tmp_path / "poster.png"
    write_bytes_exclusive(target, b"first")
    with pytest.raises(GenLessonError) as excinfo:
        write_bytes_exclusive(target, b"second")
    assert excinfo.value.code == "output_exists"
    assert target.read_bytes() == b"first"


def test_describe_source_rejects_unsupported_and_missing(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("not supported", encoding="utf-8")
    with pytest.raises(GenLessonError) as excinfo:
        describe_source(notes)
    assert excinfo.value.code == "unsupported_file_type"

    with pytest.raises(GenLessonError) as excinfo:
        describe_source(tmp_path / "missing.pdf")
    assert excinfo.value.code == "input_not_found"


def test_status_of_new_run_is_created(tmp_path: Path) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["Lesson text."])
    store = RunStore(tmp_path / "runs")
    paths, manifest = store.create(
        describe_source(pdf), manifest_config(default_options()), run_id="new"
    )
    reloaded = read_model(paths.manifest_path, RunManifest)
    assert reloaded.status == "created"
    assert paths.run_id == "new"
