from __future__ import annotations

import json
from pathlib import Path

import pytest

from genlesson import tracing
from genlesson.config import VerificationOptions
from genlesson.errors import GenLessonError
from genlesson.models import (
    FormatVerification,
    GeneratedFormat,
    GenerationDocument,
    PromptBundle,
    RunManifest,
    SaveDocument,
)
from genlesson.save import _cost, _write_pair, run_save
from genlesson.storage import read_model
from genlesson.verify import run_verify

from .helpers import FakeVisionTranscriber
from .test_verify import _exact_transcript, _generated_run


def _verified_run(
    tmp_path: Path,
    *,
    formats=("portrait", "landscape"),
    fail_formats=None,
    run_id: str = "test-run",
):
    store, paths, manifest, spec = _generated_run(
        tmp_path, formats=formats, fail_formats=fail_formats, run_id=run_id
    )
    vision = FakeVisionTranscriber(_exact_transcript(spec))
    run_verify(
        paths,
        manifest,
        store,
        VerificationOptions(api_key="test-key", similarity_threshold=0.9, max_retries=1),
        transcriber=vision,
    )
    return store, paths, manifest, spec


def _save(paths, manifest, store, out_dir=None):
    return run_save(
        paths,
        manifest,
        store,
        out_dir=out_dir or (paths.run_dir.parent.parent / "outputs"),
    )


def test_save_writes_filenames_and_sidecars(tmp_path: Path) -> None:
    store, paths, manifest, spec = _verified_run(tmp_path)

    result = _save(paths, manifest, store)

    assert len(result.files) == 2
    assert all(path.is_file() for path in result.files)
    assert read_model(paths.manifest_path, RunManifest).stages["save"].status == "complete"
    assert read_model(paths.manifest_path, RunManifest).status == "complete"
    output_dir = tmp_path / "outputs" / "lesson"
    assert Path(result.document.out_dir) == output_dir
    for item in result.document.files:
        assert item.filename == f"lesson-{item.format}.png"
        image_path = Path(item.image_path)
        sidecar_path = Path(item.sidecar_path)
        assert image_path.parent == output_dir
        assert image_path.name == item.filename
        assert sidecar_path.name == item.sidecar_filename
        assert sidecar_path.name == f"lesson-{item.format}.json"
        assert sidecar_path.is_file()
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert "spec" in sidecar
        assert "prompt" in sidecar
        assert "models" in sidecar
        assert "attempts" in sidecar
        assert "verification" in sidecar
        assert "cost" in sidecar
        assert sidecar["cost"]["amount_usd"] is None
        assert sidecar["cost"]["estimated"] is False
        assert sidecar["cost"]["image_calls"] >= 1
        assert sidecar["tracing"]["trace_id"] == tracing.trace_id_for_run(paths.run_id)
        assert sidecar["tracing"]["url"].endswith(
            tracing.trace_id_for_run(paths.run_id)
        )
        assert sidecar["verification"]["status"] == item.verification_status
        assert sidecar["spec"]["central_idea"]["title"] == spec.central_idea.title
        assert "TEXT TO RENDER" in sidecar["prompt"]
    save_doc = read_model(paths.stage_dir("save") / "save.json", SaveDocument)
    assert save_doc.stats.saved == 2
    assert save_doc.stats.skipped == 0


def test_save_needs_attention_is_still_saved(tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    bad = _exact_transcript(spec).replace("Түр 2", "", 1)
    run_verify(
        paths,
        manifest,
        store,
        VerificationOptions(api_key="test-key", similarity_threshold=0.9, max_retries=0),
        transcriber=FakeVisionTranscriber(bad),
    )

    result = _save(paths, manifest, store)

    assert len(result.files) == 1
    assert result.document.files[0].verification_status == "needs_attention"
    assert result.document.stats.needs_attention == 1
    sidecar = json.loads(Path(result.document.files[0].sidecar_path).read_text())
    assert sidecar["verification"]["status"] == "needs_attention"


def test_save_skips_generation_failed_format(tmp_path: Path) -> None:
    store, paths, manifest, spec = _verified_run(
        tmp_path, formats=("portrait", "landscape"), fail_formats={"landscape"}
    )

    result = _save(paths, manifest, store)

    assert len(result.files) == 1
    assert result.document.files[0].format == "portrait"
    assert result.document.skipped[0].format == "landscape"
    assert result.document.stats.skipped == 1
    assert read_model(paths.manifest_path, RunManifest).status == "complete"


def test_save_with_no_images_fails(tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    generation = read_model(
        paths.stage_dir("generate") / "generation.json", GenerationDocument
    )
    for item in generation.formats:
        paths.resolve(item.image_path).unlink()
    run_verify(
        paths,
        manifest,
        store,
        VerificationOptions(api_key="test-key", max_retries=0),
        transcriber=FakeVisionTranscriber(_exact_transcript(spec)),
    )

    with pytest.raises(GenLessonError) as excinfo:
        _save(paths, manifest, store)

    assert excinfo.value.code == "no_posters_saved"
    assert read_model(paths.manifest_path, RunManifest).stages["save"].status == "failed"


def test_save_resume_reuses_files(tmp_path: Path) -> None:
    store, paths, manifest, spec = _verified_run(tmp_path, formats=("portrait",))
    first = _save(paths, manifest, store)
    saved_manifest = read_model(paths.manifest_path, RunManifest)

    second = _save(paths, saved_manifest, store)

    assert second.resumed is True
    assert [item.filename for item in second.document.files] == [
        item.filename for item in first.document.files
    ]


def test_save_filename_collision_adds_counter(tmp_path: Path) -> None:
    store, paths, manifest, spec = _verified_run(tmp_path, formats=("portrait",))
    result = _save(paths, manifest, store)
    record = result.document.files[0]
    prompt = read_model(
        paths.stage_dir("prompts") / "prompts.json", PromptBundle
    ).formats[0]
    png = Path(record.image_path).read_bytes()
    sidecar = {"saved": {}, "verification_sha256": "x"}
    stage_dir = paths.stage_dir("save")

    first_path, _, _ = _write_pair(
        paths,
        stage_dir,
        spec=spec,
        prompt=prompt,
        png=png,
        width=record.width,
        height=record.height,
        timestamp="20261104-153012",
        shortid="a3f9c1",
        sidecar=sidecar,
    )
    second_path, _, _ = _write_pair(
        paths,
        stage_dir,
        spec=spec,
        prompt=prompt,
        png=png,
        width=record.width,
        height=record.height,
        timestamp="20261104-153012",
        shortid="a3f9c1",
        sidecar=sidecar,
    )

    assert first_path.name == "communication-types.png"
    assert second_path.name == "communication-types2.png"
    assert first_path.with_suffix(".json").name == "communication-types.json"
    assert second_path.with_suffix(".json").name == "communication-types2.json"
    assert first_path.is_file() and second_path.is_file()


def test_rerun_uses_same_input_folder_and_increments(tmp_path: Path) -> None:
    store, paths, manifest, spec = _verified_run(tmp_path, formats=("portrait",))
    first = _save(paths, manifest, store)
    first_image = Path(first.document.files[0].image_path)
    first_sidecar = Path(first.document.files[0].sidecar_path)
    assert first_image.parent.name == "lesson"
    assert first_image.name == "lesson.png"
    assert first_sidecar.name == "lesson.json"

    store2, paths2, manifest2, spec2 = _verified_run(
        tmp_path, formats=("portrait",), run_id="test-run-2"
    )
    second = _save(paths2, manifest2, store2)
    second_image = Path(second.document.files[0].image_path)
    second_sidecar = Path(second.document.files[0].sidecar_path)

    assert second_image.parent == first_image.parent
    assert second_image.name == "lesson2.png"
    assert second_sidecar.name == "lesson2.json"
    assert second_image.is_file() and second_sidecar.is_file()


def test_cost_uses_a_generated_estimate() -> None:
    generated = GeneratedFormat(
        format="portrait",
        width=1024,
        height=1536,
        size="1024x1536",
        status="failed",
        attempts=1,
        prompt_sha256="a" * 64,
        cost_usd=0.12,
    )
    verification = FormatVerification(format="portrait", status="needs_attention")

    cost = _cost(verification, generated)

    assert cost.amount_usd == 0.12
    assert cost.estimated is True
    assert "Estimated" in cost.note
