"""Stage 1 orchestration: extract, render, transcribe, and persist."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import IngestOptions
from ..errors import GenLessonError, input_changed
from ..models import LessonDocument, RunManifest
from ..runs import RunPaths, RunStore
from ..storage import read_model, sha256_file, write_json_atomic
from ..vision import Transcriber
from .docx_ingest import extract_docx
from .pdf import extract_pdf

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document: LessonDocument
    output_path: Path
    resumed: bool


def run_ingest(
    paths: RunPaths,
    manifest: RunManifest,
    store: RunStore,
    options: IngestOptions,
    *,
    transcriber: Transcriber | None = None,
) -> IngestResult:
    source = manifest.source
    if source is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing source metadata.")

    stage = manifest.stages.get("ingest")
    if stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the ingest stage.")
    output_path = paths.stage_dir("ingest") / "lesson.json"

    if stage.status == "complete":
        document = read_model(output_path, LessonDocument)
        if document.source.sha256 != source.sha256:
            raise GenLessonError(
                "artifact_invalid",
                "lesson.json does not match the source recorded in the manifest.",
            )
        return IngestResult(document=document, output_path=output_path, resumed=True)

    current_hash = sha256_file(Path(source.path))
    if current_hash != source.sha256:
        raise input_changed(source.sha256, current_hash)

    stage_dir = paths.stage_dir("ingest")
    stage_dir.mkdir(parents=True, exist_ok=True)
    store.set_stage(paths, manifest, "ingest", "running")

    try:
        if source.media_type == "pdf":
            document = extract_pdf(
                source,
                options,
                paths=paths,
                stage_dir=stage_dir,
                transcriber=transcriber,
            )
        elif source.media_type == "docx":
            document = extract_docx(source, options)
        else:
            raise GenLessonError(
                "unsupported_file_type",
                f"Unsupported input type: {source.media_type}",
            )
        write_json_atomic(output_path, document.model_dump(mode="json"))
    except GenLessonError as exc:
        store.set_stage(paths, manifest, "ingest", "failed", error=exc.to_dict())
        raise
    except Exception as exc:
        log.exception("Unexpected ingestion failure")
        wrapped = GenLessonError(
            "internal_error",
            f"Ingestion failed unexpectedly: {exc}",
            details={"exception": type(exc).__name__},
        )
        store.set_stage(paths, manifest, "ingest", "failed", error=wrapped.to_dict())
        raise wrapped from exc

    store.set_stage(
        paths,
        manifest,
        "ingest",
        "complete",
        output=paths.relative(output_path),
    )
    return IngestResult(document=document, output_path=output_path, resumed=False)
