"""Run directories and the manifest that makes stages resumable."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import STAGES, stage_dir_name
from .errors import GenLessonError, input_not_found, run_exists, unsupported_file_type
from .models import (
    MediaType,
    RunManifest,
    RunStatus,
    SourceInfo,
    StageRecord,
    StageStatus,
    utc_now,
)
from .storage import read_model, sha256_file, write_json_atomic

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_SUFFIXES: dict[str, MediaType] = {".pdf": "pdf", ".docx": "docx"}


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise GenLessonError(
            "run_invalid_id",
            f"Invalid run id: {run_id!r}",
            hint="Use letters, digits, dots, underscores, or dashes (max 100 chars).",
        )
    return run_id


def media_type_for(path: Path) -> MediaType:
    suffix = path.suffix.lower()
    try:
        return _SUFFIXES[suffix]
    except KeyError:
        raise unsupported_file_type(suffix) from None


def describe_source(path: Path) -> SourceInfo:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise input_not_found(str(path))
    if not resolved.is_file():
        raise GenLessonError("input_not_file", f"Input is not a regular file: {resolved}")
    stat = resolved.stat()
    if stat.st_size == 0:
        raise GenLessonError("empty_file", f"Input file is empty: {resolved}")
    media_type = media_type_for(resolved)
    return SourceInfo(
        path=str(resolved),
        filename=resolved.name,
        media_type=media_type,
        sha256=sha256_file(resolved),
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    )


def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit].rstrip("-")) or "lesson"


@dataclass(frozen=True)
class RunPaths:
    runs_root: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def stage_dir(self, stage: str) -> Path:
        return self.run_dir / stage_dir_name(stage)

    def relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.run_dir.resolve()).as_posix()
        except ValueError as exc:
            raise GenLessonError(
                "path_outside_run",
                f"Refusing to reference a path outside the run: {path}",
            ) from exc

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.run_dir / relative_path).resolve()
        if not candidate.is_relative_to(self.run_dir.resolve()):
            raise GenLessonError(
                "path_outside_run",
                f"Refusing to resolve a path outside the run: {relative_path}",
            )
        return candidate


class RunStore:
    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root.expanduser().resolve()

    def create(
        self,
        source: SourceInfo,
        config: dict,
        *,
        run_id: str | None = None,
    ) -> tuple[RunPaths, RunManifest]:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        if run_id is not None:
            paths = RunPaths(self.runs_root, validate_run_id(run_id))
            try:
                paths.run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise run_exists(str(paths.run_dir)) from exc
        else:
            paths = self._create_unique(source.filename)

        now = utc_now()
        manifest = RunManifest(
            run_id=paths.run_id,
            created_at=now,
            updated_at=now,
            status="created",
            source=source,
            config=config,
            stages={stage.name: StageRecord() for stage in STAGES},
        )
        write_json_atomic(paths.manifest_path, manifest.model_dump(mode="json"))
        return paths, manifest

    def _create_unique(self, filename: str) -> RunPaths:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{_slug(Path(filename).stem)}"
        for attempt in range(1, 101):
            candidate = base if attempt == 1 else f"{base}-{attempt:02d}"
            paths = RunPaths(self.runs_root, candidate)
            try:
                paths.run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return paths
        raise GenLessonError(
            "run_id_exhausted",
            "Could not allocate a unique run directory; pass an explicit --run-id.",
        )

    def exists(self, run_id: str) -> bool:
        return RunPaths(self.runs_root, validate_run_id(run_id)).run_dir.is_dir()

    def load(self, run_id: str) -> tuple[RunPaths, RunManifest]:
        paths = RunPaths(self.runs_root, validate_run_id(run_id))
        if not paths.run_dir.is_dir():
            raise GenLessonError(
                "run_not_found",
                f"Run does not exist: {paths.run_dir}",
                hint="Check --runs-dir and the run id.",
            )
        manifest = read_model(paths.manifest_path, RunManifest)
        if manifest.run_id != run_id:
            raise GenLessonError(
                "artifact_invalid",
                f"Manifest run id {manifest.run_id!r} does not match directory {run_id!r}.",
            )
        return paths, manifest

    def save(self, paths: RunPaths, manifest: RunManifest) -> None:
        manifest.updated_at = utc_now()
        manifest.status = _run_status(manifest)
        write_json_atomic(paths.manifest_path, manifest.model_dump(mode="json"))

    def set_stage(
        self,
        paths: RunPaths,
        manifest: RunManifest,
        stage: str,
        status: StageStatus,
        *,
        output: str | None = None,
        error: dict | None = None,
    ) -> None:
        if stage not in manifest.stages:
            raise GenLessonError("unknown_stage", f"Unknown stage: {stage!r}")
        record = manifest.stages[stage]
        record.status = status
        if status in {"running", "awaiting_review"}:
            record.started_at = utc_now()
            record.finished_at = None
            record.error = None
        elif status in {"complete", "failed", "skipped"}:
            if record.started_at is None:
                record.started_at = utc_now()
            record.finished_at = utc_now()
        if output is not None:
            record.output = output
        if error is not None:
            record.error = error
        self.save(paths, manifest)


def _run_status(manifest: RunManifest) -> RunStatus:
    statuses = [record.status for record in manifest.stages.values()]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "awaiting_review" for status in statuses):
        return "awaiting_review"
    if all(status in {"complete", "skipped"} for status in statuses):
        return "complete"
    if any(status in {"running", "complete"} for status in statuses):
        return "in_progress"
    return "created"
