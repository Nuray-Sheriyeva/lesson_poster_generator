"""Small filesystem helpers with atomic, crash-safe writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .errors import GenLessonError

T = TypeVar("T", bound=BaseModel)
_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GenLessonError(
            "unreadable_input", f"Could not read input file: {path}", details={"reason": str(exc)}
        ) from exc
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes to ``path`` atomically, replacing an existing artifact.

    Callers must ensure ``path`` is an internal stage artifact. Final posters use
    an exclusive write instead (see ``write_bytes_exclusive``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    _atomic_write(path, data)


def write_text_atomic(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"))


def write_json_atomic(path: Path, data: Any) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    _atomic_write(path, payload.encode("utf-8"))


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise GenLessonError("artifact_missing", f"Expected artifact is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise GenLessonError(
            "artifact_unreadable",
            f"Could not read artifact: {path}",
            details={"reason": str(exc)},
        ) from exc


def read_model(path: Path, model: type[T]) -> T:
    try:
        return model.model_validate(read_json(path))
    except GenLessonError:
        raise
    except Exception as exc:  # pydantic.ValidationError and friends
        raise GenLessonError(
            "artifact_invalid",
            f"Artifact does not match its schema: {path}",
            details={"reason": str(exc)},
        ) from exc


def write_bytes_exclusive(path: Path, data: bytes) -> None:
    """Create a new file and never replace an existing one.

    This is the primitive for final deliverables such as posters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise GenLessonError(
            "output_exists",
            f"Refusing to overwrite an existing file: {path}",
        ) from exc
    _fsync_dir(path.parent)
