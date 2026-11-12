"""Typed, user-facing errors for the pipeline.

The CLI is the only place that turns these into output. Library callers can catch
``GenLessonError`` and inspect ``code`` without parsing prose.
"""

from __future__ import annotations

from typing import Any


class GenLessonError(Exception):
    """An expected failure with an actionable message.

    ``code`` is stable and machine-readable; ``message`` is for humans; ``hint``
    suggests a fix. ``details`` is optional structured context for logs/tests.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message} ({self.hint})"
        return self.message


def invalid_arguments(message: str, *, hint: str | None = None) -> GenLessonError:
    return GenLessonError("invalid_arguments", message, hint=hint)


def unsupported_file_type(suffix: str) -> GenLessonError:
    shown = suffix or "(none)"
    return GenLessonError(
        "unsupported_file_type",
        f"Unsupported input type {shown!r}. Stage 1 accepts PDF (.pdf) and Word (.docx).",
        hint="Convert the lesson to PDF or DOCX and try again.",
    )


def input_not_found(path: str) -> GenLessonError:
    return GenLessonError(
        "input_not_found",
        f"Input file does not exist: {path}",
    )


def run_exists(run_dir: str) -> GenLessonError:
    return GenLessonError(
        "run_exists",
        f"Run directory already exists: {run_dir}",
        hint="Choose a new --run-id, or pass --resume to continue the existing run.",
    )


def input_changed(expected: str, actual: str) -> GenLessonError:
    return GenLessonError(
        "input_changed",
        "The input file changed since this run was created; refusing to mix artifacts.",
        hint="Start a new run for the current file.",
        details={"expected_sha256": expected, "actual_sha256": actual},
    )
