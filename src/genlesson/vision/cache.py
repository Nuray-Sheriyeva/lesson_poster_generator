"""Transcription cache so resuming a run does not pay for the same page twice."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..models import utc_now
from ..storage import read_json, write_json_atomic, write_text_atomic


class TranscriptionCache:
    """Cache keyed by page image hash and model name.

    If the source changes, the image hash changes and the cached text is ignored.
    If the model changes, the cache is ignored too, so switching OCR models does
    not silently mix outputs.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _stem(self, page_number: int) -> str:
        return f"page-{page_number:04d}"

    def get(self, page_number: int, *, image_sha256: str, model: str) -> str | None:
        stem = self._stem(page_number)
        text_path = self.directory / f"{stem}.txt"
        meta_path = self.directory / f"{stem}.json"
        if not text_path.is_file() or not meta_path.is_file():
            return None
        try:
            meta = read_json(meta_path)
            if meta.get("image_sha256") != image_sha256 or meta.get("model") != model:
                return None
            return text_path.read_text(encoding="utf-8")
        except Exception:
            return None

    def put(
        self,
        page_number: int,
        *,
        image_sha256: str,
        model: str,
        text: str,
    ) -> None:
        stem = self._stem(page_number)
        write_text_atomic(self.directory / f"{stem}.txt", text)
        write_json_atomic(
            self.directory / f"{stem}.json",
            {
                "page": page_number,
                "image_sha256": image_sha256,
                "model": model,
                "chars": len(text),
                "created_at": utc_now(),
            },
        )

    def clear(self, page_number: int) -> None:
        stem = self._stem(page_number)
        (self.directory / f"{stem}.txt").unlink(missing_ok=True)
        (self.directory / f"{stem}.json").unlink(missing_ok=True)


class VisionCallLog:
    """Tiny append-only log of actual model calls, useful for cost auditing."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, *, page: int, model: str, chars: int, latency_ms: int) -> None:
        entry = {
            "page": page,
            "model": model,
            "chars": chars,
            "latency_ms": latency_ms,
            "at": utc_now(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)
