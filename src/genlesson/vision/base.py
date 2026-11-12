"""Vision transcription boundary.

Stage 1 only needs page-level OCR for PDFs without a usable text layer. Keeping
this a protocol means the pipeline can be tested without network calls and a
different vision provider can be added later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Transcriber(Protocol):
    model: str

    def transcribe(self, image_path: Path, page_number: int) -> str:
        """Return the page text verbatim, in the page's original language."""
        ...
