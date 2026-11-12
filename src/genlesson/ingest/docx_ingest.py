"""DOCX ingestion with ordered paragraphs and tables."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterator

from docx import Document
from docx.document import Document as DocumentObject
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..config import IngestOptions
from ..errors import GenLessonError
from ..models import IngestStats, LessonDocument, PageArtifact, SourceInfo, utc_now
from .pdf import count_chars, normalize_text

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_WHITESPACE_RE = re.compile(r"\s+")


def extract_docx(
    source: SourceInfo,
    options: IngestOptions,
) -> LessonDocument:
    path = Path(source.path)
    _validate_docx_container(path)

    try:
        document = Document(str(path))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise GenLessonError(
            "corrupt_docx",
            f"Could not open Word document: {source.filename}",
            hint="Re-save the file as .docx and try again.",
            details={"reason": str(exc)},
        ) from exc
    except OSError as exc:
        raise GenLessonError(
            "unreadable_input",
            f"Could not read Word document: {source.filename}",
            details={"reason": str(exc)},
        ) from exc

    try:
        blocks = list(_iter_blocks(document))
    except Exception as exc:
        raise GenLessonError(
            "corrupt_docx",
            f"Could not read the contents of {source.filename}.",
            details={"reason": str(exc)},
        ) from exc

    text_blocks: list[str] = []
    for block in blocks:
        if isinstance(block, Paragraph):
            text = normalize_text(block.text)
            if text:
                text_blocks.append(text)
        elif isinstance(block, Table):
            table_text = _table_text(block)
            if table_text:
                text_blocks.append(table_text)

    text = "\n\n".join(text_blocks).strip()
    if not text:
        raise GenLessonError(
            "no_text",
            f"No text was found in {source.filename}.",
            hint="A Word document containing only images needs to be exported as PDF first.",
        )

    title = _docx_title(document, source)
    page = PageArtifact(
        index=1,
        label="Document",
        text=text,
        text_source="text_layer",
        char_count=count_chars(text),
        image_path=None,
        transcription_model=None,
    )
    stats = IngestStats(
        page_count=1,
        text_layer_pages=1,
        vision_pages=0,
        blank_pages=0,
        char_count=count_chars(text),
        scanned_document=False,
    )
    return LessonDocument(
        source=source,
        title=title,
        pages=[page],
        text=text,
        char_count=stats.char_count,
        warnings=["DOCX has no fixed pages; it was ingested as one text segment."],
        stats=stats,
        config=options.public_config(),
        extracted_at=utc_now(),
    )


def _validate_docx_container(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError as exc:
        raise GenLessonError(
            "unreadable_input",
            f"Could not read Word document: {path.name}",
            details={"reason": str(exc)},
        ) from exc

    if head.startswith(_OLE_MAGIC):
        raise GenLessonError(
            "encrypted_docx",
            f"Word document is password-protected or uses the legacy .doc format: {path.name}",
            hint="Remove the password and save as .docx, then try again.",
        )
    if not head.startswith(b"PK"):
        raise GenLessonError(
            "corrupt_docx",
            f"File is not a valid .docx container: {path.name}",
            hint="Re-export the lesson as .docx or PDF.",
        )


def _iter_blocks(document: DocumentObject) -> Iterator[Paragraph | Table]:
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _table_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            value = _WHITESPACE_RE.sub(" ", cell.text).strip()
            cells.append(value)
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _docx_title(document: DocumentObject, source: SourceInfo) -> str:
    try:
        title = (document.core_properties.title or "").strip()
    except Exception:
        title = ""
    return title or Path(source.filename).stem
