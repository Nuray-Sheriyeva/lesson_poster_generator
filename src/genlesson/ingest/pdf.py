"""PDF ingestion: text layer, page rendering, and scanned-page transcription."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..config import IngestOptions
from ..errors import GenLessonError
from ..models import (
    IngestStats,
    LessonDocument,
    PageArtifact,
    SourceInfo,
    utc_now,
)
from ..runs import RunPaths
from ..storage import sha256_bytes, write_bytes_atomic
from ..vision import Transcriber, build_transcriber
from ..vision.cache import TranscriptionCache, VisionCallLog, monotonic_ms


PDF_MAGIC = b"%PDF-"
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class _RawPage:
    index: int
    text: str
    char_count: int
    has_visual: bool
    needs_vision: bool = False
    used_vision: bool = False


@dataclass
class _RenderedPage:
    relative_path: str
    sha256: str
    width: int
    height: int


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    # Preserve paragraph breaks but avoid long runs of empty lines.
    compact: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip():
            compact.append(line)
            blank_run = 0
        else:
            blank_run += 1
            if blank_run <= 1:
                compact.append("")
    return "\n".join(compact).strip()


def count_chars(text: str) -> int:
    return len(_WHITESPACE_RE.sub("", text))


def extract_pdf(
    source: SourceInfo,
    options: IngestOptions,
    *,
    paths: RunPaths,
    stage_dir: Path,
    transcriber: Transcriber | None = None,
) -> LessonDocument:
    pdf_path = Path(source.path)
    if not _looks_like_pdf(pdf_path):
        raise GenLessonError(
            "corrupt_pdf",
            f"File does not start with a PDF header: {source.filename}",
            hint="The file may be renamed or damaged; export it as PDF again.",
        )

    doc = _open_pdf(pdf_path)
    try:
        if doc.page_count == 0:
            raise GenLessonError("empty_pdf", f"PDF has no pages: {source.filename}")
        title = _pdf_title(doc, source)
        raw_pages = _read_pages(doc)
    finally:
        doc.close()

    total_chars = sum(page.char_count for page in raw_pages)
    max_page_chars = max(page.char_count for page in raw_pages)
    has_text_page = max_page_chars >= options.min_page_chars
    scanned = (
        total_chars < options.min_document_chars
        and not has_text_page
        and any(page.has_visual for page in raw_pages)
    )

    warnings: list[str] = []
    for page in raw_pages:
        if scanned:
            page.needs_vision = page.has_visual
        elif page.char_count < options.min_page_chars and page.has_visual:
            page.needs_vision = True

    vision_pages = [page for page in raw_pages if page.needs_vision]
    if vision_pages and transcriber is None:
        if scanned or options.vision_mode == "openai":
            transcriber = build_transcriber(options)
        else:
            for page in vision_pages:
                warnings.append(
                    f"Page {page.index} has visual content but little or no text, and was not "
                    "transcribed because vision is disabled."
                )
                page.needs_vision = False
            vision_pages = []

    rendered = _render_pages(
        doc_path=pdf_path, stage_dir=stage_dir, paths=paths, dpi=options.render_dpi
    )

    if vision_pages and transcriber is not None:
        warnings.extend(
            _transcribe_pages(
                vision_pages,
                raw_pages=raw_pages,
                rendered=rendered,
                transcriber=transcriber,
                stage_dir=stage_dir,
                paths=paths,
            )
        )
        if scanned:
            warnings.append(
                "The PDF has no usable text layer; lesson text comes from vision "
                f"transcription ({transcriber.model})."
            )

    artifacts: list[PageArtifact] = []
    for page in raw_pages:
        image = rendered[page.index]
        if page.used_vision:
            text_source = "vision"
        elif page.text:
            text_source = "text_layer"
        else:
            text_source = "blank"
        artifacts.append(
            PageArtifact(
                index=page.index,
                label=f"Page {page.index}",
                text=page.text,
                text_source=text_source,
                char_count=count_chars(page.text),
                image_path=image.relative_path,
                image_sha256=image.sha256,
                width=image.width,
                height=image.height,
                transcription_model=(
                    transcriber.model if page.used_vision and transcriber else None
                ),
            )
        )

    assembled = "\n\n".join(page.text for page in artifacts if page.text.strip())
    if not assembled:
        raise GenLessonError(
            "no_text",
            f"No extractable text was found in {source.filename}.",
            hint="If this is a scanned PDF, set OPENAI_API_KEY so pages can be transcribed.",
        )

    blank_pages = sum(1 for page in artifacts if page.text_source == "blank")
    stats = IngestStats(
        page_count=len(artifacts),
        text_layer_pages=sum(1 for page in artifacts if page.text_source == "text_layer"),
        vision_pages=sum(1 for page in artifacts if page.text_source == "vision"),
        blank_pages=blank_pages,
        char_count=count_chars(assembled),
        scanned_document=scanned,
    )

    return LessonDocument(
        source=source,
        title=title,
        pages=artifacts,
        text=assembled,
        char_count=stats.char_count,
        warnings=warnings,
        stats=stats,
        config=options.public_config(),
        extracted_at=utc_now(),
    )


def _looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(4096)
    except OSError as exc:
        raise GenLessonError(
            "unreadable_input", f"Could not read PDF: {path}", details={"reason": str(exc)}
        ) from exc
    return PDF_MAGIC in head


def _open_pdf(path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        if _has_encrypt_marker(path):
            raise GenLessonError(
                "encrypted_pdf",
                f"PDF is password-protected: {path.name}",
                hint="Remove the password and try again; passwords are never stored.",
            ) from exc
        raise GenLessonError(
            "corrupt_pdf",
            f"Could not open PDF: {path.name}",
            hint="The PDF may be damaged; export a fresh copy.",
            details={"reason": str(exc)},
        ) from exc
    if doc.needs_pass:
        doc.close()
        raise GenLessonError(
            "encrypted_pdf",
            f"PDF is password-protected: {path.name}",
            hint="Remove the password and try again; passwords are never stored.",
        )
    if not doc.is_pdf:
        doc.close()
        raise GenLessonError("corrupt_pdf", f"File is not a valid PDF: {path.name}")
    return doc


def _has_encrypt_marker(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            return b"/Encrypt" in handle.read()
    except OSError:
        return False


def _pdf_title(doc: pymupdf.Document, source: SourceInfo) -> str:
    metadata = doc.metadata or {}
    title = (metadata.get("title") or "").strip()
    return title or Path(source.filename).stem


def _read_pages(doc: pymupdf.Document) -> list[_RawPage]:
    pages: list[_RawPage] = []
    for index in range(1, doc.page_count + 1):
        try:
            page = doc.load_page(index - 1)
            text = normalize_text(page.get_text("text", sort=True))
            has_visual = _has_visual_content(page)
        except Exception as exc:
            raise GenLessonError(
                "corrupt_pdf",
                f"Could not read page {index} of the PDF.",
                details={"reason": str(exc)},
            ) from exc
        pages.append(
            _RawPage(
                index=index,
                text=text,
                char_count=count_chars(text),
                has_visual=has_visual,
            )
        )
    return pages


def _has_visual_content(page: pymupdf.Page) -> bool:
    """Detect images and vector graphics, including inline images.

    A page with no text layer but visible content is a scanned (or otherwise
    image-only) page that needs transcription. ``get_images`` misses inline
    images and vector line art, so inspect the page's display list instead.
    """
    try:
        blocks = page.get_bboxlog()
    except Exception:
        return bool(page.get_images(full=True))
    return any(
        str(kind).endswith(("image", "path", "shade")) for kind, _bbox in blocks
    )


def _render_pages(
    *,
    doc_path: Path,
    stage_dir: Path,
    paths: RunPaths,
    dpi: int,
) -> dict[int, _RenderedPage]:
    """Render every page, not just pages that need vision.

    The images are part of the run artifacts so a reviewer or a later vision
    step can inspect exactly what the model saw.
    """
    pages_dir = stage_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    output: dict[int, _RenderedPage] = {}

    doc = _open_pdf(doc_path)
    try:
        for index in range(1, doc.page_count + 1):
            page = doc.load_page(index - 1)
            try:
                pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                data = pixmap.tobytes("png")
            except Exception as exc:
                raise GenLessonError(
                    "render_failed",
                    f"Could not render page {index} at {dpi} DPI.",
                    hint="Lower --dpi and resume the run.",
                    details={"reason": str(exc)},
                ) from exc
            image_path = pages_dir / f"page-{index:04d}.png"
            write_bytes_atomic(image_path, data)
            output[index] = _RenderedPage(
                relative_path=paths.relative(image_path),
                sha256=sha256_bytes(data),
                width=pixmap.width,
                height=pixmap.height,
            )
    finally:
        doc.close()
    return output


def _transcribe_pages(
    pages: list[_RawPage],
    *,
    raw_pages: list[_RawPage],
    rendered: dict[int, _RenderedPage],
    transcriber: Transcriber,
    stage_dir: Path,
    paths: RunPaths,
) -> list[str]:
    cache = TranscriptionCache(stage_dir / "transcripts")
    call_log = VisionCallLog(stage_dir / "vision_calls.jsonl")
    warnings: list[str] = []
    by_index = {page.index: page for page in raw_pages}

    for page in pages:
        image = rendered[page.index]
        cached = cache.get(page.index, image_sha256=image.sha256, model=transcriber.model)
        if cached is not None:
            text = cached
        else:
            image_path = paths.resolve(image.relative_path)
            started = monotonic_ms()
            text = transcriber.transcribe(image_path, page.index)
            text = normalize_text(text)
            cache.put(
                page.index,
                image_sha256=image.sha256,
                model=transcriber.model,
                text=text,
            )
            call_log.record(
                page=page.index,
                model=transcriber.model,
                chars=len(text),
                latency_ms=monotonic_ms() - started,
            )
        if not text:
            warnings.append(f"Page {page.index} produced an empty transcription.")
        by_index[page.index].text = text
        by_index[page.index].char_count = count_chars(text)
        by_index[page.index].used_vision = True
    return warnings
