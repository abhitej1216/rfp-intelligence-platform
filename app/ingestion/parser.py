from __future__ import annotations

import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.ingestion.addenda import determine_addendum_relationship
from app.ingestion.security import DEFAULT_MAX_FILE_BYTES, IngestionError, read_source_bytes, relative_source_path, resolve_source_path, sha256_hex
from app.schemas.document import DocumentKind, DocumentRecord, RelationshipStatus, TextSegment

PDF_MAGIC = b"%PDF-"


def _clean_text(value: str) -> str:
    value = value.replace("\x00", "")
    return "\n".join(line for line in (re.sub(r"[ \t]+", " ", part).strip() for part in value.splitlines()) if line).strip()


def _extract_pdf(data: bytes):
    if not data.startswith(PDF_MAGIC):
        raise IngestionError("File extension is PDF but the content signature is invalid.")
    try:
        reader = PdfReader(BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise IngestionError("Encrypted PDFs are not supported.")
        segments: list[TextSegment] = []
        warnings: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = _clean_text(page.extract_text() or "")
            except Exception as exc:  # malformed pages should not discard other pages
                text = ""
                warnings.append(f"Native text extraction failed on PDF page {number} ({type(exc).__name__}).")
            if text:
                segments.append(TextSegment(text=text, page_number=number))
            else:
                warnings.append(f"PDF page {number} yielded no usable text; OCR may be required.")
        metadata = reader.metadata
        title = str(metadata.title).strip() if metadata and metadata.title else None
        return "\n\n".join(segment.text for segment in segments), segments, len(reader.pages), warnings, title or None
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"PDF parsing failed ({type(exc).__name__}).") from exc


def _extract_html(data: bytes):
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            raw = data.decode("windows-1252")
        except UnicodeDecodeError as exc:
            raise IngestionError("HTML encoding could not be decoded safely.") from exc
    soup = BeautifulSoup(raw, "html.parser")
    for element in soup(["script", "style", "noscript", "template", "svg"]):
        element.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = _clean_text(main.get_text("\n", strip=True))
    warnings = ["HTML parsing produced no visible text."] if not text else []
    return text, ([TextSegment(text=text)] if text else []), None, warnings, title or None


def ingest_file(input_root: str | Path, candidate: str | Path, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> DocumentRecord:
    root, candidate_path = Path(input_root), Path(candidate)
    source = resolve_source_path(root, candidate_path)
    data = read_source_bytes(source, max_file_bytes)
    if source.suffix.lower() == ".pdf":
        kind = DocumentKind.PDF
        text, segments, page_count, warnings, title = _extract_pdf(data)
        mime = "application/pdf"
    else:
        kind = DocumentKind.HTML
        text, segments, page_count, warnings, title = _extract_html(data)
        mime = "text/html"
    if not text:
        warnings.append("No text was extracted; downstream extraction must not treat this as evidence.")
    return DocumentRecord(
        source_filename=source.name,
        source_relative_path=relative_source_path(root, source),
        source_sha256=sha256_hex(data),
        source_size_bytes=len(data),
        source_kind=kind,
        source_mime_type=mime,
        ingested_at_utc=datetime.now(timezone.utc),
        title=title,
        text=text,
        segments=segments,
        page_count=page_count,
        extraction_warnings=warnings,
        addendum_relationship=determine_addendum_relationship(source.name, text, {source.name: text}),
    )


def ingest_directory(input_root: str | Path, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> list[DocumentRecord]:
    root = Path(input_root).resolve(strict=True)
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".pdf", ".html", ".htm"})
    records = [ingest_file(root, path, max_file_bytes=max_file_bytes) for path in paths]
    texts = {record.source_filename: record.text for record in records}
    if len(texts) != len(records):
        for record in records:
            if record.addendum_relationship.is_addendum:
                record.addendum_relationship.status = RelationshipStatus.UNRESOLVED
                record.addendum_relationship.bid_identifier = None
                record.addendum_relationship.unresolved_reason = "Duplicate source filenames make relationship matching ambiguous."
        return records
    for record in records:
        record.addendum_relationship = determine_addendum_relationship(record.source_filename, record.text, texts)
    return records
