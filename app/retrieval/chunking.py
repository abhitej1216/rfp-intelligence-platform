
from __future__ import annotations

import hashlib

from app.schemas.chunk import ChunkRecord
from app.schemas.document import DocumentRecord, TextSegment

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def _stable_id(*parts: str) -> str:
    """Create a deterministic SHA-256 identifier."""
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text with overlap, preferring whitespace boundaries."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        next_start = max(start + 1, end - chunk_overlap)

        # Avoid starting the next chunk in the middle of a word.
        whitespace = text.find(" ", next_start, end)
        if whitespace != -1:
            next_start = whitespace + 1

        start = next_start

    return chunks


def _document_segments(document: DocumentRecord) -> list[TextSegment]:
    """Use page-aware segments where available."""
    if document.segments:
        return document.segments

    if document.text.strip():
        return [TextSegment(text=document.text.strip())]

    return []


def chunk_document(
    document: DocumentRecord,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    """Convert one ingested document into traceable search chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    document_id = _stable_id(
        document.source_relative_path,
        document.source_sha256,
    )

    relationship = document.addendum_relationship
    chunks: list[ChunkRecord] = []
    chunk_index = 0

    for segment in _document_segments(document):
        pieces = _split_text(
            segment.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for piece in pieces:
            chunk_id = _stable_id(
                document_id,
                str(chunk_index),
                str(segment.page_number or ""),
                piece,
            )

            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_filename=document.source_filename,
                    source_relative_path=document.source_relative_path,
                    source_sha256=document.source_sha256,
                    source_kind=document.source_kind.value,
                    title=document.title,
                    text=piece,
                    chunk_index=chunk_index,
                    page_number=segment.page_number,
                    section_hint=segment.section_hint,
                    bid_identifier=relationship.bid_identifier,
                    is_addendum=relationship.is_addendum,
                    addendum_status=relationship.status.value,
                )
            )
            chunk_index += 1

    return chunks


def chunk_documents(
    documents: list[DocumentRecord],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    """Chunk multiple documents independently."""
    chunks: list[ChunkRecord] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks
