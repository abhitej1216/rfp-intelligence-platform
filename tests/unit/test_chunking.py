
from datetime import datetime, timezone

from app.retrieval.chunking import (
    _split_text,
    chunk_document,
)
from app.schemas.document import (
    AddendumRelationship,
    DocumentKind,
    DocumentRecord,
    RelationshipStatus,
    TextSegment,
)


def make_document(
    text: str,
    *,
    segments: list[TextSegment] | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        source_filename="example.pdf",
        source_relative_path="bid-a/example.pdf",
        source_sha256="a" * 64,
        source_size_bytes=100,
        source_kind=DocumentKind.PDF,
        source_mime_type="application/pdf",
        ingested_at_utc=datetime.now(timezone.utc),
        text=text,
        segments=segments or [],
        page_count=1,
        addendum_relationship=AddendumRelationship(
            is_addendum=False,
            status=RelationshipStatus.NOT_AN_ADDENDUM,
        ),
    )


def test_short_document_produces_one_chunk():
    document = make_document("A short RFP passage.")

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].text == "A short RFP passage."


def test_long_text_is_split_with_overlap():
    text = "word " * 500

    chunks = _split_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 100 for chunk in chunks)


def test_page_numbers_are_preserved():
    document = make_document(
        "Page one text. Page two text.",
        segments=[
            TextSegment(text="Page one text.", page_number=1),
            TextSegment(text="Page two text.", page_number=2),
        ],
    )

    chunks = chunk_document(document)

    assert [chunk.page_number for chunk in chunks] == [1, 2]


def test_chunk_ids_are_deterministic():
    document = make_document("Stable identifiers.")

    first = chunk_document(document)
    second = chunk_document(document)

    assert [c.chunk_id for c in first] == [
        c.chunk_id for c in second
    ]


def test_empty_document_produces_no_chunks():
    document = make_document("")

    assert chunk_document(document) == []
