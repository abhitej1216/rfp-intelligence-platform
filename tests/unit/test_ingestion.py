from pathlib import Path

import pytest

from app.ingestion.parser import ingest_directory, ingest_file
from app.ingestion.security import IngestionError, resolve_source_path


def test_html_extracts_text_and_metadata(tmp_path: Path):
    (tmp_path / "bid.html").write_text(
        "<html><head><title>Example RFP</title></head><body><script>ignore_me()</script>"
        "<main><h1>Bid 12345</h1><p>Submission due June 1.</p></main></body></html>",
        encoding="utf-8",
    )
    record = ingest_file(tmp_path, "bid.html")
    assert record.title == "Example RFP"
    assert "Bid 12345" in record.text and "Submission due June 1." in record.text
    assert "ignore_me" not in record.text
    assert len(record.source_sha256) == 64
    assert record.source_relative_path == "bid.html"


def test_invalid_pdf_signature_rejected(tmp_path: Path):
    (tmp_path / "fake.pdf").write_bytes(b"not a pdf")
    with pytest.raises(IngestionError, match="signature"):
        ingest_file(tmp_path, "fake.pdf")


def test_unsupported_file_rejected(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("not supported", encoding="utf-8")
    with pytest.raises(IngestionError, match="Unsupported"):
        ingest_file(tmp_path, "notes.txt")


def test_path_outside_root_rejected(tmp_path: Path):
    root = tmp_path / "input"
    root.mkdir()
    outside = tmp_path / "secret.html"
    outside.write_text("<p>private</p>", encoding="utf-8")
    with pytest.raises(IngestionError, match="outside"):
        resolve_source_path(root, outside)


def test_addendum_without_matching_id_is_unresolved(tmp_path: Path):
    (tmp_path / "Addendum 1.html").write_text(
        "<html><body><p>Changes to submission instructions.</p></body></html>",
        encoding="utf-8",
    )
    record = ingest_directory(tmp_path)[0]
    assert record.addendum_relationship.is_addendum
    assert record.addendum_relationship.status.value == "unresolved"
    assert record.addendum_relationship.bid_identifier is None
    assert record.addendum_relationship.unresolved_reason


def test_matching_explicit_bid_identifier_links_addendum(tmp_path: Path):
    (tmp_path / "main.html").write_text(
        "<html><body><p>RFP # JA-207652 Student Devices</p></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "Addendum 1 RFP JA-207652.html").write_text(
        "<html><body><p>Addendum 1 to RFP JA-207652</p></body></html>", encoding="utf-8"
    )
    records = ingest_directory(tmp_path)
    addendum = next(r for r in records if r.source_filename.startswith("Addendum"))
    assert addendum.addendum_relationship.status.value == "linked"
    assert addendum.addendum_relationship.bid_identifier == "JA-207652"


def test_oversized_file_rejected(tmp_path: Path):
    (tmp_path / "large.html").write_text("<p>123456789</p>", encoding="utf-8")
    with pytest.raises(IngestionError, match="size limit"):
        ingest_file(tmp_path, "large.html", max_file_bytes=5)


def test_main_rfp_mentioning_addendum_is_not_classified_as_addendum(
    tmp_path: Path,
):
    (tmp_path / "Final RFP.html").write_text(
        "<html><body><main>"
        "<h1>RFP JA-207652 Student and Staff Computing Devices</h1>"
        "<p>This final RFP includes changes described in Addendum 1.</p>"
        "</main></body></html>",
        encoding="utf-8",
    )

    record = ingest_directory(tmp_path)[0]

    assert record.addendum_relationship.is_addendum is False
    assert record.addendum_relationship.status.value == "not_an_addendum"


def test_both_addenda_link_to_same_rfp(tmp_path: Path):
    (tmp_path / "Final RFP JA-207652.html").write_text(
        "<html><body><main>"
        "<h1>RFP JA-207652 Student Devices</h1>"
        "</main></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "Addendum 1 RFP JA-207652.html").write_text(
        "<html><body><main>"
        "<h1>ADDENDUM No. 1</h1>"
        "<p>RFP JA-207652</p>"
        "</main></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "Addendum 2 RFP JA-207652.html").write_text(
        "<html><body><main>"
        "<h1>ADDENDUM No. 2</h1>"
        "<p>RFP JA-207652</p>"
        "</main></body></html>",
        encoding="utf-8",
    )

    records = ingest_directory(tmp_path)
    addenda = [
        record
        for record in records
        if record.source_filename.startswith("Addendum")
    ]

    assert len(addenda) == 2
    for record in addenda:
        assert record.addendum_relationship.status.value == "linked"
        assert record.addendum_relationship.bid_identifier == "JA-207652"


def test_conflicting_heading_identifiers_remain_unresolved(
    tmp_path: Path,
):
    (tmp_path / "Main RFP.html").write_text(
        "<html><body><main>"
        "<h1>RFP JA-207652 Student Devices</h1>"
        "</main></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "Other RFP.html").write_text(
        "<html><body><main>"
        "<h1>RFP JA-207999 Other Procurement</h1>"
        "</main></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "Addendum 1.html").write_text(
        "<html><body><main>"
        "<h1>ADDENDUM No. 1</h1>"
        "<p>RFP JA-207652</p>"
        "<p>Project JA-207999</p>"
        "</main></body></html>",
        encoding="utf-8",
    )

    records = ingest_directory(tmp_path)
    addendum = next(
        record
        for record in records
        if record.source_filename == "Addendum 1.html"
    )

    assert addendum.addendum_relationship.status.value == "unresolved"
    assert addendum.addendum_relationship.bid_identifier is None

def test_html_navigation_label_is_not_an_addendum(tmp_path: Path):
    (tmp_path / "BidNet listing.html").write_text(
        "<html><body><main>"
        "<h1>JA-207652 - Student and Staff Computing Devices</h1>"
        "<nav><a>Addendums</a><a>Documents</a></nav>"
        "<p>Solicitation Number JA-207652</p>"
        "</main></body></html>",
        encoding="utf-8",
    )

    record = ingest_directory(tmp_path)[0]

    assert record.addendum_relationship.is_addendum is False
    assert record.addendum_relationship.status.value == "not_an_addendum"
