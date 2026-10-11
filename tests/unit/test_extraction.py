from app.extraction.extractor import build_bid_record


def test_empty_evidence_returns_null_with_reason():
    record = build_bid_record("test", [])
    assert len(record.fields) == 20
    for field in record.fields.values():
        assert field.value is None
        assert "Not found" in (field.notes or "")


def test_extracts_matching_due_date_with_file_page_citation():
    record = build_bid_record("JA-207652", [{
        "text": "The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.",
        "chunk_id": "abc123",
        "metadata": {"source_filename": "Addendum 2.pdf", "page_number": 1},
    }])
    due_date = record.fields["Due Date"]
    assert due_date.value
    assert "July 9, 2024" in due_date.value
    assert due_date.sources[0].file == "Addendum 2.pdf"
    assert due_date.sources[0].page == 1


def test_nonmatching_chunk_is_not_reused_for_every_field():
    record = build_bid_record("test", [{"text": "Question about USB ports. Answer: USB 3.1"}])
    assert record.fields["Bid Bond Requirement"].value is None
    assert record.fields["Due Date"].value is None


def test_unseen_bid_folder_no_code_changes():
    record = build_bid_record("unseen", [{"text": "Sample bid text", "chunk_id": "x"}])
    assert record.bid_id == "unseen"
    assert len(record.fields) == 20
