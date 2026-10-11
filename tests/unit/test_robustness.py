from app.extraction.extractor import build_bid_record


def test_empty_evidence_returns_null_with_reason():
    record = build_bid_record("test", [])
    for name in record.fields:
        assert record.fields[name].value is None
        assert "Not found" in (record.fields[name].notes or "")

def test_unseen_bid_folder_no_code_changes():
    # Confirms extractor works without hardcoded filenames
    record = build_bid_record("unseen", [{"text": "Sample bid text", "chunk_id": "x"}])
    assert record.bid_id == "unseen"
    assert len(record.fields) == 20
