
from app.schemas.chunk import ChunkRecord
from scripts.evaluate_retrieval import (
    evaluate_ranked_results,
    recall_at_k,
    reciprocal_rank,
    resolve_relevant_chunks,
)


def test_recall_at_k():
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}

    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_reciprocal_rank_respects_cutoff():
    assert reciprocal_rank(["a"] * 10 + ["target"], {"target"}, k=10) == 0.0


def test_normalized_pdf_line_break_matching():
    chunk_text = "Due Date: July 9, 2024\n2:00 PM CST"
    chunk = ChunkRecord(
        chunk_id="a" * 64,
        document_id="b" * 64,
        source_filename="test.pdf",
        source_relative_path="test.pdf",
        source_sha256="c" * 64,
        source_kind="pdf",
        text=chunk_text,
        chunk_index=0,
        page_number=1,
        is_addendum=False,
        addendum_status="not_an_addendum",
    )
    cases = [{"id": "t1", "source": "test.pdf", "evidence_terms": ["July 9, 2024 2:00 PM CST"]}]
    result = resolve_relevant_chunks(cases, [chunk])
    assert result["t1"] == {chunk.chunk_id}

def test_evaluation_reports_first_relevant_rank():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "target"},
        {"chunk_id": "c"},
    ]

    metrics = evaluate_ranked_results(results, {"target"})

    assert metrics["first_relevant_rank"] == 2
    assert metrics["recall"]["recall@1"] == 0.0
    assert metrics["recall"]["recall@3"] == 1.0
    assert metrics["rr@10"] == 0.5
