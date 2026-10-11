
from __future__ import annotations

import pytest

from app.retrieval.bm25_store import BM25Store, tokenize
from app.retrieval.hybrid_search import (
    HybridSearch,
    reciprocal_rank_fusion,
)
from app.schemas.chunk import ChunkRecord


def make_chunk(
    chunk_id: str,
    text: str,
    index: int = 0,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id * 64,
        document_id="a" * 64,
        source_filename="sample.pdf",
        source_relative_path="sample.pdf",
        source_sha256="b" * 64,
        source_kind="pdf",
        title="Sample RFP",
        text=text,
        chunk_index=index,
        page_number=1,
        section_hint=None,
        bid_identifier="E20P4600040",
        is_addendum=False,
        addendum_status="not_an_addendum",
    )


def test_tokenize_preserves_procurement_identifiers():
    tokens = tokenize("PORFP E20P4600040, Dell WD22TB4 and 210-BLYZ")

    assert "e20p4600040" in tokens
    assert "wd22tb4" in tokens
    assert "210-blyz" in tokens


def test_bm25_returns_keyword_matches():
    chunks = [
        make_chunk("1", "Laptop warranty is three years"),
        make_chunk("2", "Delivery must occur within 45 days", 1),
        make_chunk("3", "Dock model WD22TB4", 2),
    ]
    store = BM25Store(chunks)

    results = store.search("WD22TB4", top_k=3)

    assert results
    assert results[0]["chunk_id"] == chunks[2].chunk_id
    assert results[0]["metadata"]["source_filename"] == "sample.pdf"


def test_bm25_rejects_empty_query():
    store = BM25Store([make_chunk("1", "Laptop warranty")])

    with pytest.raises(ValueError, match="Query"):
        store.search("  ")


def test_bm25_empty_corpus_returns_no_results():
    assert BM25Store([]).search("laptop") == []


def test_rrf_rewards_documents_found_by_both_retrievers():
    shared = {
        "chunk_id": "shared",
        "text": "Shared result",
        "metadata": {},
    }
    vector_only = {
        "chunk_id": "vector-only",
        "text": "Vector result",
        "metadata": {},
    }
    bm25_only = {
        "chunk_id": "bm25-only",
        "text": "BM25 result",
        "metadata": {},
    }

    fused = reciprocal_rank_fusion(
        [
            [shared, vector_only],
            [shared, bm25_only],
        ],
        top_k=3,
    )

    assert fused[0]["chunk_id"] == "shared"
    assert len({item["chunk_id"] for item in fused}) == 3


def test_rrf_deduplicates_repeated_ids_within_a_list():
    results = reciprocal_rank_fusion(
        [[
            {"chunk_id": "same", "text": "A", "metadata": {}},
            {"chunk_id": "same", "text": "A duplicate", "metadata": {}},
        ]],
        top_k=5,
    )

    assert len(results) == 1


def test_rrf_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([], top_k=0)


def test_hybrid_search_combines_vector_and_keyword_results():
    class FakeVectorStore:
        def search(self, query, top_k=5):
            return [
                {
                    "chunk_id": "vector-hit",
                    "score": 0.9,
                    "text": "Semantic match",
                    "metadata": {},
                }
            ]

    class FakeBM25Store:
        def search(self, query, top_k=5):
            return [
                {
                    "chunk_id": "keyword-hit",
                    "score": 4.2,
                    "text": "Exact model number match",
                    "metadata": {},
                }
            ]

    search = HybridSearch(FakeVectorStore(), FakeBM25Store())
    results = search.search("WD22TB4", top_k=2)

    assert len(results) == 2
    assert {item["chunk_id"] for item in results} == {
        "vector-hit",
        "keyword-hit",
    }
    assert all("rrf_score" in item for item in results)
