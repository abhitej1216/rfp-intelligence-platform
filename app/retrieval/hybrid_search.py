
from __future__ import annotations

from typing import Any

from app.retrieval.bm25_store import BM25Store
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.vector_store import VectorStore

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists using one-based reciprocal ranks."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    fused: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()

        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = result.get("chunk_id")
            if not chunk_id or chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "retrieval_sources": [],
                }

            fused[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

            source = result.get("retrieval_source")
            if source and source not in fused[chunk_id]["retrieval_sources"]:
                fused[chunk_id]["retrieval_sources"].append(source)

    ranked = sorted(
        fused.values(),
        key=lambda item: (-item["rrf_score"], item["chunk_id"]),
    )
    return ranked[:top_k]


class HybridSearch:
    """Combine vector and BM25 retrieval using reciprocal rank fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        filters: dict[str, Any] | None = None,
        reranker: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve candidates from both indexes and fuse their rankings."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        candidate_k = candidate_k or max(top_k * 3, 10)
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k.")

        vector_results = self.vector_store.search(
            query,
            top_k=candidate_k,
            **({"filters": filters} if filters is not None else {}),
        )
        bm25_results = self.bm25_store.search(
            query,
            top_k=candidate_k,
            **({"filters": filters} if filters is not None else {}),
        )

        for result in vector_results:
            result["retrieval_source"] = "vector"
        for result in bm25_results:
            result["retrieval_source"] = "bm25"

        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            top_k=top_k,
            rrf_k=self.rrf_k,
        )
        if reranker:
            fused = reranker.rerank(query, fused, top_k=top_k)
        return fused
