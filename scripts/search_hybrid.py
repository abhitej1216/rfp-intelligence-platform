
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.vector_store import VectorStore
from app.schemas.chunk import ChunkRecord


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search RFP documents using hybrid retrieval."
    )
    parser.add_argument(
        "query",
        help="Question or search phrase",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/chunks.json"),
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=Path("data/qdrant"),
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if args.candidate_k < args.top_k:
        parser.error("--candidate-k must be at least --top-k.")
    if not args.chunks.is_file():
        raise FileNotFoundError(f"Chunk file not found: {args.chunks}")

    with args.chunks.open("r", encoding="utf-8") as file:
        raw_chunks = json.load(file)

    chunks = [ChunkRecord.model_validate(item) for item in raw_chunks]
    bm25_store = BM25Store(chunks)

    with VectorStore(storage_path=args.storage) as vector_store:
        search = HybridSearch(vector_store, bm25_store)
        results = search.search(
            args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(f"\n{'=' * 80}")
        print(f"Rank: {rank}")
        print(f"RRF score: {result['rrf_score']:.6f}")
        print(f"Retrievers: {', '.join(result['retrieval_sources'])}")
        print(f"Source: {metadata.get('source_filename')}")
        print(f"Page: {metadata.get('page_number')}")
        print(f"Section: {metadata.get('section_hint')}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(result["text"][:1800])


if __name__ == "__main__":
    main()
