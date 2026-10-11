
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from statistics import mean
from typing import Any

from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.vector_store import VectorStore
from app.schemas.chunk import ChunkRecord

CHUNKS_PATH = Path("data/chunks.json")
EVAL_PATH = Path("data/evaluation/retrieval_eval.json")
OUTPUT_PATH = Path("data/evaluation/retrieval_results.json")
K_VALUES = (1, 3, 5)
MRR_K = 10

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def normalize_text(value: str) -> str:
    """Normalize whitespace introduced by PDF text extraction."""
    return re.sub(r"\s+", " ", value).strip().casefold()



def resolve_relevant_chunks(
    cases: list[dict[str, Any]],
    chunks: list[ChunkRecord],
) -> dict[str, set[str]]:
    """Resolve evidence phrases to chunk IDs, ignoring PDF line breaks."""
    resolved: dict[str, set[str]] = {}
    errors: list[str] = []

    for case in cases:
        source = case["source"]
        terms = [
            normalize_text(term)
            for term in case["evidence_terms"]
        ]

        matches = {
            chunk.chunk_id
            for chunk in chunks
            if chunk.source_filename == source
            and all(
                term in normalize_text(chunk.text)
                for term in terms
            )
        }

        if not matches:
            errors.append(
                f'{case["id"]}: no chunk in "{source}" contains all '
                f'evidence terms {case["evidence_terms"]!r}'
            )
        else:
            resolved[case["id"]] = matches

    if errors:
        raise ValueError(
            "Evaluation labels need review:\n- " + "\n- ".join(errors)
        )

    return resolved



def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of relevant chunks found in the first k results."""
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = MRR_K,
) -> float:
    """Reciprocal rank of the first relevant result within k."""
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_ranked_results(
    ranked_results: list[dict[str, Any]],
    relevant_ids: set[str],
) -> dict[str, Any]:
    ranked_ids = [item["chunk_id"] for item in ranked_results]

    return {
        "recall": {
            f"recall@{k}": recall_at_k(ranked_ids, relevant_ids, k)
            for k in K_VALUES
        },
        f"rr@{MRR_K}": reciprocal_rank(ranked_ids, relevant_ids),
        "ranked_chunk_ids": ranked_ids,
        "first_relevant_rank": next(
            (
                rank
                for rank, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None,
        ),
    }


def aggregate(
    per_query: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    methods = ("vector", "bm25", "hybrid")
    summary: dict[str, Any] = {}

    for method in methods:
        method_results = [
            query_result[method]
            for query_result in per_query.values()
        ]

        summary[method] = {
            f"recall@{k}": round(
                mean(
                    result["recall"][f"recall@{k}"]
                    for result in method_results
                ),
                4,
            )
            for k in K_VALUES
        }
        summary[method][f"mrr@{MRR_K}"] = round(
            mean(result[f"rr@{MRR_K}"] for result in method_results),
            4,
        )

    return summary


def main() -> None:
    raw_chunks = load_json(CHUNKS_PATH)
    chunks = [ChunkRecord.model_validate(item) for item in raw_chunks]
    cases = load_json(EVAL_PATH)

    if len(cases) < 15:
        raise ValueError(
            f"Need at least 15 evaluation cases; found {len(cases)}."
        )

    relevant_by_case = resolve_relevant_chunks(cases, chunks)
    bm25 = BM25Store(chunks)
    per_query: dict[str, dict[str, dict[str, Any]]] = {}

    with VectorStore() as vector_store:
        hybrid = HybridSearch(vector_store, bm25)

        for case in cases:
            query = case["query"]
            case_id = case["id"]
            relevant_ids = relevant_by_case[case_id]

            vector_results = vector_store.search(query, top_k=20)
            bm25_results = bm25.search(query, top_k=20)
            hybrid_results = hybrid.search(
                query,
                top_k=20,
                candidate_k=20,
            )

            per_query[case_id] = {
                "vector": evaluate_ranked_results(
                    vector_results, relevant_ids
                ),
                "bm25": evaluate_ranked_results(
                    bm25_results, relevant_ids
                ),
                "hybrid": evaluate_ranked_results(
                    hybrid_results, relevant_ids
                ),
            }

            logger.info(
                "%s | V rank=%s | BM25 rank=%s | Hybrid rank=%s",
                case_id,
                per_query[case_id]["vector"]["first_relevant_rank"],
                per_query[case_id]["bm25"]["first_relevant_rank"],
                per_query[case_id]["hybrid"]["first_relevant_rank"],
            )

    summary = aggregate(per_query)
    report = {
        "evaluation_cases": len(cases),
        "relevant_chunks_per_case": {
            case_id: sorted(chunk_ids)
            for case_id, chunk_ids in relevant_by_case.items()
        },
        "metrics": summary,
        "per_query": per_query,
        "metric_definitions": {
            "recall@k": "Relevant labelled chunks retrieved in top k / "
            "total relevant labelled chunks",
            "mrr@10": "Mean reciprocal rank of the first relevant chunk "
            "within the first 10 results",
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nAggregate metrics")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
