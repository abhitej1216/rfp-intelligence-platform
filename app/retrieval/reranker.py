from __future__ import annotations

from typing import Any

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = None
        if CrossEncoder:
            try:
                self.model = CrossEncoder(model_name)
            except Exception:  # noqa: BLE001,S110 - safe fallback for optional reranker
                pass

    def rerank(self, query: str, results: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not self.model or not results:
            return results[:top_k]
        pairs = [(query, r.get("text", "")) for r in results]
        try:
            scores = self.model.predict(pairs)
            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
            results.sort(key=lambda x: (-x.get("rerank_score", 0), x.get("chunk_id", "")))
        except Exception:  # noqa: BLE001,S110 - safe fallback for optional reranker
            pass
        return results[:top_k]
