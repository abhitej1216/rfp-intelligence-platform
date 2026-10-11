
from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.schemas.chunk import ChunkRecord

TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:[-./][a-z0-9]+)*",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    """Tokenize text while preserving common procurement identifiers."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class BM25Store:
    """In-memory BM25 index built from validated document chunks."""

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = list(chunks)
        self._chunk_by_id = {
            chunk.chunk_id: chunk for chunk in self.chunks
        }

        if len(self._chunk_by_id) != len(self.chunks):
            raise ValueError("Duplicate chunk IDs in BM25 input.")

        self._tokenized = [tokenize(chunk.text) for chunk in self.chunks]
        self._index = (
            BM25Okapi(self._tokenized) if self._tokenized else None
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return BM25-ranked chunks with source metadata."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self._index is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )

        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            chunk = self.chunks[index]
            # Apply metadata filters post-ranking.
            if filters and not all(
                getattr(chunk, k, None) == v for k, v in filters.items()
            ):
                continue
            score = float(scores[index])
            if score <= 0:
                continue
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "score": score,
                    "text": chunk.text,
                    "metadata": chunk.model_dump(mode="json"),
                }
            )

            if len(results) >= top_k:
                break

        return results
