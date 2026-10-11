
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.schemas.chunk import ChunkRecord

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "rfp_chunks"


def _point_id(chunk_id: str) -> str:
    """Map a stable SHA-256 chunk ID to a deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _payload(chunk: ChunkRecord) -> dict[str, Any]:
    """Build a Qdrant payload while preserving chunk provenance."""
    return chunk.model_dump(mode="json")


class VectorStore:
    """Persistent local vector index for RFP document chunks."""

    def __init__(
        self,
        storage_path: str | Path = "data/qdrant",
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        embedding_model: Any | None = None,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else SentenceTransformer(model_name)
        )

        self.vector_size = int(
            self.embedding_model.get_sentence_embedding_dimension()
        )
        if self.vector_size <= 0:
            raise ValueError("Embedding dimension must be positive.")

        self.client = QdrantClient(path=str(self.storage_path))

        if self.client.collection_exists(collection_name):
            collection = self.client.get_collection(collection_name)
            configured_size = collection.config.params.vectors.size

            if configured_size != self.vector_size:
                self.client.close()
                raise ValueError(
                    f"Collection '{collection_name}' uses vector size "
                    f"{configured_size}, but the embedding model produces "
                    f"{self.vector_size}. Use a new collection or rebuild "
                    "the existing index."
                )
        else:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode text locally and validate embedding output."""
        vectors = np.asarray(
            self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

        if vectors.shape != (len(texts), self.vector_size):
            raise ValueError(
                "Embedding model returned an unexpected shape: "
                f"{vectors.shape}; expected "
                f"({len(texts)}, {self.vector_size})."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain non-finite values.")

        return vectors

    def index_chunks(
        self,
        chunks: list[ChunkRecord],
        batch_size: int = 32,
    ) -> int:
        """Upsert chunks in batches; rerunning is safe for stable IDs."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if not chunks:
            return 0

        seen: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen:
                raise ValueError(
                    f"Duplicate chunk ID in input: {chunk.chunk_id}"
                )
            seen.add(chunk.chunk_id)

        indexed = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self._embed([chunk.text for chunk in batch])

            points = [
                models.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vector.tolist(),
                    payload=_payload(chunk),
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            indexed += len(points)

        return indexed

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked matches with their source and page metadata."""
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self._embed([query])[0]

        query_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v),
                    )
                )
            if conditions:
                query_filter = models.Filter(
                    must=conditions
                )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=query_filter,
        )

        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = dict(point.payload or {})
            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "score": float(point.score),
                    "text": payload.get("text", ""),
                    "metadata": payload,
                }
            )

        return results

    def count(self) -> int:
        """Return the number of indexed points."""
        return int(
            self.client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
        )

    def close(self) -> None:
        """Close the local Qdrant client."""
        self.client.close()

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
