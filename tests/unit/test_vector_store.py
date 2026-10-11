
from __future__ import annotations

import numpy as np
import pytest

from app.retrieval.vector_store import VectorStore
from app.schemas.chunk import ChunkRecord


class FakeEmbeddingModel:
    """Small deterministic embedding model for unit tests."""

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [
                float("laptop" in lowered),
                float("warranty" in lowered),
                float("delivery" in lowered),
            ]
            if not any(vector):
                vector = [0.1, 0.1, 0.1]
            vectors.append(vector)

        result = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / norms
        return result


def make_chunk(
    chunk_id: str,
    text: str,
    chunk_index: int = 0,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id * 64,
        document_id="a" * 64,
        source_filename="example.pdf",
        source_relative_path="example.pdf",
        source_sha256="b" * 64,
        source_kind="pdf",
        title="Example RFP",
        text=text,
        chunk_index=chunk_index,
        page_number=2,
        section_hint="Technical Requirements",
        bid_identifier="RFP-12345",
        is_addendum=False,
        addendum_status="not_an_addendum",
    )


def make_store(path) -> VectorStore:
    return VectorStore(
        storage_path=path,
        collection_name="test_chunks",
        embedding_model=FakeEmbeddingModel(),
    )


def test_index_and_search_preserves_metadata(tmp_path):
    with make_store(tmp_path / "qdrant") as store:
        chunks = [
            make_chunk("1", "Laptop with extended warranty"),
            make_chunk("2", "Delivery must occur within 45 days", 1),
        ]

        assert store.index_chunks(chunks) == 2
        assert store.count() == 2

        results = store.search("laptop", top_k=1)

        assert len(results) == 1
        assert results[0]["chunk_id"] == chunks[0].chunk_id
        assert results[0]["metadata"]["source_filename"] == "example.pdf"
        assert results[0]["metadata"]["page_number"] == 2
        assert results[0]["metadata"]["bid_identifier"] == "RFP-12345"


def test_reindexing_does_not_create_duplicate_points(tmp_path):
    with make_store(tmp_path / "qdrant") as store:
        chunk = make_chunk("3", "Laptop requirements")

        store.index_chunks([chunk])
        store.index_chunks([chunk])

        assert store.count() == 1


def test_empty_query_is_rejected(tmp_path):
    with make_store(tmp_path / "qdrant") as store:
        with pytest.raises(ValueError, match="Query"):
            store.search("   ")


def test_invalid_top_k_is_rejected(tmp_path):
    with make_store(tmp_path / "qdrant") as store:
        with pytest.raises(ValueError, match="top_k"):
            store.search("laptop", top_k=0)


def test_empty_chunk_list_is_safe(tmp_path):
    with make_store(tmp_path / "qdrant") as store:
        assert store.index_chunks([]) == 0
        assert store.count() == 0


def test_duplicate_chunk_ids_in_input_are_rejected(tmp_path):
    with make_store(tmp_path / "qdrant") as store:
        chunk = make_chunk("4", "Laptop requirements")

        with pytest.raises(ValueError, match="Duplicate chunk ID"):
            store.index_chunks([chunk, chunk])
