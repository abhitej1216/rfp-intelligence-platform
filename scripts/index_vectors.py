
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.retrieval.vector_store import VectorStore
from app.schemas.chunk import ChunkRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the local RFP vector index."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/chunks.json"),
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=Path("data/qdrant"),
    )
    parser.add_argument(
        "--collection",
        default="rfp_chunks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Chunk file not found: {args.input}"
        )

    with args.input.open("r", encoding="utf-8") as file:
        raw_chunks = json.load(file)

    if not isinstance(raw_chunks, list):
        raise ValueError("The chunk JSON root must be a list.")

    chunks = [
        ChunkRecord.model_validate(item)
        for item in raw_chunks
    ]

    logger.info("Loaded %d chunks.", len(chunks))

    with VectorStore(
        storage_path=args.storage,
        collection_name=args.collection,
    ) as store:
        indexed = store.index_chunks(
            chunks,
            batch_size=args.batch_size,
        )
        logger.info("Processed %d chunks.", indexed)
        logger.info("Total points in collection: %d", store.count())


if __name__ == "__main__":
    main()
