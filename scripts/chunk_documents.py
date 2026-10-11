
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

from app.retrieval.chunking import chunk_documents
from app.schemas.document import DocumentRecord

logger = logging.getLogger("chunk_documents")


def load_documents(path: Path) -> list[DocumentRecord]:
    """Load and validate serialized document records."""
    with path.open("r", encoding="utf-8") as file:
        raw_documents = json.load(file)

    if not isinstance(raw_documents, list):
        raise ValueError("Input JSON must contain a list of documents.")

    return [
        DocumentRecord.model_validate(item)
        for item in raw_documents
    ]


def write_chunks(chunks, output_path: Path) -> None:
    """Write chunk records atomically as a JSON array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            json.dump(
                [chunk.model_dump(mode="json") for chunk in chunks],
                temporary_file,
                ensure_ascii=False,
                indent=2,
            )
            temporary_file.write("\n")

        os.replace(temporary_path, output_path)

    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create searchable chunks from ingested RFP documents."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/documents_ocr.json"),
        help="Path to the ingested document JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chunks.json"),
        help="Output path for chunk records.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum approximate characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Character overlap between chunks in a segment.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.chunk_size < 1:
        parser.error("--chunk-size must be greater than zero.")

    if not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error(
            "--chunk-overlap must be non-negative and smaller than --chunk-size."
        )

    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    documents = load_documents(args.input)
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    write_chunks(chunks, args.output)

    logger.info("Loaded %d document(s).", len(documents))
    logger.info("Created %d chunk(s).", len(chunks))
    logger.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
