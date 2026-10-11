#!/usr/bin/env python
"""Generate final JSON output files for Bid1 and Bid2."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from app.extraction.extractor import build_bid_record
from app.ingestion.parser import ingest_directory
from app.retrieval.chunking import chunk_documents

def stable_bid_id(folder_name: str, records) -> str:
    from app.ingestion.addenda import extract_explicit_bid_identifiers

    identifiers = set()
    for record in records:
        identifiers.update(extract_explicit_bid_identifiers(record.text))
        identifiers.update(extract_explicit_bid_identifiers(record.source_filename))
    if len(identifiers) == 1:
        return next(iter(identifiers))
    return folder_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cited extraction JSON for a bid folder.")
    parser.add_argument("input_dir", type=Path, nargs="?", default=Path("Assignment-Data-Statements (AI Engineer-Emplay Inc)"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    folders = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    for folder in folders:
        records = ingest_directory(folder)
        chunks = chunk_documents(records)
        evidence = [
            {"text": chunk.text, "chunk_id": chunk.chunk_id, "metadata": chunk.model_dump(mode="json")}
            for chunk in chunks
        ]
        bid_id = stable_bid_id(folder.name, records)
        record = build_bid_record(bid_id, evidence)
        # RFP facts can be far apart; extraction evaluates all evidence, bounded per field.
        output_path = args.output_dir / f"{folder.name}_output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        print(f"Generated: {output_path} ({len(records)} documents, {len(chunks)} chunks)")


if __name__ == "__main__":
    main()
