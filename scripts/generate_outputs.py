#!/usr/bin/env python
"""Generate final JSON output files for Bid1 and Bid2."""
from pathlib import Path

from app.extraction.extractor import GeminiExtractor
from app.ingestion.parser import ingest_directory
from app.retrieval.chunking import chunk_documents

for folder_name in ["Bid1", "Bid2"]:
    input_path = Path("Assignment-Data-Statements (AI Engineer-Emplay Inc)") / folder_name
    records = ingest_directory(str(input_path))
    chunks = chunk_documents(records)
    # Build evidence from chunks for extraction
    evidence = [{"text": c.text, "chunk_id": c.chunk_id} for c in chunks[:10]]
    bid_id = records[0].source_filename.split()[0] if records else folder_name
    extractor = GeminiExtractor()
    record = extractor.extract(bid_id, "\n".join([e.get("text", "") for e in evidence]))
    output_path = Path("data") / f"{folder_name}_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    print(f"Generated: {output_path}")
