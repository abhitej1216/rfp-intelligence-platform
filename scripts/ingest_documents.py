from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running this script directly (e.g. `python scripts/ingest_documents.py`)
# by making the project root importable regardless of the current directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.parser import ingest_directory
from app.ingestion.security import IngestionError
from app.ingestion.serialization import write_records_json

LOGGER = logging.getLogger("rfp_ingestion")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text and provenance from PDF/HTML bid documents."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-file-mb", type=int, default=25)
    args = parser.parse_args()
    if args.max_file_mb < 1:
        parser.error("--max-file-mb must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        records = ingest_directory(
            args.input_dir, max_file_bytes=args.max_file_mb * 1024 * 1024
        )
        write_records_json(records, args.output)
    except (OSError, IngestionError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("Ingested %d document(s). Output: %s", len(records), args.output)
    for record in records:
        LOGGER.info(
            "%s | %s | %s",
            record.source_filename,
            record.source_kind.value,
            record.addendum_relationship.status.value,
        )
        for warning in record.extraction_warnings:
            LOGGER.warning("%s: %s", record.source_filename, warning)
    return 0


if __name__ == "__main__":
    sys.exit(main())
