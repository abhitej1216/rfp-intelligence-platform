import json
from collections.abc import Iterable
from pathlib import Path

from app.schemas.document import DocumentRecord


def write_records_json(records: Iterable[DocumentRecord], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump(mode="json") for record in records]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(destination)
