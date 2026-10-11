from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reports"


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", str(name).strip()) or "report"
    return cleaned[:120]


def persist_report(payload: dict[str, Any], *, prefix: str = "report") -> Path:
    """Always write the final report JSON under data/reports/."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bid = payload.get("bid_id") or prefix
    if isinstance(bid, str) and (len(bid) > 48 or " " in bid):
        bid = prefix
    filename = _safe_filename(f"{prefix}_{bid}_{stamp}.json")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Also keep a stable "latest" pointer for quick inspection.
    latest = REPORTS_DIR / f"{prefix}_latest.json"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path
