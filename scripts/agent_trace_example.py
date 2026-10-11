from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.orchestrator import Orchestrator
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.vector_store import VectorStore
from app.schemas.chunk import ChunkRecord


def main() -> None:
    bid_id = sys.argv[1] if len(sys.argv) > 1 else "Bid1"
    chunks_path = ROOT / "data" / "chunks.json"
    chunks = []
    if chunks_path.is_file():
        chunks = [ChunkRecord.model_validate(item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))]

    with VectorStore() as store:
        engine = HybridSearch(store, BM25Store(chunks))
        state = Orchestrator(engine).run_extraction(bid_id=bid_id)

    print("Plan:", state.plan)
    print("Retries:", state.retries)
    print("\nAgent steps:")
    for step in state.steps:
        print(f"- {step.agent}.{step.action} [{step.status}] {step.detail}")
    print("\nValidation:", (state.final_output or {}).get("validation"))
    out = ROOT / "data" / f"{bid_id}_agent_trace.json"
    out.write_text(json.dumps(state.final_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
