from __future__ import annotations

from app.agents.state import AgentState
from app.retrieval.hybrid_search import HybridSearch


class RetrievalAgent:
    def __init__(self, hybrid: HybridSearch):
        self.hybrid = hybrid

    def retrieve(self, query: str, state: AgentState, bid_id: str | None = None) -> AgentState:
        filters = {"bid_identifier": bid_id} if bid_id else None
        results = self.hybrid.search(query, top_k=10, filters=filters)
        # Fall back when metadata filter matches nothing (common for folder labels like Bid1).
        if not results and bid_id:
            results = self.hybrid.search(query, top_k=10, filters=None)
        existing = {item.get("chunk_id") for item in state.evidence}
        state.evidence.extend(item for item in results if item.get("chunk_id") not in existing)
        return state
