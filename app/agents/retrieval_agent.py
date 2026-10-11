from __future__ import annotations

from app.agents.state import AgentState
from app.retrieval.hybrid_search import HybridSearch


class RetrievalAgent:
    def __init__(self, hybrid: HybridSearch):
        self.hybrid = hybrid

    def retrieve(self, query: str, state: AgentState) -> AgentState:
        results = self.hybrid.search(query, top_k=5)
        state.evidence.extend(results)
        return state
