from __future__ import annotations

from app.agents.state import AgentState
from app.extraction.extractor import build_bid_record


class ExtractionAgent:
    def extract(self, state: AgentState, bid_id: str | None = None) -> AgentState:
        identifier = bid_id or str((state.final_output or {}).get("bid_id", "unknown"))
        state.draft_fields = build_bid_record(identifier, state.evidence).model_dump(mode="json")
        return state
