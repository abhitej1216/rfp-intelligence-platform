from __future__ import annotations

from app.agents.state import AgentState


class ExtractionAgent:
    def extract(self, state: AgentState) -> AgentState:
        # Uses evidence to fill draft_fields; placeholder for 20-field logic
        state.draft_fields = {"status": "extracted_from_evidence"}
        return state
