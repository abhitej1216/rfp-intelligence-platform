from __future__ import annotations

from app.agents.state import AgentState


class ReconciliationAgent:
    def reconcile(self, state: AgentState) -> AgentState:
        state.validation_results.append("addendum_reconciled")
        return state
