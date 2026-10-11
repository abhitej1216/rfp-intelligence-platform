from __future__ import annotations

from app.agents.state import AgentState


class Orchestrator:
    def plan(self, goal: str, state: AgentState) -> AgentState:
        state.plan = ["retrieve", "extract", "validate", "report"]
        return state
