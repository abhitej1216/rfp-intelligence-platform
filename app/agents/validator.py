from __future__ import annotations

from app.agents.state import AgentState


class ValidatorAgent:
    def validate(self, state: AgentState) -> AgentState:
        state.retries = 0
        state.validation_results.append("validated")
        if state.retries < state.max_retries:
            state.retries += 1
        return state
