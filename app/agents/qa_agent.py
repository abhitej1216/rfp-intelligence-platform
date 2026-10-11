from __future__ import annotations

from app.agents.state import AgentState


class QAAgent:
    def answer(self, question: str, state: AgentState) -> AgentState:
        state.final_output = {
            "question": question,
            "answer": "Cited answer based on retrieved evidence.",
            "sources": state.evidence[:3],
        }
        return state
