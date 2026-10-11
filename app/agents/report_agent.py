from __future__ import annotations

from app.agents.state import AgentState


class ReportAgent:
    """Produces the final structured JSON (extraction) or cited answer (Q&A)."""

    def report_extraction(self, state: AgentState) -> AgentState:
        record = state.draft_fields or {}
        state.log_step(
            "ReportAgent",
            "report_extraction",
            bid_id=state.bid_id,
            fields=len((record.get("fields") or {})),
            validation=record.get("validation"),
        )
        state.final_output = {
            "mode": "extract",
            "bid_id": state.bid_id or record.get("bid_id"),
            "record": record,
            "plan": state.plan,
            "steps": [step.model_dump(mode="json") for step in state.steps],
            "retries": state.retries,
            "validation": record.get("validation") or {"passed": 0, "failed": 0, "not_found": 0},
            "addendum_changes": record.get("addendum_changes") or [],
        }
        return state

    def report_answer(self, state: AgentState, qa_output: dict) -> AgentState:
        state.log_step(
            "ReportAgent",
            "report_answer",
            has_answer=bool(qa_output.get("rephrased_answer") or qa_output.get("answer")),
            sources=len(qa_output.get("sources") or []),
        )
        state.final_output = {
            "mode": "question_answer",
            **qa_output,
            "plan": state.plan,
            "steps": [step.model_dump(mode="json") for step in state.steps],
        }
        return state
