from __future__ import annotations

from app.agents.state import AgentState


class ValidatorAgent:
    def validate(self, state: AgentState) -> AgentState:
        record = state.draft_fields
        fields = record.get("fields", {})
        passed = failed = not_found = 0
        rejected = []
        for name, field in fields.items():
            value = field.get("value")
            sources = field.get("sources") or []
            if value is None:
                not_found += 1
            elif sources:
                passed += 1
            else:
                failed += 1
                rejected.append(name)
                field["value"] = None
                field["confidence"] = 0.0
                field["notes"] = "Rejected: extracted value has no source citation"
        record["validation"] = {"passed": passed, "failed": failed, "not_found": not_found}
        state.draft_fields = record
        state.rejected_fields = rejected
        state.validation_results.append(f"passed={passed}; failed={failed}; not_found={not_found}")
        return state
