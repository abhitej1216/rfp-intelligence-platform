from __future__ import annotations

from app.agents.state import AgentState


class ReconciliationAgent:
    def reconcile(self, state: AgentState) -> AgentState:
        """Prefer evidence from later numbered addenda for date and requirement changes."""
        record = state.draft_fields
        fields = record.get("fields", {})
        addenda = [
            item for item in state.evidence
            if (item.get("metadata") or {}).get("is_addendum")
        ]
        addenda.sort(key=lambda item: (item.get("metadata") or {}).get("addendum_number") or 0)
        changes = []
        for field_name in ("Due Date", "Delivery Date", "Bid Submission Type", "Product Specification"):
            candidates = []
            for item in addenda:
                metadata = item.get("metadata") or {}
                number = metadata.get("addendum_number") or 0
                snippet = item.get("text", "")
                if field_name.lower().split()[0] in snippet.casefold() or (field_name == "Due Date" and "due date" in snippet.casefold()):
                    candidates.append((number, item, snippet))
            if candidates:
                _, item, text = candidates[-1]
                metadata = item.get("metadata") or {}
                from app.extraction.extractor import build_bid_record
                single = build_bid_record(record.get("bid_id", "unknown"), [item]).fields.get(field_name)
                if single and single.value:
                    fields[field_name] = single.model_dump(mode="json")
                    changes.append(f"{metadata.get('source_filename', 'Addendum')} updated {field_name}: {single.value}")
        record["addendum_changes"] = changes
        state.draft_fields = record
        state.validation_results.append("addendum_reconciled")
        return state
