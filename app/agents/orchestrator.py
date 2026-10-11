from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from app.agents.catalog import AGENT_CATALOG, get_agent, list_specialist_agents
from app.agents.extraction_agent import ExtractionAgent
from app.agents.ingestion_agent import IngestionAgent
from app.agents.qa_agent import QAAgent
from app.agents.reconciliation import ReconciliationAgent
from app.agents.report_agent import ReportAgent
from app.agents.report_store import persist_report
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.state import AgentState
from app.agents.validator import ValidatorAgent
from app.retrieval.hybrid_search import HybridSearch

EXTRACTION_QUERIES = (
    "bid number title solicitation due date submission deadline",
    "bid bond payment terms delivery installation pre-bid meeting",
    "product specification model part number warranty affidavit",
    "contact procurement company cooperative contract manufacturer registration",
)


class Orchestrator:
    """Master planner: picks specialist agents from the query and coordinates them."""

    def __init__(self, hybrid: HybridSearch | None = None):
        self.hybrid = hybrid
        self.ingestion = IngestionAgent()
        self.extraction = ExtractionAgent()
        self.reconciliation = ReconciliationAgent()
        self.validator = ValidatorAgent()
        self.qa = QAAgent()
        self.report = ReportAgent()

    def select_agents(self, goal: str) -> dict[str, Any]:
        text = goal.casefold().strip()
        selected: list[str] = []
        reasons: list[str] = []

        def add(agent_id: str, reason: str) -> None:
            if agent_id not in selected:
                selected.append(agent_id)
                reasons.append(reason)

        # Explicit full extraction / final bid-report mode
        if any(
            token in text
            for token in (
                "extract bid",
                "full extract",
                "structured json",
                "all fields",
                "bid record",
                "final report",
                "bid summary",
                "summary export",
            )
        ):
            for agent_id in ("retrieval", "extraction", "reconciliation", "validator", "report"):
                add(agent_id, "Full extraction / final bid report requested")
            if any(token in text for token in ("ingest", "index", "folder", "incoming")):
                selected.insert(0, "ingestion")
                reasons.insert(0, "Ingestion requested with extraction")
            return self._route_payload(goal, "extract", selected, reasons)

        # Score specialists by keyword hits (skip orchestrator itself).
        scores: dict[str, int] = {}
        for agent in list_specialist_agents():
            score = sum(1 for kw in agent["keywords"] if kw in text)
            if score:
                scores[agent["id"]] = score

        # Question-style goals default toward Q&A unless extraction keywords dominate.
        is_question = "?" in goal or text.startswith(("what", "which", "when", "where", "who", "how", "why", "is ", "are ", "compare"))
        if is_question and scores.get("extraction", 0) < 2:
            add("retrieval", "Questions need grounded evidence from search")
            add("qa", "Natural-language question routed to Q&A Agent")
            add("report", "Report Agent packages the cited answer")
            if "addendum" in text or "changed" in text:
                add("reconciliation", "Addendum change mentioned in the question")
            return self._route_payload(goal, "question_answer", selected, reasons)

        search_focus = any(token in text for token in ("search", "retrieve", "find evidence", "lookup", "passages"))
        extract_focus = any(token in text for token in ("extract", "structured", "json", "all fields", "bid record", "fill fields"))

        if "ingestion" in scores:
            add("ingestion", "Query matches Ingestion Agent keywords")
        if search_focus or "retrieval" in scores:
            add("retrieval", "Query matches Retrieval / Search Agent keywords")
            if search_focus and not extract_focus and "validator" not in scores and "reconciliation" not in scores:
                # Pure search requests stop at retrieval + light report.
                add("report", "Report Agent returns cited passages")
                return self._route_payload(goal, "search", selected, reasons)
        if "extraction" in scores or extract_focus:
            add("retrieval", "Extraction requires retrieved evidence")
            add("extraction", "Query matches Extraction Agent keywords")
            add("report", "Report Agent emits the structured output")
        if "reconciliation" in scores:
            add("retrieval", "Reconciliation needs current evidence")
            add("extraction", "Reconciliation updates extracted fields")
            add("reconciliation", "Query matches Addendum Reconciliation Agent keywords")
            add("report", "Report Agent emits updated fields")
        if "validator" in scores:
            add("retrieval", "Validation needs evidence-backed fields")
            add("extraction", "Validator critiques extracted fields")
            add("validator", "Query matches Validator / Critic Agent keywords")
            add("report", "Report Agent includes validation summary")
        if "qa" in scores and "qa" not in selected:
            add("retrieval", "Q&A requires retrieval grounding")
            add("qa", "Query matches Q&A Agent keywords")
            add("report", "Report Agent packages the cited answer")
        if "report" in scores and "report" not in selected:
            # Standalone report requests still gather evidence + answer first.
            add("retrieval", "Report Agent needs retrieved evidence")
            add("qa", "Report Agent summarizes grounded Q&A output")
            add("report", "Query matches Report Agent keywords")

        if not selected:
            # Safe default for free-form goals.
            add("retrieval", "Default route: retrieve evidence first")
            add("qa", "Default route: answer with Q&A Agent")
            add("report", "Default route: finalize with Report Agent")
            mode = "question_answer"
        else:
            mode = "extract" if "extraction" in selected and "qa" not in selected else "question_answer"

        return self._route_payload(goal, mode, selected, reasons)

    def _route_payload(self, goal: str, mode: str, selected: list[str], reasons: list[str]) -> dict[str, Any]:
        agents = []
        for agent_id in selected:
            meta = get_agent(agent_id)
            if not meta:
                continue
            agents.append(
                {
                    "id": meta["id"],
                    "name": meta["name"],
                    "title": meta["title"],
                    "responsibility": meta["responsibility"],
                    "output": meta["output"],
                }
            )
        return {
            "goal": goal,
            "mode": mode,
            "selected_agent_ids": selected,
            "selected_agents": agents,
            "reasons": reasons,
            "orchestrator": get_agent("orchestrator"),
        }

    def plan(self, goal: str, state: AgentState) -> AgentState:
        route = self.select_agents(goal)
        state.goal = goal
        state.plan = list(route["selected_agent_ids"])
        state.log_step(
            "Orchestrator",
            "route",
            mode=route["mode"],
            selected_agents=route["selected_agent_ids"],
            reasons=route["reasons"],
            goal=goal,
        )
        return state

    def run_validation_loop(
        self,
        state: AgentState,
        retrieve: Callable[[AgentState], AgentState],
        extract: Callable[[AgentState], AgentState],
        validate: Callable[[AgentState], AgentState],
    ) -> AgentState:
        state = extract(state)
        state = validate(state)
        while state.rejected_fields and state.retries < state.max_retries:
            state.retries += 1
            state.log_step(
                "Orchestrator",
                "validation_retry",
                retry=state.retries,
                rejected_fields=list(state.rejected_fields),
            )
            state = retrieve(state)
            state = extract(state)
            state = validate(state)
        return state

    def iter_routed(
        self,
        goal: str,
        *,
        bid_id: str | None = None,
        input_dir: str | None = None,
        index_callback: Callable[[list[dict[str, Any]]], int] | None = None,
        force_agent_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield UI/API progress events while executing the routed agent chain."""
        if force_agent_id:
            meta = get_agent(force_agent_id)
            if not meta or meta["id"] == "orchestrator":
                raise ValueError(f"Unknown specialist agent: {force_agent_id}")
            deps = {
                "ingestion": ["ingestion"],
                "retrieval": ["retrieval"],
                "extraction": ["retrieval", "extraction", "report"],
                "reconciliation": ["retrieval", "extraction", "reconciliation", "report"],
                "validator": ["retrieval", "extraction", "validator", "report"],
                "qa": ["retrieval", "qa", "report"],
                "report": ["retrieval", "qa", "report"],
            }
            selected_ids = deps.get(meta["id"], [meta["id"]])
            route = self._route_payload(
                goal,
                "single_agent",
                selected_ids,
                [f"Direct run from {meta['title']} page"],
            )
        else:
            route = self.select_agents(goal)

        yield {"type": "route", "data": route}

        state = AgentState(bid_id=bid_id, goal=goal)
        state.plan = list(route["selected_agent_ids"])
        state.log_step(
            "Orchestrator",
            "route",
            mode=route["mode"],
            selected_agents=route["selected_agent_ids"],
            reasons=route["reasons"],
            goal=goal,
        )

        selected = set(route["selected_agent_ids"])

        if "ingestion" in selected:
            yield self._agent_event("ingestion", "running")
            if input_dir:
                state = self.ingestion.ingest(input_dir, state)
                chunks = (state.final_output or {}).pop("ingested_chunks", [])
                if index_callback and chunks:
                    indexed = index_callback(chunks)
                    state.log_step("IngestionAgent", "index_chunks", indexed=indexed)
            else:
                state.log_step("IngestionAgent", "skip", reason="No input_dir provided")
            yield self._agent_event("ingestion", "done", documents=len(state.documents))

        def retrieve_fn(current: AgentState) -> AgentState:
            if self.hybrid is None:
                raise ValueError("HybridSearch engine is required.")
            retrieval = RetrievalAgent(self.hybrid)
            before = len(current.evidence)
            queries = [goal] if route["mode"] != "extract" else list(EXTRACTION_QUERIES)
            if current.rejected_fields:
                queries = [f"{field} {bid_id or ''}" for field in current.rejected_fields] + queries
            for query in queries:
                current = retrieval.retrieve(query, current, bid_id=bid_id)
            current.log_step(
                "RetrievalAgent",
                "retrieve",
                queries=len(queries),
                evidence_added=len(current.evidence) - before,
                evidence_total=len(current.evidence),
                bid_id=bid_id,
            )
            return current

        if "retrieval" in selected:
            yield self._agent_event("retrieval", "running")
            state = retrieve_fn(state)
            yield self._agent_event("retrieval", "done", evidence=len(state.evidence))

        needs_fields = bool(selected & {"extraction", "reconciliation", "validator"})
        if needs_fields and "retrieval" not in selected and not state.evidence:
            yield self._agent_event("retrieval", "running", note="Auto-selected dependency")
            state = retrieve_fn(state)
            yield self._agent_event("retrieval", "done", evidence=len(state.evidence))

        if "extraction" in selected or needs_fields:
            yield self._agent_event("extraction", "running")
            state = self.extraction.extract(state, bid_id=bid_id or "unknown")
            state.log_step(
                "ExtractionAgent",
                "extract",
                fields=len((state.draft_fields or {}).get("fields") or {}),
            )
            yield self._agent_event(
                "extraction",
                "done",
                fields=len((state.draft_fields or {}).get("fields") or {}),
            )

        if "reconciliation" in selected:
            yield self._agent_event("reconciliation", "running")
            if not state.draft_fields:
                state = self.extraction.extract(state, bid_id=bid_id or "unknown")
            state = self.reconciliation.reconcile(state)
            state.log_step(
                "ReconciliationAgent",
                "reconcile",
                changes=len((state.draft_fields or {}).get("addendum_changes") or []),
            )
            yield self._agent_event(
                "reconciliation",
                "done",
                changes=len((state.draft_fields or {}).get("addendum_changes") or []),
            )

        if "validator" in selected:
            yield self._agent_event("validator", "running")

            def extract_fn(current: AgentState) -> AgentState:
                current = self.extraction.extract(current, bid_id=bid_id or "unknown")
                current.log_step(
                    "ExtractionAgent",
                    "extract",
                    fields=len((current.draft_fields or {}).get("fields") or {}),
                )
                return current

            def validate_fn(current: AgentState) -> AgentState:
                current = self.validator.validate(current)
                current.log_step(
                    "ValidatorAgent",
                    "validate",
                    validation=(current.draft_fields or {}).get("validation"),
                    rejected_fields=list(current.rejected_fields),
                )
                return current

            if not state.draft_fields:
                state = extract_fn(state)
            state = self.run_validation_loop(state, retrieve_fn, extract_fn, validate_fn)
            yield self._agent_event(
                "validator",
                "done",
                validation=(state.draft_fields or {}).get("validation"),
            )

        qa_output: dict[str, Any] | None = None
        if "qa" in selected:
            yield self._agent_event("qa", "running")
            if not state.evidence:
                state = retrieve_fn(state)
            snippets, citations, raw_evidence = self.qa.collect_evidence(state.evidence, goal)
            parts: list[str] = []
            rephrase_source = "offline"
            if snippets:
                for event in self.qa.iter_rephrase_events(goal, snippets):
                    if event["type"] == "token":
                        parts.append(event["text"])
                        yield {"type": "token", "data": {"text": event["text"]}}
                    elif event["type"] == "source":
                        rephrase_source = event["source"]
                rephrased = self.qa._normalize_rephrased("".join(parts), goal, snippets, rephrase_source)
                if rephrased != "".join(parts).strip():
                    yield {"type": "replace", "data": {"text": rephrased}}
                if rephrase_source == "gemini" and rephrased.startswith("I could not find a direct answer"):
                    rephrase_source = "gemini+offline"
                blended = self.qa._blend_answer(rephrased, snippets)
            else:
                rephrased = "I could not find an answer to that question in the retrieved documents."
                blended = rephrased
                yield {"type": "token", "data": {"text": rephrased}}
            qa_output = {
                "question": goal,
                "answer": blended,
                "rephrased_answer": rephrased,
                "raw_evidence": raw_evidence,
                "sources": citations,
                "rephrase_source": rephrase_source,
            }
            state.final_output = qa_output
            state.log_step(
                "QAAgent",
                "answer",
                rephrase_source=rephrase_source,
                sources=len(citations),
            )
            yield self._agent_event("qa", "done", sources=len(citations))

        if "report" in selected or needs_fields or qa_output is not None:
            yield self._agent_event("report", "running")
            if needs_fields or "extraction" in selected:
                state = self.report.report_extraction(state)
                if qa_output is not None:
                    state.final_output = {
                        **(state.final_output or {}),
                        "question": goal,
                        "rephrased_answer": qa_output.get("rephrased_answer"),
                        "answer": qa_output.get("answer"),
                        "sources": qa_output.get("sources"),
                        "raw_evidence": qa_output.get("raw_evidence"),
                        "rephrase_source": qa_output.get("rephrase_source"),
                    }
            else:
                if qa_output is None:
                    snippets, citations, raw_evidence = self.qa.collect_evidence(state.evidence, goal)
                    summary = "\n\n".join(snippets[:3]) if snippets else "No evidence passages found."
                    qa_output = {
                        "question": goal,
                        "answer": summary,
                        "rephrased_answer": summary,
                        "raw_evidence": raw_evidence,
                        "sources": citations,
                        "rephrase_source": "retrieval",
                    }
                    yield {"type": "token", "data": {"text": summary}}
                state = self.report.report_answer(state, qa_output)
            yield self._agent_event("report", "done")

        final = state.final_output or {}
        final["route"] = route
        final["steps"] = [step.model_dump(mode="json") for step in state.steps]
        try:
            saved = persist_report(final, prefix="orchestrator_report")
            final["report_path"] = str(saved)
        except OSError as exc:
            final["report_path_error"] = str(exc)
        yield {"type": "done", "data": final}

    def run_extraction(
        self,
        *,
        bid_id: str,
        input_dir: str | None = None,
        index_callback: Callable[[list[dict[str, Any]]], int] | None = None,
    ) -> AgentState:
        events = list(
            self.iter_routed(
                f"extract bid {bid_id}",
                bid_id=bid_id,
                input_dir=input_dir,
                index_callback=index_callback,
            )
        )
        done = next((event["data"] for event in events if event["type"] == "done"), {})
        state = AgentState(bid_id=bid_id, goal=f"extract bid {bid_id}", final_output=done)
        state.plan = list((done.get("route") or {}).get("selected_agent_ids") or [])
        return state

    def run_ask(self, question: str, bid_id: str | None = None) -> AgentState:
        events = list(self.iter_routed(question, bid_id=bid_id))
        done = next((event["data"] for event in events if event["type"] == "done"), {})
        state = AgentState(bid_id=bid_id, goal=question, final_output=done)
        state.plan = list((done.get("route") or {}).get("selected_agent_ids") or [])
        return state

    def _agent_event(self, agent_id: str, status: str, **extra: Any) -> dict[str, Any]:
        meta = get_agent(agent_id) or {"id": agent_id, "name": agent_id, "title": agent_id, "responsibility": ""}
        return {
            "type": "agent",
            "data": {
                "id": meta.get("id", agent_id),
                "name": meta.get("name", agent_id),
                "title": meta.get("title", agent_id),
                "responsibility": meta.get("responsibility", ""),
                "status": status,
                **extra,
            },
        }


def catalog_payload() -> list[dict[str, Any]]:
    return list(AGENT_CATALOG)
