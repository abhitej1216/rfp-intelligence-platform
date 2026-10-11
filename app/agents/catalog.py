from __future__ import annotations

from typing import Any

# Responsibilities taken from the assignment (Part C – Required Agents).
AGENT_CATALOG: list[dict[str, Any]] = [
    {
        "id": "orchestrator",
        "name": "Orchestrator",
        "class_name": "Orchestrator",
        "title": "Orchestrator / Planner",
        "responsibility": (
            "Receives the user goal (\"extract bid\" or a question), breaks it into sub-tasks, "
            "routes them to the right agents, and decides when the job is done."
        ),
        "output": "Execution plan, final response",
        "keywords": (),
    },
    {
        "id": "ingestion",
        "name": "IngestionAgent",
        "class_name": "IngestionAgent",
        "title": "Ingestion Agent",
        "responsibility": (
            "Detects file type, parses HTML/PDF (including tables), cleans text, "
            "attaches metadata and triggers indexing."
        ),
        "output": "Normalized documents + chunks",
        "keywords": (
            "ingest", "index", "parse", "upload", "import", "folder", "html", "pdf", "incoming",
        ),
    },
    {
        "id": "retrieval",
        "name": "RetrievalAgent",
        "class_name": "RetrievalAgent",
        "title": "Retrieval / Search Agent",
        "responsibility": (
            "Rewrites and expands queries, calls the search engine, applies metadata filters "
            "and reranking, returns evidence with citations."
        ),
        "output": "Ranked evidence passages",
        "keywords": (
            "search", "retrieve", "find", "lookup", "evidence", "passage", "hybrid", "bm25",
        ),
    },
    {
        "id": "extraction",
        "name": "ExtractionAgent",
        "class_name": "ExtractionAgent",
        "title": "Extraction Agent",
        "responsibility": (
            "Fills the structured fields using only retrieved evidence. May cover dates & logistics, "
            "commercial & legal, and product & specs."
        ),
        "output": "Field values + source citations",
        "keywords": (
            "extract", "structured", "fields", "json", "bid record",
            "model_no", "part_no", "fill fields", "all fields", "field values",
        ),
    },
    {
        "id": "reconciliation",
        "name": "ReconciliationAgent",
        "class_name": "ReconciliationAgent",
        "title": "Addendum Reconciliation Agent",
        "responsibility": (
            "Compares the base RFP with its addendums and applies the latest valid change to each field "
            "(e.g. extended due date)."
        ),
        "output": "Change log + updated fields",
        "keywords": (
            "addendum", "reconcile", "override", "supersede", "changed", "addenda", "revision",
        ),
    },
    {
        "id": "validator",
        "name": "ValidatorAgent",
        "class_name": "ValidatorAgent",
        "title": "Validator / Critic Agent",
        "responsibility": (
            "Checks each field for evidence support, format (dates, numbers), consistency and hallucination. "
            "Sends failed fields back for re-extraction."
        ),
        "output": "Validation report, confidence scores",
        "keywords": (
            "validate", "validation", "critic", "check fields", "hallucination", "confidence",
        ),
    },
    {
        "id": "qa",
        "name": "QAAgent",
        "class_name": "QAAgent",
        "title": "Q&A Agent",
        "responsibility": (
            "Answers free-form user questions over the bids with citations grounded in retrieved evidence."
        ),
        "output": "Answer with citations",
        "keywords": (
            "what", "which", "when", "where", "who", "how", "why", "is ", "are ", "compare", "?",
        ),
    },
    {
        "id": "report",
        "name": "ReportAgent",
        "class_name": "ReportAgent",
        "title": "Report Agent",
        "responsibility": (
            "Produces the final JSON bid record and/or a human-readable summary from validated agent outputs."
        ),
        "output": "Answer with citations, JSON output",
        "keywords": (
            "report", "summary", "final json", "export", "bid summary",
            "final report", "summary export",
        ),
    },
]


def get_agent(agent_id: str) -> dict[str, Any] | None:
    for agent in AGENT_CATALOG:
        if agent["id"] == agent_id or agent["name"].casefold() == agent_id.casefold():
            return agent
    return None


def list_specialist_agents() -> list[dict[str, Any]]:
    return [agent for agent in AGENT_CATALOG if agent["id"] != "orchestrator"]
