from __future__ import annotations

from pathlib import Path

from app.agents.state import AgentState
from app.ingestion.parser import ingest_directory
from app.retrieval.chunking import chunk_documents


class IngestionAgent:
    """Detects file types, parses HTML/PDF, cleans text, attaches metadata, and prepares chunks."""

    def ingest(self, input_dir: str, state: AgentState) -> AgentState:
        path = Path(input_dir)
        if not path.exists():
            state.log_step(
                "IngestionAgent",
                "ingest_directory",
                status="error",
                input_dir=input_dir,
                error="Directory does not exist",
            )
            return state

        records = ingest_directory(str(path))
        chunks = chunk_documents(records)
        state.documents = [
            {
                "source_filename": record.source_filename,
                "document_type": record.document_type,
                "bid_identifier": record.addendum_relationship.bid_identifier,
                "page_count": record.page_count,
            }
            for record in records
        ]
        state.log_step(
            "IngestionAgent",
            "ingest_directory",
            input_dir=input_dir,
            documents=len(records),
            chunks=len(chunks),
            sources=[record.source_filename for record in records],
        )
        # Stash chunks for the orchestrator / indexer without bloating API responses.
        state.final_output = {
            **(state.final_output or {}),
            "ingested_chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            "ingested_documents": len(records),
        }
        return state
