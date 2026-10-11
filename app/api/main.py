from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.agents.catalog import AGENT_CATALOG, get_agent
from app.agents.orchestrator import Orchestrator
from app.api.schemas import AgentRunRequest, AskRequest, ExtractRequest, IndexRequest, RouteRequest
from app.ingestion.parser import ingest_directory
from app.retrieval.bm25_store import BM25Store
from app.retrieval.chunking import chunk_documents
from app.retrieval.hybrid_search import HybridSearch
from app.schemas.chunk import ChunkRecord

load_dotenv()

app = FastAPI(title="RFP Intelligence Platform")
CHUNKS_PATH = Path("data/chunks.json")


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _load_chunks() -> list[ChunkRecord]:
    if not CHUNKS_PATH.is_file():
        return []
    return [ChunkRecord.model_validate(item) for item in json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))]


def _write_chunks(chunks: list[ChunkRecord]) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {chunk.chunk_id: chunk for chunk in _load_chunks()}
    existing.update({chunk.chunk_id: chunk for chunk in chunks})
    temp = CHUNKS_PATH.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in existing.values()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(CHUNKS_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agents")
def agents() -> dict[str, object]:
    return {"agents": AGENT_CATALOG, "count": len(AGENT_CATALOG)}


@app.post("/route")
def route(req: RouteRequest) -> dict[str, object]:
    return Orchestrator().select_agents(req.question)


@app.post("/index")
def index(req: IndexRequest) -> dict[str, object]:
    try:
        records = ingest_directory(req.input_dir)
        chunks = chunk_documents(records)
        _write_chunks(chunks)
        try:
            from app.retrieval.vector_store import VectorStore

            with VectorStore() as store:
                indexed_chunks = store.index_chunks(chunks)
        except ImportError:
            indexed_chunks = 0
        return {
            "documents": len(records),
            "chunks": len(chunks),
            "vector_chunks_indexed": indexed_chunks,
            "sources": [record.source_filename for record in records],
        }
    except ImportError:
        return {
            "documents": len(records),
            "chunks": len(chunks),
            "vector_chunks_indexed": 0,
            "sources": [record.source_filename for record in records],
            "warning": "Install optional vector dependencies to index embeddings.",
        }
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/search")
def search(
    q: Annotated[str, Query(min_length=1)],
    bid_id: str | None = None,
    top_k: Annotated[int, Query(ge=1, le=50)] = 5,
) -> dict[str, object]:
    chunks = _load_chunks()
    try:
        from app.retrieval.vector_store import VectorStore

        with VectorStore() as store:
            engine = HybridSearch(store, BM25Store(chunks))
            results = engine.search(q, top_k=top_k, filters={"bid_identifier": bid_id} if bid_id else None)
        return {"query": q, "results": results, "count": len(results)}
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Vector dependencies are unavailable; install requirements.") from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, object]:
    chunks = _load_chunks()
    try:
        from app.retrieval.vector_store import VectorStore

        with VectorStore() as store:
            engine = HybridSearch(store, BM25Store(chunks))
            events = list(
                Orchestrator(engine).iter_routed(
                    req.question,
                    bid_id=req.bid_id,
                    force_agent_id=req.agent_id,
                )
            )
        done = next((event["data"] for event in events if event["type"] == "done"), {})
        return done or {"question": req.question, "answer": "Not found in documents.", "sources": []}
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Vector dependencies are unavailable; install requirements.") from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent/run")
def agent_run(req: AgentRunRequest) -> dict[str, object]:
    if not get_agent(req.agent_id) or req.agent_id == "orchestrator":
        raise HTTPException(status_code=404, detail=f"Unknown specialist agent: {req.agent_id}")
    chunks = _load_chunks()
    try:
        from app.retrieval.vector_store import VectorStore

        with VectorStore() as store:
            engine = HybridSearch(store, BM25Store(chunks))
            orch = Orchestrator(engine)

            def index_and_refresh(raw_chunks: list[dict]) -> int:
                parsed = [ChunkRecord.model_validate(item) for item in raw_chunks]
                _write_chunks(parsed)
                indexed = store.index_chunks(parsed)
                refreshed = _load_chunks()
                orch.hybrid = HybridSearch(store, BM25Store(refreshed))
                return indexed

            events = list(
                orch.iter_routed(
                    req.question,
                    bid_id=req.bid_id,
                    input_dir=req.input_dir,
                    index_callback=index_and_refresh if req.input_dir else None,
                    force_agent_id=req.agent_id,
                )
            )
        done = next((event["data"] for event in events if event["type"] == "done"), {})
        route = next((event["data"] for event in events if event["type"] == "route"), {})
        return {"route": route, "result": done, "events": events}
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Vector dependencies are unavailable; install requirements.") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/extract")
def extract(req: ExtractRequest) -> dict[str, object]:
    chunks = _load_chunks()
    try:
        from app.retrieval.vector_store import VectorStore

        with VectorStore() as store:
            engine = HybridSearch(store, BM25Store(chunks))
            orch = Orchestrator(engine)

            def index_and_refresh(raw_chunks: list[dict]) -> int:
                parsed = [ChunkRecord.model_validate(item) for item in raw_chunks]
                _write_chunks(parsed)
                indexed = store.index_chunks(parsed)
                refreshed = _load_chunks()
                orch.hybrid = HybridSearch(store, BM25Store(refreshed))
                return indexed

            state = orch.run_extraction(
                bid_id=req.bid_id,
                input_dir=req.input_dir,
                index_callback=index_and_refresh if req.input_dir else None,
            )
        return state.final_output or {"bid_id": req.bid_id, "record": {}, "steps": []}
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Vector dependencies are unavailable; install requirements.") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        try:
            chunks = _load_chunks()
            from app.retrieval.vector_store import VectorStore

            with VectorStore() as store:
                engine = HybridSearch(store, BM25Store(chunks))
                for event in Orchestrator(engine).iter_routed(
                    req.question,
                    bid_id=req.bid_id,
                    force_agent_id=req.agent_id,
                ):
                    yield _sse(event["type"], event["data"])
        except ImportError as exc:
            yield _sse("error", {"detail": f"Vector dependencies unavailable: {exc}"})
        except (ValueError, RuntimeError, OSError) as exc:
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
