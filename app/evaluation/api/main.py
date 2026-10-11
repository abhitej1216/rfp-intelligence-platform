
from fastapi import FastAPI

from app.agents.extraction_agent import ExtractionAgent
from app.agents.orchestrator import Orchestrator
from app.agents.qa_agent import QAAgent
from app.agents.reconciliation import ReconciliationAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.state import AgentState
from app.agents.validator import ValidatorAgent
from app.api.schemas import AskRequest, IndexRequest, SearchRequest
from app.ingestion.parser import ingest_directory
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.vector_store import VectorStore
from app.retrieval.chunking import chunk_documents
from app.schemas.chunk import ChunkRecord

app = FastAPI(title="RFP Intelligence Platform")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/index")
def index(req: IndexRequest):
    records = ingest_directory(req.input_dir)
    return {"indexed": len(records), "documents": [r.source_filename for r in records]}

@app.get("/search")
def search(req: SearchRequest):
    chunks_path = Path("data/chunks.json")
    chunks = []
    if chunks_path.exists():
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [ChunkRecord.model_validate(item) for item in raw]
    with VectorStore() as vs:
        bm25 = BM25Store(chunks)
        hybrid = HybridSearch(vs, bm25)
        results = hybrid.search(req.q, top_k=req.top_k, filters={"bid_identifier": req.bid_id} if req.bid_id else None)
    return {"query": req.q, "results": results, "count": len(results)}

import json
from pathlib import Path

@app.post("/ask")
def ask(req: AskRequest):
    # Load real chunks for retrieval
    chunks_path = Path("data/chunks.json")
    chunks = []
    if chunks_path.exists():
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [ChunkRecord.model_validate(item) for item in raw]
    state = AgentState()
    orch = Orchestrator()
    state = orch.plan(req.question, state)
    with VectorStore() as vs:
        bm25 = BM25Store(chunks)
        hybrid = HybridSearch(vs, bm25)
        ret_agent = RetrievalAgent(hybrid)
        state = ret_agent.retrieve(req.question, state)
    ext_agent = ExtractionAgent()
    state = ext_agent.extract(state)
    rec_agent = ReconciliationAgent()
    state = rec_agent.reconcile(state)
    val_agent = ValidatorAgent()
    state = val_agent.validate(state)
    qa_agent = QAAgent()
    state = qa_agent.answer(req.question, state)
    return {
        "question": req.question,
        "answer": state.final_output.get("answer") if state.final_output else "No answer generated.",
        "sources": state.final_output.get("sources", []) if state.final_output else [],
        "plan": state.plan,
        "retries": state.retries,
    }
