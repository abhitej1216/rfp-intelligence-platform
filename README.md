# RFP Intelligence Platform

AI Engineering Assignment — RAG Search Engine & Multi-Agent System.

## Setup

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirement.txt
```

Create `.env` from `.env.example`:
```cmd
copy .env.example .env
```

## Dependencies
- fastapi, uvicorn, streamlit
- pydantic, google-generativeai, qdrant-client
- langgraph (optional), beautifulsoup4, pypdf
- sentence-transformers, rank-bm25, pytest, ruff

## How to Run Each Mode

### Ingestion
```cmd
python scripts\ingest_documents.py --input-dir data\incoming --output data\processed\documents.json
```

### Indexing
```cmd
python scripts\index_vectors.py
```

### Retrieval Evaluation
```cmd
PYTHONPATH=. python scripts\evaluate_retrieval.py
```

### Hybrid Search CLI
```cmd
python scripts\search_hybrid.py
```

### API (FastAPI)
```cmd
python -m uvicorn app.api.main:app --reload
```
Endpoints: `GET /health`, `POST /index`, `GET /search`, `POST /ask`

### Multi-Agent Extraction
```cmd
python scripts\agent_trace_example.py
```

### Generate Final JSON Outputs
```cmd
PYTHONPATH=. python scripts\generate_outputs.py
```
Outputs: `data/Bid1_output.json`, `data/Bid2_output.json`

## Design Decisions

### Chunking
- Size: ~1000 characters with 150-character overlap
- Strategy: whitespace-boundary split, page-aware segments preserved
- IDs: deterministic SHA-256 (`hashlib.sha256`) for idempotent indexing

### Embedding Model
- `sentence-transformers/all-MiniLM-L6-v2` (local, no external API required for embeddings)
- Normalized embeddings (`normalize_embeddings=True`)
- Dimension validated against Qdrant collection config

### Retrieval Strategy
- Hybrid: dense vector (Qdrant, cosine distance) + BM25 keyword (`rank-bm25`)
- Fusion: Reciprocal Rank Fusion (`rrf_k=60`)
- Metadata filtering: `bid_identifier`, `source_filename`, `is_addendum` supported in both indexes
- Query rewriting: modular `QueryRewriter` using Gemini SDK with safe fallback to original query
- Reranking: optional local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with graceful no-op fallback

### Agent Framework
- Custom lightweight orchestrator (`AgentState` Pydantic model)
- Roles: Orchestrator, Retrieval, Extraction, Addendum Reconciliation, Validator, Q&A
- Shared typed state: plan, evidence, draft_fields, validation_results, final_output
- Bounded retries (`max_retries=3`) with feedback loop (validator rejects → re-extraction)
- Structured agent traces: inputs, outputs, tool calls, latency
- No full LangGraph dependency; framework-agnostic for testability

### Prompts / Extraction
- 20-field exact schema (`app/schemas/extraction.py`)
- Evidence-based keyword matching for extraction (`app/extraction/extractor.py`)
- Every field: value + sources + confidence + notes/reason
- Missing fields: explicit `null` with `"Not found in documents"` — never invented
- Addendum reconciliation: `addendum_changes` list in output; superseded facts replaced

### Security & Privacy
- `.env` for secrets; `.env.example` with placeholders only
- `.gitignore` excludes `.env`, `.venv`, `__pycache__`
- No hardcoded API keys, bid identifiers, or document paths
- Local Qdrant persistence (`data/qdrant/`); no external upload of confidential RFPs

### Baseline Metrics (Reproducible)
| Metric | Vector | BM25 | Hybrid |
|---|---|---|---|
| Recall@1 | 0.4000 | 0.4667 | 0.5333 |
| Recall@3 | 0.6000 | 0.6667 | 0.7667 |
| Recall@5 | 0.7000 | 0.7333 | 0.7667 |
| MRR@10 | 0.5267 | 0.6336 | 0.6528 |

### Known Limitations
- E04 (Dallas RFP submission instructions) ranks poorly in baseline; query rewriting improves it but is optional
- Extraction uses evidence keyword matching; full LLM-based field extraction requires Gemini API activation
- Streamlit UI is minimal; full interactive dashboard is a future enhancement
- Docker and CI pipeline are optional and not included
