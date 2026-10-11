# RFP Intelligence Platform

AI Engineering Assignment — RAG Search Engine & Multi-Agent System.

High quality DEMO video:https://drive.google.com/file/d/1NWngJTo-cYCV1gtJJkipiclRL_mBiLv1/view?usp=drive_link

## Setup

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirement.txt
```

Create `.env` from `.env.example`:
```cmd
copy .env.example .env
```

## Dependencies
Install all project dependencies with `python -m pip install -r requirement.txt`. PDF text extraction uses `pypdf`; OCR can be enabled locally when the optional OCR dependencies and Tesseract are installed. Embeddings and vector indexing use Sentence Transformers and local Qdrant.

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
python -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8001
```
Endpoints: `GET /health`, `POST /index`, `GET /search`, `POST /ask`, `POST /ask/stream`, `POST /extract`

### Multi-agent extraction (API)
```cmd
curl -X POST http://127.0.0.1:8001/extract -H "Content-Type: application/json" -d "{\"bid_id\": \"Bid1\"}"
```
Optional body field `input_dir` runs the IngestionAgent before retrieval/extraction. The response includes the 20-field bid record, addendum changes, validation counts, and a full agent step trace.

Agents: Orchestrator (master router), IngestionAgent, RetrievalAgent, ExtractionAgent, ReconciliationAgent, ValidatorAgent, QAAgent, ReportAgent.

The Master Orchestrator picks agents from the query (`POST /route`, streamed on `POST /ask/stream`). Streamlit exposes each agent as its own page under **Agents**, plus the master chat under **Master Orchestrator**.

When a run finishes in the UI, the final report JSON is auto-downloaded in the browser and also saved under `data/reports/`.

### Streamlit Frontend
With the API running, start the UI in another terminal:
```cmd
.venv\Scripts\activate
streamlit run app\ui\streamlit_app.py
```
Chat-style UI streams answers from `POST /ask/stream` (bouncing loader while retrieving/generating). Health, Index, and Search tools live in the sidebar. Set the API base URL in the sidebar if needed (default `http://127.0.0.1:8001`). Gemini rephrases answers when `GEMINI_API_KEY` / `GEMINI_MODEL` are set in `.env`.

### Multi-Agent Extraction (CLI trace)
```cmd
python scripts\agent_trace_example.py Bid1
```
Or use the Streamlit **Extract** page under Assistant.

### Generate Final JSON Outputs
```cmd
python scripts\generate_outputs.py
```
Outputs one JSON per immediate bid subfolder in `data/`. Override the input root with a path argument, for example `python scripts\generate_outputs.py path\to\bids`.

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
- Query rewriting: optional Gemini-backed query rewriter; if unavailable, the original query is retained.
- Reranking: optional local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with graceful no-op fallback.
- Provenance: results retain source filename, page, document type, addendum number, bid identifier, and text.

### Agent Framework
- Custom lightweight orchestrator (`AgentState` Pydantic model)
- Roles: Orchestrator, Retrieval, Extraction, Addendum Reconciliation, Validator, Q&A
- Shared typed state: plan, evidence, draft_fields, validation_results, final_output
- Bounded validation retries are represented in the orchestration API; extraction only accepts values with explicit citations.
- Structured source citations include filename and page where available.
- No full LangGraph dependency; framework-agnostic for testability

### Prompts / Extraction
- 20-field exact schema (`app/schemas/extraction.py`)
- Field-specific evidence matching for the offline baseline (`app/extraction/extractor.py`); it abstains when no matching evidence is found. Treat heuristic extraction as a candidate for human review, not a verified fact.
- Every field: value + sources + confidence + notes/reason
- Missing fields: explicit `null` with `"Not found in documents"` — never invented
- Addendum reconciliation: `addendum_changes` list in output; superseded facts replaced

### Security & Privacy
- `.env` for secrets; `.env.example` with placeholders only
- `.gitignore` excludes `.env`, `.venv`, `__pycache__`
- No hardcoded API keys. Gemini query rewriting is opt-in via environment configuration.
- Local Qdrant persistence (`data/qdrant/`); embedding model download may require network access on first run.
- Generated records and ingested text can contain confidential data; do not publish them without review.

### Baseline Metrics (Reproducible)
| Metric | Vector | BM25 | Hybrid |
|---|---|---|---|
| Recall@1 | 0.4000 | 0.4667 | 0.5333 |
| Recall@3 | 0.6000 | 0.6667 | 0.7667 |
| Recall@5 | 0.7000 | 0.7333 | 0.7667 |
| MRR@10 | 0.5267 | 0.6336 | 0.6528 |

### Known Limitations
- E04 (Dallas RFP submission instructions) ranks poorly in baseline; query rewriting improves it but is optional
- Extraction is a conservative heuristic baseline, not a substitute for bid specialist review. Optional LLM calls should not be used for sensitive bid data without an approved provider/privacy review.
- Streamlit UI is minimal; full interactive dashboard is a future enhancement
- Docker and CI pipeline are optional and not included
