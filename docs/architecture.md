graph TD
    A[Ingestion] --> B[Chunking]
    B --> C[Vector Index Qdrant]
    B --> D[BM25 Index]
    C --> E[Hybrid Search RRF]
    D --> E
    E --> F[Agent Orchestrator]
    F --> G[Retrieval Agent]
    F --> H[Extraction Agent]
    F --> I[Validator Agent]
    F --> J[Q&A Agent]
    G --> E
    H --> K[20-Field JSON]
    I --> H
    J --> L[Cited Answer]
