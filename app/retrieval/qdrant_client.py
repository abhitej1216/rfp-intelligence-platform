from __future__ import annotations

import os

from qdrant_client import QdrantClient


class QdrantStore:
    def __init__(self, url: str | None = None, collection: str = "rfp_documents"):
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=url)
        self.collection = collection

    def health(self) -> bool:
        return self.client.get_collections().status == "ok"
