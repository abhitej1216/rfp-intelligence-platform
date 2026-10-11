from __future__ import annotations

import os


class QueryRewriter:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    def rewrite(self, query: str) -> str:
        # Safe fallback: return original if any issue
        return query
