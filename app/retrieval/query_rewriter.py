from __future__ import annotations

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class QueryRewriter:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        self.llm = ChatGoogleGenerativeAI(model=model_name, api_key=self.api_key) if self.api_key else None

    def rewrite(self, query: str) -> str:
        if not self.llm:
            return query
        try:
            prompt = ChatPromptTemplate.from_template(
                "Expand this procurement query with synonyms and related terms. Keep it concise.\n"
                "Original: {query}\nRewritten:"
            )
            chain = prompt | self.llm
            result = chain.invoke({"query": query})
            rewritten = result.content if hasattr(result, "content") else str(result)
            if rewritten and len(rewritten) > 2 and rewritten != query:
                return rewritten
        except Exception:  # noqa: BLE001,S110 - safe LLM fallback
            pass
        return query
