from __future__ import annotations

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    input_dir: str


class SearchRequest(BaseModel):
    q: str
    bid_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class AskRequest(BaseModel):
    question: str
    bid_id: str | None = None
    agent_id: str | None = None


class ExtractRequest(BaseModel):
    bid_id: str
    input_dir: str | None = None


class RouteRequest(BaseModel):
    question: str


class AgentRunRequest(BaseModel):
    agent_id: str
    question: str
    bid_id: str | None = None
    input_dir: str | None = None
