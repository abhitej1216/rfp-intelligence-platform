from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AgentState(BaseModel):
    plan: list[str] = []
    evidence: list[dict[str, Any]] = []
    draft_fields: dict[str, Any] = {}
    validation_results: list[str] = []
    final_output: dict[str, Any] | None = None
    retries: int = 0
    max_retries: int = 3
