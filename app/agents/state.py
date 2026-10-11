from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    agent: str
    action: str
    status: str = "ok"
    detail: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    goal: str = ""
    bid_id: str | None = None
    plan: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    draft_fields: dict[str, Any] = Field(default_factory=dict)
    validation_results: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    final_output: dict[str, Any] | None = None
    retries: int = 0
    max_retries: int = 3
    rejected_fields: list[str] = Field(default_factory=list)

    def log_step(self, agent: str, action: str, status: str = "ok", **detail: Any) -> None:
        self.steps.append(AgentStep(agent=agent, action=action, status=status, detail=detail))
