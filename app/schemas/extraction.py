from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BidField(BaseModel):
    value: str | None
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None

class BidRecord(BaseModel):
    bid_id: str
    fields: dict[
        Literal[
            "Bid Number", "Title", "Due Date", "Bid Submission Type",
            "Term of Bid", "Pre Bid Meeting", "Installation",
            "Bid Bond Requirement", "Delivery Date", "Payment Terms",
            "Additional Documentation Required", "MFG for Registration",
            "Contract or Cooperative to use", "Model_no", "Part_no",
            "Product", "contact_info", "company_name", "Bid Summary",
            "Product Specification"
        ],
        BidField
    ]
    addendum_changes: list[str] = Field(default_factory=list)
    validation: dict = Field(default_factory=lambda: {"passed": 0, "failed": 0, "not_found": 0})
