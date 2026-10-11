
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChunkRecord(BaseModel):
    """A searchable text chunk with source provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    document_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_filename: str
    source_relative_path: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_kind: Literal["pdf", "html"]
    title: str | None = None
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)
    section_hint: str | None = None
    bid_identifier: str | None = None
    is_addendum: bool
    addendum_status: Literal[
        "linked", "unresolved", "not_an_addendum"
    ]
