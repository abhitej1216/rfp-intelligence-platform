from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentKind(str, Enum):
    PDF = "pdf"
    HTML = "html"


class RelationshipStatus(str, Enum):
    LINKED = "linked"
    UNRESOLVED = "unresolved"
    NOT_AN_ADDENDUM = "not_an_addendum"


class TextSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    section_hint: str | None = None


class AddendumRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_addendum: bool
    status: RelationshipStatus
    bid_identifier: str | None = None
    evidence: list[str] = Field(default_factory=list)
    unresolved_reason: str | None = None


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    source_filename: str
    source_relative_path: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_size_bytes: int = Field(ge=0)
    source_kind: DocumentKind
    source_mime_type: str
    ingested_at_utc: datetime
    title: str | None = None
    text: str
    segments: list[TextSegment] = Field(default_factory=list)
    page_count: int | None = Field(default=None, ge=0)
    extraction_warnings: list[str] = Field(default_factory=list)
    addendum_relationship: AddendumRelationship
