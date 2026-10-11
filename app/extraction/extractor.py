from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from app.schemas.extraction import BidField, BidRecord, SourceCitation

FIELDS = (
    "Bid Number", "Title", "Due Date", "Bid Submission Type", "Term of Bid",
    "Pre Bid Meeting", "Installation", "Bid Bond Requirement", "Delivery Date",
    "Payment Terms", "Additional Documentation Required", "MFG for Registration",
    "Contract or Cooperative to use", "Model_no", "Part_no", "Product",
    "contact_info", "company_name", "Bid Summary", "Product Specification",
)

# Match relevant evidence, not every chunk containing an overly broad keyword.
FIELD_PATTERNS = {
    "Bid Number": (r"\b(?:solicitation|rfp|rfq|porfp)\s*(?:number|no\.?|#)?\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{3,}",),
    "Title": (r"(?im)^.{0,30}(?:request for proposals?|invitation to bid|student and staff computing devices|dell laptops?).{0,160}$",),
    "Due Date": (r"(?i).{0,100}(?:new due date|solicitation due|proposal due|submission deadline|due date).{0,180}",),
    "Bid Submission Type": (r"(?i).{0,120}(?:submit|submission|iSupplier|sealed (?:bid|proposal)|eMMA portal).{0,180}",),
    "Term of Bid": (r"(?i).{0,100}(?:initial term|contract term|renewal option|three years|one-year renewal).{0,180}",),
    "Pre Bid Meeting": (r"(?i).{0,100}(?:pre[- ](?:proposal|bid) meeting|mandatory meeting).{0,180}",),
    "Installation": (r"(?i).{0,100}(?:installation|deployment|imaging|white glove).{0,180}",),
    "Bid Bond Requirement": (r"(?i).{0,100}(?:bid bond|bid security|proposal security).{0,180}",),
    "Delivery Date": (r"(?i).{0,100}(?:delivery within|delivery .*days|within \d+ days of award|delivery deadline).{0,180}",),
    "Payment Terms": (r"(?i).{0,100}(?:invoice|invoicing|payment terms|net \d+|within \d+ days of delivering).{0,180}",),
    "Additional Documentation Required": (r"(?i).{0,100}(?:affidavit|certificate|insurance|w-9|letter of authorization|loa).{0,180}",),
    "MFG for Registration": (r"(?i).{0,100}(?:manufacturer|master contractor|authorized reseller|letter of authorization).{0,180}",),
    "Contract or Cooperative to use": (r"(?i).{0,100}(?:cooperative|master contract|contract vehicle|state contract).{0,180}",),
    "Model_no": (r"\b(?:Latitude\s*\d{4}|WD\d{2}TB\d|[A-Z]{2,}\d{2,}[A-Z0-9-]*)\b",),
    "Part_no": (r"(?i).{0,50}\b(?:sku|part\s*(?:number|no\.?))\b.{0,100}",),
    "Product": (r"(?i).{0,120}(?:qty|quantity|laptops?|dock|chromebook|desktop|monitor).{0,120}",),
    "contact_info": (r"(?i).{0,100}(?:contact|email|phone|buyer|procurement officer).{0,180}",),
    "company_name": (r"(?i).{0,100}(?:Dallas ISD|Dallas Independent School District|Maryland Department|Department of [A-Z][A-Za-z ]+).{0,100}",),
    "Product Specification": (r"(?i).{0,100}(?:processor|CPU|RAM|memory|SSD|storage|display|warranty|Latitude|WD22TB4).{0,180}",),
}


def _evidence_records(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for item in evidence:
        metadata = item.get("metadata") or {}
        normalized.append({
            "text": str(item.get("text", "")),
            "file": metadata.get("source_filename") or item.get("source_filename") or "Unknown source",
            "page": metadata.get("page_number") or item.get("page_number"),
            "chunk_id": item.get("chunk_id"),
        })
    return normalized


def _patterns(field: str) -> tuple[str, ...]:
    if field == "Bid Summary":
        return (r"(?i).{0,160}(?:procure|purchase|seeking|requests|contract|award).{0,180}",)
    return FIELD_PATTERNS.get(field, ())


def _extract_field(name: str, evidence: list[dict[str, Any]]) -> BidField:
    matches: list[tuple[str, dict[str, Any]]] = []
    patterns = _patterns(name)
    for source in evidence:
        text = source["text"]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                snippet = re.sub(r"\s+", " ", match.group(0)).strip(" \t\r\n:;,-")
                if len(snippet) >= 5:
                    matches.append((snippet, source))

    # Deduplicate equivalent repeated lines while preserving document order.
    seen: set[str] = set()
    unique: list[tuple[str, dict[str, Any]]] = []
    for snippet, source in matches:
        key = snippet.casefold()
        if key not in seen:
            seen.add(key)
            unique.append((snippet, source))

    if not unique:
        return BidField(value=None, confidence=0.0, notes="Not found in retrieved evidence")

    if name == "Bid Summary":
        text = " ".join(value for value, _ in unique[:3])
        sentences = re.split(r"(?<=[.!?])\s+", text)
        value = " ".join(sentences[:3])[:650]
    elif name == "Additional Documentation Required":
        labels = sorted({re.sub(r"\s+", " ", value).strip() for value, _ in unique[:8]})
        value = "; ".join(labels)
    elif name in {"Model_no", "Part_no"}:
        vals = []
        for snippet, _ in unique:
            vals.extend(re.findall(r"\b(?:Latitude\s*\d{4}|WD\d{2}TB\d|\d{3}-[A-Z0-9]{4}|[A-Z]{2,}\d{2,}[A-Z0-9-]*)\b", snippet, re.I))
        value = ", ".join(dict.fromkeys(vals)) if vals else unique[0][0]
    else:
        value = unique[0][0]

    citations = []
    for _, source in unique[:4]:
        citation = SourceCitation(file=source["file"], page=source["page"], chunk_id=source["chunk_id"])
        if citation not in citations:
            citations.append(citation)
    confidence = min(0.88, 0.58 + 0.08 * (len(citations) - 1))
    return BidField(value=value, sources=citations, confidence=confidence, notes="Matched field-specific evidence; verify the cited passage")


def build_bid_record(bid_id: str, evidence: list[dict[str, Any]]) -> BidRecord:
    normalized = _evidence_records(evidence)
    fields = {name: _extract_field(name, normalized) for name in FIELDS}
    return BidRecord(
        bid_id=bid_id,
        fields=fields,
        validation={
            "passed": sum(field.value is not None and bool(field.sources) for field in fields.values()),
            "failed": 0,
            "not_found": sum(field.value is None for field in fields.values()),
        },
    )
