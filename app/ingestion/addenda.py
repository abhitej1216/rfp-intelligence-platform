
from __future__ import annotations

import re
from pathlib import Path

from app.schemas.document import AddendumRelationship, RelationshipStatus

ADDENDUM_FILENAME_PATTERN = re.compile(
    r"^\s*addend(?:um|a)\b",
    re.IGNORECASE,
)

ADDENDUM_HEADING_PATTERN = re.compile(
    r"^\s*addend(?:um|a)\s+"
    r"(?:no\.?\s*)?[0-9]+\b",
    re.IGNORECASE,
)

# Alphanumeric and hyphenated procurement identifiers.

CODE_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"(?:"
    r"[A-Z]{2,}[0-9][A-Z0-9]*(?:-[A-Z0-9]+)*"
    r"|[A-Z0-9]+-[A-Z0-9]+(?:-[A-Z0-9]+)*"
    r")"
    r"(?![A-Z0-9])",
    re.IGNORECASE,
)


# Numeric identifiers are accepted only when explicitly labeled.
NUMERIC_ID_PATTERN = re.compile(
    r"(?<![A-Z0-9-])\d{5,}(?![A-Z0-9-])",
    re.IGNORECASE,
)
# A label can be followed by its identifier on the same or next line.
LABEL_PATTERN = re.compile(
    r"\b("
    r"porfp\s*(?:number|no\.?)?"
    r"|eMMA\s*(?:project\s*)?(?:number|no\.?)?"
    r"|(?:bid|solicitation|rfp|rfq|project|reference)"
    r"\s*(?:number|no\.?)?"
    r"|request\s+for\s+proposal"
    r")\s*(?:number|no\.?)?\s*[:#-]?\s*(.*)$",
    re.IGNORECASE,
)


def _opening_lines(text: str, limit: int = 60) -> list[str]:
    """Return non-empty opening lines for heading-based matching."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ][:limit]


def _normalise_identifier(value: str) -> str:
    """Normalize whitespace and surrounding punctuation."""
    return value.strip().strip(" \t\r\n.,:;#()[]{}").upper()


def _identifier_from_value(value: str, allow_numeric: bool) -> set[str]:
    """Extract plausible identifier tokens from a labeled value."""
    value = value.strip()
    if not value:
        return set()

    # Remove a leading hash commonly used before identifiers.
    value = value.lstrip("#").strip()

    identifiers = {
        _normalise_identifier(match.group(0))
        for match in CODE_PATTERN.finditer(value)
    }

    if allow_numeric:
        identifiers.update(
            match.group(0)
            for match in NUMERIC_ID_PATTERN.finditer(value)
        )

    return {
        identifier
        for identifier in identifiers
        if identifier
    }


def _heading_identifiers(text: str) -> set[str]:
    """Extract explicit identifiers from opening heading context."""
    lines = _opening_lines(text)
    identifiers: set[str] = set()

    for index, line in enumerate(lines):
        match = LABEL_PATTERN.search(line)
        if not match:
            continue

        label = match.group(1).lower()
        value = match.group(2).strip()

        # Numeric IDs are valid for explicit solicitation/RFP numbers,
        # but not arbitrary project or reference numbers.
        allow_numeric = bool(
            re.search(
                r"solicitation|request\s+for\s+proposal|\brfp\b|\brfq\b",
                label,
                re.IGNORECASE,
            )
        )

        if value:
            identifiers.update(
                _identifier_from_value(value, allow_numeric)
            )
        elif index + 1 < len(lines):
            next_line = lines[index + 1]
            # Do not consume a subsequent label as an identifier.
            if not LABEL_PATTERN.search(next_line):
                identifiers.update(
                    _identifier_from_value(
                        next_line,
                        allow_numeric,
                    )
                )

    return identifiers


def extract_explicit_bid_identifiers(text: str) -> set[str]:
    """Extract explicit procurement identifiers from heading context."""
    return _heading_identifiers(text)


def classify_addendum(
    filename: str,
    text: str,
) -> tuple[bool, str | None]:
    """Classify a document using its filename or explicit heading."""
    if ADDENDUM_FILENAME_PATTERN.search(filename):
        return True, "Filename explicitly identifies an addendum."

    for line in _opening_lines(text):
        if ADDENDUM_HEADING_PATTERN.search(line):
            return True, f"Opening heading identifies an addendum: {line[:200]}"

    return False, None


def determine_addendum_relationship(
    filename: str,
    text: str,
    all_source_texts: dict[str, str],
) -> AddendumRelationship:
    """Link an addendum only when an identifier is corroborated."""
    is_addendum, evidence = classify_addendum(filename, text)

    if not is_addendum:
        return AddendumRelationship(
            is_addendum=False,
            status=RelationshipStatus.NOT_AN_ADDENDUM,
        )

    own_ids = _heading_identifiers(text)
    own_ids.update(_heading_identifiers(Path(filename).stem))

    matching_sources: dict[str, set[str]] = {}

    for other_name, other_text in all_source_texts.items():
        if other_name == filename:
            continue

        other_ids = _heading_identifiers(other_text)
        other_ids.update(_heading_identifiers(Path(other_name).stem))

        shared_ids = own_ids & other_ids
        if shared_ids:
            matching_sources[other_name] = shared_ids

    matched_ids = {
        identifier
        for identifiers in matching_sources.values()
        for identifier in identifiers
    }

    if len(matched_ids) == 1:
        bid_id = next(iter(matched_ids))
        names = sorted(
            name
            for name, identifiers in matching_sources.items()
            if bid_id in identifiers
        )

        return AddendumRelationship(
            is_addendum=True,
            status=RelationshipStatus.LINKED,
            bid_identifier=bid_id,
            evidence=[
                evidence or "Addendum heading detected.",
                (
                    f"Heading identifier {bid_id} is corroborated by: "
                    f"{', '.join(names)}"
                ),
            ],
        )

    if len(matched_ids) > 1:
        reason = (
            "Multiple procurement identifiers matched across heading "
            "evidence; automatic linking was withheld."
        )
    elif not own_ids:
        reason = (
            "Addendum detected, but no explicit procurement identifier "
            "was found in the opening headings."
        )
    else:
        reason = (
            "No matching heading identifier was found in another "
            "ingested source."
        )

    return AddendumRelationship(
        is_addendum=True,
        status=RelationshipStatus.UNRESOLVED,
        evidence=[evidence] if evidence else [],
        unresolved_reason=reason,
    )
