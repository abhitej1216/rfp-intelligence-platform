from __future__ import annotations

import hashlib
from pathlib import Path

ALLOWED_SUFFIXES = {".pdf", ".html", ".htm"}
DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024


class IngestionError(ValueError):
    """Raised when a source cannot be safely ingested."""


def resolve_source_path(root: Path, candidate: Path) -> Path:
    root_resolved = root.resolve(strict=True)
    raw = candidate if candidate.is_absolute() else root_resolved / candidate
    try:
        relative = raw.relative_to(root_resolved)
    except ValueError:
        relative = None
    if relative is not None:
        current = root_resolved
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise IngestionError("Symlink sources are not allowed.")
    resolved = raw.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise IngestionError(
            "Source path is outside the configured input directory."
        ) from exc
    if not resolved.is_file():
        raise IngestionError("Source must be a regular file.")
    if resolved.suffix.lower() not in ALLOWED_SUFFIXES:
        raise IngestionError("Unsupported file type; only PDF and HTML are accepted.")
    return resolved


def read_source_bytes(
    path: Path, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
) -> bytes:
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive.")
    if path.stat().st_size == 0:
        raise IngestionError("Source file is empty.")
    if path.stat().st_size > max_file_bytes:
        raise IngestionError(
            f"Source exceeds the configured size limit ({max_file_bytes} bytes)."
        )
    with path.open("rb") as stream:
        data = stream.read(max_file_bytes + 1)
    if len(data) > max_file_bytes:
        raise IngestionError(
            "Source grew beyond the configured size limit while reading."
        )
    return data


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative_source_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
