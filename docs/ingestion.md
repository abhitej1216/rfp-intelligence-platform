# Document ingestion

## Run
Put bid documents in a dedicated input folder, such as `data/incoming/`, then run:

```cmd
python scripts\ingest_documents.py --input-dir data\incoming --output data\processed\documents.json
```

Default file limit is 25 MiB. Output JSON contains source text and may be confidential; do not commit it.

## Provenance
Records include source filename/path, SHA-256, byte size, format/MIME type, UTC ingestion time,
title when available, extracted text, PDF page segments, warnings, and addendum relationship.

## Addenda policy
Addendum wording is detected in source filename/text. Linking requires a unique explicit bid
identifier shared with another ingested source. Title similarity alone is insufficient. Ambiguous
relationships remain `unresolved`. This does not yet interpret supersession language such as
“replaces section 4”; that belongs in a later evidence-aware analysis stage.

## Security and limitations
Only PDF and HTML are accepted. Paths must stay under the input root; symlinks are rejected.
Files are size-limited and PDFs must have a PDF signature. Encrypted PDFs are rejected. Text-only
PDF extraction does not OCR scanned documents; affected pages are flagged. HTML scripts, styles,
templates, SVG and noscript elements are removed. Documents are untrusted data, never instructions.
For production, add isolated worker limits, optional malware scanning, and authentication around
uploads.
