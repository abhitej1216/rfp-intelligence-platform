Agent Trace Example (Extraction Mode):

Input: Bid folder = Bid1 (JA-207652)

Step 1 - Orchestrator:
  Input: goal="extract structured bid record"
  Output: plan=["retrieve", "extract", "reconcile", "validate", "report"]
  Latency: ~0.01s

Step 2 - Retrieval Agent:
  Tool call: HybridSearch.search(query="due date submission deadline", top_k=5, filters={"bid_identifier":"JA-207652"})
  Output: 5 evidence chunks (chunk_ids: a1, b2, c3, d4, e5)
  Latency: ~0.45s

Step 3 - Extraction Agent:
  Input: evidence=[chunk texts with citations]
  Output: draft_fields (20 fields filled or null with reason)
  Tool: build_bid_record(bid_id="JA-207652", evidence=...)
  Latency: ~0.12s

Step 4 - Reconciliation Agent:
  Input: draft_fields + addendum evidence
  Output: addendum_changes=["Addendum 2 extends due date to July 9, 2024 2:00 PM CST"]
  Latency: ~0.05s

Step 5 - Validator Agent:
  Input: draft_fields
  Output: validation={"passed":18, "failed":0, "not_found":2}, retries=0
  Latency: ~0.08s

Step 6 - Q&A / Report Agent:
  Input: validated fields + evidence
  Output: final_output={"bid_id":"JA-207652", "fields":{...}, "addendum_changes":[...], "validation":{...}}
  Latency: ~0.03s

Total pipeline latency: ~0.74s
No hallucinations detected; all values have citations or explicit null reasons.
