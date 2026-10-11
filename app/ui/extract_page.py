from __future__ import annotations

import json

import requests
import streamlit as st

from app.ui.common import api_request, auto_download_report, render_sidebar_connection, show_bounce


def render_extract_page() -> None:
    with st.sidebar:
        render_sidebar_connection()

    st.title("Multi-agent extraction")
    st.caption(
        "Runs Orchestrator → Ingestion → Retrieval → Extraction → "
        "Reconciliation → Validator → Report"
    )

    bid_id = st.text_input("Bid ID", value="Bid1", help="Logical bid identifier for the structured record")
    input_dir = st.text_input(
        "Optional input directory",
        value="",
        help="If set, IngestionAgent parses this folder and indexes before extraction",
    )
    run = st.button("Run extraction pipeline", type="primary")

    if not run:
        st.info(
            "Required agents: Orchestrator, IngestionAgent, RetrievalAgent, ExtractionAgent, "
            "ReconciliationAgent, ValidatorAgent, ReportAgent / QAAgent."
        )
        return

    status = st.empty()
    show_bounce(status, "Running multi-agent extraction...")
    body: dict[str, object] = {"bid_id": bid_id.strip() or "unknown"}
    if input_dir.strip():
        body["input_dir"] = input_dir.strip()

    try:
        response = api_request("POST", "/extract", json=body, timeout=600)
        status.empty()
        st.write(f"Status code: {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            st.code(response.text)
            return

        if not response.ok:
            st.error(payload)
            return

        st.success("Extraction complete")
        cols = st.columns(3)
        validation = payload.get("validation") or {}
        cols[0].metric("Passed", validation.get("passed", 0))
        cols[1].metric("Failed", validation.get("failed", 0))
        cols[2].metric("Not found", validation.get("not_found", 0))

        st.markdown("### Agent plan")
        st.write(payload.get("plan") or [])

        st.markdown("### Agent trace")
        for idx, step in enumerate(payload.get("steps") or [], start=1):
            title = f"{idx}. {step.get('agent', 'Agent')} · {step.get('action', '')}"
            with st.expander(title, expanded=idx <= 3):
                st.json(step)

        changes = payload.get("addendum_changes") or []
        if changes:
            st.markdown("### Addendum changes")
            for change in changes:
                st.write(f"- {change}")

        st.markdown("### Structured bid record")
        record = payload.get("record") or {}
        st.json(record)

        auto_download_report(payload, prefix="extract_report")
        st.download_button(
            "Download JSON again",
            data=json.dumps(payload, ensure_ascii=False, indent=2),
            file_name=f"{payload.get('bid_id', bid_id)}_extract.json",
            mime="application/json",
        )
    except requests.RequestException as exc:
        status.empty()
        st.error(str(exc))
