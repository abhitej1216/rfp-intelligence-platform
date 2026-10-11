from __future__ import annotations

import requests
import streamlit as st

from app.agents.catalog import AGENT_CATALOG, get_agent
from app.ui.common import (
    BOUNCE_CSS,
    api_request,
    auto_download_report,
    render_sidebar_connection,
    show_bounce,
    stream_ask,
)


def render_agents_overview() -> None:
    st.markdown(BOUNCE_CSS, unsafe_allow_html=True)
    with st.sidebar:
        render_sidebar_connection()

    st.title("Specialist agents")
    st.caption("Each agent matches a role from the assignment. Open an agent page to run it directly.")

    for agent in AGENT_CATALOG:
        with st.container(border=True):
            st.markdown(f"### {agent['title']}")
            st.write(agent["responsibility"])
            st.caption(f"Output: {agent['output']}")
            if agent["id"] == "orchestrator":
                st.info("Use **Master Orchestrator** chat to let this agent route queries.")
            else:
                st.caption(f"Sidebar → Agents → {agent['title']}")


def _render_agent_page(agent_id: str) -> None:
    meta = get_agent(agent_id)
    if not meta:
        st.error(f"Unknown agent: {agent_id}")
        return

    st.markdown(BOUNCE_CSS, unsafe_allow_html=True)
    with st.sidebar:
        render_sidebar_connection()

    st.title(meta["title"])
    st.markdown(f"**Class:** `{meta['name']}`")
    st.write(meta["responsibility"])
    st.caption(f"Expected output: {meta['output']}")

    if agent_id == "orchestrator":
        st.info("The Orchestrator is the master router. Use the Master Orchestrator chat page.")
        sample = st.text_input("Preview routing for a query", value="What is the submission deadline?")
        if st.button("Show selected agents", type="primary"):
            try:
                payload = api_request("POST", "/route", json={"question": sample}).json()
                st.json(payload)
            except requests.RequestException as exc:
                st.error(str(exc))
        return

    default_prompt = {
        "ingestion": "Ingest and index documents from data/incoming",
        "retrieval": "Search for bid submission deadline evidence",
        "extraction": "Extract all structured bid fields as JSON",
        "reconciliation": "Reconcile addendum changes for due date and requirements",
        "validator": "Validate extracted fields for citations and consistency",
        "qa": "What is the bid submission deadline after addendums?",
        "report": "Produce the final cited report for this bid",
    }.get(agent_id, "Run this agent")

    prompt = st.text_area("Query / goal for this agent", value=default_prompt)
    bid_id = st.text_input("Optional bid id", value="")
    input_dir = ""
    if agent_id == "ingestion":
        input_dir = st.text_input("Input directory", value="data/incoming")

    if not st.button(f"Run {meta['title']}", type="primary"):
        return

    route_box = st.empty()
    agent_box = st.empty()
    status = st.empty()
    show_bounce(status, f"Starting {meta['title']}...")
    answer_box = st.empty()

    # Ingestion + Report use the non-stream endpoint so the full JSON is available
    # before we trigger download (more reliable than mid-stream iframe downloads).
    if agent_id in {"ingestion", "report"}:
        try:
            response = api_request(
                "POST",
                "/agent/run",
                json={
                    "agent_id": agent_id,
                    "question": prompt,
                    "bid_id": bid_id or None,
                    "input_dir": input_dir or None,
                },
                timeout=600,
            )
            status.empty()
            if response.status_code >= 400:
                st.error(response.text)
                return
            payload = response.json()
            route = payload.get("route") or {}
            result = payload.get("result") or payload
            with route_box.container():
                st.markdown("**Agents in this run**")
                for agent in route.get("selected_agents") or []:
                    st.write(f"- {agent.get('title')}: {agent.get('responsibility')}")
            agent_box.markdown(f"**Active:** {meta['title']} · done")
            answer_text = result.get("rephrased_answer") or result.get("answer")
            if answer_text:
                answer_box.markdown(answer_text)
            if result.get("record"):
                st.json(result.get("record"))
            with st.expander("Full result", expanded=agent_id == "report"):
                st.json(result)
            auto_download_report(result, prefix=f"{agent_id}_report")
        except requests.RequestException as exc:
            status.empty()
            st.error(str(exc))
        return

    assembled = ""
    final_payload: dict | None = None
    try:
        for event_name, payload in stream_ask(prompt, agent_id=agent_id, bid_id=bid_id or None):
            if event_name == "route":
                with route_box.container():
                    st.markdown("**Agents in this run**")
                    for agent in payload.get("selected_agents") or []:
                        st.write(f"- {agent.get('title')}: {agent.get('responsibility')}")
            elif event_name == "agent":
                show_bounce(
                    status,
                    f"{payload.get('title') or payload.get('name')} · {payload.get('status')}",
                )
                agent_box.markdown(
                    f"**Active:** {payload.get('title')}  \n{payload.get('responsibility')}"
                )
            elif event_name == "token":
                status.empty()
                assembled += payload.get("text") or ""
                answer_box.markdown(assembled)
            elif event_name == "done":
                status.empty()
                final_payload = payload
                assembled = payload.get("rephrased_answer") or payload.get("answer") or assembled
                if assembled:
                    answer_box.markdown(assembled)
                if payload.get("record"):
                    st.json(payload.get("record"))
                if payload.get("validation"):
                    st.caption(f"Validation: {payload.get('validation')}")
                with st.expander("Full result"):
                    st.json(payload)
            elif event_name == "error":
                status.empty()
                st.error(payload.get("detail") or "Request failed")
                return
        if final_payload is not None:
            auto_download_report(final_payload, prefix=f"{agent_id}_report")
        else:
            st.warning("No completed report payload was returned.")
    except requests.RequestException as exc:
        status.empty()
        st.error(str(exc))


def make_agent_renderer(agent_id: str):
    def _render() -> None:
        _render_agent_page(agent_id)

    _render.__name__ = f"render_{agent_id}_agent"
    return _render
