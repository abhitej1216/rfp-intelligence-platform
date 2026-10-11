from __future__ import annotations

import requests
import streamlit as st

from app.ui.common import BOUNCE_CSS, auto_download_report, render_sidebar_connection, show_bounce, stream_ask


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "I am the **Master Orchestrator**. Tell me your goal and I will pick the right "
                    "specialist agents (Ingestion, Retrieval, Extraction, Reconciliation, Validator, "
                    "Q&A, Report) based on your query."
                ),
                "sources": [],
                "raw_evidence": "",
                "route": None,
                "agent_events": [],
            }
        ]


def _render_route(route: dict | None) -> None:
    if not route:
        return
    agents = route.get("selected_agents") or []
    st.markdown("**Orchestrator selected:**")
    for agent in agents:
        st.markdown(
            f"- **{agent.get('title') or agent.get('name')}** — {agent.get('responsibility', '')}"
        )
    reasons = route.get("reasons") or []
    if reasons:
        with st.expander("Routing reasons"):
            for reason in reasons:
                st.write(f"- {reason}")


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("route"):
            _render_route(message["route"])
        st.markdown(message.get("content") or "")
        sources = message.get("sources") or []
        if sources:
            chips = []
            for source in sources:
                page = source.get("page")
                label = source.get("file", "Unknown")
                if page is not None:
                    label = f"{label} · p.{page}"
                chips.append(f'<span class="source-chip">{label}</span>')
            st.markdown("".join(chips), unsafe_allow_html=True)
        raw_evidence = message.get("raw_evidence") or ""
        if raw_evidence:
            with st.expander("Supporting chunks"):
                st.markdown(raw_evidence)


def render_chat_page() -> None:
    st.markdown(BOUNCE_CSS, unsafe_allow_html=True)
    _init_state()

    with st.sidebar:
        render_sidebar_connection(show_clear_chat=True)
        st.caption("Master chat uses the Orchestrator to route each query.")

    st.markdown(
        '<div class="chat-hero"><h1>Master Orchestrator</h1>'
        "<p>Routes your query to the specialist agents defined in the assignment.</p></div>",
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        _render_message(message)

    prompt = st.chat_input("Ask a question, extract a bid, reconcile addendums, validate fields...")
    if not prompt:
        return

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": [], "raw_evidence": "", "route": None, "agent_events": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        route_box = st.empty()
        agent_box = st.empty()
        status = st.empty()
        show_bounce(status, "Orchestrator is choosing agents...")
        answer_box = st.empty()
        meta_box = st.empty()

        assembled = ""
        sources: list[dict] = []
        raw_evidence = ""
        route: dict | None = None
        agent_events: list[dict] = []
        final_payload: dict | None = None
        error_text = ""

        try:
            for event_name, payload in stream_ask(prompt):
                if event_name == "route":
                    route = payload
                    with route_box.container():
                        _render_route(route)
                    show_bounce(status, "Agents selected — starting pipeline...")
                elif event_name == "agent":
                    agent_events.append(payload)
                    title = payload.get("title") or payload.get("name")
                    state = payload.get("status", "")
                    show_bounce(status, f"{title} · {state}")
                    lines = []
                    for item in agent_events[-8:]:
                        mark = "▶" if item.get("status") == "running" else "✓"
                        lines.append(
                            f"{mark} **{item.get('title') or item.get('name')}** "
                            f"({item.get('status')}) — {item.get('responsibility', '')}"
                        )
                    agent_box.markdown("\n\n".join(lines))
                elif event_name == "token":
                    status.empty()
                    assembled += payload.get("text") or ""
                    answer_box.markdown(assembled)
                elif event_name == "replace":
                    status.empty()
                    assembled = payload.get("text") or assembled
                    answer_box.markdown(assembled)
                elif event_name == "done":
                    status.empty()
                    final_payload = payload
                    assembled = (
                        payload.get("rephrased_answer")
                        or payload.get("answer")
                        or assembled
                    )
                    if not assembled and payload.get("record"):
                        assembled = (
                            "Extraction complete. See the structured record and agent trace below."
                        )
                    sources = payload.get("sources") or []
                    raw_evidence = payload.get("raw_evidence") or ""
                    if payload.get("route"):
                        route = payload.get("route")
                        with route_box.container():
                            _render_route(route)
                    answer_box.markdown(assembled or "")
                    if payload.get("record"):
                        with st.expander("Structured bid record", expanded=True):
                            st.json(payload.get("record"))
                    if payload.get("validation"):
                        st.caption(f"Validation: {payload.get('validation')}")
                elif event_name == "error":
                    error_text = payload.get("detail") or "Request failed."
                    break
        except requests.RequestException as exc:
            error_text = str(exc)

        if error_text:
            status.empty()
            answer_box.error(error_text)
            assembled = f"Sorry — I couldn't complete that request.\n\n{error_text}"
        else:
            status.empty()
            if not assembled:
                assembled = "Pipeline finished. Check the selected agents and outputs above."
                answer_box.markdown(assembled)
            chips = []
            for source in sources:
                page = source.get("page")
                label = source.get("file", "Unknown")
                if page is not None:
                    label = f"{label} · p.{page}"
                chips.append(f'<span class="source-chip">{label}</span>')
            if chips:
                meta_box.markdown("".join(chips), unsafe_allow_html=True)
            if raw_evidence:
                with st.expander("Supporting chunks"):
                    st.markdown(raw_evidence)
            if final_payload is not None:
                report = {
                    **final_payload,
                    "question": prompt,
                    "route": route or final_payload.get("route"),
                    "agent_events": agent_events,
                }
                auto_download_report(report, prefix="orchestrator_report")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assembled,
                "sources": sources,
                "raw_evidence": raw_evidence,
                "route": route,
                "agent_events": agent_events,
            }
        )
