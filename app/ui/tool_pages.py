from __future__ import annotations

import requests
import streamlit as st

from app.ui.common import api_request, render_sidebar_connection


def render_health_page() -> None:
    with st.sidebar:
        render_sidebar_connection()

    st.title("Health")
    st.caption("GET /health — use the sidebar Tools / Assistant links to switch pages")

    if st.button("Check API health", type="primary"):
        try:
            response = api_request("GET", "/health")
            st.write(f"Status code: {response.status_code}")
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(str(exc))


def render_index_page() -> None:
    with st.sidebar:
        render_sidebar_connection()

    st.title("Index documents")
    st.caption("POST /index — use the sidebar to switch pages")

    input_dir = st.text_input("Input directory", value="data/incoming")
    if st.button("Run indexing", type="primary"):
        try:
            response = api_request("POST", "/index", json={"input_dir": input_dir})
            st.write(f"Status code: {response.status_code}")
            try:
                st.json(response.json())
            except ValueError:
                st.code(response.text)
        except requests.RequestException as exc:
            st.error(str(exc))


def render_search_page() -> None:
    with st.sidebar:
        render_sidebar_connection()

    st.title("Hybrid search")
    st.caption("GET /search — use the sidebar to switch pages")

    query = st.text_input("Search query", value="bid submission deadline")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
    if st.button("Search documents", type="primary"):
        try:
            response = api_request("GET", "/search", params={"q": query, "top_k": int(top_k)})
            st.write(f"Status code: {response.status_code}")
            payload = response.json()
            st.caption(f"{payload.get('count', 0)} results")
            for idx, item in enumerate(payload.get("results") or [], start=1):
                metadata = item.get("metadata") or {}
                title = metadata.get("source_filename") or f"Result {idx}"
                with st.expander(f"{idx}. {title}"):
                    st.write(item.get("text", ""))
        except requests.RequestException as exc:
            st.error(str(exc))
