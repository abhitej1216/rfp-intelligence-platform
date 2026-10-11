from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.agents.catalog import list_specialist_agents
from app.ui.agent_pages import make_agent_renderer, render_agents_overview
from app.ui.chat_page import render_chat_page
from app.ui.extract_page import render_extract_page
from app.ui.tool_pages import render_health_page, render_index_page, render_search_page

st.set_page_config(page_title="RFP Assistant", page_icon="💬", layout="centered")

master = st.Page(render_chat_page, title="Master Orchestrator", icon="🎯", default=True)
overview = st.Page(render_agents_overview, title="Agents overview", icon="🗂️")
extract = st.Page(render_extract_page, title="Full extract", icon="🧠")

agent_pages = [
    st.Page(make_agent_renderer("orchestrator"), title="Orchestrator", icon="🧭"),
]
for agent in list_specialist_agents():
    icons = {
        "ingestion": "📥",
        "retrieval": "🔎",
        "extraction": "🧩",
        "reconciliation": "🔁",
        "validator": "✅",
        "qa": "💬",
        "report": "📄",
    }
    agent_pages.append(
        st.Page(
            make_agent_renderer(agent["id"]),
            title=agent["title"],
            icon=icons.get(agent["id"], "🤖"),
        )
    )

health = st.Page(render_health_page, title="Health", icon="💚")
index_docs = st.Page(render_index_page, title="Index", icon="📦")
search = st.Page(render_search_page, title="Search", icon="🔍")

navigation = st.navigation(
    {
        "Master": [master, overview, extract],
        "Agents": agent_pages,
        "Tools": [health, index_docs, search],
    }
)
navigation.run()
