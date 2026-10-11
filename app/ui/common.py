from __future__ import annotations

import base64
import json
import os
import re
import shutil
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_API_URL = os.getenv("RFP_API_URL", "http://127.0.0.1:8001")
REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reports"

BOUNCE_CSS = """
<style>
  .block-container { padding-top: 1.4rem; max-width: 860px; }
  .chat-hero h1 {
    font-family: "Source Sans Pro", "Segoe UI", sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.15rem;
    color: var(--text-color);
  }
  .chat-hero p {
    color: var(--text-color);
    opacity: 0.7;
    margin-top: 0;
  }
  .source-chip {
    display: inline-block;
    margin: 0.15rem 0.35rem 0.15rem 0;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    background: var(--secondary-background-color);
    color: var(--text-color);
    opacity: 0.9;
    font-size: 0.8rem;
  }
  .rfp-typing {
    display: inline-flex !important;
    align-items: center;
    gap: 0.55rem;
    margin: 0.1rem 0 0.25rem 0;
    padding: 0;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-color);
    opacity: 0.72;
    font-size: 1rem;
    line-height: 1.6;
  }
  .rfp-typing .rfp-dots {
    display: inline-flex;
    align-items: center;
    gap: 0.22rem;
  }
  .rfp-typing .rfp-dot {
    width: 0.4rem;
    height: 0.4rem;
    border-radius: 50%;
    background: var(--text-color);
    opacity: 0.45;
    display: inline-block;
    animation: rfp-bounce 1.05s infinite ease-in-out;
  }
  .rfp-typing .rfp-dot:nth-child(2) { animation-delay: 0.14s; }
  .rfp-typing .rfp-dot:nth-child(3) { animation-delay: 0.28s; }
  @keyframes rfp-bounce {
    0%, 70%, 100% { transform: translateY(0); opacity: 0.3; }
    35% { transform: translateY(-0.26rem); opacity: 0.9; }
  }
</style>
"""


def api_base() -> str:
    return st.session_state.get("api_base", DEFAULT_API_URL).rstrip("/")


def api_request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{api_base()}{path}"
    return requests.request(method, url, timeout=kwargs.pop("timeout", 180), **kwargs)


def bounce_html(message: str) -> str:
    """Inline typing indicator styled like normal assistant chat text."""
    safe = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        "<style>"
        ".rfp-typing{display:inline-flex!important;align-items:center;gap:.55rem;"
        "margin:.1rem 0 .25rem;padding:0;background:transparent!important;border:none!important;"
        "box-shadow:none!important;color:var(--text-color);opacity:.72;font-size:1rem;line-height:1.6}"
        ".rfp-typing .rfp-dots{display:inline-flex;align-items:center;gap:.22rem}"
        ".rfp-typing .rfp-dot{width:.4rem;height:.4rem;border-radius:50%;background:var(--text-color);"
        "display:inline-block;animation:rfp-bounce 1.05s infinite ease-in-out}"
        ".rfp-typing .rfp-dot:nth-child(2){animation-delay:.14s}"
        ".rfp-typing .rfp-dot:nth-child(3){animation-delay:.28s}"
        "@keyframes rfp-bounce{0%,70%,100%{transform:translateY(0);opacity:.3}"
        "35%{transform:translateY(-.26rem);opacity:.9}}"
        "</style>"
        '<div class="rfp-typing">'
        '<span class="rfp-dots" aria-hidden="true">'
        '<span class="rfp-dot"></span>'
        '<span class="rfp-dot"></span>'
        '<span class="rfp-dot"></span>'
        "</span>"
        f"<span>{safe}</span>"
        "</div>"
    )


def show_bounce(placeholder, message: str) -> None:
    placeholder.markdown(bounce_html(message), unsafe_allow_html=True)


def parse_sse(response: requests.Response) -> Iterator[tuple[str, dict]]:
    event_name = "message"
    data_lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                payload_text = "\n".join(data_lines)
                data_lines = []
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    payload = {"text": payload_text}
                yield event_name, payload
                event_name = "message"
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())


def stream_ask(question: str, *, agent_id: str | None = None, bid_id: str | None = None) -> Iterator[tuple[str, dict]]:
    body: dict[str, object] = {"question": question}
    if agent_id:
        body["agent_id"] = agent_id
    if bid_id:
        body["bid_id"] = bid_id
    with api_request(
        "POST",
        "/ask/stream",
        json=body,
        stream=True,
        headers={"Accept": "text/event-stream"},
        timeout=300,
    ) as response:
        if response.status_code >= 400:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            yield "error", {"detail": str(detail)}
            return
        yield from parse_sse(response)


def render_sidebar_connection(*, show_clear_chat: bool = False) -> None:
    st.markdown("### Connection")
    st.text_input("API base URL", value=DEFAULT_API_URL, key="api_base")
    if show_clear_chat and st.button("Clear chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Ask the Master Orchestrator a new goal.",
                "sources": [],
                "raw_evidence": "",
                "route": None,
                "agent_events": [],
            }
        ]
        st.rerun()


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", name.strip()) or "report"
    return cleaned[:120]


def _user_downloads_dir() -> Path | None:
    home = Path.home()
    candidates = [
        home / "Downloads",
        home / "Download",
        Path(os.environ.get("USERPROFILE", "")) / "Downloads",
    ]
    for candidate in candidates:
        if candidate and candidate.is_dir():
            return candidate
    return None


def auto_download_report(payload: dict, *, prefix: str = "report") -> Path:
    """Persist the report and trigger a browser download (with Downloads-folder fallback)."""
    from app.agents.report_store import persist_report

    # Prefer server-written path when orchestrator already saved the file.
    existing = payload.get("report_path")
    if existing and Path(str(existing)).is_file():
        path = Path(str(existing))
        text = path.read_text(encoding="utf-8")
        filename = path.name
    else:
        path = persist_report(payload, prefix=prefix)
        text = path.read_text(encoding="utf-8")
        filename = path.name

    downloads_dir = _user_downloads_dir()
    if downloads_dir is not None:
        try:
            shutil.copyfile(path, downloads_dir / filename)
        except OSError:
            pass

    # Quiet fallback download control (no path messaging in the chat response).
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    st.download_button(
        label="Download report",
        data=text.encode("utf-8"),
        file_name=filename,
        mime="application/json",
        key=f"download_{prefix}_{stamp}",
    )

    # Attempt iframe auto-click download via Blob (no visible status text).
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    components.html(
        f"""
        <html>
        <body style="margin:0;">
          <a id="rfp-dl" download="{filename}" style="display:none;">download</a>
          <script>
            (function() {{
              try {{
                const b64 = "{encoded}";
                const binary = atob(b64);
                const bytes = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                const blob = new Blob([bytes], {{ type: "application/json" }});
                const url = URL.createObjectURL(blob);
                const link = document.getElementById("rfp-dl");
                link.href = url;
                setTimeout(function() {{
                  link.click();
                  URL.revokeObjectURL(url);
                }}, 150);
              }} catch (err) {{
                console.error(err);
              }}
            }})();
          </script>
        </body>
        </html>
        """,
        height=0,
    )
    return path
