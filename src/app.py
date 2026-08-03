"""OpsAssist AI — Streamlit chat UI.

Implements Phase 7 of the project plan:

* Chat input with a mode toggle (Docs Q&A vs Explain Log).
* Streaming-style answer + numbered source list with provenance.
* 👍 / 👎 feedback buttons persisted to ``data/feedback/feedback.csv``.
* Sidebar with example questions and the monitoring dashboard link.

The UI is intentionally defensive — it survives missing OpenAI keys,
missing Qdrant connections, and a not-yet-built index by either showing a
degraded-mode message or by switching to the offline log explainer.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from ingest_core import (  # noqa: E402
    EMBED_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    collection_count,
    ensure_collection,
    get_qdrant_client,
)
from rag import rag_pipeline  # noqa: E402
from tools.log_explainer import explain_log  # noqa: E402

FEEDBACK_DIR = REPO_ROOT / "data" / "feedback"
FEEDBACK_PATH = FEEDBACK_DIR / "feedback.csv"
REQUIRED_COLUMNS = [
    "timestamp",
    "question",
    "answer",
    "rating",
    "mode",
    "latency_ms",
    "tokens",
]


def _ensure_feedback_dir() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    if not FEEDBACK_PATH.exists():
        with FEEDBACK_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(REQUIRED_COLUMNS)


def _append_feedback(record: dict[str, Any]) -> None:
    _ensure_feedback_dir()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writerow(record)


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("mode", "Docs Q&A")
    st.session_state.setdefault("pending_log", "")


def _format_sources(sources: list[dict[str, Any]]) -> None:
    """Render retrieved sources as a numbered list.

    Note: this is always called from inside the caller's ``st.expander``
    ("Sources") in ``main()``, so we must NOT use ``st.expander`` here —
    Streamlit rejects nested expanders with a ``StreamlitAPIException``.
    """
    if not sources:
        st.info("No source passages matched this query.")
        return
    for idx, src in enumerate(sources, start=1):
        score = src.get("rerank_score") or src.get("score") or 0
        source = src.get("source") or "unknown"
        chunk_index = src.get("chunk_index")
        rank = src.get("rank")
        header = (
            f"**[{idx}] · score {float(score):.3f} · "
            f"`{source}` · chunk {chunk_index}**"
        )
        st.markdown(header)
        st.write((src.get("text") or "").strip())
        if rank is not None:
            st.caption(f"rank {rank}")


EXAMPLE_QUESTIONS = [
    "How do I scale a service in docker compose?",
    "Why does Nginx use an event-driven architecture?",
    "How do I set memory limits on a Docker container?",
    "What firewall ports does Nginx need open?",
    "Explain this log: OOMKilled — exit code 137",
]


def _sidebar() -> None:
    with st.sidebar:
        st.header("OpsAssist AI")
        st.caption(
            f"Qdrant: `{QDRANT_URL}`\n\n"
            f"Collection: `{QDRANT_COLLECTION}`\n\n"
            f"Embed model: `{EMBED_MODEL}`"
        )
        try:
            client = get_qdrant_client()
            ensure_collection(client)
            st.metric("Indexed chunks", collection_count(client))
        except Exception as exc:
            st.warning(f"Qdrant unreachable: {exc}")

        st.session_state.mode = st.radio(
            "Mode", ["Docs Q&A", "Explain Log"], index=0
        )
        st.markdown("**Example questions**")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"example:{example}"):
                st.session_state["pending_input"] = example

        st.markdown("---")
        st.caption(
            "Run `streamlit run src/monitoring.py` for the dashboard."
        )


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                cols = st.columns([1, 1, 6])
                if cols[0].button("👍", key=f"up:{id(message)}"):
                    _append_feedback(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "question": message.get("user_question", ""),
                            "answer": message["content"],
                            "rating": 5,
                            "mode": message.get("mode", "docs"),
                            "latency_ms": message.get("latency_ms", 0),
                            "tokens": message.get("tokens", 0),
                        }
                    )
                    st.toast("Thanks for the feedback!")
                if cols[1].button("👎", key=f"down:{id(message)}"):
                    _append_feedback(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "question": message.get("user_question", ""),
                            "answer": message["content"],
                            "rating": 1,
                            "mode": message.get("mode", "docs"),
                            "latency_ms": message.get("latency_ms", 0),
                            "tokens": message.get("tokens", 0),
                        }
                    )
                    st.toast("Logged — we'll look into it.")
                with cols[2].expander("Sources"):
                    _format_sources(message.get("sources", []))


def _handle_docs_query(question: str) -> dict[str, Any]:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-6:]
        if m["role"] in ("user", "assistant")
    ]
    t0 = time.perf_counter()
    try:
        result = rag_pipeline(query=question, history=history)
    except Exception as exc:
        return {
            "answer": f"The RAG pipeline failed: {exc}",
            "sources": [],
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "tokens": 0,
        }
    latency_ms = int((time.perf_counter() - t0) * 1000)
    timings = result.get("timings", {})
    # Rough token estimate from answer length (4 chars ≈ 1 token) so the
    # monitoring dashboard has *something* to plot.
    tokens = int(len(result.get("answer", "")) / 4)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "rewritten_query": result.get("rewritten_query", ""),
        "timings": timings,
        "latency_ms": latency_ms,
        "tokens": tokens,
    }


def _handle_log_query(log_text: str, environment: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = explain_log(log_text=log_text, environment=environment)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "latency_ms": latency_ms,
        "tokens": int(len(result.get("answer", "")) / 4),
        "environment": result.get("environment", environment),
    }


def main() -> None:
    """Render the Streamlit chat UI."""
    st.set_page_config(page_title="OpsAssist AI", page_icon="🛠️", layout="wide")
    _init_state()
    _sidebar()

    st.title("OpsAssist AI — DevOps Q&A + Log Explainer")
    st.caption(
        "Ask a question about Docker / Nginx, or paste a log block and pick "
        "the environment on the left to get a structured incident report."
    )

    _render_history()

    if st.session_state.mode == "Docs Q&A":
        prompt = st.chat_input("Ask a Docker or Nginx question")
    else:
        prompt = None
        log_text = st.text_area(
            "Paste a log block",
            value=st.session_state.get("pending_log", ""),
            height=200,
            placeholder="kubectl logs, systemd journal, nginx error.log ...",
        )
        environment = st.selectbox(
            "Environment", ["auto", "kubernetes", "docker", "nginx", "systemd"]
        )
        if st.button("Explain log", type="primary"):
            if log_text.strip():
                st.session_state["pending_log"] = log_text
                prompt = log_text
                st.session_state["_env"] = environment

    if st.session_state.get("pending_input"):
        prompt = st.session_state.pop("pending_input")
        st.session_state["_env"] = "auto"

    if not prompt:
        return

    user_message = prompt if st.session_state.mode == "Explain Log" else prompt
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    if st.session_state.mode == "Explain Log":
        environment = st.session_state.get("_env", "auto")
        response = _handle_log_query(user_message, environment=environment)
    else:
        response = _handle_docs_query(user_message)

    answer = response["answer"]
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Sources"):
            _format_sources(response.get("sources", []))
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "user_question": user_message,
            "sources": response.get("sources", []),
            "mode": "log" if st.session_state.mode == "Explain Log" else "docs",
            "latency_ms": response.get("latency_ms", 0),
            "tokens": response.get("tokens", 0),
        }
    )


def cli(argv: list[str] | None = None) -> int:  # noqa: ARG001
    """Headless entry point used by ``python src/app.py`` from CLI tests."""
    print(json.dumps({"ui_only": True, "hint": "use streamlit run src/app.py"}))
    return 0


if __name__ == "__main__":
    # Streamlit re-runs this module and ``__name__`` is ``"__main__"`` for
    # every page render, so we must call ``main()`` here, not the CLI shim.
    main()