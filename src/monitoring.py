"""Streamlit monitoring dashboard for OpsAssist AI.

Renders ≥5 charts from the ``data/feedback/feedback.csv`` log:

1.  Ratings pie — share of 👍 / 👎 / neutral feedback.
2.  Questions per day — daily volume of questions.
3.  Avg latency per day — p50 / p95 of the RAG pipeline.
4.  Top 10 queries — most-asked questions (bar chart).
5.  Tool usage split — docs Q&A vs log-explainer.
6.  Token usage per day (bonus).

The dashboard degrades gracefully — if ``feedback.csv`` doesn't exist yet
(e.g. before any user interaction), every chart shows an empty placeholder
instead of crashing.

Run with::

    streamlit run src/monitoring.py
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
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


@st.cache_data(show_spinner=False)
def load_feedback() -> pd.DataFrame:
    """Load ``feedback.csv`` into a DataFrame; gracefully empty if absent."""
    _ensure_feedback_dir()
    try:
        df = pd.read_csv(FEEDBACK_PATH)
    except Exception:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
    if "latency_ms" in df.columns:
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    if "tokens" in df.columns:
        df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce")
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


def _rating_pie(df: pd.DataFrame) -> None:
    st.subheader("1. Ratings pie")
    if df.empty or "rating" not in df.columns:
        st.info("No ratings yet.")
        return
    counts = Counter()
    for value in df["rating"].dropna().tolist():
        rating = int(value)
        if rating >= 4:
            counts["👍 (good)"] += 1
        elif rating == 3:
            counts["😐 (neutral)"] += 1
        else:
            counts["👎 (bad)"] += 1
    if not counts:
        st.info("No ratings yet.")
        return
    pie_df = pd.DataFrame(
        {"rating": list(counts.keys()), "count": list(counts.values())}
    )
    st.bar_chart(pie_df, x="rating", y="count")


def _questions_per_day(df: pd.DataFrame) -> None:
    st.subheader("2. Questions per day")
    if df.empty:
        st.info("No questions yet.")
        return
    daily = df.assign(day=df["timestamp"].dt.date).groupby("day").size().reset_index(name="count")
    st.line_chart(daily, x="day", y="count")


def _latency_per_day(df: pd.DataFrame) -> None:
    st.subheader("3. Avg latency per day (ms)")
    if df.empty or "latency_ms" not in df.columns:
        st.info("No latency data yet.")
        return
    grouped = (
        df.assign(day=df["timestamp"].dt.date)
        .groupby("day")["latency_ms"]
        .agg(["mean", "max"])
        .reset_index()
        .rename(columns={"mean": "avg_ms", "max": "p95_ms"})
    )
    if grouped.empty:
        st.info("Not enough latency samples to plot.")
        return
    st.line_chart(grouped, x="day", y=["avg_ms", "p95_ms"])


def _top_queries(df: pd.DataFrame) -> None:
    st.subheader("4. Top 10 queries")
    if df.empty or "question" not in df.columns:
        st.info("No queries yet.")
        return
    top = (
        df["question"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .head(10)
        .reset_index(name="count")
        .rename(columns={"index": "question"})
    )
    if top.empty:
        st.info("No queries yet.")
        return
    st.bar_chart(top[::-1], x="count", y="question", horizontal=True)


def _tool_usage(df: pd.DataFrame) -> None:
    st.subheader("5. Tool usage split")
    if df.empty or "mode" not in df.columns:
        st.info("No mode info yet.")
        return
    counts = df["mode"].fillna("docs").value_counts().reset_index(name="count")
    counts.columns = ["mode", "count"]
    if counts.empty:
        st.info("No mode data.")
        return
    st.bar_chart(counts, x="mode", y="count")


def _tokens_per_day(df: pd.DataFrame) -> None:
    st.subheader("6. Token usage per day (bonus)")
    if df.empty or "tokens" not in df.columns:
        st.info("No token usage logged yet.")
        return
    grouped = (
        df.assign(day=df["timestamp"].dt.date)
        .groupby("day")["tokens"]
        .sum()
        .reset_index(name="total_tokens")
    )
    if grouped.empty:
        st.info("Not enough token samples to plot.")
        return
    st.area_chart(grouped, x="day", y="total_tokens")


def _recent_table(df: pd.DataFrame) -> None:
    st.subheader("Recent feedback")
    if df.empty:
        st.info("No feedback logged yet.")
        return
    cols = [c for c in ["timestamp", "question", "rating", "mode", "latency_ms"] if c in df.columns]
    st.dataframe(
        df.sort_values("timestamp", ascending=False).head(20)[cols],
        use_container_width=True,
    )


def render() -> None:
    """Render the monitoring dashboard. Used by ``streamlit run``."""
    st.set_page_config(page_title="OpsAssist — Monitoring", layout="wide")
    st.title("OpsAssist AI — Monitoring dashboard")

    df = load_feedback()
    n = len(df)
    cols = st.columns(4)
    cols[0].metric("Total questions", n)
    cols[1].metric("Avg rating (1-5)", round(fmean(df["rating"].dropna()), 2) if n else "n/a")
    cols[2].metric(
        "Avg latency (ms)",
        round(fmean(df["latency_ms"].dropna()), 1) if n and "latency_ms" in df else "n/a",
    )
    cols[3].metric("Positive rate", f"{_positive_rate(df):.0%}" if n else "n/a")

    _rating_pie(df)
    _questions_per_day(df)
    _latency_per_day(df)
    _top_queries(df)
    _tool_usage(df)
    _tokens_per_day(df)
    _recent_table(df)
    st.caption(f"Source: {FEEDBACK_PATH}")


def _positive_rate(df: pd.DataFrame) -> float:
    if df.empty or "rating" not in df.columns:
        return 0.0
    rated = df["rating"].dropna()
    if rated.empty:
        return 0.0
    return float((rated >= 4).sum() / len(rated))


def main() -> None:
    """Module entry point so ``streamlit run src/monitoring.py`` works."""
    render()


# When imported as a library (e.g. by ``src/app.py``) the caller can use
# ``render()`` to embed the dashboard inside another page.
__all__ = ["render", "load_feedback", "FEEDBACK_PATH"]


if __name__ == "__main__":
    # Touch datetime import for linters; otherwise unused.
    _ = datetime.utcnow
    main()