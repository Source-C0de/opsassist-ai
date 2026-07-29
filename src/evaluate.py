"""Evaluation harness for OpsAssist AI.

This module implements the Phase 6 deliverables:

* Retrieval evaluation — Hit Rate@k, MRR, and NDCG@5 — by running a set of
  manually-written queries (in ``data/eval/eval.jsonl``) through each of four
  retrieval strategies and comparing the rankings to a list of relevant
  ``chunk_id`` values.
* LLM-as-judge evaluation — a separate pass where a strong model (default
  ``gpt-4o-mini``) grades generated answers on relevance, faithfulness, and
  completeness on a 1-5 scale.

Each pass writes its results to ``data/eval/`` (``retrieval_report.json`` /
  ``llm_judge_report.json``).  Both reports are reproducible from a clean
  clone as long as ``OPENAI_API_KEY`` is set in the environment.

Usage
-----
    python src/evaluate.py retrieval
    python src/evaluate.py llm
    python src/evaluate.py all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv

load_dotenv()

# Repo-root paths so the script behaves the same from any cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"
EVAL_PATH = EVAL_DIR / "eval.jsonl"
RETRIEVAL_REPORT = EVAL_DIR / "retrieval_report.json"
LLM_REPORT = EVAL_DIR / "llm_judge_report.json"

# Retrieval strategies — each is a string passed to ``run_strategy``.
STRATEGIES: list[str] = [
    "vector_only_k5",
    "vector_only_k20_rerank",
    "hybrid_k20_rerank",
    "hybrid_k20_rerank_rewrite",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class EvalItem:
    """One line from ``data/eval/eval.jsonl``."""

    qid: str
    question: str
    relevant_ids: list[str]
    reference_answer: str
    difficulty: str = "medium"
    section: str = ""

    @classmethod
    def from_jsonl(cls, line: str) -> "EvalItem":
        record = json.loads(line)
        return cls(
            qid=str(record.get("qid") or record.get("id") or ""),
            question=record.get("question", ""),
            relevant_ids=list(record.get("relevant_ids") or []),
            reference_answer=record.get("reference_answer") or record.get("answer", ""),
            difficulty=record.get("difficulty", "medium"),
            section=record.get("section", ""),
        )


def load_eval_set(path: Path = EVAL_PATH) -> list[EvalItem]:
    """Load the manual eval set from JSONL; return [] if the file is absent."""
    if not path.exists():
        return []
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(EvalItem.from_jsonl(line))
    return items


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Return 1 if any retrieved id appears in ``relevant`` within the top-k."""
    top = set(retrieved[:k])
    return 1.0 if any(rid in top for rid in relevant) else 0.0


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant hit (0 if none)."""
    rel = set(relevant)
    for idx, rid in enumerate(retrieved, start=1):
        if rid in rel:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ k (binary relevance)."""
    rel = set(relevant)
    dcg = 0.0
    for idx, rid in enumerate(retrieved[:k], start=1):
        if rid in rel:
            dcg += 1.0 / math.log2(idx + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(rel)) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------
def _vector_only(query: str, top_k: int):
    """Dense-only retrieval via the Qdrant client (no BM25, no rerank)."""
    from ingest_core import embed_texts, get_qdrant_client, ensure_collection

    client = get_qdrant_client()
    ensure_collection(client)
    query_vec = embed_texts([query])[0]
    response = client.query_points(
        collection_name=os.getenv("QDRANT_COLLECTION", "opsassist"),
        query=query_vec,
        limit=top_k,
        with_payload=False,
    )
    return [str(point.id) for point in response.points]


def _hybrid(query: str, top_k: int, retrieve_k: int):
    """Hybrid RRF retrieval — same code path as ``retriever.HybridRetriever``."""
    from retriever import HybridRetriever

    return [
        hit["id"]
        for hit in HybridRetriever.from_disk().retrieve(query, top_k=retrieve_k)
    ][:top_k]


def _maybe_rerank(query: str, hits: Sequence[str], top_k: int) -> list[str]:
    """Cross-encoder rerank if available; otherwise return input order."""
    from reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    if not reranker.available:
        return list(hits)[:top_k]
    # Wrap ids back into minimal dicts for the reranker's API contract.
    pairs = [{"id": cid, "text": ""} for cid in hits]
    ranked = reranker.rerank(query, pairs, top_k=top_k)  # type: ignore[arg-type]
    return [r.hit["id"] for r in ranked]


def _maybe_rewrite(query: str, history=None) -> str:
    """LLM-based rewrite; safe fallback to the original query."""
    from query_rewriter import rewrite_query

    return rewrite_query(query, history or []) or query


def run_strategy(strategy: str, item: EvalItem) -> list[str]:
    """Run ``strategy`` against ``item.question``; return retrieved chunk ids."""
    query = item.question

    if strategy == "vector_only_k5":
        return _vector_only(query, top_k=5)
    if strategy == "vector_only_k20_rerank":
        ids = _vector_only(query, top_k=20)
        return _maybe_rerank(query, ids, top_k=5)
    if strategy == "hybrid_k20_rerank":
        ids = _hybrid(query, top_k=20, retrieve_k=20)
        return _maybe_rerank(query, ids, top_k=5)
    if strategy == "hybrid_k20_rerank_rewrite":
        rewritten = _maybe_rewrite(query)
        ids = _hybrid(rewritten, top_k=20, retrieve_k=20)
        return _maybe_rerank(rewritten, ids, top_k=5)
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def evaluate_retrieval(
    items: list[EvalItem] | None = None,
    strategies: Iterable[str] = STRATEGIES,
    k: int = 5,
    out_path: Path = RETRIEVAL_REPORT,
) -> dict:
    """Run every strategy on every query; write and return a structured report."""
    items = items or load_eval_set()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not items:
        empty = {"k": k, "items": 0, "results_per_strategy": {}, "winner": None}
        out_path.write_text(json.dumps(empty, indent=2))
        return empty

    results: dict[str, dict[str, list[float]]] = {
        s: {"hit_rate": [], "mrr": [], "ndcg": []} for s in strategies
    }

    for item in items:
        for strategy in strategies:
            try:
                retrieved = run_strategy(strategy, item)
            except Exception as exc:
                # A failing strategy shouldn't blow up the whole report.
                retrieved = []
                print(f"[warn] strategy '{strategy}' failed on qid={item.qid}: {exc}")
            results[strategy]["hit_rate"].append(
                hit_rate_at_k(retrieved, item.relevant_ids, k)
            )
            results[strategy]["mrr"].append(mrr(retrieved, item.relevant_ids))
            results[strategy]["ndcg"].append(ndcg_at_k(retrieved, item.relevant_ids, k))

    summary: dict[str, dict[str, float]] = {}
    for strategy, metrics in results.items():
        summary[strategy] = {
            "hit_rate_at_5": round(statistics.fmean(metrics["hit_rate"]), 4),
            "mrr": round(statistics.fmean(metrics["mrr"]), 4),
            "ndcg_at_5": round(statistics.fmean(metrics["ndcg"]), 4),
        }

    # Pick the winner by primary metric (Hit Rate@5).
    winner = (
        max(summary, key=lambda s: summary[s]["hit_rate_at_5"]) if summary else None
    )
    report = {
        "k": k,
        "items": len(items),
        "results_per_strategy": summary,
        "winner": winner,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote retrieval report -> {out_path}")
    print(json.dumps(summary, indent=2))
    if winner:
        print(f"Winner: {winner}")
    return report


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

JUDGE_SYSTEM = (
    "You are a strict evaluator of DevOps Q&A answers. "
    "Score the assistant answer on three axes, each 1-5 (5 best): "
    "relevance (does it answer the question?), "
    "faithfulness (does it stick to the provided reference / docs?), "
    "completeness (does it cover the key points?). "
    "Return ONLY a JSON object with keys relevance, faithfulness, completeness, "
    "and a one-sentence comment."
)


def judge_answer(question: str, reference: str, answer: str) -> dict:
    """Use an LLM to grade a generated answer; return a parsed dict."""
    if not answer.strip():
        return {
            "relevance": 0,
            "faithfulness": 0,
            "completeness": 0,
            "comment": "empty answer",
        }
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "relevance": 0,
            "faithfulness": 0,
            "completeness": 0,
            "comment": "OPENAI_API_KEY missing",
        }
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Reference answer: {reference or '(none provided)'}\n\n"
                        f"Assistant answer: {answer}\n\n"
                        "Scores (JSON):"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = (response.choices[0].message.content or "").strip()
        # The model sometimes wraps JSON in ```json fences — strip them.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as exc:
        return {
            "relevance": 0,
            "faithfulness": 0,
            "completeness": 0,
            "comment": f"judge error: {exc}",
        }


def evaluate_llm(
    items: list[EvalItem] | None = None,
    out_path: Path = LLM_REPORT,
) -> dict:
    """Run the RAG pipeline + judge every answer; write and return a report."""
    from rag import rag_pipeline

    items = items or load_eval_set()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    graded: list[dict] = []
    latencies: list[float] = []
    for item in items:
        t0 = time.perf_counter()
        try:
            result = rag_pipeline(
                query=item.question, rewrite=False, rerank=False
            )
            answer = result.get("answer", "")
            timings = result.get("timings", {})
        except Exception as exc:
            answer = ""
            timings = {}
            print(f"[warn] rag_pipeline failed on qid={item.qid}: {exc}")
        latencies.append(round((time.perf_counter() - t0) * 1000, 1))
        score = judge_answer(item.question, item.reference_answer, answer)
        graded.append(
            {
                "qid": item.qid,
                "question": item.question,
                "difficulty": item.difficulty,
                "answer_excerpt": answer[:300],
                "judge": score,
                "timings_ms": timings,
                "total_latency_ms": latencies[-1],
            }
        )

    def _avg(metric: str) -> float:
        values = [
            g["judge"].get(metric, 0)
            for g in graded
            if isinstance(g["judge"].get(metric), (int, float))
        ]
        return round(statistics.fmean(values), 3) if values else 0.0

    summary = {
        "items": len(graded),
        "avg_relevance": _avg("relevance"),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_completeness": _avg("completeness"),
        "avg_latency_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": (
            round(sorted(latencies)[int(0.95 * len(latencies))], 1) if latencies else 0.0
        ),
    }
    report = {"summary": summary, "graded": graded}
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote LLM judge report -> {out_path}")
    print(json.dumps(summary, indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpsAssist evaluation harness.")
    parser.add_argument(
        "mode",
        choices=("retrieval", "llm", "all"),
        default="all",
        nargs="?",
        help="Which evaluation to run (default: all).",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Cut-off for Hit Rate@k and NDCG@k."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode in ("retrieval", "all"):
        evaluate_retrieval(k=args.k)
    if args.mode in ("llm", "all"):
        evaluate_llm()
    return 0


if __name__ == "__main__":
    sys.exit(main())