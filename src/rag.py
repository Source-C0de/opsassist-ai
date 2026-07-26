"""RAG orchestrator: rewrite -> retrieve -> rerank -> answer.

Wires together:
  - ``query_rewriter.rewrite_query`` (LLM-based contextual expansion)
  - ``retriever.HybridRetriever.retrieve`` (Qdrant dense + BM25 sparse, RRF-fused)
  - ``reranker.CrossEncoderReranker.rerank`` (cross-encoder re-scoring)
  - OpenAI chat completion with a citation-aware prompt

Returns a dict with the LLM answer, the cited sources, and per-stage timings
so the UI can show transparency panels and the monitoring layer can plot
latencies.
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

from dotenv import load_dotenv

from query_rewriter import rewrite_query
from reranker import CrossEncoderReranker
from retriever import HybridRetriever

load_dotenv()

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "600"))

SYSTEM_PROMPT = (
    "You are OpsAssist, a concise ops/devops assistant. Answer the user's "
    "question using ONLY the provided context passages. If the context does "
    "not contain the answer, say so explicitly — do not invent. Cite the "
    "passage numbers in square brackets, e.g. [1], [2]. Keep answers tight: "
    "lead with the answer, then 1-3 sentences of detail."
)


def _format_context(hits: Sequence[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Render retrieved hits as a numbered citation list."""
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    for idx, hit in enumerate(hits, start=1):
        text = (hit.get("text") or "").strip()
        source = hit.get("source") or "unknown"
        # Keep the context window manageable.
        snippet = text[:1500]
        blocks.append(f"[{idx}] ({source}) {snippet}")
        sources.append(
            {
                "id": hit.get("id"),
                "rank": idx,
                "source": source,
                "chunk_index": hit.get("chunk_index"),
                "score": hit.get("score"),
                "rerank_score": hit.get("rerank_score"),
                "text": text,
            }
        )
    return "\n\n".join(blocks), sources


def _build_messages(
    query: str,
    context: str,
    history: Sequence[dict] | None = None,
) -> list[dict]:
    """Compose chat-completion messages with system + history + current turn."""
    history_lines: list[str] = []
    for turn in (history or [])[-6:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines)

    user_prompt = (
        f"Context passages:\n{context}\n\n"
        + (f"Conversation so far:\n{history_block}\n\n" if history_block else "")
        + f"Question: {query}\n\n"
        "Answer with citations:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _llm_answer(messages: list[dict], model: str = ANSWER_MODEL) -> str:
    """Call the OpenAI chat completion API; return the assistant text."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=ANSWER_MAX_TOKENS,
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def _identity_rerank(query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict]:
    """No-op rerank used when the cross-encoder isn't available."""
    for h in hits:
        h["rerank_score"] = h.get("score", 0.0)
    return hits[:top_k]


def rag_pipeline(
    query: str,
    history: Sequence[dict] | None = None,
    top_k: int = 5,
    retrieve_k: int = 20,
    rewrite: bool = True,
    rerank: bool = True,
    retriever: HybridRetriever | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> dict[str, Any]:
    """Run the full RAG flow and return answer + sources + timings."""
    timings: dict[str, float] = {}
    retriever = retriever or HybridRetriever.from_disk()
    reranker = reranker or CrossEncoderReranker()

    # 1. Rewrite
    t0 = time.perf_counter()
    rewritten = rewrite_query(query, history) if rewrite else query
    timings["rewrite_s"] = round(time.perf_counter() - t0, 4)

    # 2. Retrieve (over-fetch for rerank headroom).
    t0 = time.perf_counter()
    raw_hits = retriever.retrieve(rewritten, top_k=max(retrieve_k, top_k))
    timings["retrieve_s"] = round(time.perf_counter() - t0, 4)

    # 3. Rerank (optional).
    t0 = time.perf_counter()
    if rerank and reranker.available and raw_hits:
        ranked = reranker.rerank(rewritten, raw_hits, top_k=top_k)
        hits = [
            {**r.hit, "rerank_score": r.rerank_score, "score": r.rerank_score}
            for r in ranked
        ]
    else:
        hits = _identity_rerank(rewritten, raw_hits, top_k)
    timings["rerank_s"] = round(time.perf_counter() - t0, 4)

    # 4. Build the citation-aware prompt.
    context, sources = _format_context(hits)
    if not context:
        answer = "I could not find any relevant passages in the knowledge base."
        timings["answer_s"] = 0.0
        return {
            "query": query,
            "rewritten_query": rewritten,
            "answer": answer,
            "sources": [],
            "timings": timings,
        }

    messages = _build_messages(query, context, history)

    # 5. LLM answer.
    t0 = time.perf_counter()
    if not os.getenv("OPENAI_API_KEY"):
        # Degraded mode — return the most relevant chunks as the answer.
        answer = (
            "OPENAI_API_KEY is not set, so I can't generate an answer. "
            "Here are the most relevant passages instead:\n\n" + context
        )
    else:
        try:
            answer = _llm_answer(messages)
        except Exception as exc:  # pragma: no cover - network/SDK errors.
            answer = f"LLM call failed: {exc}\n\nRelevant passages:\n\n{context}"
    timings["answer_s"] = round(time.perf_counter() - t0, 4)

    return {
        "query": query,
        "rewritten_query": rewritten,
        "answer": answer,
        "sources": sources,
        "timings": timings,
    }


def main() -> int:
    """Tiny CLI so the pipeline can be exercised from a shell."""
    import argparse
    import json as jsonlib

    parser = argparse.ArgumentParser(description="OpsAssist RAG pipeline")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    result = rag_pipeline(
        query=args.query,
        top_k=args.top_k,
        retrieve_k=args.retrieve_k,
        rewrite=not args.no_rewrite,
        rerank=not args.no_rerank,
    )
    print(jsonlib.dumps(result, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())