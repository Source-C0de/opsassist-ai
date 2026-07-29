"""Configuration review tool for OpsAssist AI.

The config reviewer is a stretch tool that highlights anti-patterns in pasted
Dockerfile, docker-compose.yml, Nginx, or Terraform HCL snippets and returns
fix suggestions grounded in the same retrieval backbone as the RAG flow.

It does not parse the file strictly — instead it asks an LLM to compare the
input against retrieved documentation. Outputs are advisory and the operator
must always review before applying changes to production.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag import _format_context, _llm_answer  # noqa: E402
from retriever import HybridRetriever  # noqa: E402
from reranker import CrossEncoderReranker  # noqa: E402

CONFIG_SYSTEM_PROMPT = (
    "You are OpsAssist's config reviewer. You are given a configuration snippet "
    "(Dockerfile, docker-compose, nginx.conf, or Terraform HCL) and a context "
    "passage set. Identify concrete anti-patterns, explain why each is a risk, "
    "and propose a minimal-diff fix. Structure the output as a Markdown list "
    "with 'Issue', 'Why', 'Suggested fix'. End with a short 'Overall' verdict. "
    "Cite documentation passages as [1], [2]. Do not invent new features."
)

CONFIG_HINTS = {
    "dockerfile": "Dockerfile best practices multi-stage USER HEALTHCHECK COPY",
    "compose": "docker-compose services depends_on restart healthcheck networks volumes secrets",
    "nginx": "Nginx configuration server location upstream security headers",
    "terraform": "Terraform HCL state backend module best practices",
    "auto": "DevOps configuration best practices anti-pattern",
}


def _detect_kind(text: str) -> str:
    """Return the most likely config flavour based on simple heuristics."""
    head = text[:2000].lower()
    if re.search(r"^\s*from\s+\w+", head, re.MULTILINE) and ("dockerfile" in head or "run " in head):
        return "dockerfile"
    if "services:" in head or "image:" in head or "version:" in head:
        return "compose"
    if "server {" in head or "location " in head or "upstream " in head:
        return "nginx"
    if re.search(r"\b(resource|provider|module)\s+\"?\w+\"?\s*\{", head):
        return "terraform"
    return "auto"


def _normalise_kind(kind: str) -> str:
    return kind if kind in CONFIG_HINTS else "auto"


def build_config_messages(
    snippet: str, kind: str, context: str, history: Sequence[dict] | None = None
) -> list[dict[str, str]]:
    """Compose the chat messages sent to the answer model."""
    history_lines = []
    for turn in (history or [])[-4:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines)
    prompt = (
        f"Detected configuration kind: {kind}\n\n"
        f"Configuration snippet:\n---\n{snippet[:8000]}\n---\n\n"
        f"Documentation context:\n{context or '(no matching documentation)'}\n\n"
        + (f"Prior conversation:\n{history_block}\n\n" if history_block else "")
        + "Produce the review now."
    )
    return [
        {"role": "system", "content": CONFIG_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _heuristic_review(snippet: str, kind: str) -> str:
    """Offline fallback that flags a few well-known anti-patterns."""
    issues: list[str] = []
    lower = snippet.lower()
    if kind == "dockerfile":
        if "user root" not in lower and not re.search(r"^user\s+\w+", lower, re.MULTILINE):
            issues.append("- Issue: container still runs as `root`. **Why:** a container escape becomes a host escalation. **Suggested fix:** add `USER appuser` near the end of the Dockerfile.")
        if ":latest" in lower:
            issues.append("- Issue: image uses the `:latest` tag. **Why:** builds become non-reproducible and supply-chain risk rises. **Suggested fix:** pin to a specific tag or digest.")
        if "add " in lower and "copy" in lower:
            issues.append("- Issue: legacy `ADD` used where `COPY` would suffice. **Why:** `ADD` silently extracts archives and fetches URLs. **Suggested fix:** switch to `COPY` unless you specifically need archive extraction.")
    elif kind == "compose":
        if "privileged: true" in lower:
            issues.append("- Issue: `privileged: true` in compose. **Why:** disables most isolation. **Suggested fix:** use `cap_add` with the minimum capability required.")
        if "ports:\n      - 80:80" in lower and "127.0.0.1" not in lower:
            issues.append("- Issue: binding published ports on all interfaces. **Why:** exposes services to the LAN. **Suggested fix:** restrict to `127.0.0.1:80:80` when not behind a reverse proxy.")
    elif kind == "nginx":
        if "server_tokens on" in lower or "server_tokens on;" in lower:
            issues.append("- Issue: `server_tokens on;`. **Why:** leaks Nginx version in error pages. **Suggested fix:** set `server_tokens off;`.")
        if "ssl_protocols" in lower and "tlsv1" in lower:
            issues.append("- Issue: TLS 1.0/1.1 enabled. **Why:** legacy protocols are vulnerable. **Suggested fix:** allow only TLSv1.2 and TLSv1.3.")
    if not issues:
        issues.append("- No high-confidence anti-patterns detected from heuristics. Run with `OPENAI_API_KEY` set for a richer review.")
    return "## Heuristic review\n\n" + "\n".join(issues) + "\n\n## Overall\nReview the snippet with the documentation context above."


def review_config(
    snippet: str,
    kind: str = "auto",
    top_k: int = 5,
    retrieve_k: int = 20,
    history: Sequence[dict] | None = None,
    retriever: HybridRetriever | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> dict:
    """Review ``snippet`` and return the analysis."""
    text = (snippet or "").strip()
    if not text:
        return {"kind": "auto", "answer": "Paste a non-empty configuration snippet.", "sources": []}
    detected = _detect_kind(text)
    chosen = _normalise_kind(kind if kind and kind != "auto" else detected)
    query = f"{CONFIG_HINTS[chosen]}\n{text[:4000]}"
    retriever = retriever or HybridRetriever.from_disk()
    reranker = reranker or CrossEncoderReranker()
    raw = retriever.retrieve(query, top_k=max(retrieve_k, top_k))
    if reranker.available:
        ranked = reranker.rerank(query, raw, top_k=top_k)
        hits = [{**r.hit, "score": r.rerank_score, "rerank_score": r.rerank_score} for r in ranked]
    else:
        hits = raw[:top_k]
    context, sources = _format_context(hits)
    if os.getenv("OPENAI_API_KEY"):
        try:
            answer = _llm_answer(build_config_messages(text[:8000], chosen, context, history))
        except Exception as exc:
            answer = _heuristic_review(text, chosen) + f"\n\n_Model call failed: {exc}_"
    else:
        answer = _heuristic_review(text, chosen)
    return {"kind": chosen, "detected_kind": detected, "answer": answer, "sources": sources, "query": query}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review a configuration snippet.")
    parser.add_argument("--kind", choices=tuple(CONFIG_HINTS), default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--file", help="Read the snippet from this path instead of stdin.")
    args = parser.parse_args(argv)
    if args.file:
        snippet = Path(args.file).read_text(encoding="utf-8", errors="replace")
    else:
        snippet = sys.stdin.read()
    print(json.dumps(review_config(snippet, kind=args.kind, top_k=args.top_k), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())