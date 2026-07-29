"""Log explanation tool for OpsAssist AI.

The tool accepts logs from Kubernetes, Docker, systemd, or Nginx and returns a
structured incident explanation grounded in the same documentation retriever as
the normal RAG flow:

* probable cause(s)
* why the error happened
* recommended, ordered fix steps
* safety / verification notes
* citations to the retrieved documentation

It deliberately does not execute commands or mutate infrastructure. The
output is advisory and every suggested command should be reviewed before use.

Python API
----------
    from tools.log_explainer import explain_log
    result = explain_log(log_text, environment="kubernetes")

CLI
---
    python src/tools/log_explainer.py --environment docker < error.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Sequence

from dotenv import load_dotenv

load_dotenv()

from rag import _format_context, _llm_answer  # noqa: E402
from retriever import HybridRetriever  # noqa: E402
from reranker import CrossEncoderReranker  # noqa: E402

LOG_SYSTEM_PROMPT = (
    "You are OpsAssist's incident-response log explainer. Analyze the supplied "
    "log using ONLY the documentation context. Return a practical Markdown "
    "report with exactly these headings: ## Probable cause, ## Why it happened, "
    "## Recommended fix, ## Verify, ## Safety notes. Quote short log evidence "
    "where useful and cite documentation passages as [1], [2]. If evidence is "
    "insufficient, say what extra diagnostic information is needed. Never claim "
    "you executed a command; all commands are suggestions for the operator."
)

# Lightweight environment hints improve retrieval when logs contain terse error
# codes that otherwise have little semantic context.
ENVIRONMENT_HINTS = {
    "kubernetes": "Kubernetes kubectl logs pod container event deployment",
    "docker": "Docker container daemon compose image network volume",
    "nginx": "Nginx web server reverse proxy access error configuration",
    "systemd": "Linux systemd service journalctl unit daemon",
    "auto": "DevOps infrastructure production log error troubleshooting",
}


def _redact_secrets(text: str) -> str:
    """Redact common credentials before logs reach an external LLM."""
    patterns = [
        (r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+", r"\1[REDACTED]"),
        (r"(?i)(password|passwd|secret|token|api[_-]?key)\s*[=:]\s*([^\s,;]+)", r"\1=[REDACTED]"),
        (r"(?i)(aws_secret_access_key\s*[=:]\s*)[^\s]+", r"\1[REDACTED]"),
    ]
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result


def _normalise_environment(environment: str) -> str:
    value = (environment or "auto").strip().lower()
    aliases = {"k8s": "kubernetes", "kubectl": "kubernetes", "systemd/journal": "systemd"}
    return aliases.get(value, value if value in ENVIRONMENT_HINTS else "auto")


def build_log_messages(
    log_text: str,
    context: str,
    environment: str = "auto",
    history: Sequence[dict] | None = None,
) -> list[dict[str, str]]:
    """Build the structured prompt sent to the answer model."""
    env = _normalise_environment(environment)
    history_lines = []
    for turn in (history or [])[-4:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines)
    prompt = (
        f"Environment: {env}\n\n"
        f"Log block (possibly redacted):\n---\n{log_text}\n---\n\n"
        f"Documentation context:\n{context or '(no matching documentation)'}\n\n"
        + (f"Prior conversation:\n{history_block}\n\n" if history_block else "")
        + "Produce the incident report now."
    )
    return [
        {"role": "system", "content": LOG_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _heuristic_report(log_text: str, environment: str) -> str:
    """Useful offline fallback when no OpenAI key is configured."""
    lower = log_text.lower()
    observations: list[str] = []
    if "crashloopbackoff" in lower or "back-off restarting failed container" in lower:
        observations.append("The workload is repeatedly exiting; Kubernetes is backing off restarts.")
    if "connection refused" in lower:
        observations.append("A client reached the host but the target process is not accepting connections.")
    if "permission denied" in lower or "eacces" in lower:
        observations.append("The process lacks permission for a file, directory, socket, or operation.")
    if "no such file" in lower or "not found" in lower:
        observations.append("A referenced path, executable, image, or configuration object is missing.")
    if "oomkilled" in lower or "out of memory" in lower:
        observations.append("The process exceeded an available memory limit and was terminated.")
    if "bind()" in lower and "address already in use" in lower:
        observations.append("Another process is already listening on the requested port.")
    if not observations:
        observations.append("The log needs correlation with service status, recent changes, and surrounding lines.")
    cause = " ".join(observations)
    return (
        f"## Probable cause\n{cause}\n\n"
        "## Why it happened\n"
        "This offline explanation is based on recognizable log patterns only; "
        "retrieve the relevant runbook and inspect the full event history before changing production.\n\n"
        "## Recommended fix\n"
        "1. Capture the complete error and the 30–60 seconds around it.\n"
        "2. Check service/container status, recent deployments, resource limits, and configuration syntax.\n"
        "3. Apply the smallest reversible change, then observe logs and health checks.\n\n"
        "## Verify\n"
        "Confirm the service is healthy, the error no longer repeats, and dependent requests succeed.\n\n"
        "## Safety notes\n"
        f"Environment detected: `{environment}`. No commands were executed."
    )


def explain_log(
    log_text: str,
    environment: str = "auto",
    top_k: int = 5,
    retrieve_k: int = 20,
    history: Sequence[dict] | None = None,
    retriever: HybridRetriever | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> dict[str, Any]:
    """Explain ``log_text`` with retrieved docs and return report metadata."""
    redacted = _redact_secrets((log_text or "").strip())
    if not redacted:
        return {
            "environment": _normalise_environment(environment),
            "answer": "Paste a non-empty log block so I can analyze it.",
            "sources": [],
            "redacted": False,
        }

    env = _normalise_environment(environment)
    # Use the first ~4,000 chars for retrieval; preserve the full redacted log
    # in the answer prompt only up to a safe bound.
    query = f"{ENVIRONMENT_HINTS[env]}\n{redacted[:4000]}"
    retriever = retriever or HybridRetriever.from_disk()
    reranker = reranker or CrossEncoderReranker()
    raw_hits = retriever.retrieve(query, top_k=max(retrieve_k, top_k))
    if reranker.available:
        ranked = reranker.rerank(query, raw_hits, top_k=top_k)
        hits = [{**r.hit, "score": r.rerank_score, "rerank_score": r.rerank_score} for r in ranked]
    else:
        hits = raw_hits[:top_k]
    context, sources = _format_context(hits)

    if os.getenv("OPENAI_API_KEY"):
        try:
            answer = _llm_answer(build_log_messages(redacted[:12000], context, env, history))
        except Exception as exc:
            answer = _heuristic_report(redacted, env) + f"\n\n_Model call failed: {exc}_"
    else:
        answer = _heuristic_report(redacted, env)

    return {
        "environment": env,
        "answer": answer,
        "sources": sources,
        "redacted": redacted != log_text,
        "query": query,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain a DevOps log block.")
    parser.add_argument("--environment", choices=tuple(ENVIRONMENT_HINTS), default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    log_text = sys.stdin.read()
    result = explain_log(log_text, environment=args.environment, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
