"""Smoke tests for the RAG pipeline + log explainer (no live network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag import _format_context  # noqa: E402
from reranker import CrossEncoderReranker, RankedHit  # noqa: E402
from tools.log_explainer import _redact_secrets, _normalise_environment  # noqa: E402


# --- rag._format_context ---------------------------------------------------
def test_format_context_numbers_sources_and_truncates() -> None:
    hits = [
        {"id": "x", "text": "A" * 5000, "source": "doc1.md", "chunk_index": 1, "score": 0.9},
        {"id": "y", "text": "short text", "source": "doc2.md", "chunk_index": 2, "score": 0.7},
    ]
    context, sources = _format_context(hits)
    assert "[1] (doc1.md)" in context
    assert "[2] (doc2.md)" in context
    # Snippets must be truncated to <=1500 chars per source.
    assert context.count("A") <= 1500
    assert sources[0]["rank"] == 1
    assert sources[1]["rank"] == 2


# --- reranker.CrossEncoderReranker ----------------------------------------
def test_identity_rerank_preserves_input_order_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the model can't be loaded, the reranker must still order correctly."""
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "stub"
    reranker._model = None  # noqa: SLF001 — test only
    hits = [
        {"id": "a", "text": "alpha", "score": 0.1},
        {"id": "b", "text": "beta", "score": 0.9},
        {"id": "c", "text": "gamma", "score": 0.5},
    ]
    ranked = reranker.rerank("alpha", hits, top_k=2)
    assert isinstance(ranked[0], RankedHit)
    # Without a model, the score equals the input score → beta wins.
    assert [r.hit["id"] for r in ranked] == ["b", "c"]


# --- log_explainer helpers -------------------------------------------------
def test_redact_secrets_replaces_bearer_and_passwords() -> None:
    raw = "Authorization: Bearer abc.def.ghi\npassword=hunter2 log line"
    out = _redact_secrets(raw)
    assert "abc.def.ghi" not in out
    assert "hunter2" not in out
    assert "[REDACTED]" in out
    # Non-secret text is preserved.
    assert "log line" in out


def test_redact_secrets_handles_empty_input() -> None:
    assert _redact_secrets("") == ""


def test_normalise_environment_uses_aliases() -> None:
    assert _normalise_environment("k8s") == "kubernetes"
    assert _normalise_environment("kubectl") == "kubernetes"
    assert _normalise_environment("nginx") == "nginx"
    # Unknown environments fall back to "auto".
    assert _normalise_environment("not-a-real-env") == "auto"