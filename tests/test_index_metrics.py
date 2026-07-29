"""Unit tests for the BM25 index + evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate import hit_rate_at_k, mrr, ndcg_at_k  # noqa: E402
from index import BM25Index, build_index, tokenize  # noqa: E402


# --- Tokenisation -----------------------------------------------------------
def test_tokenize_lowercases_and_strips_stopwords() -> None:
    tokens = tokenize("The Docker Container is RUNNING! Stop?")
    assert "docker" in tokens
    assert "container" in tokens
    assert "running" in tokens
    # 'the', 'is' are stopwords; punctuation stripped.
    assert "the" not in tokens
    assert "is" not in tokens


def test_tokenize_handles_empty_input() -> None:
    assert tokenize("") == []
    assert tokenize(None or "") == []  # type: ignore[arg-type]


# --- BM25Index -------------------------------------------------------------
def _build_sample_index() -> BM25Index:
    # Note: tokenisation strips short tokens; "s1" is dropped, but "docker"
    # appears in two documents and "nginx" in one.
    chunks = [
        {"text": "docker compose services network port", "source": "doc-a", "document_id": "d1", "chunk_index": 0},
        {"text": "nginx reverse proxy load balancer", "source": "doc-b", "document_id": "d2", "chunk_index": 0},
        {"text": "dockerfile build cache multi-stage", "source": "doc-c", "document_id": "d3", "chunk_index": 0},
    ]
    return build_index(chunks)


def test_build_index_counts_unique_terms() -> None:
    index = _build_sample_index()
    assert index.n_docs == 3
    assert "docker" in index.df
    assert "nginx" in index.df
    # 'docker' and 'dockerfile' are different tokens — 'docker' is unique to chunk 1.
    assert index.df["docker"] == 1
    assert index.df["dockerfile"] == 1
    assert index.df["compose"] == 1


def test_top_k_returns_relevant_doc_first() -> None:
    from ingest_core import chunk_id

    index = _build_sample_index()
    top = index.top_k(tokenize("docker compose"), k=3)
    assert top, "expected non-empty hits"
    top_id = top[0][0]
    expected_id = chunk_id("doc-a", "d1", 0)
    # The first chunk talks about docker compose; it should outrank the others.
    assert top_id == expected_id


# --- Retrieval metrics ------------------------------------------------------
def test_hit_rate_at_k_positive() -> None:
    assert hit_rate_at_k(["a", "b"], {"b"}, k=2) == 1.0


def test_hit_rate_at_k_negative() -> None:
    assert hit_rate_at_k(["a", "b"], {"c"}, k=2) == 0.0


def test_mrr_reciprocal_rank() -> None:
    assert mrr(["a", "b", "c"], {"c"}) == 1 / 3
    assert mrr(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_at_k_perfect_ranking() -> None:
    # Both relevant at the top → ideal gain.
    score = ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=2)
    assert score == pytest.approx(1.0)


def test_ndcg_at_k_drops_with_worse_ranking() -> None:
    good = ndcg_at_k(["a", "b", "c"], {"a"}, k=3)
    bad = ndcg_at_k(["c", "b", "a"], {"a"}, k=3)
    assert good > bad