"""Cross-encoder re-ranker for retrieved chunks.

The retriever pulls ``top_k`` candidates (vector + BM25 fused). The reranker
takes those candidates and re-scores each ``(query, chunk)`` pair with a
cross-encoder so the most relevant chunks bubble to the top.

We default to ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — small (~90 MB),
runs on CPU in tens of milliseconds per query, and is a standard baseline
for MS MARCO passage ranking.

If ``sentence-transformers`` is not installed (e.g. minimal dev env), the
module degrades to a *no-op* reranker that preserves input order so the
rest of the pipeline still works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Lazy import so the rest of the module is usable without torch / transformers.
try:  # pragma: no cover - exercised when sentence-transformers is present.
    from sentence_transformers import CrossEncoder  # type: ignore

    _CROSS_ENCODER_AVAILABLE = True
except Exception:  # ImportError, OSError (missing torch), etc.
    CrossEncoder = None  # type: ignore
    _CROSS_ENCODER_AVAILABLE = False


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RankedHit:
    """A chunk after reranking. Wraps the input hit dict with a new score."""

    hit: dict
    rerank_score: float


class CrossEncoderReranker:
    """Wrap a sentence-transformers CrossEncoder with safe fallbacks."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self._model = None
        self._device = device
        if _CROSS_ENCODER_AVAILABLE:
            try:
                self._model = CrossEncoder(model_name, device=device)
            except Exception:
                # Model download / load failure — fall back to identity rerank.
                self._model = None

    @property
    def available(self) -> bool:
        """True iff a cross-encoder model is loaded and ready."""
        return self._model is not None

    def rerank(
        self,
        query: str,
        hits: Sequence[dict],
        top_k: int | None = None,
    ) -> list[RankedHit]:
        """Score each ``(query, hit['text'])`` pair; return sorted by score."""
        if not hits:
            return []
        if self._model is None:
            # Identity rerank — keep input order, copy the dense score if any.
            ranked = [RankedHit(hit=h, rerank_score=float(h.get("score", 0.0))) for h in hits]
        else:
            pairs = [[query, h.get("text", "")] for h in hits]
            scores = self._model.predict(pairs, show_progress_bar=False)
            ranked = [RankedHit(hit=h, rerank_score=float(s)) for h, s in zip(hits, scores)]
        ranked.sort(key=lambda r: r.rerank_score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


def rerank(
    query: str,
    hits: Sequence[dict],
    top_k: int | None = None,
    model_name: str = DEFAULT_MODEL,
) -> list[RankedHit]:
    """Convenience function: build a fresh reranker and rerank ``hits``."""
    reranker = CrossEncoderReranker(model_name=model_name)
    return reranker.rerank(query, hits, top_k=top_k)