"""Hybrid retriever: dense (Qdrant) + sparse (BM25) fused with RRF.

The dense index lives in Qdrant (production system) and the sparse index
lives in ``data/bm25.pkl`` (built by ``src/index.py``). Both rank chunks by
their own scoring scale; we fuse them with Reciprocal Rank Fusion, which is
robust to score-scale differences and is the textbook hybrid-search bonus.

Pipeline:
  1. Encode the query with the same OpenAI model used for indexing.
  2. Run dense top-``vector_k`` search against Qdrant.
  3. Run BM25 top-``bm25_k`` search against the in-memory index.
  4. RRF-fuse the two ranked lists into a single top-``k``.
  5. Return hits with provenance metadata so the LLM can cite sources.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ingest_core import (
    QDRANT_COLLECTION,
    embed_texts,
    ensure_collection,
    get_qdrant_client,
)
from index import INDEX_PATH, BM25Index, load_index, tokenize


@dataclass
class HybridRetriever:
    """Holds a Qdrant client and a BM25 index; runs hybrid search on demand."""

    qdrant_url: str
    collection: str
    bm25_index: BM25Index
    rrf_k: int = 60  # RRF constant — 60 is the classic choice.
    _client: object | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_disk(
        cls,
        bm25_path: Path = INDEX_PATH,
        collection: str = QDRANT_COLLECTION,
        qdrant_url: str | None = None,
    ) -> "HybridRetriever":
        """Load the BM25 index from disk and prepare the Qdrant client."""
        from ingest_core import QDRANT_URL

        return cls(
            qdrant_url=qdrant_url or QDRANT_URL,
            collection=collection,
            bm25_index=load_index(bm25_path),
        )

    def _dense_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run dense vector search against Qdrant; return hits with metadata."""
        client = self._client or get_qdrant_client(self.qdrant_url)
        self._client = client
        ensure_collection(client, name=self.collection)
        query_vector = embed_texts([query])[0]
        response = client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "id": str(point.id),
                "score": float(point.score),
                "text": (point.payload or {}).get("text", ""),
                "source": (point.payload or {}).get("source", ""),
                "chunk_index": (point.payload or {}).get("chunk_index"),
                "metadata": (point.payload or {}).get("metadata", {}),
            }
            for point in response.points
        ]

    def _sparse_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run BM25 search; look up payloads from the local chunks.jsonl."""
        tokens = tokenize(query)
        ranked = self.bm25_index.top_k(tokens, top_k)
        if not ranked:
            return []
        # O(1) chunk lookup by id.
        catalog = _load_chunk_catalog()
        hits: list[dict[str, Any]] = []
        for chunk_id_str, score in ranked:
            payload = catalog.get(chunk_id_str)
            if not payload:
                continue
            hits.append(
                {
                    "id": chunk_id_str,
                    "score": float(score),
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "chunk_index": payload.get("chunk_index"),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return hits

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_k: int | None = None,
        bm25_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run hybrid search; return top-``k`` chunks with provenance."""
        if not query.strip():
            return []
        # Over-fetch from each side so fusion has enough headroom.
        vector_k = vector_k or max(top_k * 4, 20)
        bm25_k = bm25_k or max(top_k * 4, 20)

        dense = self._dense_search(query, vector_k)
        sparse = self._sparse_search(query, bm25_k)
        return self._fuse_rrf(dense, sparse, top_k)

    def _fuse_rrf(
        self,
        dense: list[dict[str, Any]],
        sparse: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion — sum ``1 / (k + rank)`` across both lists."""
        scores: dict[str, float] = {}
        catalog: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense, start=1):
            cid = hit["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            catalog[cid] = hit

        for rank, hit in enumerate(sparse, start=1):
            cid = hit["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            # Sparse may surface chunks that aren't in the dense top-k; keep them.
            catalog.setdefault(cid, hit)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        fused: list[dict[str, Any]] = []
        for cid, fused_score in ranked:
            hit = dict(catalog[cid])
            hit["score"] = fused_score
            fused.append(hit)
        return fused


_CHUNK_CATALOG: dict[str, dict[str, Any]] | None = None
_CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks.jsonl"


def _load_chunk_catalog() -> dict[str, dict[str, Any]]:
    """Lazy-load a chunk_id -> chunk payload dict from data/chunks.jsonl."""
    global _CHUNK_CATALOG
    if _CHUNK_CATALOG is not None:
        return _CHUNK_CATALOG
    catalog: dict[str, dict[str, Any]] = {}
    if not _CHUNKS_PATH.exists():
        _CHUNK_CATALOG = catalog
        return catalog
    from ingest_core import chunk_id as make_chunk_id

    with _CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            cid = make_chunk_id(
                chunk.get("source", ""),
                chunk.get("document_id", ""),
                chunk.get("chunk_index", 0),
            )
            catalog[cid] = chunk
    _CHUNK_CATALOG = catalog
    return catalog


def retrieve(
    query: str,
    top_k: int = 5,
    vector_k: int | None = None,
    bm25_k: int | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper: load the retriever from disk and run a query."""
    return HybridRetriever.from_disk().retrieve(
        query, top_k=top_k, vector_k=vector_k, bm25_k=bm25_k
    )