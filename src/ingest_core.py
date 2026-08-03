"""Core helpers shared by the ingestion pipeline and the Streamlit app.

This module is intentionally tiny: it only contains the chunking, embedding,
and Qdrant primitives that both ``src/ingest.py`` (batch pipeline) and
``src/app.py`` (interactive playground) need. Keeping them in one place
ensures the two entry points stay in sync.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import tiktoken
from dotenv import load_dotenv

# SDKs are imported lazily inside the helpers that use them, so the
# chunking primitives can be exercised in environments without the SDKs
# installed (e.g. dry-runs of the collect/clean stages).

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "opsassist")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
# Upper bound on the number of *items* sent in a single embeddings request.
# The hard constraint is actually the token budget below — this just caps the
# item count so a runaway chunk size can't issue arbitrarily large requests.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
# Maximum input tokens per embeddings request. Stays comfortably under the
# OpenAI Tier-1 default of 1,000,000 TPM (per-minute) and the 40,000 TPM
# limit seen on free / low-spend accounts. Override via env if your org has
# a higher quota.
EMBED_TOKEN_BUDGET = int(os.getenv("EMBED_TOKEN_BUDGET", "30000"))
# Retry settings for 429 / transient errors.
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "5"))
EMBED_RETRY_BASE_SECONDS = float(os.getenv("EMBED_RETRY_BASE_SECONDS", "1.0"))


def require_openai_key() -> None:
    """Raise if no OpenAI key is configured; the embedding step cannot proceed."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )


def get_qdrant_client(url: str = QDRANT_URL):
    """Connect to a persistent Qdrant server; never falls back to memory."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=30.0)


def ensure_collection(client, name: str = QDRANT_COLLECTION, dim: int = EMBED_DIM) -> None:
    """Create the collection if absent and guard against dimension mismatch."""
    from qdrant_client.http import models as qmodels
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        info = client.get_collection(collection_name=name)
    except UnexpectedResponse as exc:
        if exc.status_code != 404:
            raise
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=dim, distance=qmodels.Distance.COSINE
            ),
        )
        return

    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, qmodels.VectorParams):
        existing_dim = vectors_config.size
    else:
        existing_dim = next(iter(vectors_config.values())).size
    if existing_dim != dim:
        raise RuntimeError(
            f"Collection '{name}' uses {existing_dim}-dimensional vectors, but "
            f"EMBED_DIM={dim}. Restore the matching model/dimension or use a new "
            "QDRANT_COLLECTION name."
        )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    metadata: dict[str, Any] | None = None,
    source: str = "manual",
) -> list[dict[str, Any]]:
    """Split text into overlapping token windows; empty input returns []."""
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    stride = chunk_size - overlap
    chunks: list[dict[str, Any]] = []

    for start in range(0, len(tokens), stride):
        chunk_tokens = tokens[start : start + chunk_size]
        chunk_value = enc.decode(chunk_tokens).strip()
        if chunk_value:
            chunks.append(
                {
                    "text": chunk_value,
                    "chunk_index": len(chunks),
                    "token_count": len(chunk_tokens),
                    "source": source,
                    "metadata": metadata or {},
                }
            )
        if start + chunk_size >= len(tokens):
            break
    return chunks


def _plan_token_batches(
    texts: list[str],
    token_budget: int = EMBED_TOKEN_BUDGET,
    item_cap: int = EMBED_BATCH_SIZE,
) -> list[list[int]]:
    """Group text indices into batches that respect both a token budget and
    an item-count cap.

    Each chunk is counted with ``cl100k_base`` (the tokenizer used by
    ``text-embedding-3-small``). A single oversized chunk that exceeds the
    token budget on its own gets its own batch — OpenAI will return a 400
    with a clear error, which is preferable to silently dropping data.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    counts = [len(enc.encode(t)) for t in texts]

    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx, n_tokens in enumerate(counts):
        # If this chunk alone is too large, flush whatever we have and ship
        # it on its own so the caller sees the error rather than losing data.
        if n_tokens > token_budget:
            if current:
                batches.append(current)
                current, current_tokens = [], 0
            batches.append([idx])
            continue
        # Would adding this chunk overflow either limit? Flush first.
        if current and (
            current_tokens + n_tokens > token_budget
            or len(current) >= item_cap
        ):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(idx)
        current_tokens += n_tokens
    if current:
        batches.append(current)
    return batches


def _embed_batch_with_retry(client, model: str, batch: list[str]) -> list[list[float]]:
    """Call ``embeddings.create`` for one batch, retrying on 429 / transient
    errors with exponential backoff."""
    from openai import APIError, APITimeoutError, RateLimitError

    last_exc: Exception | None = None
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            response = client.embeddings.create(model=model, input=batch)
            return [item.embedding for item in response.data]
        except (RateLimitError, APITimeoutError, APIError) as exc:
            last_exc = exc
            # If the SDK gave us a Retry-After hint, honour it; otherwise
            # exponential backoff: 1s, 2s, 4s, 8s, 16s (capped).
            retry_after: float | None = None
            retry_after_attr = getattr(exc, "retry_after", None)
            if retry_after_attr is not None:
                try:
                    retry_after = float(retry_after_attr)
                except (TypeError, ValueError):
                    retry_after = None
            wait = retry_after if retry_after is not None else EMBED_RETRY_BASE_SECONDS * (2 ** attempt)
            time.sleep(wait)
    # All retries exhausted.
    assert last_exc is not None
    raise last_exc


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed strings in batches using OpenAI.

    Batches are sized to stay within ``EMBED_TOKEN_BUDGET`` tokens (per
    request) and ``EMBED_BATCH_SIZE`` items, so we never blow past the
    organisation's TPM limit. Transient 429/5xx errors are retried with
    exponential backoff up to ``EMBED_MAX_RETRIES`` times.
    """
    if not texts:
        return []
    from openai import OpenAI  # imported lazily so collect/clean stages don't need it

    require_openai_key()
    client = OpenAI(api_key=OPENAI_API_KEY)

    indices = _plan_token_batches(texts)
    vectors: list[list[float]] = [[] for _ in range(len(texts))]
    for batch_indices in indices:
        batch_texts = [texts[i] for i in batch_indices]
        batch_vectors = _embed_batch_with_retry(client, model, batch_texts)
        # Defensive: OpenAI returns vectors in input order.
        if len(batch_vectors) != len(batch_indices):
            raise RuntimeError(
                f"OpenAI returned {len(batch_vectors)} embeddings for "
                f"{len(batch_indices)} inputs; aborting to avoid misaligned vectors."
            )
        for original_idx, vector in zip(batch_indices, batch_vectors, strict=True):
            vectors[original_idx] = vector
    return vectors


def chunk_id(source: str, document_id: str, chunk_index: int) -> str:
    """Create an idempotent point ID for one source document chunk."""
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{document_id}:{chunk_index}")
    )


def collection_count(client) -> int:
    """Return the number of stored points; absent collection counts as zero."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        return int(client.count(collection_name=QDRANT_COLLECTION).count)
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            return 0
        raise
