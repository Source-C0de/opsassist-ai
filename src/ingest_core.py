"""Core helpers shared by the ingestion pipeline and the Streamlit app.

This module is intentionally tiny: it only contains the chunking, embedding,
and Qdrant primitives that both ``src/ingest.py`` (batch pipeline) and
``src/app.py`` (interactive playground) need. Keeping them in one place
ensures the two entry points stay in sync.
"""

from __future__ import annotations

import os
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
EMBED_BATCH_SIZE = 96


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


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed strings in batches using OpenAI."""
    if not texts:
        return []
    from openai import OpenAI  # imported lazily so collect/clean stages don't need it

    require_openai_key()
    client = OpenAI(api_key=OPENAI_API_KEY)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        response = client.embeddings.create(
            model=model, input=texts[start : start + EMBED_BATCH_SIZE]
        )
        vectors.extend(item.embedding for item in response.data)
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
