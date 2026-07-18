"""OpsAssist AI: chunk, embed, store, and search arbitrary text."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "opsassist")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
EMBED_BATCH_SIZE = 96


def _require_openai_key() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
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
    _require_openai_key()
    client = OpenAI(api_key=OPENAI_API_KEY)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        response = client.embeddings.create(
            model=model, input=texts[start : start + EMBED_BATCH_SIZE]
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def get_qdrant_client(url: str = QDRANT_URL) -> QdrantClient:
    """Connect to a persistent Qdrant server; never falls back to memory."""
    return QdrantClient(url=url, timeout=30.0)


def ensure_collection(
    client: QdrantClient,
    name: str = QDRANT_COLLECTION,
    dim: int = EMBED_DIM,
) -> None:
    """Create the collection if absent and guard against dimension mismatch."""
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


def _chunk_id(source: str, document_id: str, chunk_index: int) -> str:
    """Create an idempotent point ID for one source document chunk."""
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{document_id}:{chunk_index}")
    )


def ingest(
    client: QdrantClient,
    texts: list[str],
    source: str = "manual",
    metadata: dict[str, Any] | None = None,
) -> int:
    """Chunk, embed, and upsert texts; return the number of points written."""
    ensure_collection(client)
    chunks: list[dict[str, Any]] = []
    for text in texts:
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, text.strip()))
        for chunk in chunk_text(text, source=source, metadata=metadata):
            chunk["document_id"] = document_id
            chunks.append(chunk)
    if not chunks:
        return 0

    vectors = embed_texts([chunk["text"] for chunk in chunks])
    points = [
        qmodels.PointStruct(
            id=_chunk_id(chunk["source"], chunk["document_id"], chunk["chunk_index"]),
            vector=vector,
            payload=chunk,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    return len(points)


def search(
    client: QdrantClient, query: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """Embed a query and return its closest stored chunks."""
    if not query.strip():
        return []
    ensure_collection(client)
    query_vector = embed_texts([query])[0]
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    results: list[dict[str, Any]] = []
    for hit in response.points:
        payload = hit.payload or {}
        results.append(
            {
                "score": float(hit.score),
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "chunk_index": payload.get("chunk_index"),
                "metadata": payload.get("metadata", {}),
            }
        )
    return results


def collection_count(client: QdrantClient) -> int:
    """Return the number of stored points; absent collection counts as zero."""
    try:
        return int(client.count(collection_name=QDRANT_COLLECTION).count)
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            return 0
        raise


def main() -> None:
    """Render the Streamlit ingestion and search UI."""
    import streamlit as st

    st.set_page_config(page_title="OpsAssist AI", layout="wide")
    st.title("OpsAssist AI — vector DB playground")

    @st.cache_resource
    def cached_qdrant_client() -> QdrantClient:
        return get_qdrant_client()

    with st.sidebar:
        st.header("Status")
        try:
            client = cached_qdrant_client()
            ensure_collection(client)
            st.success("Connected to persistent Qdrant")
            st.metric("Points in collection", collection_count(client))
        except Exception as exc:
            st.error(f"Qdrant connection failed: {exc}")
            st.info("Start Qdrant first, then click Recheck connection.")
            if st.button("Recheck connection"):
                st.rerun()
            st.stop()

        st.caption(
            f"URL: `{QDRANT_URL}`  \nCollection: `{QDRANT_COLLECTION}`  \n"
            f"Model: `{EMBED_MODEL}` ({EMBED_DIM}d)  \n"
            f"Chunks: `{CHUNK_SIZE}` tokens / `{CHUNK_OVERLAP}` overlap"
        )
        if st.button("Recheck connection"):
            st.rerun()

    add_tab, search_tab = st.tabs(["Add data", "Search"])

    with add_tab:
        text = st.text_area("Text to add", height=220)
        source = st.text_input("Source label", value="manual")
        metadata_raw = st.text_input(
            "Metadata (optional JSON)", placeholder='{"author": "me"}'
        )
        if st.button("Ingest", type="primary"):
            if not text.strip():
                st.warning("Enter some text first.")
            else:
                try:
                    metadata = json.loads(metadata_raw) if metadata_raw.strip() else None
                    count = ingest(client, [text], source=source, metadata=metadata)
                    st.success(f"Upserted {count} chunk(s).")
                except json.JSONDecodeError as exc:
                    st.error(f"Metadata is not valid JSON: {exc}")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    with search_tab:
        query = st.text_input("Query")
        top_k = st.slider("Top results", 1, 20, 5)
        if st.button("Search", type="primary"):
            if not query.strip():
                st.warning("Enter a query first.")
            else:
                try:
                    hits = search(client, query, top_k)
                    if not hits:
                        st.info("No matching chunks found.")
                    for rank, hit in enumerate(hits, 1):
                        label = (
                            f"#{rank} · score {hit['score']:.4f} · "
                            f"{hit['source']} · chunk {hit['chunk_index']}"
                        )
                        with st.expander(label, expanded=rank == 1):
                            st.write(hit["text"])
                            if hit["metadata"]:
                                st.json(hit["metadata"])
                except Exception as exc:
                    st.error(f"Search failed: {exc}")


def cli(argv: list[str] | None = None) -> int:
    """Expose ingestion and search for quick headless testing."""
    parser = argparse.ArgumentParser(description="OpsAssist AI vector DB playground")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest")
    ingest_parser.add_argument("--text", action="append", required=True)
    ingest_parser.add_argument("--source", default="cli")
    ingest_parser.add_argument("--metadata", help="Optional JSON object")

    query_parser = commands.add_parser("query")
    query_parser.add_argument("--text", required=True)
    query_parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args(argv)
    client = get_qdrant_client()
    if args.command == "ingest":
        metadata = json.loads(args.metadata) if args.metadata else None
        count = ingest(client, args.text, source=args.source, metadata=metadata)
        print(f"Upserted {count} chunk(s).")
    else:
        for rank, hit in enumerate(search(client, args.text, args.top_k), 1):
            print(
                f"#{rank} score={hit['score']:.4f} source={hit['source']} "
                f"chunk={hit['chunk_index']}\n{hit['text'][:500]}\n---"
            )
    return 0


if __name__ == "__main__":
    sys.exit(cli())
