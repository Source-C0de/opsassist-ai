"""OpsAssist AI: streamlit playground for adding and searching vector chunks.

The embedding, chunking, and Qdrant primitives live in ``ingest_core`` so the
batch pipeline (``src/ingest.py``) and this interactive UI stay in sync.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingest_core import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_DIM,
    EMBED_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    chunk_id,
    chunk_text,
    collection_count,
    embed_texts,
    ensure_collection,
    get_qdrant_client,
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
            id=chunk_id(chunk["source"], chunk["document_id"], chunk["chunk_index"]),
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
