"""Batch ingestion pipeline for opsassist-ai.

This is the Phase 2 deliverable: an idempotent pipeline that turns the raw
documentation corpus under ``data/raw/`` into overlapping token chunks and
uploads them to a persistent Qdrant collection.

Pipeline stages (run from the repo root):

    python src/ingest.py collect     # walk data/raw/*.md -> data/raw_manifest.jsonl
    python src/ingest.py clean       # raw markdown -> data/chunks.jsonl
    python src/ingest.py upload      # chunks.jsonl -> Qdrant (re-embeds + upserts)
    python src/ingest.py run         # all three, in order

The pipeline is idempotent:
- ``collect`` overwrites the manifest but skips files that no longer exist.
- ``clean`` overwrites ``chunks.jsonl`` deterministically.
- ``upload`` uses content-hashed point IDs so re-running upserts the same
  points instead of creating duplicates.

Skip the upload step with ``--dry-run`` (useful for testing without an OpenAI
key or a running Qdrant instance).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

from ingest_core import (
    QDRANT_COLLECTION,
    chunk_id,
    chunk_text,
    embed_texts,
    ensure_collection,
    get_qdrant_client,
)

# Paths are anchored to the repo root, not the src/ directory, so the pipeline
# behaves the same whether you run it from src/ or from the project root.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "raw_manifest.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

# Files we never want to ingest.
EXCLUDE_DIR_NAMES = {"images", ".git", "__pycache__"}
EXCLUDE_FILE_NAMES = {"_index.md"}  # Hugo navigation stubs, not useful content

# Hugo-style YAML front-matter at the start of a file.
FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
# Hugo shortcodes like {{< grid >}} or {{< tabs >}}.
SHORTCODE_RE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.DOTALL)
# Any HTML tag (very aggressive — fine for embedding).
HTML_TAG_RE = re.compile(r"<[^>]+>")
# Image-markdown syntax: ![alt](url) — keep the alt text, drop the URL.
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
# Standard markdown link: [text](url) -> keep the text.
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# Markdown headers, lists — keep text but normalise markers.
HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
LIST_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
LIST_ORDERED_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
# Collapse 3+ blank lines into one.
BLANK_LINES_RE = re.compile(r"\n{3,}")


def iter_markdown_files(raw_dir: Path) -> Iterable[Path]:
    """Yield every .md file under ``raw_dir`` that we want to ingest."""
    for path in sorted(raw_dir.rglob("*.md")):
        if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        if path.name in EXCLUDE_FILE_NAMES:
            continue
        if path.stat().st_size == 0:
            # Skip empty placeholder files like "introduction_nginx.md".
            continue
        yield path


def document_id(text: str) -> str:
    """Stable per-document UUID derived from the cleaned text."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sha256:{digest}"))


def clean_markdown(raw: str) -> str:
    """Strip Hugo front-matter, shortcodes, HTML, and Markdown chrome."""
    if not raw:
        return ""
    text = FRONT_MATTER_RE.sub("", raw, count=1)
    text = SHORTCODE_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    text = MD_IMAGE_RE.sub(r"\1", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = HEADER_RE.sub("", text)
    text = LIST_BULLET_RE.sub("", text)
    text = LIST_ORDERED_RE.sub("", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def collect_documents(raw_dir: Path = RAW_DIR, manifest: Path = MANIFEST_PATH) -> int:
    """Walk raw markdown and write a manifest; return count of documents."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for path in iter_markdown_files(raw_dir):
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_markdown(raw)
        if not cleaned:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        # Top-level section is the first directory under raw/ (e.g. "docker", "nginx").
        section = (
            path.relative_to(raw_dir).parts[0]
            if raw_dir in path.parents
            else "misc"
        )
        entries.append(
            {
                "path": rel,
                "section": section,
                "size_bytes": len(raw),
                "cleaned_chars": len(cleaned),
            }
        )

    with manifest.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(entries)


def load_cleaned_documents(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Read raw markdown, clean it, and return per-document dicts."""
    documents: list[dict[str, Any]] = []
    for path in iter_markdown_files(raw_dir):
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_markdown(raw)
        if not cleaned:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        section = (
            path.relative_to(raw_dir).parts[0]
            if raw_dir in path.parents
            else "misc"
        )
        documents.append(
            {
                "source": rel,
                "section": section,
                "cleaned_text": cleaned,
                "document_id": document_id(cleaned),
            }
        )
    return documents


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunk every document and tag each chunk with provenance metadata."""
    chunks: list[dict[str, Any]] = []
    for doc in documents:
        for chunk in chunk_text(
            doc["cleaned_text"],
            source=doc["source"],
            metadata={"section": doc["section"]},
        ):
            chunk["document_id"] = doc["document_id"]
            chunks.append(chunk)
    return chunks


def write_chunks_jsonl(
    chunks: list[dict[str, Any]], chunks_path: Path = CHUNKS_PATH
) -> int:
    """Write chunks to JSONL; return number of chunks written."""
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return len(chunks)


def upload_chunks(
    chunks: list[dict[str, Any]],
    collection: str = QDRANT_COLLECTION,
    batch_size: int = 96,
) -> int:
    """Embed chunks and upsert them; return the number of points written."""
    if not chunks:
        return 0
    from qdrant_client.http import models as qmodels

    client = get_qdrant_client()
    ensure_collection(client, name=collection)

    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)

    points = [
        qmodels.PointStruct(
            id=chunk_id(chunk["source"], chunk["document_id"], chunk["chunk_index"]),
            vector=vector,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "token_count": chunk["token_count"],
                "metadata": chunk.get("metadata", {}),
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    # Upsert in batches so very large corpora don't blow memory.
    for start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection,
            points=points[start : start + batch_size],
            wait=True,
        )
    return len(points)


def run_pipeline(
    raw_dir: Path = RAW_DIR,
    chunks_path: Path = CHUNKS_PATH,
    collection: str = QDRANT_COLLECTION,
    dry_run: bool = False,
) -> dict[str, int]:
    """Collect -> clean -> upload; return per-stage counts."""
    doc_count = collect_documents(raw_dir)
    documents = load_cleaned_documents(raw_dir)
    chunks = chunk_documents(documents)
    chunk_count = write_chunks_jsonl(chunks, chunks_path)
    if dry_run:
        uploaded = 0
    else:
        uploaded = upload_chunks(chunks, collection=collection)
    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "uploaded": uploaded,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpsAssist AI batch ingestion pipeline."
    )
    parser.add_argument(
        "command",
        choices=("collect", "clean", "upload", "run"),
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory of raw markdown files (default: data/raw).",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=CHUNKS_PATH,
        help="Output JSONL path (default: data/chunks.jsonl).",
    )
    parser.add_argument(
        "--collection",
        default=QDRANT_COLLECTION,
        help="Qdrant collection name (default: env QDRANT_COLLECTION).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the upload step (no OpenAI calls, no Qdrant writes).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "collect":
        n = collect_documents(args.raw_dir)
        print(f"Collected {n} document(s) -> {MANIFEST_PATH}")
    elif args.command == "clean":
        documents = load_cleaned_documents(args.raw_dir)
        chunks = chunk_documents(documents)
        n = write_chunks_jsonl(chunks, args.chunks_path)
        print(f"Wrote {n} chunk(s) -> {args.chunks_path}")
    elif args.command == "upload":
        documents = load_cleaned_documents(args.raw_dir)
        chunks = chunk_documents(documents)
        n = upload_chunks(chunks, collection=args.collection)
        print(f"Uploaded {n} chunk(s) to '{args.collection}'.")
    elif args.command == "run":
        result = run_pipeline(
            raw_dir=args.raw_dir,
            chunks_path=args.chunks_path,
            collection=args.collection,
            dry_run=args.dry_run,
        )
        print(f"Documents: {result['documents']}")
        print(f"Chunks:    {result['chunks']}")
        if args.dry_run:
            print("Upload:    skipped (--dry-run)")
        else:
            print(f"Uploaded:  {result['uploaded']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
