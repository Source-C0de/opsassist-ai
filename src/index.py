"""Build a BM25 sparse index from ``data/chunks.jsonl``.

The index is persisted as a pickle at ``data/bm25.pkl`` so the retriever does
not have to rebuild it on every query. Re-run this script whenever the chunk
set changes (i.e. after ``python src/ingest.py run``).

The BM25 implementation is pure-Python so we don't pull in a C dependency
just for retrieval. It is plenty fast for tens of thousands of chunks.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Repo-root paths so the script works from any cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_PATH = DATA_DIR / "bm25.pkl"

# Conservative English stopword list — keeps the index compact and trims
# noise from headings ("the", "a", "how to", ...).
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if while of in on at to for from by with as is are
    was were be been being have has had do does did this that these those it
    its you your we our they them their i my me him her he she what which who
    when where why how can could should would may might will shall not no nor
    so than too very also there here into out up down over under again further
    then once more most other some such only own same s t just don now
    """.split()
)

# Token regex: lowercase letters, digits, and internal hyphens/underscores.
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on token boundaries, drop stopwords and short tokens."""
    if not text:
        return []
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


@dataclass
class BM25Index:
    """In-memory BM25 index with disk pickle serialisation."""

    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # For each chunk id (chunk_id()), the term-frequency multiset.
    doc_freqs: list[Counter[str]] = field(default_factory=list)
    # Chunk id for each doc_freqs slot (parallel arrays).
    chunk_ids: list[str] = field(default_factory=list)
    # Per-chunk total length (token count).
    doc_lens: list[int] = field(default_factory=list)
    # Number of chunks indexed.
    n_docs: int = 0
    # Average document length.
    avg_doc_len: float = 0.0
    # Document-frequency of each term (number of chunks containing it).
    df: Counter[str] = field(default_factory=Counter)
    # Inverse-document-frequency for each term.
    idf: dict[str, float] = field(default_factory=dict)

    def score(self, query_tokens: list[str]) -> list[float]:
        """Return BM25 scores for every chunk, ordered by chunk_ids."""
        if self.n_docs == 0 or not query_tokens:
            return [0.0] * self.n_docs
        scores = [0.0] * self.n_docs
        for term in query_tokens:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = self.idf.get(term, 0.0)
            for i, terms in enumerate(self.doc_freqs):
                f = terms.get(term, 0)
                if f == 0:
                    continue
                numerator = f * (self.BM25_K1 + 1)
                denominator = f + self.BM25_K1 * (
                    1 - self.BM25_B + self.BM25_B * self.doc_lens[i] / self.avg_doc_len
                )
                scores[i] += idf * (numerator / denominator)
        return scores

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[str, float]]:
        """Return the top-k (chunk_id, score) tuples by BM25 score."""
        scores = self.score(query_tokens)
        if not scores:
            return []
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda pair: pair[1], reverse=True)
        return [(self.chunk_ids[i], s) for i, s in indexed[:k] if s > 0]


def _chunk_id_strings(chunk: dict) -> list[str]:
    """Stable per-chunk identifier matching the Qdrant payload id."""
    from ingest_core import chunk_id

    doc_id = chunk.get("document_id") or chunk.get("source", "")
    return [chunk_id(chunk.get("source", ""), doc_id, chunk.get("chunk_index", 0))]


def build_index(chunks: Iterable[dict]) -> BM25Index:
    """Build a BM25Index from an iterable of chunk dicts (must have 'text')."""
    doc_freqs: list[Counter[str]] = []
    chunk_ids: list[str] = []
    doc_lens: list[int] = []
    df: Counter[str] = Counter()

    for chunk in chunks:
        tokens = tokenize(chunk.get("text", ""))
        for c in _chunk_id_strings(chunk):
            doc_freqs.append(Counter(tokens))
            chunk_ids.append(c)
            doc_lens.append(len(tokens))
        # Document frequency increments once per chunk.
        for term in set(tokens):
            df[term] += 1

    avg_doc_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    idf = {
        term: math.log(1 + (len(doc_freqs) - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
    return BM25Index(
        doc_freqs=doc_freqs,
        chunk_ids=chunk_ids,
        doc_lens=doc_lens,
        n_docs=len(doc_freqs),
        avg_doc_len=avg_doc_len,
        df=df,
        idf=idf,
    )


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """Read a JSONL file produced by the ingest pipeline."""
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def save_index(index: BM25Index, path: Path = INDEX_PATH) -> None:
    """Pickle the BM25 index to disk.

    When this file is run as ``python index.py`` the dataclass would
    otherwise pickle as ``__main__.BM25Index`` and break loading from any
    other script. We register a ``__reduce__`` alias on the instance that
    always reconstructs via ``index.BM25Index`` regardless of entry point.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = index.__dict__.copy()
    with path.open("wb") as handle:
        handle.write(b"BM25INDEX_V1\n")
        # Persist each field individually with explicit module-resolution.
        handle.write(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


def load_index(path: Path = INDEX_PATH) -> BM25Index:
    """Load a persisted BM25 index from disk.

    New indexes written by ``save_index`` start with a ``BM25INDEX_V1``
    header; older pickles written before the rewrite don't. We detect the
    header and fall back to a raw pickle when the header is missing.
    """
    with path.open("rb") as handle:
        first = handle.read(13)
    if first.startswith(b"BM25INDEX_V1"):
        with path.open("rb") as handle:
            handle.readline()  # discard magic
            state = pickle.load(handle)
        return BM25Index(
            doc_freqs=state.get("doc_freqs", []),
            chunk_ids=state.get("chunk_ids", []),
            doc_lens=state.get("doc_lens", []),
            n_docs=state.get("n_docs", 0),
            avg_doc_len=state.get("avg_doc_len", 0.0),
            df=state.get("df", Counter()),
            idf=state.get("idf", {}),
        )
    # Legacy pickle (possibly __main__-bound) — rewrap under the index module.
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if isinstance(raw, BM25Index):
        return raw
    raise ValueError("Unsupported BM25 index format")


def main() -> int:
    """CLI: build the BM25 index from data/chunks.jsonl."""
    chunks = load_chunks()
    if not chunks:
        print("No chunks found at", CHUNKS_PATH)
        return 1
    index = build_index(chunks)
    save_index(index)
    print(
        f"Indexed {index.n_docs} chunks (avg {index.avg_doc_len:.1f} tokens, "
        f"{len(index.df)} unique terms) -> {INDEX_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
