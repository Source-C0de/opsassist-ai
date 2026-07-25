# Data Sources

The knowledge base for opsassist-ai is a curated subset of public ops/devops
documentation. Both sources are downloaded once into `data/raw/` and tracked
in this repo so the corpus is reproducible.

| Source                                       | Section       | License                        | Accessed   |
|----------------------------------------------|---------------|--------------------------------|------------|
| `docker/` — Docker docs mirror               | Docker engine, Compose, concepts, intro workshop | Apache-2.0 (Docker documentation) | 2025-07    |
| `nginx/` — KodeKloud "Nginx for Beginners" notes | Nginx         | KodeKloud educational notes (CC-style attribution, see upstream) | 2025-07    |

Raw files are intentionally committed (see `data/raw/.gitignore` below) so a
fresh clone can build the Qdrant collection without re-downloading anything.
The pipeline still treats `data/raw/` as the single source of truth — if you
add or remove a markdown file, just re-run `python src/ingest.py run`.

## Why these two?

- **Docker docs** give broad, current coverage of containers, Compose,
  networking, security, and storage — exactly the topics an ops assistant
  should be able to answer.
- **Nginx notes** add web-server / reverse-proxy material that the Docker
  docs do not cover, so the assistant can field questions like "how does
  Nginx's event loop work?" or "which ports does Nginx use?".

## Adding more sources

1. Drop a folder of `.md` files under `data/raw/<your-source>/`.
2. Make sure filenames end in `.md` and aren't in `EXCLUDE_FILE_NAMES`
   (`_index.md`) or empty placeholders.
3. Re-run the pipeline:
   ```bash
   python src/ingest.py run
   ```
4. If you changed the embedding model or dimensions, also bump
   `EMBED_DIM` in `.env` and pick a fresh `QDRANT_COLLECTION` to avoid a
   mismatch error.

## Upstream URLs (for provenance only)

- Docker docs: <https://docs.docker.com/>
- KodeKloud Nginx notes: <https://notes.kodekloud.com/>

We do not redistribute the upstream content beyond what is needed to build
the search index in this repo; if you intend to publish the corpus, double
check the upstream licenses first.