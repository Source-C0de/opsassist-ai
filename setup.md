# OpsAssist AI setup guide

This guide takes a fresh clone from zero to a working local application.

## 1. Prerequisites

Choose either the native Python path or Docker Compose. Docker is recommended
because it supplies Qdrant, Elasticsearch, and PostgreSQL consistently.

- Python 3.11+
- Docker Desktop 4.30+ / Docker Engine 24+
- An OpenAI API key for embeddings and generated answers (optional for offline
  ingestion and the log-pattern fallback)
- At least 6 GB free RAM when running Elasticsearch + the reranker

## 2. Configure the environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

Never commit `.env` or paste production credentials into logs. The log
explainer redacts common `token`, `password`, `secret`, and bearer values before
sending a log to the model, but operators should still remove sensitive data.

## 3. Docker Compose (recommended)

Start all services:

```bash
docker compose up --build
```

Open <http://localhost:8501>. The supporting services are available at:

| Service | Address |
|---|---|
| Streamlit | http://localhost:8501 |
| Qdrant | http://localhost:6333 |
| Elasticsearch | http://localhost:9200 |
| PostgreSQL | localhost:5432 |

The dashboard is available as an opt-in profile:

```bash
docker compose --profile monitoring up --build
# dashboard: http://localhost:8502
```

Stop services without deleting persistent volumes:

```bash
docker compose down
```

## 4. Build the knowledge index

The committed raw corpus is under `data/raw/`. The idempotent pipeline cleans
and chunks it without making network calls, then uploads embeddings to Qdrant:

```bash
# Native Python, from the repository root
python -m pip install -r requriement.txt
PYTHONPATH=src python src/ingest.py run --dry-run
PYTHONPATH=src python src/index.py
PYTHONPATH=src python src/ingest.py upload
```

When running inside the app container, use the service hostname instead of
`localhost`:

```bash
docker compose exec app python src/ingest.py run
```

`data/chunks.jsonl` and `data/bm25.pkl` are generated artifacts. They are
ignored by Git and can always be rebuilt from `data/raw/`.

## 5. Native Streamlit

To run without the app container, start the databases first:

```bash
docker compose up -d qdrant elasticsearch postgres
python -m pip install -r requriement.txt
PYTHONPATH=src streamlit run src/app.py
```

## 6. Evaluation

The golden set contains 44 questions across Docker and Nginx topics:

```bash
PYTHONPATH=src python src/evaluate.py retrieval
PYTHONPATH=src python src/evaluate.py llm
```

Retrieval evaluation compares:

1. vector-only top-5;
2. vector-only top-20 + cross-encoder;
3. hybrid BM25 + vector RRF top-20 + cross-encoder;
4. hybrid + reranker + query rewrite.

Metrics are written to `data/eval/retrieval_report.json`. LLM-as-judge results
are written to `data/eval/llm_judge_report.json`. The LLM evaluation requires
`OPENAI_API_KEY`.

## 7. Log explainer CLI

```bash
printf '2025-01-01 ERROR: connection refused to upstream\n' \
  | PYTHONPATH=src python src/tools/log_explainer.py --environment nginx
```

The CLI returns JSON. The Streamlit UI exposes the same capability through the
**Explain Log** mode.

## 8. Tests and linting

```bash
PYTHONPATH=src pytest -q
python -m compileall src
```

## Troubleshooting

### Qdrant connection refused

Check `docker compose ps` and ensure `QDRANT_URL` is `http://localhost:6333`
for native execution or `http://qdrant:6333` inside Compose.

### Empty search results

Run the dry-run pipeline, build the BM25 index, and upload vectors:

```bash
PYTHONPATH=src python src/ingest.py run --dry-run
PYTHONPATH=src python src/index.py
PYTHONPATH=src python src/ingest.py upload
```

### OpenAI errors

Confirm the key is present in `.env`, the account can access the configured
models, and the embedding dimension matches `EMBED_DIM`. Use `--dry-run` or
the local log fallback when working without credentials.

### Elasticsearch is not used by the retriever

This repository uses a pure-Python persisted BM25 index for portability. The
Compose Elasticsearch service is included for the planned production index;
RRF currently fuses Qdrant dense retrieval with the local BM25 ranker.
