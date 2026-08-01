# Cloud Deployment Guide

OpsAssist AI ships with three cloud-ready configurations so reviewers (and
future-me) can pick whichever platform matches their constraints.

| File                  | Platform                                | One-line use                                                      |
|-----------------------|------------------------------------------|--------------------------------------------------------------------|
| `Dockerfile.cloud`    | any Docker host (incl. Fly, Render, AWS) | slim Streamlit image referenced by `fly.toml` and `render.yaml`   |
| `fly.toml`            | Fly.io                                   | `fly launch` then `fly deploy` from the repo root                 |
| `render.yaml`         | Render                                   | "New Blueprint" in the Render dashboard, point at this repo       |
| `docker-compose.prod.yml` | Self-hosted VPS                       | `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d` |

In all three cases the Streamlit process listens on `$PORT` (defaults to
`8501`) and the URL `/` serves the chat UI.

## Prerequisites you must provide

| Secret                       | Where used                              | Notes                                            |
|------------------------------|-----------------------------------------|--------------------------------------------------|
| `OPENAI_API_KEY`             | `app` (everywhere)                      | gpt-4o-mini for the answer LLM + judge           |
| `QDRANT_URL` (+`QDRANT_API_KEY`) | `app`, `fly.toml` service block      | Qdrant Cloud free tier works                     |
| `ELASTIC_URL`                | `app`                                   | only needed if you keep BM25 on                  |
| `POSTGRES_*`                 | `docker-compose.prod.yml`               | write the password to `secrets/pg_password.txt`  |

For Fly: `fly secrets set OPENAI_API_KEY=sk-... QDRANT_URL=https://...`.
For Render: paste the values in the dashboard when the Blueprint prompts.

## Fly.io path (recommended for the rubric)

```bash
# 0. Install flyctl: https://fly.io/docs/hands-on/install-flyctl/
brew install flyctl        # or scoop/choco/curl installer

# 1. One-time bootstrap (skips the first deploy so you can review)
fly launch --no-deploy --copy-config
fly secrets set OPENAI_API_KEY=sk-... QDRANT_URL=https://YOUR-QDRANT.qdrant.io QDRANT_API_KEY=...

# 2. Pre-warm the index (one-shot machine so you don't have to ingest on the app box)
fly machine run . \
    --release-command "python src/index.py" \
    --env PYTHONPATH=/app/src --env OPENAI_API_KEY=$OPENAI_API_KEY \
    --env QDRANT_URL=$QDRANT_URL \
    --rm

# 3. Ship it
fly deploy
```

Notes:
* `fly.toml` already declares a Qdrant volume + the `release_command` so a
  green deploy automatically runs `python src/index.py` against the
  configured `QDRANT_URL`.
* `auto_stop_machines = "stop"` lets the app idle at $0 when not in use.
* Health check: `/` returns 200, Fly will reroute traffic if the machine fails.

## Render path

1. Push the repo to GitHub.
2. Render dashboard → **New +** → **Blueprint** → select the repo.
3. Set the secrets (`OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`).
4. Click **Apply**. Render builds `Dockerfile.cloud`, exposes port `8501`,
   and gives you `https://opsassist-ai.onrender.com`.

## Self-hosted Docker path

```bash
# Set up secrets (never commit this file)
echo "mysecret" > secrets/pg_password.txt

# Bring the full stack online (qdrant + elasticsearch + postgres + streamlit)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Visit `http://<your-host>:8501`. The `app` container automatically depends on
the local Qdrant and Elasticsearch services being healthy, so first boot
takes ~20s.

## Smoke testing the deploy

Once the app is reachable:

```bash
curl -fs https://<app-url>/_stcore/health   # -> {"status":"ok"}
```

Then open the URL in a browser and ask something like
`"How do I expose a port on a Docker container?"`. You should see a
hybrid-retrieval answer (RRF over BM25 + dense, reranked, LLM-as-judged).

## Rolling back

* Fly:    `fly releases list && fly releases rollback <version>`
* Render: redeploy a previous commit from the dashboard
* Compose: `docker compose -f docker-compose.yml -f docker-compose.prod.yml pull && ... up -d`

## Cost notes

* Fly free allowance covers ~3 shared-cpu-1x machines at 256MB each month.
* Render's starter plan (~$7/mo) keeps the app on 24/7.
* The cheapest Qdrant Cloud cluster (1GB free) is enough for this corpus.
