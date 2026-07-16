# LLM Zoomcamp Project — Build Plan & Checklist

> A complete step-by-step roadmap for the DataTalksClub LLM Zoomcamp capstone project.
> Target: **16/16 + all bonus points** on the peer-review rubric.

---

## Table of Contents

1. [Phase 0 — Idea Generation & Selection](#phase-0--idea-generation--selection)
2. [Phase 1 — Repository Setup](#phase-1--repository-setup)
3. [Phase 2 — Data Ingestion Pipeline](#phase-2--data-ingestion-pipeline)
4. [Phase 3 — RAG Pipeline](#phase-3--rag-pipeline)
5. [Phase 4 — Evaluation](#phase-4--evaluation)
6. [Phase 5 — Interface](#phase-5--interface)
7. [Phase 6 — Monitoring](#phase-6--monitoring)
8. [Phase 7 — Containerization](#phase-7--containerization)
9. [Phase 8 — Reproducibility & Documentation](#phase-8--reproducibility--documentation)
10. [Phase 9 — Pre-Submission Checklist](#phase-9--pre-submission-checklist)
11. [Phase 10 — Submission & Peer Review](#phase-10--submission--peer-review)
12. [Common Pitfalls](#common-pitfalls)
13. [Quick-Start Cheatsheet](#quick-start-cheatsheet)

---

## Phase 0 — Idea Generation & Selection

### Goal
Pick a dataset and problem you can defend and finish in 2–3 weeks.

### Steps

1. **Brainstorm 5 candidate ideas.** High-scoring dataset types:
   - DTC podcast transcripts
   - Book of the Week archive
   - Wikipedia subset
   - YouTube playlist transcripts
   - GitHub repo READMEs
   - Generated synthetic dataset (full control)

2. **Apply the SMART filter** to each idea:
   - **S**pecific — what question does the user ask?
   - **M**easurable — can you write ≥20 eval Q&A pairs?
   - **A**ttainable — can you finish in 2–3 weeks?
   - **R**elevant — interests you enough to debug at 2 AM
   - **T**ime-bounded — clear scope (e.g., "2024 episodes only")

3. **Pick ONE idea** and write a 1-paragraph problem statement.

   > Example: *"This project builds a Q&A assistant for the DataTalksClub Podcast. Users can ask natural-language questions like 'What did Alexey say about RAG evaluation in episode 42?' and get answers grounded in the transcript corpus."*

---

## Phase 1 — Repository Setup

### Goal
A clean, professional, reproducible GitHub repo.

### Steps

4. **Create a dedicated GitHub repo.** Meaningful name (e.g., `podcast-rag-assistant`), not `project1`.
   - Initialize with `.gitignore` (Python template), `LICENSE` (MIT), empty `README.md`.
   - Make the **first commit** before any code.

5. **Define the project structure:**

   ```
   project/
   ├── README.md
   ├── setup.md
   ├── docker-compose.yml
   ├── Dockerfile
   ├── pyproject.toml          # or requirements.txt
   ├── .env.example
   ├── data/                   # raw + processed (gitignored or DVC)
   ├── notebooks/              # exploration + eval
   ├── src/
   │   ├── ingest.py
   │   ├── index.py
   │   ├── rag.py
   │   ├── evaluate.py
   │   └── app.py
   ├── tests/
   └── images/                 # screenshots for README
   ```

6. **Lock the stack with exact versions:**

   | Layer         | Tool                              | Why                                 |
   |---------------|-----------------------------------|-------------------------------------|
   | LLM           | OpenAI `gpt-4o-mini`              | Cheap, fast, course-friendly        |
   | Embeddings    | `text-embedding-3-small`          | Cheap, strong                       |
   | Vector DB     | **Qdrant** or ElasticSearch       | Enables hybrid search bonus         |
   | Orchestration | **Mage** or plain Python script   | 2 pts for ingestion                 |
   | Interface     | **Streamlit**                     | 2 pts, fastest to build             |
   | Monitoring    | Streamlit feedback + Grafana      | 2 pts with 5+ charts                |
   | Container     | docker-compose                    | 2 pts                               |

   Pin every version in `requirements.txt` (e.g., `openai==1.51.0`, `qdrant-client==1.12.0`).

---

## Phase 2 — Data Ingestion Pipeline

### Goal
An automated, idempotent pipeline that turns raw data into indexed vectors.

### Steps

7. **Acquire raw data.** Download from public source (URL, Kaggle, HuggingFace). Store in `data/raw/`. Add `data_sources.md` listing URLs and access dates.

8. **Clean and chunk.**
   - Strip HTML, normalize whitespace, remove boilerplate.
   - Chunk size: **500 tokens with 100 token overlap** (good default).
   - Preserve metadata: `source_file`, `chunk_index`, `timestamp`, `speaker`.
   - Save processed chunks to `data/chunks.jsonl`.

9. **Build the automated ingestion pipeline.**

   **Path A — Mage / dlt / Airflow:** DAG with scheduled runs.
   **Path B — Plain Python script (`src/ingest.py`):**
   ```
   ingest.py
     ├── download_data()        # idempotent, skips if already downloaded
     ├── clean_and_chunk()      # writes chunks.jsonl
     └── upload_to_qdrant()     # creates collection, embeds, upserts
   ```
   - Must be **idempotent** (safe to re-run).
   - Run once → commit the resulting Qdrant snapshot **OR** document regeneration.

---

## Phase 3 — RAG Pipeline

### Goal
A retrieval + generation flow with hybrid search, re-ranking, and query rewriting.

### Steps

10. **Implement the core RAG flow** in `src/rag.py`:

    - `retrieve(query, top_k=5, method="hybrid")` → list of chunks
    - `build_prompt(query, chunks)` → formatted prompt with citations
    - `llm_answer(prompt)` → final response
    - `rag_pipeline(query)` → orchestrator returning `{answer, sources, timings}`

11. **Hybrid search (+1 bonus point).**
    - Vector search + BM25 (ElasticSearch) **or** dense + sparse vectors (Qdrant).
    - Combine scores via **RRF** (Reciprocal Rank Fusion) or weighted sum.
    - Compare vector-only vs hybrid in evaluation.

12. **Re-ranking (+1 bonus point).**
    - Use Cohere Rerank, cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), or LLM reranking.
    - Pipeline: retrieve 20 → rerank → keep top 5.

13. **Query rewriting (+1 bonus point).**
    - Use LLM to rewrite ambiguous follow-ups (`"what about that?"` → `"what about chunking?"`).
    - Use conversation history when rewriting.

---

## Phase 4 — Evaluation

### Goal
The highest-impact phase — full retrieval + LLM evaluation, picking the best approach.

### Steps

14. **Build a retrieval evaluation set.**
    - Write **30+ question-answer pairs** manually.
    - Mix difficulty: 10 easy, 10 medium, 10 hard (multi-hop, paraphrase).
    - Save as `data/eval/retrieval_eval.jsonl`:
      ```json
      {"question": "...", "relevant_chunk_ids": ["chunk_42", "chunk_87"], "answer": "..."}
      ```

15. **Implement retrieval metrics.**
    - **Hit Rate** (Recall@k)
    - **MRR** (Mean Reciprocal Rank)
    - **NDCG@10**

16. **Compare multiple retrieval approaches (2 pts)** in `notebooks/02_retrieval_eval.ipynb`:
    - Baseline: vector-only, k=5
    - Variant: vector-only, k=20 → rerank → 5
    - Variant: hybrid search
    - Variant: hybrid + rerank + query rewrite

    Produce a **table + bar chart** → choose the winner for the final pipeline.

17. **Evaluate the LLM layer (2 pts)** in `notebooks/03_llm_eval.ipynb`:
    - Test **3+ prompts** (concise vs CoT vs with-citations vs no-citations).
    - LLM-as-judge (GPT-4 grades relevance, faithfulness, completeness on 1–5 scale).
    - Pick the best prompt and document why.

---

## Phase 5 — Interface

### Goal
A polished Streamlit UI with source citations and feedback capture.

### Steps

18. **Build the Streamlit app (`src/app.py`)** with:
    - Chat-style input box
    - Streaming responses
    - Display **source chunks with similarity scores** (clickable)
    - **Feedback buttons** 👍 / 👎 writing to CSV/DB
    - Sidebar showing retrieved chunks (transparency)

19. **Polish the UI:** title, description, example questions, logo/favicon, mobile-friendly layout.

20. **Take screenshots:** landing page, mid-conversation, source-citation view, feedback button. Save to `images/`.

---

## Phase 6 — Monitoring

### Goal
Feedback collection + dashboard with at least 5 charts.

### Steps

21. **Capture user feedback.** Each 👍/👎 click → append
    `{timestamp, question, answer, rating, comment}` to `data/feedback/feedback.csv`.

22. **Build a monitoring dashboard with ≥5 charts.**
    Options: Streamlit dashboard page (simplest) or Grafana connected to Postgres (more impressive).

    Required charts:
    1. Feedback rating distribution (pie)
    2. Questions per day (line)
    3. Avg response latency over time (line)
    4. Top 10 most-asked questions (bar)
    5. Retrieval hit-rate over time (line)
    6. (Bonus) Token usage per day (bar)

---

## Phase 7 — Containerization

### Goal
A single-command setup via docker-compose.

### Steps

23. **Write a `Dockerfile`** for the app:
    - Base: `python:3.11-slim`.
    - Copy code, install pinned requirements, expose Streamlit port `8501`.

24. **Write a `docker-compose.yml`** including all services:

    ```yaml
    services:
      app:
        build: .
        ports: ["8501:8501"]
        env_file: .env
        depends_on: [qdrant]
      qdrant:
        image: qdrant/qdrant:latest
        ports: ["6333:6333"]
        volumes: [qdrant_data:/qdrant/storage]
      postgres:           # for feedback storage
        image: postgres:16
        environment:
          POSTGRES_PASSWORD: ...
    ```

    Now everything runs with **one command**: `docker-compose up`.

---

## Phase 8 — Reproducibility & Documentation

### Goal
A README that lets a stranger clone → run → use the project in under 10 minutes.

### Steps

25. **Write `README.md`** with these sections:
    1. Title + 1-sentence description
    2. Problem statement (what + why)
    3. Dataset (source, size, license, sample)
    4. Architecture diagram (Mermaid or image)
    5. Tech stack with versions
    6. How to run — copy-pasteable commands
    7. Evaluation results — link the table from Phase 4
    8. Screenshots of the UI
    9. Monitoring dashboard screenshot
    10. Limitations & future work

26. **Write `setup.md`:**
    - Prerequisites (Docker, API keys).
    - `.env.example` with placeholders.
    - Step-by-step: `git clone` → `cp .env.example .env` → fill keys → `docker-compose up`.

27. **Add a 30-second demo video** (Loom or Streamlit recorder) embedded in README.

---

## Phase 9 — Pre-Submission Checklist

Score yourself against the rubric:

| Criterion                              | Target | Self-score |
|----------------------------------------|--------|------------|
| Problem description                    | 2      | ☐          |
| Retrieval flow                         | 2      | ☐          |
| Retrieval evaluation (multi-approach)  | 2      | ☐          |
| LLM evaluation (multi-prompt)          | 2      | ☐          |
| Interface (Streamlit UI)               | 2      | ☐          |
| Ingestion pipeline (automated)         | 2      | ☐          |
| Monitoring (feedback + ≥5 charts)      | 2      | ☐          |
| Containerization (docker-compose)      | 2      | ☐          |
| Reproducibility                        | 2      | ☐          |
| Hybrid search                          | +1     | ☐          |
| Re-ranking                             | +1     | ☐          |
| Query rewriting                        | +1     | ☐          |
| Cloud deployment                       | +2     | ☐          |
| **Total possible**                     | **23** |            |

---

## Phase 10 — Submission & Peer Review

### Steps

28. **Submit.**
    - Push the final commit.
    - Note the **commit hash** — reviewers need it.
    - Submit GitHub URL + commit hash via the cohort form.

29. **Review 3 peer projects (mandatory for 9 bonus pts).**
    - Clone each at the given commit hash.
    - Score using the rubric.
    - Be specific in feedback — cite file paths and line numbers.
    - Suggest concrete improvements.

---

## Common Pitfalls

1. Starting with code before writing the problem statement.
2. Using only one retrieval approach — you need at least 2 compared.
3. Skipping prompt comparison — easy 2 points lost.
4. README without screenshots — reviewers can't verify your claims.
5. Loose dependency versions — kills reproducibility score.
6. Hardcoding API keys — use `.env` and `.env.example`.
7. Submitting before testing docker-compose from a clean machine.
8. Ignoring the "feedback collection" requirement — a simple 👍/👎 is enough.

---

## Quick-Start Cheatsheet

```bash
# Day 1 — scaffold
git init && git add . && git commit -m "init: project scaffold"

# Day 5 — ingest data
python src/ingest.py

# Day 11 — evaluate
jupyter notebook notebooks/03_llm_eval.ipynb

# Day 18 — final test from clean clone
git clone <your-repo>
cd <your-repo>
cp .env.example .env       # add keys
docker-compose up
# open http://localhost:8501
```

---

## Suggested Timeline

| Day | Phase | Deliverable |
|-----|-------|-------------|
| 1   | 0     | Idea + problem statement |
| 2   | 1     | Repo scaffold + locked stack |
| 3–5 | 2     | Ingestion pipeline + indexed data |
| 6–8 | 3     | RAG pipeline with hybrid + rerank + rewrite |
| 9–11| 4     | Eval notebooks with comparison tables |
| 12–13| 5    | Streamlit UI + screenshots |
| 14  | 6     | Feedback + dashboard with 5+ charts |
| 15  | 7     | Dockerfile + docker-compose |
| 16–17| 8    | README + setup.md + demo video |
| 18  | 9     | Pre-submission self-score |
| 19–20| 10   | Submit + review 3 peers |

---

## Notes

- Replace `<your-repo>` and API key placeholders before running.
- Keep this file updated as you complete each phase — strike through items as you finish them.