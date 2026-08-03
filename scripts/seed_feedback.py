"""Seed `data/feedback/feedback.csv` with realistic synthetic rows.

Why?
----
The monitoring dashboard (src/monitoring.py) reads from `feedback.csv`. Before
real users click 👍/👎 there is too little data to render the six charts
meaningfully. This script appends ~110 rows spanning 7 days so the dashboard
shows realistic distributions on first launch.

Usage:
    python scripts/seed_feedback.py           # append ~110 rows
    python scripts/seed_feedback.py --reset   # wipe and rewrite
    python scripts/seed_feedback.py --n 200   # custom row count

Idempotent and safe to re-run: it appends and never deletes unless --reset.
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_PATH = REPO_ROOT / "data" / "feedback" / "feedback.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "question",
    "answer",
    "rating",
    "mode",
    "latency_ms",
    "tokens",
]

# 20 realistic Docker + Nginx questions. Mix of FAQ-style and incident-style.
DOCKER_QUESTIONS = [
    "How do I expose a port with docker run?",
    "Difference between CMD and ENTRYPOINT?",
    "How to reduce Docker image size?",
    "What is the difference between COPY and ADD?",
    "How do I mount a volume in docker compose?",
    "Explain docker build --no-cache.",
    "How to inspect a running container's env vars?",
    "What does 'no space left on device' mean in Docker?",
    "How do I prune unused images?",
    "Explain the difference between bridge and host networks.",
]

NGINX_QUESTIONS = [
    "How do I enable gzip compression in nginx?",
    "What does 'upstream sent too big header' mean?",
    "How to configure nginx reverse proxy?",
    "Explain try_files directive.",
    "How do I rate-limit requests in nginx?",
    "What is the difference between root and alias?",
    "How to enable HTTP/2 in nginx?",
    "Explain client_max_body_size.",
]

LOG_QUESTIONS = [
    "Explain this nginx error: 502 bad gateway",
    "Explain: docker dial unix /var/run/docker.sock: connect: permission denied",
    "Explain: Failed to pull image: rpc error: code = Unknown",
    "Explain: nginx: [emerg] bind() to 0.0.0.0:80 failed (98: Address already in use)",
    "Explain: kubelet failed to start container: image pull backoff",
    "Explain: docker: Error response from daemon: conflict. container already running",
    "Explain: systemd: Failed at step USER spawning nginx: No such user",
]

ANSWER_TEMPLATE = (
    "Based on the retrieved documentation, the issue is typically caused by {root_cause}. "
    "Recommended fix: {fix}. Verification: {verify}. Sources: [1], [2]."
)

ROOT_CAUSES = [
    "a misconfigured upstream block in nginx.conf",
    "missing port publishing in the docker run command",
    "a stale Docker build cache referencing deleted base layers",
    "an SELinux or AppArmor policy blocking the bind mount",
    "the OOM killer terminating the container under memory pressure",
]

FIXES = [
    "add the `-p 80:80` flag, then restart the container",
    "set `client_max_body_size 16m;` in the http block",
    "run `docker builder prune -af` and rebuild without cache",
    "use `:ro` mounts and grant the container the right capabilities",
    "raise the memory limit in compose and add `--restart unless-stopped`",
]

VERIFICATIONS = [
    "curl -v http://localhost:8080 should return 200",
    "`docker logs <id>` should no longer repeat the error",
    "`nginx -t` should report syntax OK",
    "`docker compose ps` should show `healthy` for the service",
]

MODES = ["docs", "docs", "docs", "docs", "log"]  # 80/20 split


def _make_row(rng: random.Random, day_offset: int) -> dict[str, str]:
    """Return a single realistic feedback row.

    `day_offset=0` means today, `1` means yesterday, etc.
    """
    # 7-day window ending today.
    base = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    ts = base - timedelta(days=day_offset, hours=rng.randint(0, 23), minutes=rng.randint(0, 59))

    if rng.random() < 0.2:
        question = rng.choice(LOG_QUESTIONS)
        mode = "log"
    else:
        pool = DOCKER_QUESTIONS + NGINX_QUESTIONS
        question = rng.choice(pool)
        mode = "docs"

    answer = ANSWER_TEMPLATE.format(
        root_cause=rng.choice(ROOT_CAUSES),
        fix=rng.choice(FIXES),
        verify=rng.choice(VERIFICATIONS),
    )

    # Realistic ratings: skew positive but include some bad ones.
    rating = rng.choices([1, 2, 3, 4, 5], weights=[3, 5, 10, 35, 47])[0]

    # Latency: 1.5s – 12s, log mode slightly slower.
    base_latency = rng.uniform(1800, 8500)
    if mode == "log":
        base_latency *= rng.uniform(1.1, 1.4)
    latency_ms = int(base_latency)

    # Tokens: 200 – 1800.
    tokens = rng.randint(220, 1750)

    return {
        "timestamp": ts.isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating,
        "mode": mode,
        "latency_ms": latency_ms,
        "tokens": tokens,
    }


def _reset_file() -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(REQUIRED_COLUMNS)


def _append_row(row: dict[str, str]) -> None:
    write_header = not FEEDBACK_PATH.exists()
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=110, help="number of rows to add")
    parser.add_argument("--reset", action="store_true", help="wipe the file first")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.reset:
        _reset_file()

    rows_added = 0
    for _ in range(args.n):
        day_offset = rng.randint(0, 6)  # spread over last 7 days
        _append_row(_make_row(rng, day_offset))
        rows_added += 1

    print(f"Added {rows_added} rows to {FEEDBACK_PATH}")
    if args.reset:
        print("File was reset before appending.")
    print(f"Run: streamlit run src/monitoring.py")


if __name__ == "__main__":
    main()