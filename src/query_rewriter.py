"""LLM-based query rewriting for ambiguous follow-ups.

When a user types "what about that?" or "and the ports?" mid-conversation,
the raw query is useless for retrieval. This module asks the LLM to rewrite
the latest user turn into a self-contained search query, using the prior
conversation history as context.

The rewrite is intentionally conservative — it returns the original query
unchanged when no contextual gap exists, so the rest of the pipeline isn't
worse off if the LLM is unavailable.
"""

from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")
REWRITE_MAX_TOKENS = 120


SYSTEM_PROMPT = (
    "You rewrite a user's latest question into a single, self-contained "
    "search query suitable for retrieving documentation passages. "
    "Resolve pronouns and references ('it', 'that', 'those ports') using the "
    "prior conversation. Keep the result concise and technical. "
    "If the latest question is already self-contained, return it unchanged. "
    "Return ONLY the rewritten query — no quotes, no preamble."
)


def _format_history(history: Sequence[dict]) -> str:
    """Render prior turns as 'user: ... / assistant: ...' lines."""
    lines = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_rewrite_messages(
    query: str,
    history: Sequence[dict] | None = None,
) -> list[dict]:
    """Compose the chat-completion messages for the rewrite call."""
    history_block = _format_history(history or [])
    user_prompt = (
        f"Conversation so far:\n{history_block or '(none)'}\n\n"
        f"Latest question: {query}\n\n"
        "Rewritten search query:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def rewrite_query(
    query: str,
    history: Sequence[dict] | None = None,
    model: str = REWRITE_MODEL,
) -> str:
    """Rewrite ``query`` using the conversation ``history`` for context.

    Falls back to the original query if the OpenAI client is unavailable or
    the API call fails — better to search with the raw query than crash.
    """
    text = (query or "").strip()
    if not text:
        return ""
    if not os.getenv("OPENAI_API_KEY"):
        return text
    try:
        from openai import OpenAI
    except Exception:
        return text

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_rewrite_messages(text, history),  # type: ignore[arg-type]
            max_tokens=REWRITE_MAX_TOKENS,
            temperature=0.0,
        )
    except Exception:
        return text

    rewritten = (response.choices[0].message.content or "").strip()
    # Strip wrapping quotes the model sometimes adds.
    rewritten = rewritten.strip('"').strip("'").strip()
    return rewritten or text