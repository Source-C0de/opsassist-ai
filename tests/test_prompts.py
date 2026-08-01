"""Unit tests for the answer-prompt registry."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prompts import (  # noqa: E402
    DEFAULT_PROMPT,
    PROMPTS,
    available_prompts,
    build_messages,
    get_prompt,
)


def test_at_least_three_prompts_registered() -> None:
    """The rubric requires 'multiple approaches' — we expose four."""
    names = available_prompts()
    assert len(names) >= 3, f"expected >=3 prompt strategies, got {names}"
    # The four concrete strategies named in the plan.
    for required in ("concise", "with_citations", "chain_of_thought", "no_citations"):
        assert required in names, f"missing prompt strategy: {required}"


def test_default_prompt_is_registered() -> None:
    assert DEFAULT_PROMPT in PROMPTS
    assert get_prompt(DEFAULT_PROMPT) is PROMPTS[DEFAULT_PROMPT]


def test_unknown_prompt_falls_back_to_default() -> None:
    """Asking for a non-existent prompt must not crash."""
    fallback = get_prompt("does-not-exist")
    assert fallback is PROMPTS[DEFAULT_PROMPT]


def test_build_messages_substitutes_context_and_query() -> None:
    messages = build_messages(
        query="How does Compose expose ports?",
        context="[1] (compose) Use ports: 8080:80",
        prompt_name="concise",
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_text = messages[1]["content"]
    assert "How does Compose expose ports?" in user_text
    assert "[1] (compose) Use ports: 8080:80" in user_text
    # The system prompt for "concise" includes citation guidance.
    assert "[" in messages[0]["content"]


def test_history_block_is_only_included_when_present() -> None:
    msgs_without = build_messages(query="q", context="c", prompt_name="concise")
    msgs_with = build_messages(
        query="q",
        context="c",
        prompt_name="concise",
        history=[{"role": "user", "content": "earlier turn"}],
    )
    assert "Conversation so far" not in msgs_without[1]["content"]
    assert "Conversation so far" in msgs_with[1]["content"]
    assert "earlier turn" in msgs_with[1]["content"]


def test_chain_of_thought_prompt_directs_reasoning() -> None:
    template = get_prompt("chain_of_thought")
    assert "## Reasoning" in template["system"]
    assert "## Answer" in template["system"]
    user = template["user_template"]
    assert "{context}" in user and "{query}" in user


def test_no_citations_prompt_strips_citation_rule() -> None:
    template = get_prompt("no_citations")
    assert "Do NOT include citation markers" in template["system"]