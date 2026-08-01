"""Named answer-prompt strategies for the RAG pipeline.

The LLM-evaluation rubric (LLM Zoomcamp capstone) awards 2 points for comparing
multiple prompt styles and picking the best one. This module defines four
self-contained prompt variants that all consume the same retrieval context
but instruct the LLM to behave differently:

* ``concise`` — short Markdown answer, citation-anchored (production default).
* ``with_citations`` — bullet-point citations after every claim.
* ``chain_of_thought`` — explicit reasoning trace, then a clean answer.
* ``no_citations`` — same shape as ``concise`` but without the citation rule.

Every variant returns ``(system_prompt, user_prompt_template)`` so the rest
of the pipeline can keep using the same context-preparation code path.
``build_messages`` is the single entry point used by the evaluation harness.
"""

from __future__ import annotations

from typing import Sequence


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
PROMPTS: dict[str, dict[str, str]] = {
    "concise": {
        "system": (
            "You are OpsAssist, a concise ops/devops assistant. Answer the "
            "user's question using ONLY the provided context passages. If the "
            "context does not contain the answer, say so explicitly — do not "
            "invent. Cite the passage numbers in square brackets, e.g. [1], "
            "[2]. Keep answers tight: lead with the answer, then 1-3 "
            "sentences of detail."
        ),
        "user_template": (
            "Context passages:\n{context}\n\n"
            "{history_block}"
            "Question: {query}\n\n"
            "Answer with citations:"
        ),
    },
    "with_citations": {
        "system": (
            "You are OpsAssist, an ops/devops assistant. Use ONLY the "
            "provided context passages. Structure every answer as a Markdown "
            "bullet list — every bullet MUST end with a bracketed citation, "
            "e.g. \"Docker reuses the cache per instruction [1].\". Add a "
            "final line of the form \"Sources: [1], [2], [3]\" listing the "
            "unique citations you actually used. If the context does not "
            "contain the answer, say so explicitly."
        ),
        "user_template": (
            "Context passages:\n{context}\n\n"
            "{history_block}"
            "Question: {query}\n\n"
            "Answer (bullet list with citations, then a Sources line):"
        ),
    },
    "chain_of_thought": {
        "system": (
            "You are OpsAssist, an ops/devops assistant. Reason step by step "
            "through the evidence in the provided context, then give a final "
            "answer. Structure the response in two sections: \"## Reasoning\" "
            "(short bullet trace of which passages support which conclusion) "
            "and \"## Answer\" (the final answer with bracketed citations "
            "like [1], [2]). Use ONLY the provided context — never invent."
        ),
        "user_template": (
            "Context passages:\n{context}\n\n"
            "{history_block}"
            "Question: {query}\n\n"
            "Produce the two-section reasoning + answer:"
        ),
    },
    "no_citations": {
        "system": (
            "You are OpsAssist, a concise ops/devops assistant. Answer the "
            "user's question using ONLY the provided context passages. If "
            "the context does not contain the answer, say so explicitly — do "
            "not invent. Keep answers tight: lead with the answer, then 1-3 "
            "sentences of detail. Do NOT include citation markers."
        ),
        "user_template": (
            "Context passages:\n{context}\n\n"
            "{history_block}"
            "Question: {query}\n\n"
            "Answer (no citations):"
        ),
    },
}


DEFAULT_PROMPT = "concise"


def available_prompts() -> list[str]:
    """Return the list of registered prompt strategy names."""
    return list(PROMPTS.keys())


def get_prompt(name: str) -> dict[str, str]:
    """Return the prompt dict for ``name``; fall back to the default."""
    if name not in PROMPTS:
        return PROMPTS[DEFAULT_PROMPT]
    return PROMPTS[name]


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------
def _format_history(history: Sequence[dict]) -> str:
    """Render the last few conversation turns as a 'user:/assistant:' block."""
    lines: list[str] = []
    for turn in (history or [])[-6:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def build_messages(
    query: str,
    context: str,
    prompt_name: str = DEFAULT_PROMPT,
    history: Sequence[dict] | None = None,
) -> list[dict[str, str]]:
    """Compose chat-completion messages for the requested prompt style."""
    template = get_prompt(prompt_name)
    history_block = _format_history(history or [])
    user_content = template["user_template"].format(
        context=context, history_block=history_block, query=query
    )
    return [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": user_content},
    ]