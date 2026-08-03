"""Unit tests for the embedding batching + retry logic in ``ingest_core``.

We don't hit the real OpenAI API here — these tests verify the batch planner
respects the token budget and that the retry helper eventually succeeds
(or surfaces the last error) when given a fake client.
"""

from __future__ import annotations

import sys
import time as time_module
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import ingest_core  # noqa: E402
from ingest_core import _embed_batch_with_retry, _plan_token_batches  # noqa: E402


# --- _plan_token_batches ----------------------------------------------------
def _toks(text: str) -> int:
    """Token count for an arbitrary string using the same encoder as the planner."""
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def test_plan_token_batches_splits_when_token_budget_exceeded() -> None:
    """Three ~500-token chunks with a 1,200-token budget: the first two fit
    in one batch, the third overflows into its own batch."""
    # Build a chunk that's around 500 tokens with cl100k_base.
    # "token " encodes as 2 tokens, so 500 reps = ~1000 tokens. To get
    # closer to 500 tokens, use a denser string.
    chunk = "tok" * 500  # ~500 tokens
    assert 400 <= _toks(chunk) <= 600
    texts = [chunk, chunk, chunk]
    batches = _plan_token_batches(texts, token_budget=1200, item_cap=100)
    assert len(batches) == 2
    # Two chunks fit (~1000 tokens), third spills over.
    assert batches[0] == [0, 1]
    assert batches[1] == [2]


def test_plan_token_batches_respects_item_cap() -> None:
    """Small chunks should still be grouped but capped at the item cap."""
    chunk = "word " * 10  # small enough to never hit the token budget
    texts = [chunk] * 5
    batches = _plan_token_batches(texts, token_budget=10_000, item_cap=2)
    assert batches == [[0, 1], [2, 3], [4]]


def test_plan_token_batches_oversized_chunk_is_isolated() -> None:
    """A single chunk larger than the budget is shipped on its own — we want
    OpenAI to return a clear 400 rather than silently dropping data."""
    big = "token " * 2_000
    small = "token " * 50
    batches = _plan_token_batches([big, small], token_budget=1_000, item_cap=100)
    # The overflow chunk goes out alone; the small chunk is unaffected.
    assert batches[0] == [0]
    assert batches[1] == [1]


def test_plan_token_batches_empty_input() -> None:
    assert _plan_token_batches([]) == []


def test_plan_token_batches_satisfies_token_budget() -> None:
    """Whatever the inputs, every batch must be within the budget."""
    import random

    random.seed(0)
    texts = ["token " * random.randint(50, 1_500) for _ in range(40)]
    budget = 5_000
    batches = _plan_token_batches(texts, token_budget=budget, item_cap=10)
    for batch in batches:
        total = sum(_toks(texts[i]) for i in batch)
        assert total <= budget, f"batch over budget: {total}"
        assert len(batch) <= 10


# --- _embed_batch_with_retry ------------------------------------------------
class _Item:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeResponse:
    def __init__(self, data):
        self.data = [_Item(v) for v in data]


class _RateLimitStubError(Exception):
    """Stand-in for openai.RateLimitError used by tests."""


class _FakeClient:
    """Minimal OpenAI client stub driven by a list of scripted outcomes.

    Each outcome is either the string ``"raise"`` (raise a
    ``_RateLimitStubError``) or a list of embedding vectors. The retry
    helper will see them one at a time and either succeed or move on.
    """

    def __init__(self, script):
        self.script = list(script)
        self.call_count = 0
        self.last_kwargs = None

        outer = self

        class _Embeddings:
            def create(self, model, input):
                outer.last_kwargs = {"model": model, "input": list(input)}
                outcome = outer.script[outer.call_count]
                outer.call_count += 1
                if outcome == "raise":
                    raise _RateLimitStubError(f"429 attempt {outer.call_count}")
                return _FakeResponse(outcome)

        # Bind the inner class so ``client.embeddings.create`` works.
        self.embeddings = _Embeddings()


@pytest.fixture
def _patch_openai_exceptions(monkeypatch):
    """Make the retry helper treat ``_RateLimitStubError`` as the catch-all
    category so we don't have to install the real ``openai`` SDK or fabricate
    its exception hierarchy. Only applied to tests that opt in."""
    import sys
    import types

    fake_openai = types.ModuleType("openai")
    fake_openai.RateLimitError = _RateLimitStubError
    fake_openai.APITimeoutError = _RateLimitStubError
    fake_openai.APIError = _RateLimitStubError
    fake_openai.OpenAI = lambda *a, **kw: None  # tests override this
    monkeypatch.setitem(sys.modules, "openai", fake_openai)


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip the backoff sleep so tests run fast."""
    monkeypatch.setattr(time_module, "sleep", lambda _s: None)


def test_retry_succeeds_after_two_429s(_patch_openai_exceptions, no_sleep) -> None:
    """After two 429s, the third attempt should succeed."""
    client = _FakeClient(["raise", "raise", [[0.1, 0.2]]])
    vectors = _embed_batch_with_retry(client, "text-embedding-3-small", ["hi"])
    assert client.call_count == 3
    assert vectors == [[0.1, 0.2]]


def test_retry_gives_up_after_max_retries(_patch_openai_exceptions, no_sleep) -> None:
    """If all attempts raise, the last exception should propagate."""
    client = _FakeClient(["raise"] * (ingest_core.EMBED_MAX_RETRIES + 5))
    with pytest.raises(_RateLimitStubError):
        _embed_batch_with_retry(client, "text-embedding-3-small", ["hi"])
    assert client.call_count == ingest_core.EMBED_MAX_RETRIES


def test_retry_no_retry_on_immediate_success(_patch_openai_exceptions, no_sleep) -> None:
    """A clean first call should not loop."""
    client = _FakeClient([[[0.0, 0.1]]])
    _embed_batch_with_retry(client, "text-embedding-3-small", ["hi"])
    assert client.call_count == 1


def test_retry_inputs_pass_through(_patch_openai_exceptions, no_sleep) -> None:
    """The model name and full input list must reach the SDK unchanged."""
    client = _FakeClient([[[0.0], [0.1]]])
    _embed_batch_with_retry(client, "text-embedding-3-small", ["a", "b"])
    assert client.last_kwargs == {
        "model": "text-embedding-3-small",
        "input": ["a", "b"],
    }


# --- embed_texts integration (no network) ----------------------------------
def test_embed_texts_retries_real_rate_limit(
    _patch_openai_exceptions, no_sleep, monkeypatch
) -> None:
    """End-to-end: embed_texts should drive the fake client through its
    batch plan and retry on 429s."""
    monkeypatch.setattr(ingest_core, "OPENAI_API_KEY", "sk-test", raising=False)
    monkeypatch.setattr(ingest_core, "require_openai_key", lambda: None, raising=False)
    # Force a deterministic batch plan: [0, 1] then [2].
    monkeypatch.setattr(
        ingest_core, "_plan_token_batches", lambda _t: [[0, 1], [2]], raising=False
    )

    # Script: batch 1 needs 2 retries then succeeds (3 calls),
    #         batch 2 needs 1 retry then succeeds (2 calls). Total 5.
    script = [
        "raise", "raise", [[0.1], [0.2]],
        "raise", [[0.3]],
    ]
    client = _FakeClient(script)
    # Patch the OpenAI factory inside the `openai` module that the retry
    # helper imports from, so that freshly-imported references also see it.
    import sys

    sys.modules["openai"].OpenAI = lambda api_key: client

    texts = ["first", "second", "third"]
    vectors = ingest_core.embed_texts(texts, model="text-embedding-3-small")
    # Order preserved regardless of which batch was first.
    assert vectors[0] == [0.1]
    assert vectors[1] == [0.2]
    assert vectors[2] == [0.3]
    assert client.call_count == 5


def test_embed_texts_empty_input_returns_empty() -> None:
    assert ingest_core.embed_texts([]) == []
