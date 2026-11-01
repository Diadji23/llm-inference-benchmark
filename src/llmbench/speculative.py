"""Greedy speculative decoding: a small draft model proposes, the target model verifies.

Each round, the draft guesses k tokens one by one; the target scores all of them in a single
forward pass, keeps the longest prefix it agrees with, and adds one token of its own. With
greedy decoding the output is exactly what the target alone would generate: only the number
of target passes changes.

The loop only needs a model that can feed tokens and roll its KV cache back (`Model`), so it
is tested with fake models; `LlamaCppModel` adapts a llama.cpp context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from llmbench.metrics import GenerationTiming


class Model(Protocol):
    n_tokens: int  # tokens currently in the KV cache

    def forward(self, tokens: list[int], all_logits: bool = False) -> list[int]:
        """Feed tokens after the cache; greedy next-token prediction at each position
        (all of them with all_logits, otherwise only the last one)."""
        ...

    def rollback(self, n_tokens: int) -> None:
        """Keep only the first n_tokens of the cache."""
        ...


@dataclass
class SpecResult:
    tokens: list[int]
    timing: GenerationTiming
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    draft_s: float = 0.0
    verify_s: float = 0.0
    accepted_per_round: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else float("nan")

    @property
    def tokens_per_round(self) -> float:
        # Tokens produced after the prefill, per target verification pass.
        return (len(self.tokens) - 1) / self.rounds if self.rounds else float("nan")


def _timing(prompt, out, t_start, t_first, t_end) -> GenerationTiming:
    return GenerationTiming(t_start=t_start, t_first_token=t_first, t_end=t_end,
                            n_prompt_tokens=len(prompt), n_generated_tokens=len(out))


def greedy_generate(target: Model, prompt: list[int], max_new_tokens: int) -> SpecResult:
    """The target alone, through the same interface: the reference for speed and output."""
    target.rollback(0)
    t_start = time.perf_counter()
    out = [target.forward(prompt)[-1]]
    t_first = time.perf_counter()
    while len(out) < max_new_tokens:
        out.append(target.forward([out[-1]])[-1])
    t_end = time.perf_counter()
    return SpecResult(tokens=out, timing=_timing(prompt, out, t_start, t_first, t_end),
                      rounds=len(out) - 1)


def speculative_generate(
    target: Model, draft: Model, prompt: list[int], max_new_tokens: int, k: int
) -> SpecResult:
    """Greedy speculative decoding with k draft tokens per round.

    Invariant between rounds: the target's cache holds the whole sequence but its last
    token, which is fed first in the next verification pass. The draft's cache holds a
    prefix of the sequence; the rest is fed before it starts guessing.
    """
    if k < 1:
        raise ValueError("k must be >= 1; use greedy_generate for the target alone")
    target.rollback(0)
    draft.rollback(0)
    t_start = time.perf_counter()
    out = [target.forward(prompt)[-1]]
    # The draft's prefill belongs to the time to first token, not to decode.
    draft.forward(prompt)
    t_first = time.perf_counter()
    res = SpecResult(tokens=out, timing=None)  # type: ignore[arg-type]
    seq_len = len(prompt) + 1  # prompt + out

    while len(out) < max_new_tokens:
        # Draft: catch up with the accepted tokens, then guess k tokens one at a time.
        t0 = time.perf_counter()
        seq = prompt + out
        guesses = [draft.forward(seq[draft.n_tokens:])[-1]]
        for _ in range(k - 1):
            guesses.append(draft.forward([guesses[-1]])[-1])
        t1 = time.perf_counter()

        # Target: the last accepted token plus the k guesses, in one pass. preds[i] is
        # what the target would generate after guess i-1, so guess i is right if it
        # equals preds[i].
        base = target.n_tokens
        preds = target.forward([out[-1]] + guesses, all_logits=True)
        n_ok = 0
        while n_ok < k and guesses[n_ok] == preds[n_ok]:
            n_ok += 1
        res.verify_s += time.perf_counter() - t1
        res.draft_s += t1 - t0

        # Accepted guesses, then the target's own token (a correction or a bonus).
        out.extend(guesses[:n_ok] + [preds[n_ok]])
        # Drop the rejected guesses from both caches. The target keeps the old last
        # token and the accepted guesses; the draft only ever fed the first k-1 guesses.
        target.rollback(base + 1 + n_ok)
        draft.rollback(seq_len + min(n_ok, k - 1))
        seq_len += n_ok + 1
        res.rounds += 1
        res.proposed += k
        res.accepted += n_ok
        res.accepted_per_round.append(n_ok)

    t_end = time.perf_counter()
    del out[max_new_tokens:]  # the last round may overshoot
    res.timing = _timing(prompt, out, t_start, t_first, t_end)
    return res


class LlamaCppModel:
    """A llama.cpp context seen as a `Model`: batches built by hand, so that only the
    positions we need get logits (the full prompt would otherwise cost one vocabulary
    projection per token)."""

    def __init__(self, llm):
        self.llm = llm
        self.n_vocab = llm.n_vocab()

    @property
    def n_tokens(self) -> int:
        return self.llm.n_tokens

    def forward(self, tokens: list[int], all_logits: bool = False) -> list[int]:
        llm = self.llm
        llm._ctx.kv_cache_seq_rm(-1, llm.n_tokens, -1)  # drop rolled-back positions
        preds: list[int] = []
        for i in range(0, len(tokens), llm.n_batch):
            batch = tokens[i:i + llm.n_batch]
            llm._batch.set_batch(batch=batch, n_past=llm.n_tokens, logits_all=all_logits)
            llm._ctx.decode(llm._batch)
            llm.input_ids[llm.n_tokens:llm.n_tokens + len(batch)] = batch
            llm.n_tokens += len(batch)
            rows = len(batch) if all_logits else 1
            logits = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(rows * self.n_vocab,))
            preds.extend(logits.reshape(rows, self.n_vocab).argmax(axis=1).tolist())
        return preds

    def rollback(self, n_tokens: int) -> None:
        self.llm.n_tokens = n_tokens
