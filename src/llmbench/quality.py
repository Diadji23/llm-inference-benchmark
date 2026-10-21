"""Quality metrics: perplexity and agreement with a reference (F16) model.

Perplexity follows llama.cpp's `perplexity` tool: the text is split into non-overlapping
windows of `ctx` tokens and only the second half of each window is scored, so every
scored token sees at least ctx/2 tokens of context.

Agreement with the reference is measured on the same scored positions:
  - top-1 agreement: the quantized model's most likely token equals the reference's;
  - coarse KL(ref || model): the reference distribution is kept only as its top-k tokens
    plus one "tail" bucket (storing the full 150k-way distribution would cost ~1 GB per
    model). Merging outcomes can only shrink a KL divergence, so this is a lower bound
    of the true KL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from llmbench.backends import Backend


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def window_starts(n_tokens: int, ctx: int, n_windows: int) -> list[int]:
    available = n_tokens // ctx
    if available < n_windows:
        raise ValueError(f"text too short: {available} windows of {ctx} < {n_windows}")
    return [i * ctx for i in range(n_windows)]


@dataclass
class ScoredPositions:
    """Per scored position: NLL of the true next token, and the model's top-k."""

    nll: np.ndarray          # (n,)
    topk_ids: np.ndarray     # (n, k), most likely first
    topk_logp: np.ndarray    # (n, k)


def score_window(logp: np.ndarray, targets: np.ndarray, k: int) -> ScoredPositions:
    """`logp[i]` is the log-distribution predicting `targets[i]`."""
    nll = -logp[np.arange(len(targets)), targets]
    # argpartition isolates the k best in O(vocab), then only those k are sorted: a full
    # argsort over a 150k vocabulary was ~25% of the runtime.
    part = np.argpartition(-logp, k, axis=-1)[:, :k]
    order = np.argsort(-np.take_along_axis(logp, part, axis=-1), axis=-1)
    topk_ids = np.take_along_axis(part, order, axis=-1)
    topk_logp = np.take_along_axis(logp, topk_ids, axis=-1)
    return ScoredPositions(nll, topk_ids, topk_logp)


def coarse_kl(ref_ids: np.ndarray, ref_logp: np.ndarray, model_logp: np.ndarray) -> np.ndarray:
    """KL(ref || model) over {ref top-k tokens, everything else}; one value per position.

    `model_logp` is the model's full log-distribution (n, vocab).
    """
    p = np.exp(ref_logp)
    q = np.exp(np.take_along_axis(model_logp, ref_ids, axis=-1))
    p_tail = np.clip(1.0 - p.sum(axis=-1), 1e-12, None)
    q_tail = np.clip(1.0 - q.sum(axis=-1), 1e-12, None)
    head = (p * (ref_logp - np.log(np.clip(q, 1e-30, None)))).sum(axis=-1)
    return head + p_tail * np.log(p_tail / q_tail)


def evaluate_text(backend: Backend, tokens: list[int], ctx: int, n_windows: int,
                  k: int = 20, reference: ScoredPositions | None = None) -> dict:
    """Perplexity of `backend` on `tokens`, plus agreement with `reference` if given.

    Returns metrics and, under "scored", the ScoredPositions to reuse as a reference.
    """
    half = ctx // 2
    parts, kls = [], []
    for w, start in enumerate(window_starts(len(tokens), ctx, n_windows)):
        window = tokens[start:start + ctx]
        # Position i predicts token i+1: score positions half-1 .. ctx-2.
        logp = log_softmax(backend.logits(window)[half - 1:ctx - 1].astype(np.float64))
        scored = score_window(logp, np.asarray(window[half:]), k)
        parts.append(scored)
        if reference is not None:
            sl = slice(w * (ctx - half), (w + 1) * (ctx - half))
            kls.append(coarse_kl(reference.topk_ids[sl], reference.topk_logp[sl], logp))

    scored = ScoredPositions(*(np.concatenate([getattr(p, f) for p in parts])
                               for f in ("nll", "topk_ids", "topk_logp")))
    mean_nll = float(scored.nll.mean())
    # Standard error of the mean NLL, propagated to perplexity (delta method).
    se_nll = float(scored.nll.std(ddof=1) / math.sqrt(scored.nll.size))
    out = {
        "n_scored": int(scored.nll.size),
        "ppl": math.exp(mean_nll),
        "ppl_se": math.exp(mean_nll) * se_nll,
        "scored": scored,
    }
    if reference is not None:
        kl = np.concatenate(kls)
        out["top1_agreement"] = float((scored.topk_ids[:, 0] == reference.topk_ids[:, 0]).mean())
        out["kl_coarse_mean"] = float(kl.mean())
        out["kl_coarse_p99"] = float(np.percentile(kl, 99))
    return out
