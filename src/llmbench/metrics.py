"""Latency/throughput metrics and statistical aggregation.

Definitions (one generation = a prompt of n_prompt tokens -> n_generated tokens):
  - TTFT               : time to the first generated token (~ prefill + 1 decode step)
  - latency            : total generation time
  - prefill throughput : n_prompt / TTFT                         (tokens/s, compute-bound)
  - decode throughput  : (n_generated - 1) / (t_end - t_first)   (tokens/s, memory-bound)
The first token is excluded from decode throughput: it is produced by the prefill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Two-sided 97.5% quantiles of Student's t distribution, df 1 to 30. Avoids a scipy
# dependency for a single number; beyond 30 df the normal approximation is good enough.
_T_975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


@dataclass(frozen=True)
class GenerationTiming:
    """Raw timestamps of one generation (seconds, time.perf_counter clock)."""

    t_start: float
    t_first_token: float
    t_end: float
    n_prompt_tokens: int
    n_generated_tokens: int

    @property
    def ttft_s(self) -> float:
        return self.t_first_token - self.t_start

    @property
    def latency_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def prefill_tps(self) -> float:
        return self.n_prompt_tokens / self.ttft_s if self.ttft_s > 0 else math.nan

    @property
    def decode_tps(self) -> float:
        dt = self.t_end - self.t_first_token
        if self.n_generated_tokens < 2 or dt <= 0:
            return math.nan
        return (self.n_generated_tokens - 1) / dt

    def as_dict(self) -> dict:
        return {
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_generated_tokens": self.n_generated_tokens,
            "ttft_s": self.ttft_s,
            "latency_s": self.latency_s,
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
        }


def t_critical(df: int) -> float:
    if df < 1:
        return math.nan
    return _T_975[df - 1] if df <= len(_T_975) else 1.96


def summarize(values) -> dict:
    """Mean, std, median, p95 and 95% CI (Student) of a series of measurements.

    NaNs are ignored. The CI assumes independent repetitions: this is why the runner
    warms up first and repeats whole generations rather than individual tokens.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "median": math.nan,
                "p95": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else math.nan
    half = t_critical(n - 1) * std / math.sqrt(n) if n > 1 else math.nan
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }
