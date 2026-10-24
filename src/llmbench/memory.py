"""Peak memory measurement (process RSS).

ru_maxrss gives the peak over the whole process lifetime and cannot be reset: useless for
comparing several models within one run. RSS is therefore sampled from a thread over a
given window. Known limitation: a spike shorter than the sampling period can be missed
(underestimate on the order of a few MB).
"""

from __future__ import annotations

import re
import threading
from typing import Self

import psutil

MIB = 2**20

# Bytes per stored element. Q8_0 and Q4_0 store blocks of 32 values plus one F16 scale:
# 32 + 2 = 34 bytes and 16 + 2 = 18 bytes per block.
KV_BYTES_PER_ELEMENT = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}


def current_rss_mib() -> float:
    return psutil.Process().memory_info().rss / MIB


def kv_cache_mib(
    n_layers: int, n_kv_heads: int, head_dim: int, n_ctx: int, kv_type: str = "f16"
) -> float:
    """Theoretical KV cache size: K and V, for every layer, KV head and context position."""
    n_elements = 2 * n_layers * n_kv_heads * head_dim * n_ctx
    return n_elements * KV_BYTES_PER_ELEMENT[kv_type] / MIB


_LOG_SIZES = {
    "model_buffer_mib": re.compile(r"model buffer size =\s*([\d.]+) MiB"),
    "kv_cache_mib": re.compile(r"llama_kv_cache: size =\s*([\d.]+) MiB"),
    "compute_buffer_mib": re.compile(r"compute buffer size =\s*([\d.]+) MiB"),
    "output_buffer_mib": re.compile(r"output buffer size =\s*([\d.]+) MiB"),
}


def parse_llamacpp_buffers(log: str) -> dict[str, float]:
    """Memory llama.cpp reserves at load time, read from its log.

    Model buffers are summed (memory-mapped weights + weights repacked for AVX2). The
    compute buffer is printed again when the context is freed: only the first one counts.
    Reserved is not resident: RSS only counts pages that are actually touched.
    """
    sizes = {}
    for key, pattern in _LOG_SIZES.items():
        values = [float(v) for v in pattern.findall(log)]
        if values:
            sizes[key] = sum(values) if key == "model_buffer_mib" else values[0]
    return sizes


class PeakRSSSampler:
    """Context manager: `with PeakRSSSampler() as s: ...` then `s.peak_mib`."""

    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self.peak_mib = 0.0
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        rss = self._proc.memory_info().rss / MIB
        self.peak_mib = max(self.peak_mib, rss)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> Self:
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
