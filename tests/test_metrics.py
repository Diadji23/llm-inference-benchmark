import math
import time

import pytest

from llmbench.memory import PeakRSSSampler
from llmbench.metrics import GenerationTiming, summarize, t_critical


def test_timing_derived_metrics():
    t = GenerationTiming(t_start=10.0, t_first_token=10.5, t_end=12.5,
                         n_prompt_tokens=100, n_generated_tokens=41)
    assert t.ttft_s == pytest.approx(0.5)
    assert t.latency_s == pytest.approx(2.5)
    assert t.prefill_tps == pytest.approx(200.0)
    # 40 tokens après le premier, en 2 s.
    assert t.decode_tps == pytest.approx(20.0)


def test_decode_tps_undefined_for_single_token():
    t = GenerationTiming(0.0, 1.0, 1.0, n_prompt_tokens=8, n_generated_tokens=1)
    assert math.isnan(t.decode_tps)


def test_summarize_known_values():
    s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(3.0)
    assert s["std"] == pytest.approx(math.sqrt(2.5))
    assert s["median"] == pytest.approx(3.0)
    half = 2.776 * math.sqrt(2.5) / math.sqrt(5)
    assert s["ci95_low"] == pytest.approx(3.0 - half)
    assert s["ci95_high"] == pytest.approx(3.0 + half)


def test_summarize_ignores_nan_and_handles_edge_cases():
    assert summarize([float("nan"), 2.0, 4.0])["mean"] == pytest.approx(3.0)
    assert summarize([])["n"] == 0
    single = summarize([7.0])
    assert single["mean"] == 7.0 and math.isnan(single["ci95_low"])


def test_t_critical_falls_back_to_normal():
    assert t_critical(4) == pytest.approx(2.776)
    assert t_critical(1000) == pytest.approx(1.96)


def test_peak_rss_sees_allocation():
    with PeakRSSSampler(interval_s=0.005) as base:
        time.sleep(0.02)
    with PeakRSSSampler(interval_s=0.005) as mem:
        buf = bytearray(200 * 2**20)  # 200 Mio, touchés par l'initialisation à zéro
        time.sleep(0.05)
        del buf
    assert mem.peak_mib - base.peak_mib > 150
