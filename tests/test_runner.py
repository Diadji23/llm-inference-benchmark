import json

import pytest

from llmbench.env import machine_canary_ms
from llmbench.runner import measurement_units, schedule, summarize_run

MODELS = [{"name": f"m{i}"} for i in range(5)]


def test_schedule_visits_each_model_once_per_round():
    plan = schedule(MODELS, rounds=4, seed=0)
    assert len(plan) == 20
    for r in range(4):
        names = sorted(m["name"] for rnd, m in plan if rnd == r)
        assert names == [m["name"] for m in MODELS]


def test_schedule_shuffles_between_rounds_and_is_reproducible():
    plan = schedule(MODELS, rounds=4, seed=0)
    orders = {tuple(m["name"] for rnd, m in plan if rnd == r) for r in range(4)}
    assert len(orders) > 1
    assert plan == schedule(MODELS, rounds=4, seed=0)


def test_canary_returns_positive_milliseconds():
    assert 0 < machine_canary_ms(repeats=3) < 1000


def _cfg(**extra):
    cfg = {"models": [{"name": "a"}, {"name": "b"}], "prompts": {"lengths": [512, 2048]},
           "generation": {"max_new_tokens": 128}}
    cfg.update(extra)
    return cfg


def test_units_without_kv_cache_keep_one_load_per_model():
    units = measurement_units(_cfg())
    assert [u["model"]["name"] for u in units] == ["a", "b"]
    assert all(u["lengths"] == [512, 2048] and u["backend_args"] == {} for u in units)


def test_units_with_kv_cache_and_context_per_prompt():
    cfg = _cfg(kv_cache=["f16", "q8_0"])
    cfg["prompts"]["ctx_per_prompt"] = True
    units = measurement_units(cfg)
    assert len(units) == 2 * 2 * 2  # models x KV types x prompt lengths
    unit = next(u for u in units if u["model"]["name"] == "b" and u["kv_type"] == "q8_0"
                and u["lengths"] == [2048])
    assert unit["backend_args"] == {"type_k": "q8_0", "type_v": "q8_0", "n_ctx": 2048 + 128}
    assert unit["flash_attn"] is None


def test_units_with_flash_attn_on_and_off():
    units = measurement_units(_cfg(kv_cache=["f16"], flash_attn=[True, False]))
    assert len(units) == 2 * 2  # models x flash attention settings
    assert {u["backend_args"]["flash_attn"] for u in units} == {True, False}


def test_units_reject_quantized_cache_without_flash_attn():
    with pytest.raises(ValueError, match="requires flash attention"):
        measurement_units(_cfg(kv_cache=["q8_0"], flash_attn=[False]))


def _record(kv_type=None, **extra):
    rec = {"model": "a", "quant": "Q4_K_M", "prompt_len": 512, "n_generated_tokens": 128,
           "load_s": 1.0, "ttft_s": 1.0, "latency_s": 2.0, "prefill_tps": 512.0,
           "decode_tps": 127.0, "peak_rss_mib": 900.0, "rss_before_load_mib": 100.0, **extra}
    if kv_type:
        rec["kv_type"] = kv_type
    return rec


def test_summarize_run_splits_by_kv_type_and_reads_old_runs(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(_record(kv, kv_cache_mib=size))
                             for kv, size in [("f16", 24.0), ("f16", 24.0), ("q8_0", 12.75)]))
    summary = summarize_run(raw)
    assert sorted(summary.kv_type) == ["f16", "q8_0"]
    assert summary.set_index("kv_type").loc["q8_0", "kv_cache_mib"] == 12.75
    assert summary.set_index("kv_type").loc["f16", "net_peak_rss_mib_median"] == 800.0

    raw.write_text(json.dumps(_record()))  # step 1-2 record: no KV type, no buffer sizes
    summary = summarize_run(raw)
    assert summary.kv_type.isna().all() and "kv_cache_mib" not in summary


def test_summarize_run_splits_by_flash_attn(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(_record("f16", flash_attn=fa, decode_tps=tps))
                             for fa, tps in [(True, 4.0), (True, 4.0), (False, 15.0)]))
    summary = summarize_run(raw).set_index("flash_attn")
    assert summary.loc[True, "decode_tps_median"] == 4.0
    assert summary.loc[False, "decode_tps_median"] == 15.0
