"""Experiment loop: interleaved rounds of load, warmup, measurement, writing results.

Output in results/<experiment>/<run_id>/:
  config.yaml  copy of the config used
  env.json     hardware/software snapshot
  raw.jsonl    one line per measured generation
  summary.csv  aggregates per (model, KV cache type, prompt length)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from llmbench.backends import make_backend
from llmbench.env import collect_env, machine_canary_ms
from llmbench.memory import PeakRSSSampler, current_rss_mib
from llmbench.metrics import summarize

SUMMARY_METRICS = (
    "ttft_s", "latency_s", "prefill_tps", "decode_tps", "peak_rss_mib", "net_peak_rss_mib",
)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def new_run_dir(config_path: Path, cfg: dict, results_root: Path, prefix: str = "") -> Path:
    """results/<experiment>/<prefix><timestamp>/ with a copy of the config and env.json."""
    run_id = prefix + datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = results_root / cfg["experiment"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(Path(config_path).read_text())
    (out_dir / "env.json").write_text(json.dumps(collect_env(), indent=2, ensure_ascii=False))
    return out_dir


def model_label(model: dict) -> str:
    return f"{model['name']} {model['quant']}" if model.get("quant") else model["name"]


def measurement_units(cfg: dict) -> list[dict]:
    """What one load measures: a model, a KV cache type, and the prompt lengths to run.

    Without `kv_cache` in the config, the cache stays at the backend default (steps 1-2).
    With `ctx_per_prompt`, each prompt length gets its own load with n_ctx = prompt +
    generation: llama.cpp allocates the KV cache for n_ctx up front, so a context sized
    for the longest prompt would make memory look the same at every length.
    """
    lengths = cfg["prompts"]["lengths"]
    per_prompt = cfg["prompts"].get("ctx_per_prompt", False)
    units = []
    for model in cfg["models"]:
        for kv in cfg.get("kv_cache", [None]):
            for group in ([[n] for n in lengths] if per_prompt else [lengths]):
                args = {} if kv is None else {"type_k": kv, "type_v": kv}
                if per_prompt:
                    args["n_ctx"] = group[0] + cfg["generation"]["max_new_tokens"]
                units.append({"model": model, "kv_type": kv, "lengths": group,
                              "backend_args": args})
    return units


def unit_label(unit: dict) -> str:
    label = model_label(unit["model"])
    if unit["kv_type"]:
        label += f" KV {unit['kv_type']}"
    if "n_ctx" in unit["backend_args"]:
        label += f" ctx {unit['backend_args']['n_ctx']}"
    return label


def schedule(units: list[dict], rounds: int, seed: int = 0) -> list[tuple[int, dict]]:
    """Interleaved plan: every round visits each unit once, in a freshly shuffled order.

    Measuring one model in a block lets a temporary disturbance (another program, a
    background update) land entirely on that model. Interleaving spreads it over
    different models in each round, and the order is seeded so the plan is reproducible.
    """
    rng = np.random.default_rng(seed)
    return [(r, units[i]) for r in range(rounds) for i in rng.permutation(len(units))]


def run_experiment(config_path: Path, results_root: Path = Path("results")) -> Path:
    cfg = load_config(config_path)
    out_dir = new_run_dir(config_path, cfg, results_root)
    source_text = Path(cfg["prompts"]["source"]).read_text()
    plan = schedule(measurement_units(cfg), cfg["repetitions"], cfg.get("schedule_seed", 0))

    with open(out_dir / "raw.jsonl", "w") as raw:
        for i, (rnd, unit) in enumerate(plan):
            print(f"\n[{i + 1}/{len(plan)}] round {rnd}", end=" ")
            _measure_once(unit, rnd, cfg, source_text, raw)

    summary = summarize_run(out_dir / "raw.jsonl")
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n-> results in {out_dir}")
    return out_dir


def _measure_once(unit: dict, rnd: int, cfg: dict, source_text: str, raw) -> None:
    """Load one model, warm it up, measure each prompt length of the unit once, unload it."""
    gen, model = cfg["generation"], unit["model"]
    backend_args = {**cfg.get("backend_args", {}), **unit["backend_args"]}
    backend = make_backend(cfg["backend"], model_path=model["path"], **backend_args)
    rss_before = current_rss_mib()
    t0 = time.perf_counter()
    backend.load()
    load_s = time.perf_counter() - t0
    rss_loaded = current_rss_mib()
    print(f"{unit_label(unit)}: loaded in {load_s:.1f} s, RSS {rss_loaded:.0f} MiB", flush=True)

    try:
        prompts = {n: backend.truncate_to_tokens(source_text, n) for n in unit["lengths"]}
        # A short generation reads every weight once: pages the model in and warms caches.
        # Its prompt defaults to the shortest measured one; long-context runs set a short
        # one, otherwise every warmup would pay for a full long prefill.
        warmup_tokens = backend.truncate_to_tokens(
            source_text, cfg.get("warmup_prompt_len", min(unit["lengths"])))
        for _ in range(cfg["warmup"]):
            backend.generate(warmup_tokens, cfg.get("warmup_new_tokens", 16))

        for n_prompt, tokens in prompts.items():
            canary = machine_canary_ms()
            with PeakRSSSampler() as mem:
                res = backend.generate(tokens, gen["max_new_tokens"], gen["ignore_eos"])
            record = {
                "experiment": cfg["experiment"],
                "model": model["name"],
                "quant": model.get("quant"),
                "kv_type": unit["kv_type"],
                "backend": backend.name,
                "n_ctx": backend_args.get("n_ctx"),
                "prompt_len": n_prompt,
                "rep": rnd,
                **res.timing.as_dict(),
                "peak_rss_mib": mem.peak_mib,
                "rss_before_load_mib": rss_before,
                "rss_loaded_mib": rss_loaded,
                "load_s": load_s,
                # Memory the backend reserved at load time (llama.cpp only).
                **getattr(backend, "buffers", {}),
                # Machine state, to spot disturbed measurements afterwards.
                "unix_time": time.time(),
                "canary_ms": canary,
                "load_avg_1m": os.getloadavg()[0],
                "text_head": res.text[:80],
            }
            raw.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw.flush()
            t = res.timing
            print(f"  prompt {n_prompt:>4} | TTFT {t.ttft_s * 1e3:7.1f} ms | "
                  f"decode {t.decode_tps:6.1f} tok/s | RSS {mem.peak_mib:6.0f} MiB | "
                  f"canary {canary:5.1f} ms", flush=True)
    finally:
        backend.close()


def summarize_run(raw_path: Path) -> pd.DataFrame:
    df = pd.read_json(raw_path, lines=True)
    # Baseline RSS grows from one model to the next (memory not returned to the OS after
    # close()): the net peak is the only figure comparable across models of one run.
    df["net_peak_rss_mib"] = df["peak_rss_mib"] - df["rss_before_load_mib"]
    if "kv_type" not in df:  # runs from before step 3
        df["kv_type"] = None
    buffers = [c for c in ("model_buffer_mib", "kv_cache_mib", "compute_buffer_mib") if c in df]
    rows = []
    keys = ["model", "quant", "kv_type", "prompt_len"]
    for (model, quant, kv_type, n_prompt), g in df.groupby(keys, dropna=False):
        row = {"model": model, "quant": quant, "kv_type": kv_type, "prompt_len": n_prompt,
               "n_generated": int(g["n_generated_tokens"].median()),
               "load_s": g["load_s"].median(),
               # Reserved sizes are fixed for a given load configuration.
               **{c: g[c].median() for c in buffers}}
        for metric in SUMMARY_METRICS:
            s = summarize(g[metric])
            row["n"] = s["n"]
            for stat in ("mean", "std", "median", "q25", "q75", "ci95_low", "ci95_high"):
                row[f"{metric}_{stat}"] = s[stat]
        rows.append(row)
    return pd.DataFrame(rows)
