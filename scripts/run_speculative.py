"""Speculative decoding: speed, acceptance rate and output check for each (target, k).

Usage: python scripts/run_speculative.py configs/04_speculative.yaml

Writes results/<experiment>/<run_id>/ with raw.jsonl (one line per generation),
verify.jsonl (cost of one target pass by batch size), summary.csv and verify_cost.csv.
Each round loads every target in a shuffled order, with the draft; within a load, the
(prompt kind, k) combinations also run in a shuffled order. k = 0 is the target alone.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from llmbench.backends.llamacpp import LlamaCppBackend
from llmbench.env import machine_canary_ms
from llmbench.metrics import summarize
from llmbench.runner import load_config, model_label, new_run_dir
from llmbench.speculative import LlamaCppModel, greedy_generate, speculative_generate


def measure_verify_cost(target: LlamaCppModel, prompt: list[int], sizes: list[int],
                        repeats: int = 3) -> dict[int, float]:
    """Median time of one target pass over b tokens, after the prompt, for each b."""
    costs = {}
    for b in sizes:
        times = []
        for _ in range(repeats):
            target.rollback(0)
            target.forward(prompt)
            t0 = time.perf_counter()
            target.forward(prompt[:b], all_logits=True)  # any b tokens: same work
            times.append(time.perf_counter() - t0)
        costs[b] = float(np.median(times))
    return costs


def run(cfg: dict, out_dir: Path) -> None:
    gen = cfg["generation"]
    n_new = gen["max_new_tokens"]
    length = cfg["prompts"]["length"]
    ks = cfg["draft_tokens"]
    n_ctx = length + n_new + max(ks) + 1
    args = {**cfg.get("backend_args", {}), "n_ctx": n_ctx}
    texts = {kind: Path(src).read_text() for kind, src in cfg["prompts"]["sources"].items()}
    rng = np.random.default_rng(cfg.get("schedule_seed", 0))
    draft_cfg = cfg["draft"]

    with open(out_dir / "raw.jsonl", "w") as raw, open(out_dir / "verify.jsonl", "w") as ver:
        for rnd in range(cfg["repetitions"]):
            for ti in rng.permutation(len(cfg["targets"])):
                target_cfg = cfg["targets"][ti]
                tb = LlamaCppBackend(target_cfg["path"], **args)
                db = LlamaCppBackend(draft_cfg["path"], **args)
                tb.load()
                db.load()
                print(f"\nround {rnd} target {model_label(target_cfg)}, "
                      f"draft {model_label(draft_cfg)}", flush=True)
                try:
                    target, draft = LlamaCppModel(tb._llm), LlamaCppModel(db._llm)
                    prompts = {kind: tb.truncate_to_tokens(text, length)
                               for kind, text in texts.items()}
                    warm = prompts[next(iter(prompts))][:cfg.get("warmup_prompt_len", 32)]
                    for _ in range(cfg["warmup"]):
                        greedy_generate(target, warm, 16)
                        greedy_generate(draft, warm, 16)

                    costs = measure_verify_cost(target, prompts[next(iter(prompts))],
                                                cfg["verify_batch_sizes"])
                    ver.writelines(
                        json.dumps({"target_quant": target_cfg["quant"], "rep": rnd,
                                    "batch": b, "time_s": s}) + "\n"
                        for b, s in costs.items())
                    print("  one target pass: " + ", ".join(
                        f"{b} tok {s * 1e3:.0f} ms" for b, s in costs.items()), flush=True)

                    combos = [(kind, k) for kind in prompts for k in ks]
                    for ci in rng.permutation(len(combos)):
                        kind, k = combos[ci]
                        canary = machine_canary_ms()
                        if k == 0:
                            res = greedy_generate(target, prompts[kind], n_new)
                        else:
                            res = speculative_generate(target, draft, prompts[kind], n_new, k)
                        record = {
                            "experiment": cfg["experiment"],
                            "target": target_cfg["name"], "target_quant": target_cfg["quant"],
                            "draft": draft_cfg["name"], "draft_quant": draft_cfg["quant"],
                            "prompt_kind": kind, "prompt_len": length, "k": k, "rep": rnd,
                            **res.timing.as_dict(),
                            "rounds": res.rounds, "proposed": res.proposed,
                            "accepted": res.accepted,
                            "acceptance_rate": res.acceptance_rate,
                            "tokens_per_round": res.tokens_per_round,
                            "draft_s": res.draft_s, "verify_s": res.verify_s,
                            "unix_time": time.time(), "canary_ms": canary,
                            "load_avg_1m": os.getloadavg()[0],
                            "tokens": res.tokens,
                        }
                        raw.write(json.dumps(record) + "\n")
                        raw.flush()
                        acc = "" if k == 0 else (f" | accepted {res.acceptance_rate:4.0%}"
                                                 f" | {res.tokens_per_round:.2f} tok/pass")
                        print(f"  {kind:5} k={k} | decode {res.timing.decode_tps:5.1f} tok/s"
                              f"{acc} | canary {canary:4.1f} ms", flush=True)
                finally:
                    tb.close()
                    db.close()


def summarize_run(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_json(out_dir / "raw.jsonl", lines=True)
    keys = ["target_quant", "prompt_kind", "rep"]
    ref = df[df["k"] == 0].set_index(keys)
    joined = df.join(ref[["decode_tps", "tokens"]], on=keys, rsuffix="_ref")
    # Same round, same load, same prompt: the cleanest pair for a ratio.
    joined["speedup"] = joined["decode_tps"] / joined["decode_tps_ref"]
    # Exact in exact arithmetic, but llama.cpp does not compute a batch of tokens with
    # exactly the same rounding as one token at a time: near-ties can flip.
    joined["same_output"] = joined["tokens"] == joined["tokens_ref"]
    joined["first_diff"] = [
        next((i for i, (a, b) in enumerate(zip(t, r)) if a != b), None)
        for t, r in zip(joined["tokens"], joined["tokens_ref"])]
    rows = []
    for (quant, kind, k), g in joined.groupby(["target_quant", "prompt_kind", "k"]):
        diffs = g["first_diff"].dropna()
        row = {"target_quant": quant, "prompt_kind": kind, "k": k, "n": len(g),
               "same_output_rate": g["same_output"].mean(),
               "first_diff_median": diffs.median() if len(diffs) else None}
        for metric in ("decode_tps", "speedup", "acceptance_rate", "tokens_per_round",
                       "ttft_s", "latency_s"):
            s = summarize(g[metric].dropna()) if g[metric].notna().any() else {}
            for stat in ("median", "q25", "q75"):
                row[f"{metric}_{stat}"] = s.get(stat, float("nan"))
        rows.append(row)
    verify = pd.read_json(out_dir / "verify.jsonl", lines=True)
    vc = verify.groupby(["target_quant", "batch"])["time_s"].median().reset_index()
    vc["vs_one_token"] = vc["time_s"] / vc.groupby("target_quant")["time_s"].transform("first")
    return pd.DataFrame(rows), vc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = new_run_dir(args.config, cfg, args.results)
    run(cfg, out_dir)
    summary, verify = summarize_run(out_dir)
    summary.to_csv(out_dir / "summary.csv", index=False)
    verify.to_csv(out_dir / "verify_cost.csv", index=False)
    print(f"\n-> results in {out_dir}")


if __name__ == "__main__":
    main()
