"""Boucle d'expérience : chargement, warmup, répétitions, écriture des résultats.

Sortie dans results/<experiment>/<run_id>/ :
  config.yaml  copie de la config utilisée
  env.json     snapshot matériel/logiciel
  raw.jsonl    une ligne par génération mesurée
  summary.csv  agrégats par (modèle, longueur de prompt)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from llmbench.backends import make_backend
from llmbench.env import collect_env
from llmbench.memory import PeakRSSSampler, current_rss_mib
from llmbench.metrics import summarize

SUMMARY_METRICS = ("ttft_s", "latency_s", "prefill_tps", "decode_tps", "peak_rss_mib")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_experiment(config_path: Path, results_root: Path = Path("results")) -> Path:
    cfg = load_config(config_path)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = results_root / cfg["experiment"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(Path(config_path).read_text())
    (out_dir / "env.json").write_text(json.dumps(collect_env(), indent=2, ensure_ascii=False))

    source_text = Path(cfg["prompts"]["source"]).read_text()
    gen = cfg["generation"]
    reps, warmup = cfg["repetitions"], cfg["warmup"]

    with open(out_dir / "raw.jsonl", "w") as raw:
        for model in cfg["models"]:
            _run_model(model, cfg, source_text, gen, reps, warmup, raw)

    summary = summarize_run(out_dir / "raw.jsonl")
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n→ résultats dans {out_dir}")
    return out_dir


def _run_model(model: dict, cfg: dict, source_text: str, gen: dict, reps: int, warmup: int,
               raw) -> None:
    backend = make_backend(cfg["backend"], model_path=model["path"], **cfg.get("backend_args", {}))
    rss_before = current_rss_mib()
    t0 = time.perf_counter()
    backend.load()
    load_s = time.perf_counter() - t0
    rss_loaded = current_rss_mib()
    print(f"\n[{model['name']}] chargé en {load_s:.1f} s, RSS {rss_loaded:.0f} Mio")

    try:
        prompts = {n: backend.truncate_to_tokens(source_text, n) for n in cfg["prompts"]["lengths"]}
        for _ in range(warmup):
            backend.generate(prompts[min(prompts)], gen["max_new_tokens"], gen["ignore_eos"])

        # Répétitions en boucle externe : une dérive lente (chauffe CPU, throttling) se
        # répartit sur toutes les longueurs au lieu de pénaliser la dernière.
        for rep in range(reps):
            for n_prompt, tokens in prompts.items():
                with PeakRSSSampler() as mem:
                    res = backend.generate(tokens, gen["max_new_tokens"], gen["ignore_eos"])
                record = {
                    "experiment": cfg["experiment"],
                    "model": model["name"],
                    "quant": model.get("quant"),
                    "backend": backend.name,
                    "prompt_len": n_prompt,
                    "rep": rep,
                    **res.timing.as_dict(),
                    "peak_rss_mib": mem.peak_mib,
                    "rss_before_load_mib": rss_before,
                    "rss_loaded_mib": rss_loaded,
                    "load_s": load_s,
                    "text_head": res.text[:80],
                }
                raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw.flush()
                t = res.timing
                print(f"  rep {rep} | prompt {n_prompt:>4} | TTFT {t.ttft_s * 1e3:7.1f} ms | "
                      f"decode {t.decode_tps:6.1f} tok/s | RSS {mem.peak_mib:6.0f} Mio")
    finally:
        backend.close()


def summarize_run(raw_path: Path) -> pd.DataFrame:
    df = pd.read_json(raw_path, lines=True)
    rows = []
    for (model, quant, n_prompt), g in df.groupby(["model", "quant", "prompt_len"], dropna=False):
        row = {"model": model, "quant": quant, "prompt_len": n_prompt,
               "n_generated": int(g["n_generated_tokens"].median()),
               "load_s": g["load_s"].iloc[0]}
        for metric in SUMMARY_METRICS:
            s = summarize(g[metric])
            row["n"] = s["n"]
            for stat in ("mean", "std", "median", "ci95_low", "ci95_high"):
                row[f"{metric}_{stat}"] = s[stat]
        rows.append(row)
    return pd.DataFrame(rows)
