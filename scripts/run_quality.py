"""Quality of each model variant: perplexity, agreement with the reference, MMLU accuracy.

Usage: python scripts/run_quality.py configs/02_quantization.yaml

Writes results/<experiment>/quality-<timestamp>/ with quality.csv (one row per variant)
and mcq_preds.jsonl (each variant's answer to every question, for paired comparisons).
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from llmbench.backends import make_backend
from llmbench.quality import evaluate_mcq, evaluate_text
from llmbench.runner import load_config, model_label, new_run_dir


def families(models: list[dict], reference: str) -> dict[str, list[dict]]:
    """Group variants by model name, reference quant first, config order otherwise."""
    groups: dict[str, list[dict]] = {}
    for m in models:
        groups.setdefault(m["name"], []).append(m)
    for name, variants in groups.items():
        if not any(v["quant"] == reference for v in variants):
            raise ValueError(f"{name} has no {reference} variant to use as reference")
        variants.sort(key=lambda v: v["quant"] != reference)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    q = cfg["quality"]
    out_dir = new_run_dir(args.config, cfg, args.results, prefix="quality-")
    text = Path(q["perplexity"]["source"]).read_text()
    items = [json.loads(line) for line in Path(q["mcq"]["source"]).read_text().splitlines()]
    backend_args = {**cfg.get("backend_args", {}), "n_ctx": q["n_ctx"], "logits_all": True}

    rows = []
    with open(out_dir / "mcq_preds.jsonl", "w") as preds_file:
        for variants in families(cfg["models"], q["reference"]).values():
            ref_scored = ref_preds = None
            for model in variants:
                t0 = time.perf_counter()
                backend = make_backend(cfg["backend"], model_path=model["path"], **backend_args)
                backend.load()
                try:
                    ppl = evaluate_text(backend, backend.tokenize(text),
                                        ctx=q["perplexity"]["window"],
                                        n_windows=q["perplexity"]["n_windows"],
                                        k=q["perplexity"]["top_k"], reference=ref_scored)
                    mcq = evaluate_mcq(backend, items, reference_preds=ref_preds)
                finally:
                    backend.close()
                if model["quant"] == q["reference"]:
                    ref_scored, ref_preds = ppl["scored"], mcq["preds"]

                row = {"model": model["name"], "quant": model["quant"],
                       **{k: v for k, v in ppl.items() if k != "scored"},
                       **{k: v for k, v in mcq.items() if k != "preds"},
                       "elapsed_s": time.perf_counter() - t0}
                rows.append(row)
                preds_file.write(json.dumps({"model": model["name"], "quant": model["quant"],
                                             "preds": mcq["preds"].tolist()}) + "\n")
                preds_file.flush()
                print(f"[{model_label(model)}] ppl {row['ppl']:.3f} | "
                      f"top1 {row.get('top1_agreement', 1.0):.3f} | "
                      f"MMLU {row['accuracy']:.3f} | {row['elapsed_s']:.0f} s", flush=True)

    pd.DataFrame(rows).to_csv(out_dir / "quality.csv", index=False)
    print(f"\n-> results in {out_dir}")


if __name__ == "__main__":
    main()
