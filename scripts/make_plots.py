"""Generate an experiment's figures from its summary.csv.

Usage: python scripts/make_plots.py results/01_baseline/<run_id> [--out figures]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from llmbench.runner import load_config

# Reference categorical palette in fixed order: color follows the model, not its rank.
# The first 3 slots stay distinguishable for every pair (color vision deficiency included).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 9,
})


def _err(df: pd.DataFrame, metric: str):
    mean = df[f"{metric}_mean"]
    return mean, [mean - df[f"{metric}_ci95_low"], df[f"{metric}_ci95_high"] - mean]


def plot_baseline(summary: pd.DataFrame, models: list[str], out: Path) -> None:
    colors = {m: SERIES[i] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")

    # 1. TTFT vs prompt length: cost of the prefill.
    ax = axes[0]
    for m in models:
        g = summary[summary["model"] == m].sort_values("prompt_len")
        mean, err = _err(g, "ttft_s")
        ax.errorbar(g["prompt_len"], mean, yerr=err, color=colors[m], lw=2, marker="o", ms=6,
                    capsize=3, label=m)
        ax.annotate(m, (g["prompt_len"].iloc[-1], mean.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(summary["prompt_len"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("prompt (tokens)")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Time to first token")
    ax.margins(x=0.25)

    # 2. Decode throughput at the median prompt length: memory-bound.
    # 3. Net peak memory (max over lengths: the longest prompt sets the peak).
    mid_len = sorted(summary["prompt_len"].unique())[len(summary["prompt_len"].unique()) // 2]
    mid = summary[summary["prompt_len"] == mid_len].set_index("model").reindex(models)
    per_model = pd.DataFrame({
        "decode": mid["decode_tps_mean"], "decode_lo": mid["decode_tps_ci95_low"],
        "decode_hi": mid["decode_tps_ci95_high"],
        "rss": summary.groupby("model")["net_peak_rss_mib_mean"].max().reindex(models),
    })
    panels = ((axes[1], "decode", f"Decode throughput (prompt {mid_len})", "tokens/s"),
              (axes[2], "rss", "Peak memory (RSS above baseline)", "MiB"))
    for ax, col, title, unit in panels:
        vals = per_model[col]
        yerr = ([vals - per_model["decode_lo"], per_model["decode_hi"] - vals]
                if col == "decode" else None)
        bars = ax.bar(models, vals, width=0.6, color=[colors[m] for m in models],
                      edgecolor=SURFACE, linewidth=2, yerr=yerr, capsize=3,
                      error_kw={"ecolor": INK_2, "lw": 1})
        ax.bar_label(bars, fmt="%.1f" if col == "decode" else "%.0f", padding=3,
                     color=INK, fontsize=8)
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)

    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(models),
               frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    summary = pd.read_csv(args.run_dir / "summary.csv")
    # Config order (by size), not the alphabetical order of the summary's groupby.
    models = [m["name"] for m in load_config(args.run_dir / "config.yaml")["models"]]
    experiment = args.run_dir.parent.name
    plot_baseline(summary, models, args.out / f"{experiment}.png")


if __name__ == "__main__":
    main()
