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
    """Median and its interquartile range, as asymmetric error bars."""
    med = df[f"{metric}_median"]
    return med, [med - df[f"{metric}_q25"], df[f"{metric}_q75"] - med]


def plot_baseline(summary: pd.DataFrame, models: list[str], out: Path) -> None:
    colors = {m: SERIES[i] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")

    # 1. TTFT vs prompt length: cost of the prefill.
    ax = axes[0]
    for m in models:
        g = summary[summary["model"] == m].sort_values("prompt_len")
        med, err = _err(g, "ttft_s")
        ax.errorbar(g["prompt_len"], med, yerr=err, color=colors[m], lw=2, marker="o", ms=6,
                    capsize=3, label=m)
        ax.annotate(m, (g["prompt_len"].iloc[-1], med.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(summary["prompt_len"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("prompt (tokens)")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Time to first token (median, IQR)")
    ax.margins(x=0.25)

    # 2. Decode throughput at the median prompt length: memory-bound.
    # 3. Net peak memory (max over lengths: the longest prompt sets the peak).
    mid_len = sorted(summary["prompt_len"].unique())[len(summary["prompt_len"].unique()) // 2]
    mid = summary[summary["prompt_len"] == mid_len].set_index("model").reindex(models)
    per_model = pd.DataFrame({
        "decode": mid["decode_tps_median"], "decode_lo": mid["decode_tps_q25"],
        "decode_hi": mid["decode_tps_q75"],
        "rss": summary.groupby("model")["net_peak_rss_mib_median"].max().reindex(models),
    })
    panels = ((axes[1], "decode", f"Decode throughput (prompt {mid_len}, median)", "tokens/s"),
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


QUANT_ORDER = ["F16", "Q8_0", "Q4_K_M"]


def _by_format(ax, df: pd.DataFrame, models: list[str], colors: dict, y: str,
               lo: str | None = None, hi: str | None = None, fmt: str = "%.1f") -> None:
    """One line per model across quantization formats, labeled at its right end."""
    x = range(len(QUANT_ORDER))
    for m in models:
        g = df[df["model"] == m].set_index("quant").reindex(QUANT_ORDER)
        yerr = [g[y] - g[lo], g[hi] - g[y]] if lo else None
        ax.errorbar(x, g[y], yerr=yerr, color=colors[m], lw=2, marker="o", ms=6, capsize=3,
                    label=m)
        ax.annotate(fmt % g[y].iloc[-1], (len(QUANT_ORDER) - 1, g[y].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=8)
    ax.set_xticks(list(x), QUANT_ORDER)
    ax.margins(x=0.15)


def plot_quantization(summary: pd.DataFrame, quality: pd.DataFrame, models: list[str],
                      out: Path) -> None:
    colors = {m: SERIES[i] for i, m in enumerate(models)}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    long_len = summary["prompt_len"].max()
    short_len = summary["prompt_len"].min()

    ax = axes[0, 0]
    d = summary[summary["prompt_len"] == short_len]
    _by_format(ax, d, models, colors, "decode_tps_median", "decode_tps_q25", "decode_tps_q75")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("tokens/s")
    ax.set_title(f"Decode throughput (prompt {short_len}, median and IQR)")

    ax = axes[0, 1]
    d = summary[summary["prompt_len"] == long_len]
    _by_format(ax, d, models, colors, "prefill_tps_median", "prefill_tps_q25", "prefill_tps_q75",
               fmt="%.0f")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("tokens/s")
    ax.set_title(f"Prefill throughput (prompt {long_len}, median and IQR)")

    # Perplexity is only comparable within one tokenizer: show the change vs each F16.
    ax = axes[1, 0]
    q = quality.copy()
    ref = q[q["quant"] == "F16"].set_index("model")["ppl"]
    q["ppl_delta_pct"] = 100 * (q["ppl"] / q["model"].map(ref) - 1)
    _by_format(ax, q, models, colors, "ppl_delta_pct", fmt="+%.1f%%")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("% vs F16")
    ax.set_title("Perplexity increase (WikiText-2)")

    ax = axes[1, 1]
    q["acc_lo"] = 100 * (q["accuracy"] - q["accuracy_ci95"])
    q["acc_hi"] = 100 * (q["accuracy"] + q["accuracy_ci95"])
    q["acc_pct"] = 100 * q["accuracy"]
    _by_format(ax, q, models, colors, "acc_pct", "acc_lo", "acc_hi", fmt="%.1f%%")
    ax.axhline(30, color=INK_2, lw=0.8, ls="--")
    ax.annotate("always 'D' (30%)", (0, 30), xytext=(0, 4), textcoords="offset points",
                color=INK_2, fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"MMLU accuracy ({int(q['n_questions'].iloc[0])} questions, 95% CI)")

    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center",
               ncol=len(models), frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--quality", type=Path, help="quality-<timestamp> run directory")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    summary = pd.read_csv(args.run_dir / "summary.csv")
    # Config order (by size), not the alphabetical order of the summary's groupby.
    names = [m["name"] for m in load_config(args.run_dir / "config.yaml")["models"]]
    models = list(dict.fromkeys(names))
    experiment = args.run_dir.parent.name
    out = args.out / f"{experiment}.png"
    if args.quality:
        plot_quantization(summary, pd.read_csv(args.quality / "quality.csv"), models, out)
    else:
        plot_baseline(summary, models, out)


if __name__ == "__main__":
    main()
