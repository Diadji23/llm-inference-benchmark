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


def plot_context(summary: pd.DataFrame, models: list[str], out: Path) -> None:
    """Step 3a: TTFT, decode time per token and memory against the context length."""
    colors = {m: SERIES[i] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    lengths = sorted(summary["prompt_len"].unique())
    for m in models:
        g = summary[summary["model"] == m].sort_values("prompt_len")
        med, err = _err(g, "ttft_s")
        axes[0].errorbar(g["prompt_len"], med, yerr=err, color=colors[m], lw=2, marker="o",
                         ms=6, capsize=3, label=m)
        # Time per token adds up linearly, unlike tokens/s: it shows the per-context cost.
        ms_per_token = 1e3 / g["decode_tps_median"]
        axes[1].plot(g["prompt_len"], ms_per_token, color=colors[m], lw=2, marker="o", ms=6)
        axes[1].annotate(f"{ms_per_token.iloc[-1]:.0f}", (g["prompt_len"].iloc[-1], ms_per_token.iloc[-1]),
                         xytext=(6, 0), textcoords="offset points", va="center", color=INK_2,
                         fontsize=8)
        axes[2].plot(g["prompt_len"], g["net_peak_rss_mib_median"], color=colors[m], lw=2,
                     marker="o", ms=6)
        axes[2].plot(g["prompt_len"], g["kv_cache_mib"], color=colors[m], lw=1.5, ls="--",
                     marker="o", ms=4)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("TTFT (s, log)")
    axes[0].set_title("Time to first token (median, IQR)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("ms per generated token")
    axes[1].set_title("Decode time per token (median)")
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("MiB")
    axes[2].set_title("Net peak RSS (solid), KV cache (dashed)", fontsize=10)
    for ax in axes:
        # Log axis: one tick per measured length. Linear axes: evenly spaced ticks, since
        # 512 and 2,048 would overlap.
        ax.set_xticks(lengths if ax is axes[0] else range(0, max(lengths) + 1, 4096))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("context (prompt tokens)")
    axes[1].margins(x=0.1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
               ncol=len(models), frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def plot_flash_attn(summary: pd.DataFrame, models: list[str], out: Path) -> None:
    """Step 3b: TTFT, decode time per token and memory, flash attention on vs off."""
    colors = {m: SERIES[i] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    lengths = sorted(summary["prompt_len"].unique())
    for m in models:
        for fa, ls in ((True, "-"), (False, "--")):
            g = summary[(summary["model"] == m) & (summary["flash_attn"] == fa)]
            g = g.sort_values("prompt_len")
            label = m if fa else None
            med, err = _err(g, "ttft_s")
            axes[0].errorbar(g["prompt_len"], med, yerr=err, color=colors[m], lw=2, ls=ls,
                             marker="o", ms=6, capsize=3, label=label)
            ms_per_token = 1e3 / g["decode_tps_median"]
            axes[1].plot(g["prompt_len"], ms_per_token, color=colors[m], lw=2, ls=ls,
                         marker="o", ms=6)
            if fa:  # the "off" lines end too close together to label
                axes[1].annotate(f"{ms_per_token.iloc[-1]:.0f}",
                                 (g["prompt_len"].iloc[-1], ms_per_token.iloc[-1]),
                                 xytext=(6, 0), textcoords="offset points", va="center",
                                 color=INK_2, fontsize=8)
            axes[2].plot(g["prompt_len"], g["net_peak_rss_mib_median"], color=colors[m], lw=2,
                         ls=ls, marker="o", ms=6)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("TTFT (s)")
    axes[0].set_title("Time to first token (median, IQR)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("ms per generated token")
    axes[1].set_title("Decode time per token (median)")
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("MiB")
    axes[2].set_title("Net peak RSS (median)")
    for ax in axes:
        ax.set_xticks(lengths)
        ax.set_xlabel("context (prompt tokens)")
        ax.margins(x=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK_2, lw=2, ls=ls) for ls in ("-", "--")]
    labels += ["flash attention on", "flash attention off"]
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


KV_ORDER = ["f16", "q8_0", "q4_0"]


def plot_kv_quant(summary: pd.DataFrame, models: list[str], out: Path) -> None:
    """Step 3c: TTFT, decode time per token and cache size against the KV cache type."""
    colors = {m: SERIES[i] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    lengths = sorted(summary["prompt_len"].unique())
    x = range(len(KV_ORDER))
    styles = {lengths[-1]: "-", lengths[0]: "--"}
    for m in models:
        for length, ls in styles.items():
            g = summary[(summary["model"] == m) & (summary["prompt_len"] == length)]
            g = g.set_index("kv_type").reindex(KV_ORDER)
            label = m if ls == "-" else None
            med, err = _err(g, "ttft_s")
            axes[0].errorbar(x, med, yerr=err, color=colors[m], lw=2, ls=ls, marker="o", ms=6,
                             capsize=3, label=label)
            ms_per_token = 1e3 / g["decode_tps_median"]
            axes[1].plot(x, ms_per_token, color=colors[m], lw=2, ls=ls, marker="o", ms=6)
            if ls == "-":
                axes[2].plot(x, g["kv_cache_mib"], color=colors[m], lw=2, marker="o", ms=6)
                axes[2].annotate(f"{g['kv_cache_mib'].iloc[-1]:.0f}",
                                 (len(KV_ORDER) - 1, g["kv_cache_mib"].iloc[-1]), xytext=(8, 0),
                                 textcoords="offset points", va="center", color=INK_2,
                                 fontsize=8)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("TTFT (s)")
    axes[0].set_title("Time to first token (median, IQR)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("ms per generated token")
    axes[1].set_title("Decode time per token (median)")
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("MiB")
    axes[2].set_title(f"KV cache at {lengths[-1]:,} tokens")
    for ax in axes:
        ax.set_xticks(list(x), [k.upper() for k in KV_ORDER])
        ax.set_xlabel("KV cache type")
        ax.margins(x=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK_2, lw=2, ls=ls) for ls in styles.values()]
    labels += [f"{length:,} tokens" for length in styles]
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def plot_kv_hf(summary: pd.DataFrame, out: Path) -> None:
    """Step 3d: time per generated token and total latency, with vs without the KV cache."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    lengths = sorted(summary["prompt_len"].unique())
    n_gen = int(summary["n_generated"].iloc[0])
    for uc, color, label in ((True, SERIES[0], "with KV cache"),
                             (False, SERIES[1], "without KV cache")):
        g = summary[summary["use_cache"] == uc].sort_values("prompt_len")
        ms_per_token = 1e3 / g["decode_tps_median"]
        axes[0].plot(g["prompt_len"], ms_per_token, color=color, lw=2, marker="o", ms=6,
                     label=label)
        axes[0].annotate(f"{ms_per_token.iloc[-1]:,.0f}",
                         (g["prompt_len"].iloc[-1], ms_per_token.iloc[-1]), xytext=(6, 0),
                         textcoords="offset points", va="center", color=INK_2, fontsize=8)
        med, err = _err(g, "latency_s")
        axes[1].errorbar(g["prompt_len"], med, yerr=err, color=color, lw=2, marker="o", ms=6,
                         capsize=3)
        axes[1].annotate(f"{med.iloc[-1]:.0f} s", (g["prompt_len"].iloc[-1], med.iloc[-1]),
                         xytext=(6, 0), textcoords="offset points", va="center", color=INK_2,
                         fontsize=8)
    # Reference: one full prefill, what each step without the cache should cost.
    g = summary[summary["use_cache"].astype(bool)].sort_values("prompt_len")
    axes[0].plot(g["prompt_len"], 1e3 * g["ttft_s_median"], color=INK_2, lw=1.5, ls="--",
                 label="TTFT (one full prefill)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("ms per generated token (log)")
    axes[0].set_title("Decode time per token (median)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("s")
    axes[1].set_title(f"Total time for {n_gen} tokens (median, IQR)")
    # Log-log on the left: a slope of 1 means time per token proportional to the prompt.
    # Linear on the right, where that proportionality should show as a straight line.
    axes[0].set_xscale("log", base=2)
    axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    for ax in axes:
        ax.set_xticks(lengths)
        ax.set_xlabel("prompt tokens")
        ax.margins(x=0.12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3,
               frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


TARGET_COLORS = {"F16": SERIES[0], "Q4_K_M": SERIES[1]}
PROMPT_STYLES = {"code": "-", "prose": "--"}


def plot_speculative(summary: pd.DataFrame, verify: pd.DataFrame, out: Path) -> None:
    """Step 4: speedup and acceptance rate against k, and the cost of one target pass."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for quant, color in TARGET_COLORS.items():
        for kind, ls in PROMPT_STYLES.items():
            g = summary[(summary["target_quant"] == quant) & (summary["prompt_kind"] == kind)]
            g = g.sort_values("k")
            med, err = _err(g, "speedup")
            axes[0].errorbar(g["k"], med, yerr=err, color=color, ls=ls, lw=2, marker="o", ms=6,
                             capsize=3)
            spec = g[g["k"] > 0]
            axes[1].plot(spec["k"], 100 * spec["acceptance_rate_median"], color=color, ls=ls,
                         lw=2, marker="o", ms=6)
        v = verify[verify["target_quant"] == quant].sort_values("batch")
        axes[2].plot(v["batch"], 1e3 * v["time_s"], color=color, lw=2, marker="o", ms=6)
        # The two lines end a few ms apart: labels above and below to keep them apart.
        axes[2].annotate(f"×{v['vs_one_token'].iloc[-1]:.1f} one token",
                         (v["batch"].iloc[-1], 1e3 * v["time_s"].iloc[-1]),
                         xytext=(6, 8 if quant == "Q4_K_M" else -8), textcoords="offset points",
                         va="center", color=INK_2, fontsize=8)
    axes[0].axhline(1, color=INK_2, lw=1)
    axes[0].set_ylabel("decode speedup vs target alone")
    axes[0].set_title("Speedup (median, IQR)")
    axes[0].set_xlabel("draft tokens per round (k; 0 = target alone)")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("% of draft tokens accepted")
    axes[1].set_title("Acceptance rate (median)")
    axes[1].set_xlabel("draft tokens per round (k)")
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("ms")
    axes[2].set_title("One target pass (median)")
    axes[2].set_xlabel("tokens in the pass (k + 1)")
    axes[2].margins(x=0.25)
    for ax in axes[:2]:
        ax.set_xticks(sorted(summary["k"].unique()))
    handles = [plt.Line2D([], [], color=c, lw=2) for c in TARGET_COLORS.values()]
    handles += [plt.Line2D([], [], color=INK_2, lw=2, ls=ls) for ls in PROMPT_STYLES.values()]
    labels = [f"target Qwen2.5-1.5B {q}" for q in TARGET_COLORS] + [
        f"{kind} prompt" for kind in PROMPT_STYLES]
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
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
    experiment = args.run_dir.parent.name
    out = args.out / f"{experiment}.png"
    if experiment == "04_speculative":
        plot_speculative(summary, pd.read_csv(args.run_dir / "verify_cost.csv"), out)
        return
    # Config order (by size), not the alphabetical order of the summary's groupby.
    names = [m["name"] for m in load_config(args.run_dir / "config.yaml")["models"]]
    models = list(dict.fromkeys(names))
    if args.quality:
        plot_quantization(summary, pd.read_csv(args.quality / "quality.csv"), models, out)
    elif experiment == "03_kv_context":
        plot_context(summary, models, out)
    elif experiment == "03_flash_attn":
        plot_flash_attn(summary, models, out)
    elif experiment == "03_kv_quant":
        plot_kv_quant(summary, models, out)
    elif experiment == "03_kv_hf":
        plot_kv_hf(summary, out)
    else:
        plot_baseline(summary, models, out)


if __name__ == "__main__":
    main()
