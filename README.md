# llmbench — LLM inference benchmark

A reproducible benchmark of small open-source LLM inference: TTFT, latency, throughput,
memory, then the effect of quantization, the KV cache and speculative decoding.
Every experiment follows **hypothesis → protocol → measurements → conclusion**, and
negative results are kept.

## Reference hardware

| Component | Value | Consequence |
|---|---|---|
| CPU | AMD Ryzen 5 7520U (Zen 2, 4 cores / 8 threads, AVX2 + FMA + F16C, no AVX-512) | CPU inference: memory is the bottleneck, not compute |
| RAM (WSL2) | 7.4 GiB visible (host: 15.3 GB) | Models ≤ 1.5B; a 1.5B model in FP32 (~6 GB) does not fit |
| GPU | Radeon 610M iGPU, no CUDA | GPU parts (step 4 at larger scale) run in a Kaggle notebook |
| OS | Ubuntu 24.04 on WSL2 | The `env.json` snapshot is saved with every run |

## Repository layout

```
configs/            one YAML config per experiment (models, prompts, repetitions)
src/llmbench/
  env.py            hardware/software snapshot attached to every result
  backends/         common interface: llama.cpp (GGUF), Hugging Face to come
  metrics.py        TTFT, latency, throughput, aggregation (mean, std, CI)
  memory.py         peak memory (process RSS)
  runner.py         experiment loop: warmup, repetitions, JSONL output
scripts/            CLI entry points (run_experiment.py, make_plots.py, download_models.py, check_env.py)
data/prompts/       frozen prompts and texts (versioned for reproducibility)
results/<exp>/<run>/ raw.jsonl (raw measurements, not versioned) + env.json + config.yaml + summary.csv
figures/            plots generated from results/
notebooks/          Kaggle notebooks for GPU measurements
tests/              unit tests (metrics, aggregation, memory sampling)
```

## Setup

```bash
uv venv && uv pip install -e ".[dev,llamacpp,hf]"
python scripts/check_env.py
pytest
```

Run benchmarks **on AC power**: on battery the throughput drops by ~1.8× and variance
explodes (see step 1).

## Experiments

1. **Baseline**: TTFT, latency, throughput and memory of 3 small models — [results](#1-baseline)
2. **Weight quantization**: F16, Q8_0, Q4_K_M (speed, memory, perplexity, exact match) — *upcoming*
3. **KV cache**: effect of context length, quantized KV cache — *upcoming*
4. **Speculative decoding**: speedup and acceptance rate — *upcoming*

### 1. Baseline

**Hypothesis.** On CPU, decoding is bound by memory bandwidth: every token re-reads all the
weights, so decode throughput should be roughly inversely proportional to their size.
Prefill processes the prompt in batches, so TTFT should grow more slowly than prompt length.

**Protocol.** [`configs/01_baseline.yaml`](configs/01_baseline.yaml) — Qwen2.5-0.5B,
Llama-3.2-1B and Qwen2.5-1.5B (Instruct), all in Q4_K_M to isolate the effect of size;
llama.cpp, 4 threads, greedy decoding; prompts of 32, 128 and 512 tokens truncated with
each model's own tokenizer; 128 generated tokens (EOS ignored, fixed length); 2 warmup
generations then 5 repetitions. Laptop on AC power.

```bash
python scripts/download_models.py configs/01_baseline.yaml
python scripts/run_experiment.py configs/01_baseline.yaml
python scripts/make_plots.py results/01_baseline/<run_id>
```

![Baseline: TTFT, decode throughput, memory](figures/01_baseline.png)

| Model (Q4_K_M) | Weights | TTFT 32 / 128 / 512 tok | Prefill @512 | Decode @128 | Net peak RSS |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 469 MiB | 0.64 / 1.28 / 3.93 s | 131 tok/s | 42.9 ± 1.5 tok/s | 631 MiB |
| Llama-3.2-1B | 770 MiB | 1.30 / 2.55 / 8.54 s | 60 tok/s | 19.0 ± 1.4 tok/s | 1,357 MiB |
| Qwen2.5-1.5B | 1,066 MiB | 1.45 / 3.33 / 11.65 s | 44 tok/s | 13.7 ± 2.6 tok/s | 1,775 MiB |

± = half-width of the 95% CI (Student, n = 5). Summary and run metadata:
[`results/01_baseline/20261020-233651/`](results/01_baseline/20261020-233651/).

**Conclusions.**
- *Decode is memory-bound: confirmed for the two larger models.* Weight size × decode
  throughput gives almost the same effective bandwidth: 15.3 GB/s for both Llama-1B and
  Qwen-1.5B. Qwen-0.5B reaches 21 GB/s, faster than size alone predicts; the cause is not
  established yet (better cache locality on small matrices?). To be checked in step 2, where
  size varies with the architecture held fixed.
- *Prefill is sub-linear: confirmed.* A prompt 16× longer (32 → 512) multiplies TTFT by only
  6 to 8, because prefill throughput rises with batch size (0.5B: 51 → 131 tok/s). Prefill
  still dominates: at 512 tokens, waiting for the first token takes longer than generating
  the next 128 for all three models.
- *Memory: weight size alone does not predict the footprint.* The overhead beyond the GGUF
  file (KV cache for n_ctx = 2048, compute buffers) ranges from 160 MiB (0.5B) to
  ~600–700 MiB (1B, 1.5B). Breaking it down belongs to step 3 (KV cache).

**Side observations (outside the protocol, worth keeping in mind).**
- *Power.* On battery (Windows "Balanced" plan), Qwen-0.5B dropped to 75 tok/s prefill and
  24 tok/s decode with ~40% TTFT variability, versus 138 and 41 tok/s (~3%) on AC power.
  All runs are therefore done on AC power.
- *Threads.* 8 threads (SMT) gave ~12 tok/s decode versus ~24 with 4 threads (measured on
  battery): the extra threads compete for the same memory bandwidth.
- *Memory measurement.* Baseline RSS grows from one model to the next (78 → 187 MiB, memory
  not returned to the OS); peak memory is therefore reported net of the pre-load RSS.

## Key takeaways

*To be written at the end.*
