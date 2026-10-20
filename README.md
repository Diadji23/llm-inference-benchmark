# llmbench — benchmark d'inférence LLM

Benchmark reproductible de l'inférence de petits LLM open source : TTFT, latence, débit,
mémoire, puis effet de la quantification, du cache KV et du décodage spéculatif.
Chaque expérience suit la démarche **hypothèse → protocole → mesures → conclusion**,
et les résultats négatifs sont conservés.

## Matériel de référence

| Composant | Valeur | Conséquence |
|---|---|---|
| CPU | AMD Ryzen 5 7520U (Zen 2, 4 cœurs / 8 threads, AVX2 + FMA + F16C, pas d'AVX-512) | Inférence sur CPU : la mémoire est le goulot, pas le calcul |
| RAM (WSL2) | 7,4 Gio visibles (hôte : 15,3 Go) | Modèles ≤ 1,5B ; un modèle 1,5B en FP32 (~6 Go) ne tient pas |
| GPU | iGPU Radeon 610M, pas de CUDA | Les parties GPU (étape 4 à plus grande échelle) passent par un notebook Kaggle |
| OS | Ubuntu 24.04 sous WSL2 | Le snapshot `results/env.json` est enregistré avec chaque run |

## Structure du dépôt

```
configs/            une config YAML par expérience (modèles, prompts, répétitions)
src/llmbench/
  env.py            snapshot matériel/logiciel joint à chaque résultat
  backends/         interface commune : llama.cpp (GGUF) et Hugging Face
  metrics.py        TTFT, latence, débit, agrégation (moyenne, écart-type, IC)
  memory.py         mémoire de pointe (RSS du processus)
  quality.py        perplexité, exact match, LLM-as-judge
  runner.py         boucle d'expérience : warmup, répétitions, écriture JSONL
scripts/            points d'entrée CLI (run_experiment.py, make_plots.py, check_env.py)
data/prompts/       prompts et textes figés (versionnés pour la reproductibilité)
results/<exp>/<run>/ raw.jsonl (mesures brutes) + env.json + summary.csv
figures/            graphiques générés à partir de results/
notebooks/          notebooks Kaggle pour les mesures sur GPU
tests/              tests unitaires (métriques, agrégation) + smoke tests des backends
```

## Installation

```bash
uv venv && uv pip install -e ".[dev,llamacpp,hf]"
python scripts/check_env.py
pytest
```

## Expériences

1. **Socle** : TTFT, latence, débit et mémoire sur 2 à 3 petits modèles — *à venir*
2. **Quantification des poids** : F16, Q8_0, Q4_K_M (vitesse, mémoire, perplexité, exact match) — *à venir*
3. **Cache KV** : effet de la longueur de contexte, cache KV quantifié — *à venir*
4. **Décodage spéculatif** : gain de vitesse et taux d'acceptation — *à venir*

## Enseignements principaux

*À compléter à la fin.*
