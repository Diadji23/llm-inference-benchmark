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

1. **Socle** : TTFT, latence, débit et mémoire sur 3 petits modèles — [résultats](#1-socle)
2. **Quantification des poids** : F16, Q8_0, Q4_K_M (vitesse, mémoire, perplexité, exact match) — *à venir*
3. **Cache KV** : effet de la longueur de contexte, cache KV quantifié — *à venir*
4. **Décodage spéculatif** : gain de vitesse et taux d'acceptation — *à venir*

### 1. Socle

**Hypothèse.** Sur CPU, le décodage est borné par la bande passante mémoire : chaque token
relit tous les poids, donc le débit decode doit être à peu près inversement proportionnel à
leur taille. Le prefill, lui, traite le prompt par lots : le TTFT doit croître moins vite
que la longueur du prompt.

**Protocole.** [`configs/01_baseline.yaml`](configs/01_baseline.yaml) — Qwen2.5-0.5B,
Llama-3.2-1B et Qwen2.5-1.5B (Instruct), tous en Q4_K_M pour isoler l'effet de la taille ;
llama.cpp, 4 threads, décodage greedy ; prompts de 32, 128 et 512 tokens tronqués avec le
tokenizer de chaque modèle ; 128 tokens générés (EOS ignoré, longueur fixe) ; 2 générations
de warmup puis 5 répétitions. Portable sur secteur.

```bash
python scripts/download_models.py configs/01_baseline.yaml
python scripts/run_experiment.py configs/01_baseline.yaml
python scripts/make_plots.py results/01_baseline/<run_id>
```

![Socle : TTFT, débit decode, mémoire](figures/01_baseline.png)

| Modèle (Q4_K_M) | Poids | TTFT 32 / 128 / 512 tok | Prefill @512 | Decode @128 | Pic RSS net |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 469 Mio | 0,64 / 1,28 / 3,93 s | 131 tok/s | 42,9 ± 1,5 tok/s | 631 Mio |
| Llama-3.2-1B | 770 Mio | 1,30 / 2,55 / 8,54 s | 60 tok/s | 19,0 ± 1,4 tok/s | 1 357 Mio |
| Qwen2.5-1.5B | 1 066 Mio | 1,45 / 3,33 / 11,65 s | 44 tok/s | 13,7 ± 2,6 tok/s | 1 775 Mio |

± = demi-largeur de l'IC à 95 % (Student, n = 5). Résultats bruts et résumé :
[`results/01_baseline/20261020-233651/`](results/01_baseline/20261020-233651/).

**Conclusions.**
- *Decode borné par la mémoire : confirmé pour les deux plus gros modèles.* Taille des poids
  × débit decode donne une bande passante effective quasi identique : 15,3 Go/s pour
  Llama-1B comme pour Qwen-1.5B. Qwen-0.5B atteint 21 Go/s, soit plus vite que ce que
  prédit la seule taille ; cause non établie à ce stade (meilleure localité cache sur de
  petites matrices ?). À vérifier à l'étape 2, où la taille varie à architecture fixe.
- *Prefill sous-linéaire : confirmé.* Un prompt 16 × plus long (32 → 512) multiplie le TTFT
  par 6 à 8 seulement, car le débit prefill monte avec la taille du lot (0,5B : 51 → 131
  tok/s). Le prefill reste pourtant le poste dominant : à 512 tokens, attendre le premier
  token prend plus de temps que générer les 128 suivants pour les trois modèles.
- *Mémoire : les poids ne suffisent pas à prédire l'empreinte.* Le surcoût au-delà du
  fichier GGUF (cache KV pour n_ctx = 2048, buffers de calcul) va de 160 Mio (0,5B) à
  ~600–700 Mio (1B, 1,5B). Sa décomposition relève de l'étape 3 (cache KV).

**Observations annexes (hors protocole, à garder en tête).**
- *Alimentation.* Sur batterie (mode « Utilisation normale »), Qwen-0.5B tombait à 75 tok/s
  en prefill et 24 tok/s en decode, avec ~40 % de variabilité du TTFT, contre 138 et 41
  tok/s (~3 %) sur secteur. Tous les runs sont donc faits sur secteur.
- *Threads.* 8 threads (SMT) donnent ~12 tok/s en decode contre ~24 avec 4 threads sur
  batterie : les threads supplémentaires se disputent la même bande passante.
- *Mesure mémoire.* Le RSS de départ grossit d'un modèle à l'autre (78 → 187 Mio, mémoire
  non rendue à l'OS) ; le pic est donc rapporté net du RSS avant chargement.

## Enseignements principaux

*À compléter à la fin.*
