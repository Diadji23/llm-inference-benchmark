"""Affiche et sauvegarde le snapshot matériel/logiciel (étape 0).

Usage : python scripts/check_env.py [--out results/env.json]
"""

import argparse
import json
from pathlib import Path

from llmbench.env import collect_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/env.json"))
    args = parser.parse_args()

    env = collect_env()
    print(json.dumps(env, indent=2, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(env, indent=2, ensure_ascii=False))
    print(f"\n→ sauvegardé dans {args.out}")


if __name__ == "__main__":
    main()
