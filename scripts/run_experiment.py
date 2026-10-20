"""Run an experiment described by a YAML config.

Usage: python scripts/run_experiment.py configs/01_baseline.yaml
"""

import argparse
from pathlib import Path

from llmbench.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_experiment(args.config, args.results)


if __name__ == "__main__":
    main()
