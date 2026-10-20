"""Télécharge les fichiers GGUF référencés par une config dans models/.

Usage : python scripts/download_models.py configs/01_baseline.yaml
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from llmbench.runner import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    for model in load_config(args.config)["models"]:
        path = Path(model["path"])
        if path.exists():
            print(f"✓ {path}")
            continue
        print(f"↓ {model['repo']}/{path.name}")
        hf_hub_download(repo_id=model["repo"], filename=path.name, local_dir=path.parent)


if __name__ == "__main__":
    main()
