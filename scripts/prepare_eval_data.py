"""Freeze evaluation data into data/eval/ (run once; outputs are versioned).

Usage: python scripts/prepare_eval_data.py

- wikitext2_test_extract.txt: the start of the WikiText-2 (raw) test split, the standard
  text for perplexity. Rows are joined as-is, which reproduces the original raw file.
  WikiText is licensed CC BY-SA 3.0 (Merity et al., 2016).
"""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path("data/eval")
# ~50k characters ~ 11-13k tokens depending on the tokenizer: enough for 16 windows of 512.
WIKITEXT_CHARS = 50_000


def prepare_wikitext() -> None:
    src = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                          repo_type="dataset")
    text = "".join(pd.read_parquet(src)["text"])
    # Cut at a line boundary so the extract does not end mid-sentence.
    extract = text[: text.rfind("\n", 0, WIKITEXT_CHARS) + 1]
    out = OUT / "wikitext2_test_extract.txt"
    out.write_text(extract)
    print(f"-> {out} ({len(extract):,} chars)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_wikitext()


if __name__ == "__main__":
    main()
