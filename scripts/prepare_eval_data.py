"""Freeze evaluation data and long prompts (run once; outputs are versioned).

Usage: python scripts/prepare_eval_data.py

- wikitext2_test_extract.txt: the start of the WikiText-2 (raw) test split, the standard
  text for perplexity. Rows are joined as-is, which reproduces the original raw file.
  WikiText is licensed CC BY-SA 3.0 (Merity et al., 2016).
- mmlu_200.jsonl: 200 questions drawn with a fixed seed from the whole MMLU test split
  (53 of the 57 subjects end up in the sample) (Hendrycks et al., 2021; MIT license).
- data/prompts/wikitext2_valid_extract.txt: the start of the WikiText-2 validation split,
  the prompt source for step 3 (long contexts). The validation split is used so that speed
  prompts never overlap the perplexity text taken from the test split.
"""

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path("data/eval")
# ~50k characters ~ 11-13k tokens depending on the tokenizer: enough for 16 windows of 512.
WIKITEXT_CHARS = 50_000
MMLU_N, MMLU_SEED = 200, 0
# ~100k characters ~ 23-24k tokens: enough for a 16k-token prompt with every tokenizer.
LONG_PROMPT_CHARS = 100_000


def prepare_wikitext() -> None:
    src = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                          repo_type="dataset")
    text = "".join(pd.read_parquet(src)["text"])
    # Cut at a line boundary so the extract does not end mid-sentence.
    extract = text[: text.rfind("\n", 0, WIKITEXT_CHARS) + 1]
    out = OUT / "wikitext2_test_extract.txt"
    out.write_text(extract)
    print(f"-> {out} ({len(extract):,} chars)")


def prepare_long_prompt() -> None:
    src = hf_hub_download("Salesforce/wikitext",
                          "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
                          repo_type="dataset")
    text = "".join(pd.read_parquet(src)["text"])
    extract = text[: text.rfind("\n", 0, LONG_PROMPT_CHARS) + 1]
    out = Path("data/prompts/wikitext2_valid_extract.txt")
    out.write_text(extract)
    print(f"-> {out} ({len(extract):,} chars)")


def prepare_mmlu() -> None:
    src = hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet", repo_type="dataset")
    sample = pd.read_parquet(src).sample(n=MMLU_N, random_state=MMLU_SEED)
    out = OUT / f"mmlu_{MMLU_N}.jsonl"
    with open(out, "w") as f:
        for row in sample.itertuples():
            item = {"subject": row.subject, "question": row.question,
                    "choices": list(row.choices), "answer": int(row.answer)}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"-> {out} ({len(sample)} questions, {sample['subject'].nunique()} subjects)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_wikitext()
    prepare_mmlu()
    prepare_long_prompt()


if __name__ == "__main__":
    main()
