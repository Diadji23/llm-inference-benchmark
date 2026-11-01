"""Hugging Face backend (safetensors models) via transformers, on CPU."""

from __future__ import annotations

import gc
import time
from pathlib import Path

from llmbench.backends.base import Backend, GenerationResult
from llmbench.metrics import GenerationTiming

DTYPES = ("float32", "bfloat16")


class HFBackend(Backend):
    name = "hf"

    def __init__(
        self,
        model_path: str | Path,
        n_threads: int | None = None,
        dtype: str = "float32",
        use_cache: bool = True,
        seed: int = 0,
    ):
        if dtype not in DTYPES:
            raise ValueError(f"dtype {dtype!r} not in {DTYPES}")
        self.model_path = Path(model_path)
        self.n_threads = n_threads
        # float32 by default: without AVX-512 BF16, bfloat16 matmuls are slow on CPU.
        self.dtype = dtype
        # False recomputes the whole sequence at every step: what the KV cache saves.
        self.use_cache = use_cache
        self.seed = seed
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.model_path} not found: run scripts/download_models.py"
            )
        if self.n_threads:
            torch.set_num_threads(self.n_threads)
        torch.manual_seed(self.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=getattr(torch, self.dtype)
        ).eval()

    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_bos)

    def detokenize(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def generate(
        self, prompt_tokens: list[int], max_new_tokens: int, ignore_eos: bool = True
    ) -> GenerationResult:
        eos = None if ignore_eos else self._tokenizer.eos_token_id
        t_start = time.perf_counter()
        out, t_first = self._greedy(prompt_tokens, max_new_tokens, eos)
        t_end = time.perf_counter()
        timing = GenerationTiming(
            t_start=t_start,
            t_first_token=t_first if t_first is not None else t_end,
            t_end=t_end,
            n_prompt_tokens=len(prompt_tokens),
            n_generated_tokens=len(out),
        )
        return GenerationResult(timing=timing, text=self.detokenize(out))

    def _greedy(
        self, prompt_tokens: list[int], max_new_tokens: int, eos: int | None = None
    ) -> tuple[list[int], float | None]:
        """Greedy tokens and the time the first one was ready.

        Hand-written loop rather than model.generate(): each step is exactly one forward
        pass, either one new token against the cache, or the whole sequence without it.
        A fresh cache per call: nothing is reused between generations.
        """
        import torch

        ids = torch.tensor([prompt_tokens])
        out: list[int] = []
        t_first = None
        past = None
        with torch.inference_mode():
            while len(out) < max_new_tokens:
                if not out:
                    step_ids = ids  # prefill
                elif self.use_cache:
                    step_ids = ids[:, -1:]
                else:
                    step_ids = ids
                # logits_to_keep=1: only the last position goes through the vocabulary
                # projection, so without the cache only the transformer is recomputed.
                res = self._model(input_ids=step_ids, past_key_values=past,
                                  use_cache=self.use_cache, logits_to_keep=1)
                past = res.past_key_values if self.use_cache else None
                tok = int(res.logits[0, -1].argmax())
                if t_first is None:
                    t_first = time.perf_counter()
                if tok == eos:
                    break
                out.append(tok)
                ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
        return out, t_first

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
