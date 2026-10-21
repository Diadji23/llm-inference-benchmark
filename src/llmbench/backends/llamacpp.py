"""llama.cpp backend (GGUF models) via llama-cpp-python."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from llmbench.backends.base import Backend, GenerationResult
from llmbench.metrics import GenerationTiming


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_batch: int = 512,
        seed: int = 0,
        logits_all: bool = False,
    ):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.seed = seed
        # Keeps logits for every position (n_ctx x vocab floats): only for quality runs.
        self.logits_all = logits_all
        self._llm = None

    def load(self) -> None:
        from llama_cpp import Llama

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.model_path} not found: run scripts/download_models.py"
            )
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            seed=self.seed,
            logits_all=self.logits_all,
            verbose=False,
        )

    def tokenize(self, text: str) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=True)

    def detokenize(self, tokens: list[int]) -> str:
        return self._llm.detokenize(tokens).decode("utf-8", errors="replace")

    def generate(
        self, prompt_tokens: list[int], max_new_tokens: int, ignore_eos: bool = True
    ) -> GenerationResult:
        llm = self._llm
        if len(prompt_tokens) + max_new_tokens > self.n_ctx:
            raise ValueError("prompt + generation exceeds n_ctx")
        # reset() sets n_tokens back to 0: generate() then finds no common prefix and
        # redoes the full prefill (see the measurement contract in Backend).
        llm.reset()
        eos = llm.token_eos()
        out: list[int] = []
        t_first = None
        t_start = time.perf_counter()
        # Low-level API rather than create_completion: one iterator step = one token,
        # whereas streamed text chunks do not always map to exactly one token.
        for tok in llm.generate(prompt_tokens, temp=0.0, reset=True):
            if t_first is None:
                t_first = time.perf_counter()
            if tok == eos and not ignore_eos:
                break
            out.append(tok)
            if len(out) >= max_new_tokens:
                break
        t_end = time.perf_counter()
        timing = GenerationTiming(
            t_start=t_start,
            t_first_token=t_first if t_first is not None else t_end,
            t_end=t_end,
            n_prompt_tokens=len(prompt_tokens),
            n_generated_tokens=len(out),
        )
        return GenerationResult(timing=timing, text=self.detokenize(out))

    def logits(self, tokens: list[int]) -> np.ndarray:
        if not self.logits_all:
            raise RuntimeError("load the backend with logits_all=True to get logits")
        if len(tokens) > self.n_ctx:
            raise ValueError("tokens exceed n_ctx")
        self._llm.reset()
        self._llm.eval(tokens)
        return np.array(self._llm.scores[: len(tokens)])

    def close(self) -> None:
        if self._llm is not None:
            self._llm.close()
            self._llm = None
