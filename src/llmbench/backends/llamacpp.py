"""llama.cpp backend (GGUF models) via llama-cpp-python."""

from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np

from llmbench.backends.base import Backend, GenerationResult
from llmbench.memory import parse_llamacpp_buffers
from llmbench.metrics import GenerationTiming

KV_TYPES = ("f16", "q8_0", "q4_0")


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
        flash_attn: bool = False,
        type_k: str = "f16",
        type_v: str = "f16",
    ):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.seed = seed
        # Keeps logits for every position (n_ctx x vocab floats): only for quality runs.
        self.logits_all = logits_all
        # False is the llama-cpp-python default, used in steps 1-2.
        self.flash_attn = flash_attn
        for kv_type in (type_k, type_v):
            if kv_type not in KV_TYPES:
                raise ValueError(f"KV cache type {kv_type!r} not in {KV_TYPES}")
        # llama.cpp only quantizes V inside the flash attention kernel.
        if type_v != "f16" and not flash_attn:
            raise ValueError("a quantized V cache requires flash_attn=True")
        self.type_k, self.type_v = type_k, type_v
        self.buffers: dict[str, float] = {}
        self.metadata: dict[str, str] = {}
        self._llm = None

    def load(self) -> None:
        import llama_cpp
        from llama_cpp import Llama

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.model_path} not found: run scripts/download_models.py"
            )
        ggml_type = {"f16": llama_cpp.GGML_TYPE_F16, "q8_0": llama_cpp.GGML_TYPE_Q8_0,
                     "q4_0": llama_cpp.GGML_TYPE_Q4_0}
        # The buffer sizes llama.cpp reserves are only printed in its log: capture it
        # during the load, then give the log back to llama-cpp-python's own handler.
        log: list[str] = []

        @llama_cpp.llama_log_callback
        def capture(level, text, user_data):
            log.append(text.decode("utf-8", errors="replace"))

        llama_cpp.llama_log_set(capture, ctypes.c_void_p(0))
        try:
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                seed=self.seed,
                logits_all=self.logits_all,
                flash_attn=self.flash_attn,
                type_k=ggml_type[self.type_k],
                type_v=ggml_type[self.type_v],
                verbose=False,
            )
        finally:
            from llama_cpp._logger import llama_log_callback

            llama_cpp.llama_log_set(llama_log_callback, ctypes.c_void_p(0))
        self.buffers = parse_llamacpp_buffers("".join(log))
        self.metadata = dict(self._llm.metadata)

    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos)

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
