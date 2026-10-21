"""Common interface for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from llmbench.metrics import GenerationTiming


@dataclass
class GenerationResult:
    timing: GenerationTiming
    text: str


class Backend(ABC):
    """A backend loads a model and generates greedily (deterministic decoding).

    Measurement contract: `generate` must not reuse any state from a previous call (no
    prefix cache), otherwise repetitions of the same prompt would skip the prefill and
    TTFT would be wrong.
    """

    name: str

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def tokenize(self, text: str) -> list[int]: ...

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> str: ...

    @abstractmethod
    def generate(
        self, prompt_tokens: list[int], max_new_tokens: int, ignore_eos: bool = True
    ) -> GenerationResult: ...

    def logits(self, tokens: list[int]) -> np.ndarray:
        """Raw logits at every position of `tokens`, shape (len(tokens), vocab).

        Row i is the prediction for token i+1. Used by quality metrics, not by timing.
        """
        raise NotImplementedError(f"{self.name} does not expose per-position logits")

    def close(self) -> None:
        pass

    def truncate_to_tokens(self, text: str, n_tokens: int) -> list[int]:
        """The first n tokens of `text`, using the model's own tokenizer.

        Fixing the length in tokens (not characters) guarantees that every model does the
        same prefill work, whatever its tokenizer.
        """
        tokens = self.tokenize(text)
        if len(tokens) < n_tokens:
            raise ValueError(
                f"source text too short: {len(tokens)} tokens < {n_tokens} requested"
            )
        return tokens[:n_tokens]
