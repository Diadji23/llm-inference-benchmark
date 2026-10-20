"""Interface commune des backends d'inférence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llmbench.metrics import GenerationTiming


@dataclass
class GenerationResult:
    timing: GenerationTiming
    text: str


class Backend(ABC):
    """Un backend charge un modèle et génère en greedy (décodage déterministe).

    Contrat de mesure : `generate` ne doit réutiliser aucun état d'un appel précédent
    (pas de cache de préfixe), sinon les répétitions d'un même prompt sauteraient le
    prefill et le TTFT serait faussé.
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

    def close(self) -> None:
        pass

    def truncate_to_tokens(self, text: str, n_tokens: int) -> list[int]:
        """Les n premiers tokens de `text`, avec le tokenizer du modèle.

        Fixer la longueur en tokens (et non en caractères) garantit que tous les modèles
        font le même travail de prefill, quel que soit leur tokenizer.
        """
        tokens = self.tokenize(text)
        if len(tokens) < n_tokens:
            raise ValueError(
                f"texte source trop court : {len(tokens)} tokens < {n_tokens} demandés"
            )
        return tokens[:n_tokens]
