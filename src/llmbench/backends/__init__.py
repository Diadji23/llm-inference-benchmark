"""Backends d'inférence : construits par nom depuis une config d'expérience."""

from __future__ import annotations

from llmbench.backends.base import Backend, GenerationResult


def make_backend(kind: str, **kwargs) -> Backend:
    # Imports paresseux : chaque backend tire des dépendances optionnelles lourdes.
    if kind == "llamacpp":
        from llmbench.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend(**kwargs)
    raise ValueError(f"backend inconnu : {kind!r}")


__all__ = ["Backend", "GenerationResult", "make_backend"]
