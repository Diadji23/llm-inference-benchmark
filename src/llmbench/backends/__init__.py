"""Inference backends, built by name from an experiment config."""

from __future__ import annotations

from llmbench.backends.base import Backend, GenerationResult


def make_backend(kind: str, **kwargs) -> Backend:
    # Lazy imports: each backend pulls in heavy optional dependencies.
    if kind == "llamacpp":
        from llmbench.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend(**kwargs)
    raise ValueError(f"unknown backend: {kind!r}")


__all__ = ["Backend", "GenerationResult", "make_backend"]
