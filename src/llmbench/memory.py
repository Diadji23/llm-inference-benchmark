"""Mesure de la mémoire de pointe (RSS du processus).

ru_maxrss donne le pic sur toute la vie du processus et ne se réinitialise pas : inutilisable
pour comparer plusieurs modèles dans un même run. On échantillonne donc le RSS dans un
thread pendant une fenêtre donnée. Limite connue : un pic plus court que la période
d'échantillonnage peut être manqué (sous-estimation de l'ordre de quelques Mo).
"""

from __future__ import annotations

import threading
from typing import Self

import psutil

MIB = 2**20


def current_rss_mib() -> float:
    return psutil.Process().memory_info().rss / MIB


class PeakRSSSampler:
    """Context manager : `with PeakRSSSampler() as s: ...` puis `s.peak_mib`."""

    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self.peak_mib = 0.0
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        rss = self._proc.memory_info().rss / MIB
        self.peak_mib = max(self.peak_mib, rss)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> Self:
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
