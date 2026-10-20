"""Runtime environment capture.

A latency number without the hardware, package versions and machine load cannot be
interpreted. Every benchmark run therefore saves this snapshot next to its measurements.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from datetime import UTC, datetime

import psutil

# SIMD instruction sets that directly drive matmul speed on CPU
# (and therefore which quantized formats llama.cpp can accelerate).
SIMD_FLAGS = ("avx", "avx2", "fma", "f16c", "avx512f", "avx512_vnni", "avx512_bf16", "avx_vnni", "amx_tile")

TRACKED_PACKAGES = ("torch", "transformers", "llama_cpp_python", "optimum-quanto", "numpy")


def _cpu_info() -> dict:
    # platform.processor() just returns "x86_64" on Linux: /proc/cpuinfo is more precise.
    model, flags = None, set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and model is None:
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    return {
        "model": model or platform.processor() or "unknown",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "simd": [f for f in SIMD_FLAGS if f in flags],
    }


def _gpu_info() -> list[dict]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [
        {
            "name": torch.cuda.get_device_name(i),
            "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
        }
        for i in range(torch.cuda.device_count())
    ]


def _package_versions() -> dict:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_env() -> dict:
    mem = psutil.virtual_memory()
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "os": platform.platform(),
        "is_wsl": "microsoft" in platform.release().lower(),
        "python": platform.python_version(),
        "cpu": _cpu_info(),
        "ram_total_gb": round(mem.total / 2**30, 1),
        "ram_available_gb": round(mem.available / 2**30, 1),
        "gpus": _gpu_info(),
        "packages": _package_versions(),
        "git_commit": _git_commit(),
        # Load at run time: a benchmark run on a busy machine is biased.
        "load_avg_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
    }
