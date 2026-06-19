"""Best-effort local resource metrics for runner diagnostics."""

from __future__ import annotations

import subprocess
from statistics import fmean
from typing import Any


def parse_nvidia_smi_csv(text: str) -> dict[str, float]:
    """Parse ``nvidia-smi`` CSV output into aggregate GPU metrics."""

    utilizations: list[float] = []
    memory_used: list[float] = []
    memory_total: list[float] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            _index = int(float(parts[0]))
            utilization = _parse_float(parts[1])
            used = _parse_float(parts[2])
            total = _parse_float(parts[3])
        except ValueError:
            continue
        utilizations.append(utilization)
        memory_used.append(used)
        memory_total.append(total)

    if not utilizations:
        return {}
    return {
        "gpu/utilization_pct_mean": float(fmean(utilizations)),
        "gpu/utilization_pct_max": float(max(utilizations)),
        "gpu/memory_used_mb_mean": float(fmean(memory_used)),
        "gpu/memory_used_mb_max": float(max(memory_used)),
        "gpu/memory_total_mb_max": float(max(memory_total)),
        "gpu/count": float(len(utilizations)),
    }


def collect_nvidia_smi_metrics() -> dict[str, float]:
    """Collect aggregate GPU utilization metrics if ``nvidia-smi`` is present."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    return parse_nvidia_smi_csv(result.stdout)


def collect_torch_cuda_memory() -> dict[str, float]:
    """Collect current and peak CUDA allocator metrics for the active process."""

    try:
        import torch
    except ImportError:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        scale = 1024.0 * 1024.0
        return {
            "cuda/memory_allocated_mb": float(
                torch.cuda.memory_allocated(device) / scale
            ),
            "cuda/memory_reserved_mb": float(torch.cuda.memory_reserved(device) / scale),
            "cuda/memory_peak_allocated_mb": float(
                torch.cuda.max_memory_allocated(device) / scale
            ),
            "cuda/memory_peak_reserved_mb": float(
                torch.cuda.max_memory_reserved(device) / scale
            ),
        }
    except (RuntimeError, AssertionError):
        return {}


def collect_resource_metrics(prefix: str = "time") -> dict[str, float]:
    """Collect best-effort resource metrics under a normalized namespace."""

    metrics: dict[str, float] = {}
    metrics.update(collect_nvidia_smi_metrics())
    metrics.update(collect_torch_cuda_memory())
    return _namespace_metrics(metrics, prefix)


def _parse_float(value: Any) -> float:
    text = str(value).strip().replace("%", "").replace("MiB", "").strip()
    if text.upper() in {"N/A", "NA", ""}:
        raise ValueError(text)
    return float(text)


def _namespace_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    if not prefix:
        return dict(metrics)
    normalized = prefix.rstrip("/")
    return {
        f"{normalized}/{key.replace('/', '_')}": float(value)
        for key, value in metrics.items()
    }
