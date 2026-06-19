"""Local hardware discovery for Ray placement validation.

This module discovers resources for manual placement checks. It does not tune
batch size, env counts, or any other runtime knob.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AcceleratorInfo:
    """Local accelerator metadata."""

    index: int
    name: str
    total_memory_bytes: int
    kind: str = "cuda"


def discover_local_accelerators() -> list[AcceleratorInfo]:
    """Return CUDA accelerators visible to the current process."""

    if not torch.cuda.is_available():
        return []
    devices: list[AcceleratorInfo] = []
    for index in range(int(torch.cuda.device_count())):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            AcceleratorInfo(
                index=index,
                name=str(torch.cuda.get_device_name(index)),
                total_memory_bytes=int(props.total_memory),
            )
        )
    return devices


def count_local_accelerators() -> int:
    """Count CUDA accelerators visible to the current process."""

    return len(discover_local_accelerators())


__all__ = ["AcceleratorInfo", "count_local_accelerators", "discover_local_accelerators"]
