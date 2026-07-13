"""Shared render-device config parsing and validation for online rollout."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any


def parse_device_ids(value: Any) -> list[int]:
    """Normalize a Hydra/device-list value to non-negative integer ids."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        raw_items: Iterable[Any] = (item.strip() for item in value.split(","))
    elif isinstance(value, bool):
        raise ValueError("device ids must be integers, not booleans")
    elif isinstance(value, int):
        raw_items = (value,)
    else:
        raw_items = value

    devices: list[int] = []
    for item in raw_items:
        if item is None or item == "":
            continue
        if isinstance(item, bool):
            raise ValueError("device ids must be integers, not booleans")
        device = int(item)
        if device < 0:
            raise ValueError(f"device ids must be >= 0, got {device}")
        devices.append(device)
    return devices


def cuda_visible_devices_from_env() -> list[int]:
    """Physical CUDA ids from CUDA_VISIBLE_DEVICES when it is an integer list."""
    return parse_device_ids(os.environ.get("CUDA_VISIBLE_DEVICES", ""))


def validate_render_device_pool(
    *,
    render_backend: str,
    num_envs: int,
    render_devices: Any,
    compute_devices: Any,
    render_key: str,
) -> list[int]:
    """Validate explicit egl multi-env render devices and return normalized ids."""
    devices = parse_device_ids(render_devices)
    if int(num_envs) <= 1 or str(render_backend).lower() != "egl":
        return devices

    if not devices:
        raise ValueError(
            f"{render_key} must be set when render_backend=egl and num_envs>1; "
            f"set {render_key} to GPUs reserved for rendering, or use "
            "render_backend=osmesa."
        )

    compute = parse_device_ids(compute_devices)
    overlap = sorted(set(devices).intersection(compute))
    if overlap:
        raise ValueError(
            f"{render_key} must not overlap compute devices for multi-env egl; "
            f"render_devices={devices}, compute_devices={compute}, overlap={overlap}. "
            f"Set disjoint {render_key}, or use render_backend=osmesa."
        )
    return devices
