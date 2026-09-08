"""Render-device parsing, environment setup, and EGL diagnostics."""

from __future__ import annotations

import logging
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


_ZERO_GPU_EGL_ERROR = "render_backend=egl requires ngpu>=1; use render_backend=osmesa for ngpu=0"


def apply_egl_device_regime(
    egl_device_id: int | None,
    *,
    logger_name: str,
) -> None:
    """Apply the child-process EGL env vars before robosuite/mujoco import.

    ``MUJOCO_EGL_DEVICE_ID`` is an index into EGL's device enumeration, not a
    CUDA physical id. DreamerVLA still narrows ``CUDA_VISIBLE_DEVICES`` to the
    configured render id to match robosuite's import-time consistency check; the
    diagnostic below makes the EGL-index assumption visible and fails early when
    the selected index is outside the driver's EGL device list.
    """
    logger = logging.getLogger(logger_name)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    if egl_device_id is None:
        _log_egl_devices(logger, None)
        return

    device = str(int(egl_device_id))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = device
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    _log_egl_devices(logger, int(egl_device_id))


def apply_libero_render_regime(backend: str, shard_id: int, gpu_pool: list[int]) -> None:
    """Apply LIBERO render env vars before robosuite/mujoco initialization."""
    normalized = str(backend).strip().lower()
    if normalized not in {"egl", "osmesa"}:
        raise ValueError("backend must be one of: egl, osmesa")

    if normalized == "osmesa":
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        return

    devices = parse_device_ids(gpu_pool)
    if not devices:
        raise ValueError(_ZERO_GPU_EGL_ERROR)
    egl_device_id = devices[int(shard_id) % len(devices)]
    device = str(int(egl_device_id))
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = device


def log_egl_device_diagnostics_from_env(*, logger_name: str) -> None:
    """Log EGL diagnostics for a worker-level regime that is already set.

    RLinf-style Ray workers receive ``CUDA_VISIBLE_DEVICES`` and
    ``MUJOCO_EGL_DEVICE_ID`` from ``runtime_env``. This helper validates and
    reports that inherited state without mutating it.
    """
    logger = logging.getLogger(logger_name)
    raw_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if raw_device in (None, ""):
        _log_egl_devices(logger, None)
        return
    try:
        egl_device_id = int(raw_device)
    except ValueError:
        logger.warning("Invalid MUJOCO_EGL_DEVICE_ID=%r for EGL diagnostics", raw_device)
        return
    _log_egl_devices(logger, egl_device_id)


def _log_egl_devices(logger: logging.Logger, egl_device_id: int | None) -> None:
    try:
        from mujoco.egl import egl_ext as egl

        count = len(egl.eglQueryDevicesEXT())
    except Exception as exc:  # noqa: BLE001 - diagnostics must not block init
        logger.warning("EGL device diagnostics unavailable: %r", exc)
        return

    message = (
        "EGL device diagnostics: eglQueryDevicesEXT count=%d, "
        "MUJOCO_EGL_DEVICE_ID=%s (EGL enumeration index, not CUDA id)"
    )
    logger.info(message, count, egl_device_id)
    if not logger.isEnabledFor(logging.INFO):
        print(
            f"[egl_device] {message % (count, egl_device_id)}",
            flush=True,
        )
    if egl_device_id is not None and not 0 <= int(egl_device_id) < count:
        raise ValueError(
            "MUJOCO_EGL_DEVICE_ID is an EGL enumeration index, not a CUDA physical "
            f"id; got {egl_device_id}, but eglQueryDevicesEXT returned {count} "
            "device(s). Use online_rollout.render_backend=osmesa, or choose a "
            "render_devices entry that maps to a valid EGL index on this host."
        )
