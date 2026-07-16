"""EGL render-device environment setup and diagnostics."""

from __future__ import annotations

import logging
import os

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

    from dreamervla.runtime.render_device import parse_device_ids

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
