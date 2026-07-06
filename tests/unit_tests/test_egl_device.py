"""No-GPU tests for LIBERO render backend environment setup."""

from __future__ import annotations

import os

import pytest

from dreamervla.utils import egl_device

_RENDER_ENV_KEYS = (
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "MUJOCO_EGL_DEVICE_ID",
    "CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
)


@pytest.fixture(autouse=True)
def clean_render_env():
    original = {key: os.environ.get(key) for key in _RENDER_ENV_KEYS}
    for key in _RENDER_ENV_KEYS:
        os.environ.pop(key, None)
    yield
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _skip_egl_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(egl_device, "_log_egl_devices", lambda logger, device: None)


def test_libero_egl_regime_selects_render_device_by_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_egl_diagnostics(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,3,4,5,6")

    egl_device.apply_libero_render_regime("egl", shard_id=5, gpu_pool=[2, 4, 7])

    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["PYOPENGL_PLATFORM"] == "egl"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "7"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,3,4,5,6"


def test_libero_osmesa_regime_sets_backend_and_clears_egl_device() -> None:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "3"

    egl_device.apply_libero_render_regime("osmesa", shard_id=5, gpu_pool=[])

    assert os.environ["MUJOCO_GL"] == "osmesa"
    assert os.environ["PYOPENGL_PLATFORM"] == "osmesa"
    assert "MUJOCO_EGL_DEVICE_ID" not in os.environ


def test_libero_egl_regime_rejects_empty_gpu_pool() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    with pytest.raises(
        ValueError,
        match="render_backend=egl requires ngpu>=1; use render_backend=osmesa for ngpu=0",
    ):
        egl_device.apply_libero_render_regime("egl", shard_id=0, gpu_pool=[])


def test_libero_render_regime_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        egl_device.apply_libero_render_regime("glfw", shard_id=0, gpu_pool=[0])
