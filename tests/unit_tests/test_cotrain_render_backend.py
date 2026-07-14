"""build_rollout_vec_env — render_backend backend selection for cotrain (no GPU).

Asserts the two user-chosen approaches are wired correctly without constructing a real
vec env: egl -> the RLinf-vendored OnlineEglVecEnv (with an explicit render device
pool and render env vars stripped); osmesa -> the proven VecRolloutEnv
unchanged. The vec-env classes are monkeypatched to record their constructor kwargs.
"""

from __future__ import annotations

import pytest

from dreamervla.runtime import world_model_training_common as wm_runtime


class _FakeEgl:
    last: dict | None = None

    def __init__(self, **kwargs):
        type(self).last = kwargs


class _FakeOsmesa:
    last: dict | None = None

    def __init__(self, **kwargs):
        type(self).last = kwargs


@pytest.fixture
def patched(monkeypatch):
    import dreamervla.envs.libero.venv as egl_mod
    import dreamervla.runtime.vec_rollout_env as vec_mod

    _FakeEgl.last = None
    _FakeOsmesa.last = None
    monkeypatch.setattr(egl_mod, "OnlineEglVecEnv", _FakeEgl)
    monkeypatch.setattr(vec_mod, "VecRolloutEnv", _FakeOsmesa)


def test_egl_backend_uses_vendored_adapter_with_device_pool(patched, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    out = wm_runtime.build_rollout_vec_env(
        render_backend="egl",
        num_envs=3,
        cfg_kwargs={"a": 1},
        env_vars={"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "LIBERO_CONFIG_PATH": "/x"},
        render_devices=[6, 7],
    )
    assert isinstance(out, _FakeEgl)
    assert _FakeEgl.last["num_envs"] == 3
    assert _FakeEgl.last["cfg_kwargs"] == {"a": 1}
    assert _FakeEgl.last["egl_device_pool"] == [6, 7]
    # render env vars are stripped (the adapter applies the egl regime per child);
    # non-render vars are forwarded.
    assert _FakeEgl.last["env_vars"] == {"LIBERO_CONFIG_PATH": "/x"}


def test_osmesa_backend_uses_vecrolloutenv_unchanged(patched, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    env_vars = {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
    out = wm_runtime.build_rollout_vec_env(
        render_backend="osmesa",
        num_envs=2,
        cfg_kwargs={},
        env_vars=env_vars,
        render_devices=[6, 7],
    )
    assert isinstance(out, _FakeOsmesa)
    assert _FakeOsmesa.last["num_envs"] == 2
    assert _FakeOsmesa.last["env_vars"] == env_vars  # osmesa path untouched


def test_egl_backend_does_not_infer_pool_from_cuda_visible_devices(patched, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm_runtime.build_rollout_vec_env(
        render_backend="egl",
        num_envs=1,
        cfg_kwargs={},
        env_vars={},
        render_devices=[2],
    )
    assert _FakeEgl.last["egl_device_pool"] == [2]
