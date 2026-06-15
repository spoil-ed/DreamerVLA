from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf


def _load_libero_env_module(monkeypatch):
    class FakeOffscreenEnv:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def seed(self, _seed: int) -> None:
            pass

    fake_libero_core = types.ModuleType("libero.libero")
    fake_libero_core.get_libero_path = lambda _kind: "/libero"
    fake_libero_core.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {})
    fake_libero_envs = types.ModuleType("libero.libero.envs")
    fake_libero_envs.OffScreenRenderEnv = FakeOffscreenEnv
    fake_libero_pkg = types.ModuleType("libero")
    fake_libero_pkg.libero = fake_libero_core
    monkeypatch.setitem(sys.modules, "libero", fake_libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_libero_core)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", fake_libero_envs)

    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "dreamervla" / "envs" / "libero_env.py"
    spec = importlib.util.spec_from_file_location("_test_libero_env", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_libero_env_helper_uses_explicit_eval_seed(monkeypatch) -> None:
    libero_env_mod = _load_libero_env_module(monkeypatch)

    seeded: list[int] = []

    class FakeOffscreenEnv:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def seed(self, seed: int) -> None:
            seeded.append(int(seed))

    monkeypatch.setattr(libero_env_mod, "OffScreenRenderEnv", FakeOffscreenEnv)
    monkeypatch.setattr(libero_env_mod, "get_libero_path", lambda _kind: "/libero")
    task = SimpleNamespace(
        language="put the bowl on the plate",
        problem_folder="libero_goal",
        bddl_file="task.bddl",
    )

    env, task_description = libero_env_mod.get_libero_env(
        task, resolution=128, seed=7
    )

    assert task_description == "put the bowl on the plate"
    assert seeded == [7]
    assert env.kwargs == {
        "bddl_file_name": "/libero/libero_goal/task.bddl",
        "camera_heights": 128,
        "camera_widths": 128,
    }


def test_libero_image_rotation_matches_rlinf_contiguous_preprocessing(
    monkeypatch,
) -> None:
    libero_env_mod = _load_libero_env_module(monkeypatch)
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    rotated = libero_env_mod.get_libero_image({"agentview_image": image}, 256)

    assert np.array_equal(rotated, np.ascontiguousarray(image[::-1, ::-1]))
    assert rotated.flags.c_contiguous


def test_eval_libero_config_uses_rlinf_protocol_defaults() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(project_root / "configs" / "evaluation" / "libero_vla.yaml")

    assert OmegaConf.select(cfg, "eval.seed") == 7
    assert OmegaConf.select(cfg, "eval.num_steps_wait") == 10
    assert OmegaConf.select(cfg, "eval.num_episodes_per_task") == 50


def test_eval_protocol_resolver_uses_rlinf_defaults(monkeypatch) -> None:
    libero_env_mod = _load_libero_env_module(monkeypatch)

    protocol = libero_env_mod.resolve_libero_eval_protocol(
        OmegaConf.create({"seed": 17}),
        OmegaConf.create({}),
    )

    assert protocol == {"seed": 17, "num_steps_wait": 10}


def test_libero_action_chunk_selection_matches_rlinf(monkeypatch) -> None:
    libero_env_mod = _load_libero_env_module(monkeypatch)
    actions = [np.array([idx], dtype=np.float32) for idx in range(4)]

    selected = libero_env_mod.select_libero_action_chunk(actions, action_steps=2)

    assert [int(action[0]) for action in selected] == [0, 1]
    with pytest.raises(AssertionError, match="replan every 5 steps"):
        libero_env_mod.select_libero_action_chunk(actions, action_steps=5)
