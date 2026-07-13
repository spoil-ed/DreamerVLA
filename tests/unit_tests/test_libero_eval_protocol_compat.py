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
    module_path = project_root / "dreamervla" / "envs" / "libero" / "utils.py"
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

    monkeypatch.setattr(
        sys.modules["libero.libero.envs"], "OffScreenRenderEnv", FakeOffscreenEnv
    )
    monkeypatch.setattr(
        sys.modules["libero.libero"], "get_libero_path", lambda _kind: "/libero"
    )
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
    assert OmegaConf.select(cfg, "eval.num_episodes_per_task") == 3
    assert OmegaConf.select(cfg, "eval.scheme") == "rlinf_chunk"
    assert OmegaConf.select(cfg, "eval.num_envs") == 64
    assert OmegaConf.select(cfg, "eval.action_steps") == 8
    assert OmegaConf.select(cfg, "eval.history_length") == 1
    assert OmegaConf.select(cfg, "eval.render_backend") == "osmesa"


def test_eval_libero_config_uses_latent_dreamer_defaults() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(project_root / "configs" / "evaluation" / "libero_vla.yaml")

    assert OmegaConf.select(cfg, "eval.dreamer_actor_input_source") == "latent"
    assert OmegaConf.select(cfg, "eval.dreamer_latent_action_source") == "env"
    assert OmegaConf.select(cfg, "eval.dreamer_rssm_action_source", default=None) is None


def test_compat_rssm_actor_input_source_normalizes_to_latent() -> None:
    from dreamervla.runners.libero_vla_evaluation_runner import (
        normalize_dreamer_actor_input_source,
        normalize_dreamer_rollout_mode,
    )

    assert normalize_dreamer_actor_input_source("rssm") == "latent"
    assert normalize_dreamer_actor_input_source("latent") == "latent"
    assert normalize_dreamer_rollout_mode("online_rssm") == "online_latent"
    assert normalize_dreamer_rollout_mode("stateless") == "stateless"


def test_manual_ray_oft_eval_normalizer_keeps_stateless_latent_mode() -> None:
    from dreamervla.runners.libero_vla_evaluation_runner import LIBEROVLAEvaluationRunner

    cfg = OmegaConf.create(
        {
            "actor": {
                "policy_cfg": {
                    "target": "dreamervla.algorithms.actor.latent_to_openvla_hidden_state_actor.LatentToOpenVLAHiddenStateActor",
                    "hidden_dim": 8,
                    "action_dim": 7,
                }
            },
            "learner": {
                "model_cfg": {
                    "world_model": {
                        "target": "dreamervla.models.embodiment.world_model.wm_chunk.ChunkAwareWorldModel",
                        "obs_dim": 8,
                        "action_dim": 7,
                    }
                }
            },
            "rollout": {
                "encoder_cfg": {
                    "target": "dreamervla.models.embodiment.openvla_oft.oft_rollout:OFTRolloutBundle"
                }
            },
            "eval": {},
            "task": {
                "openvla_oft": {
                    "hidden_token": {
                        "expected_obs_hidden_source": "hidden_token"
                    }
                }
            },
        }
    )

    LIBEROVLAEvaluationRunner._normalize_manual_ray_dreamer_eval_cfg(cfg)

    assert OmegaConf.select(cfg, "eval.dreamer_rollout_mode") == "stateless"
    assert OmegaConf.select(cfg, "eval.dreamer_actor_input_source") == "latent"
    assert OmegaConf.select(cfg, "eval.obs_hidden_source") == "hidden_token"


def test_stateless_dreamer_eval_dispatches_to_dreamer_path(monkeypatch) -> None:
    from dreamervla.runners.libero_vla_evaluation_runner import LIBEROVLAEvaluationRunner
    from dreamervla.runtime.libero_vla_evaluation_base import LIBEROVLAEvaluationBase

    runner = LIBEROVLAEvaluationRunner.__new__(LIBEROVLAEvaluationRunner)
    runner._dreamer_eval = True
    runner._dreamer_rollout_mode = "stateless"
    called: list[int] = []

    def fake_dreamer_eval(self, epoch: int) -> dict[str, float]:
        called.append(int(epoch))
        return {"eval_success_rate": 1.0}

    def fail_base_eval(self, epoch: int):  # pragma: no cover - failure assertion path
        raise AssertionError("stateless Dreamer eval fell back to base VLA eval")

    monkeypatch.setattr(
        LIBEROVLAEvaluationRunner,
        "_evaluate_libero_online_latent",
        fake_dreamer_eval,
    )
    monkeypatch.setattr(LIBEROVLAEvaluationBase, "evaluate_libero", fail_base_eval)

    metrics = runner.evaluate_libero(epoch=5)

    assert called == [5]
    assert metrics == {"eval_success_rate": 1.0}


def test_eval_summary_averages_three_trials_per_task() -> None:
    from dreamervla.runtime.eval_metrics import summarize_libero_task_success

    metrics = summarize_libero_task_success(
        [
            {"task_id": 0, "episodes": 3, "successes": 2},
            {"task_id": 1, "episodes": 3, "successes": 1},
        ],
        episodes_per_task=3,
    )

    assert metrics["eval_success_rate"] == pytest.approx((2.0 / 3.0 + 1.0 / 3.0) / 2)
    assert metrics["eval_tasks"] == 2.0
    assert metrics["eval_episodes_per_task"] == 3.0
    assert metrics["eval_total_episodes"] == 6.0
    assert metrics["eval_total_successes"] == 3.0
    assert metrics["eval_task_0_success_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["eval_task_1_success_rate"] == pytest.approx(1.0 / 3.0)


def test_eval_summary_uses_task_macro_average_not_episode_weighted() -> None:
    from dreamervla.runtime.eval_metrics import summarize_libero_task_success

    metrics = summarize_libero_task_success(
        [
            {"task_id": 0, "episodes": 1, "successes": 1},
            {"task_id": 1, "episodes": 3, "successes": 0},
        ],
        episodes_per_task=3,
    )

    assert metrics["eval_success_rate"] == pytest.approx(0.5)
    assert metrics["results/total_success_rate"] == pytest.approx(0.5)
    assert metrics["eval_episode_weighted_success_rate"] == pytest.approx(0.25)


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


def test_eval_init_state_indices_default_uses_num_episodes() -> None:
    from dreamervla.runners.libero_vla_evaluation_runner import LIBEROVLAEvaluationRunner

    indices = LIBEROVLAEvaluationRunner._eval_init_state_indices(
        num_init_states=50,
        num_episodes=3,
        enumerate_all_init_states=False,
    )

    assert indices == [0, 1, 2]


def test_eval_init_state_indices_enumerate_visits_every_state_once_in_order() -> None:
    from dreamervla.runners.libero_vla_evaluation_runner import LIBEROVLAEvaluationRunner

    num_init_states = 7
    indices = LIBEROVLAEvaluationRunner._eval_init_state_indices(
        num_init_states=num_init_states,
        num_episodes=3,
        enumerate_all_init_states=True,
    )

    # Every init-state index requested exactly once, in ascending order.
    assert indices == list(range(num_init_states))
    assert sorted(indices) == indices
    assert len(set(indices)) == num_init_states


def test_eval_enumerate_drives_mock_env_over_all_init_states_in_order() -> None:
    from dreamervla.runners.libero_vla_evaluation_runner import LIBEROVLAEvaluationRunner

    # Distinct sentinels stand in for a task's init states.
    initial_states = [f"init_{i}" for i in range(5)]

    class MockEnv:
        def __init__(self) -> None:
            self.set_calls: list[str] = []

        def set_init_state(self, state):
            self.set_calls.append(state)
            return state

    env = MockEnv()
    indices = LIBEROVLAEvaluationRunner._eval_init_state_indices(
        num_init_states=len(initial_states),
        num_episodes=3,
        enumerate_all_init_states=True,
    )
    for episode_idx in indices:
        env.set_init_state(initial_states[episode_idx])

    assert env.set_calls == initial_states


def test_eval_libero_config_declares_enumerate_all_init_states_default_false() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(project_root / "configs" / "evaluation" / "libero_vla.yaml")

    assert OmegaConf.select(cfg, "eval.enumerate_all_init_states") is False
