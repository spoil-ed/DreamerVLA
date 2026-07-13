"""RLinf-style chunk eval driver: prev-done dedup, rolling blocks, per-task SR."""

import numpy as np
from omegaconf import OmegaConf

from dreamervla.envs.libero.libero_env import LiberoEnv
from dreamervla.runtime.libero_chunk_eval import ChunkEvalTally, run_rlinf_chunk_eval


class _ScriptedChunkEnv:
    """LiberoEnv stand-in: (task, trial) grid of 2 tasks x 3 trials,
    success iff (task_id + trial_id) is even, episodes end after 1 chunk."""

    def __init__(self, num_envs, auto_reset=False):
        self.num_envs = num_envs
        self.auto_reset = auto_reset
        self.is_start = True
        self.n_tasks, self.n_trials = 2, 3
        self.total = self.n_tasks * self.n_trials
        self._block_start = 0
        self.reset_state_ids = None
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        ids = [(self._block_start + k) % self.total for k in range(self.num_envs)]
        self._block_start = (self._block_start + self.num_envs) % self.total
        self.reset_state_ids = np.array(ids)

    def reset(self, env_idx=None, reset_state_ids=None):
        self.is_start = False
        return self._obs(), {}

    def _obs(self):
        return {"task_descriptions": ["t"] * self.num_envs}

    def _episode_info(self):
        task_ids = self.reset_state_ids // self.n_trials
        trial_ids = self.reset_state_ids % self.n_trials
        return {
            "success_once": (task_ids + trial_ids) % 2 == 0,
            "task_id": task_ids,
            "reset_state_id": self.reset_state_ids.copy(),
        }

    def chunk_step(self, chunk_actions):
        n, c = chunk_actions.shape[0], chunk_actions.shape[1]
        terms = np.zeros((n, c), dtype=bool)
        truncs = np.zeros((n, c), dtype=bool)
        truncs[:, -1] = True  # every episode ends after one chunk
        infos = {"episode": self._episode_info()}
        if self.auto_reset:
            infos = {"final_info": {"episode": self._episode_info()}}
            self.update_reset_state_ids()
        return (
            [self._obs()] * c,
            np.zeros((n, c)),
            terms,
            truncs,
            [infos] * c,
        )


class _TaskSuite:
    n_tasks = 2

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_init_states(self, _task_id):
        return [object(), object(), object()]


class _LiberoEnvNoInit(LiberoEnv):
    def _load_task_suite(self):
        return _TaskSuite()

    def _init_env(self):
        self.env = None


def _policy(obs):
    return np.zeros((len(obs["task_descriptions"]), 2, 7))


def test_tally_dedups_wrapped_reset_ids():
    tally = ChunkEvalTally()
    ep = {
        "success_once": np.array([True, False]),
        "task_id": np.array([0, 0]),
        "reset_state_id": np.array([0, 1]),
    }
    tally.add(ep, np.array([True, True]))
    # same reset ids come around again with flipped success: first result wins
    ep2 = {
        "success_once": np.array([False, True]),
        "task_id": np.array([0, 0]),
        "reset_state_id": np.array([0, 1]),
    }
    tally.add(ep2, np.array([True, True]))
    assert tally.num_episodes == 2
    assert tally.records() == {0: {0: True, 1: False}}


def test_driver_covers_grid_with_rolling_blocks():
    # 6 episodes, 4 envs -> 2 epochs, last block wraps (dedup keeps 6)
    env = _ScriptedChunkEnv(num_envs=4, auto_reset=False)
    tally = run_rlinf_chunk_eval(
        env, _policy, n_chunk_steps=3, num_epochs=2, total_episodes=6
    )
    assert tally.num_episodes == 6
    metrics = tally.summarize(episodes_per_task=3)
    # success iff (task+trial) even: task0 -> trials 0,2 succeed (2/3); task1 -> trial 1 (1/3)
    assert metrics["eval_task_0_success_rate"] == 2 / 3
    assert metrics["eval_task_1_success_rate"] == 1 / 3
    assert abs(metrics["eval_success_rate"] - 0.5) < 1e-9


def test_driver_auto_reset_counts_newly_done_each_chunk():
    env = _ScriptedChunkEnv(num_envs=3, auto_reset=True)
    tally = run_rlinf_chunk_eval(
        env, _policy, n_chunk_steps=2, num_epochs=1, total_episodes=6
    )
    assert tally.num_episodes == 6


def test_driver_records_env_chunk_and_action_step_units():
    env = _ScriptedChunkEnv(num_envs=3, auto_reset=True)
    tally = run_rlinf_chunk_eval(
        env, _policy, n_chunk_steps=2, num_epochs=1, total_episodes=6
    )
    assert tally.env_chunk_steps == 6
    assert tally.env_action_steps == 12


def test_driver_exposes_read_only_reset_and_chunk_observer_events():
    env = _ScriptedChunkEnv(num_envs=2, auto_reset=False)
    resets = []
    chunks = []

    tally = run_rlinf_chunk_eval(
        env,
        _policy,
        n_chunk_steps=2,
        num_epochs=1,
        total_episodes=2,
        on_reset=lambda **event: resets.append(event),
        on_chunk=lambda **event: chunks.append(event),
    )

    assert tally.num_episodes == 2
    assert len(resets) == 1
    assert resets[0]["env"] is env
    assert len(chunks) == 1
    assert chunks[0]["env"] is env
    assert chunks[0]["chunk_actions"].shape == (2, 2, 7)
    assert chunks[0]["newly_done"].tolist() == [True, True]
    assert chunks[0]["episode_info"]["reset_state_id"].tolist() == [0, 1]


def test_libero_env_ordered_reset_ids_tile_to_requested_num_envs():
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "group_size": 1,
            "is_eval": True,
            "use_fixed_reset_state_ids": True,
            "specific_reset_id": None,
            "task_id_filter": [0, 1],
            "ignore_terminations": False,
            "auto_reset": False,
            "use_rel_reward": False,
            "max_trials_per_task": 3,
            "init_params": {"camera_heights": 64, "camera_widths": 64},
        }
    )
    env = _LiberoEnvNoInit(cfg, num_envs=8)

    assert len(env.reset_state_ids) == 8
    assert env.reset_state_ids.tolist() == [0, 1, 2, 3, 4, 5, 0, 1]


def test_runner_rlinf_chunk_uses_configured_num_envs_without_episode_cap(monkeypatch):
    from dreamervla.runtime.libero_vla_evaluation_base import LIBEROVLAEvaluationBase

    created: dict[str, int] = {}

    class FakeLiberoEnv:
        def __init__(self, _cfg, num_envs):
            created["num_envs"] = int(num_envs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeTally:
        env_chunk_steps = 0
        env_action_steps = 0

        def summarize(self, *, episodes_per_task):
            return {
                "eval_success_rate": 0.0,
                "eval_tasks": 10.0,
                "eval_episodes_per_task": float(episodes_per_task),
            }

    monkeypatch.setattr(
        "dreamervla.envs.libero.libero_env.LiberoEnv",
        FakeLiberoEnv,
    )
    monkeypatch.setattr(
        "dreamervla.runtime.libero_chunk_eval.run_rlinf_chunk_eval",
        lambda *_args, **_kwargs: FakeTally(),
    )

    runner = LIBEROVLAEvaluationBase.__new__(LIBEROVLAEvaluationBase)
    runner.cfg = OmegaConf.create(
        {
            "eval": {
                "render_backend": "osmesa",
                "render_shard_id": 0,
                "render_gpu_pool": None,
            }
        }
    )
    runner._make_parallel_oft_slot_extractor = lambda: None

    runner._finalize_libero_eval_observer = lambda: {
        "eval/cotrain_trajectory_count": 100.0
    }

    metrics = runner._evaluate_libero_rlinf_chunk(
        epoch=-1,
        eval_cfg=OmegaConf.create(
            {
                "task_suite_name": "libero_goal",
                "render_backend": "osmesa",
                "libero_env": {
                    "reset_wait_steps": 10,
                    "reset_gripper_open": True,
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "group_size": 1,
                },
            }
        ),
        backbone=None,
        item_processor=None,
        task_ids=list(range(10)),
        num_episodes=3,
        max_steps=300,
        action_steps=8,
        history_length=1,
        resolution=256,
        seed=7,
        num_envs=64,
    )

    assert created["num_envs"] == 64
    assert metrics["eval/cotrain_trajectory_count"] == 100.0


def test_build_libero_env_cfg_maps_eval_knobs():
    from omegaconf import OmegaConf

    from dreamervla.runtime.libero_vla_evaluation_base import build_libero_env_cfg

    eval_cfg = OmegaConf.create(
        {
            "task_suite_name": "libero_goal",
            "libero_env": {
                "reset_wait_steps": 10,
                "reset_gripper_open": True,
                "auto_reset": False,
                "ignore_terminations": False,
                "group_size": 1,
            },
        }
    )
    cfg = build_libero_env_cfg(
        eval_cfg,
        task_ids=[0, 2],
        num_episodes=3,
        max_steps=300,
        seed=7,
        resolution=256,
    )
    assert cfg.task_suite_name == "libero_goal"
    assert cfg.is_eval is True
    assert cfg.use_fixed_reset_state_ids is True
    assert cfg.use_ordered_reset_state_ids is True
    assert cfg.task_id_filter == [0, 2]
    assert cfg.max_trials_per_task == 3
    assert cfg.max_episode_steps == 300
    assert cfg.seed == 7
    assert cfg.reset_wait_steps == 10
    assert cfg.init_params.camera_heights == 256
