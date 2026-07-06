"""CPU-only tests for LiberoEvalEnv (fake raw env; no LIBERO/GPU import)."""

from __future__ import annotations

import numpy as np

from dreamervla.envs.libero_eval_env import LiberoEvalEnv


def _fake_obs(val: float) -> dict:
    return {
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([val, 0.0, 0.0], dtype=np.float64),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "robot0_gripper_qpos": np.array([0.0, 0.0], dtype=np.float64),
    }


class _FakeLibero:
    def __init__(self) -> None:
        self.applied = None
        self.steps = 0
        self.closed = False

    def reset(self):
        return _fake_obs(-1.0)

    def set_init_state(self, s):
        self.applied = s
        return _fake_obs(0.0)

    def step(self, a):
        self.steps += 1
        return (_fake_obs(1.0), 0.0, self.steps >= 2, {})

    def close(self):
        self.closed = True


def _make_env(num_steps_wait: int = 1) -> tuple[LiberoEvalEnv, _FakeLibero]:
    fake = _FakeLibero()
    env = LiberoEvalEnv(
        task_suite_name="libero_goal",
        resolution=256,
        seed=0,
        num_steps_wait=num_steps_wait,
        max_steps=2,
        make_env=lambda task_id: (fake, "desc"),
        init_states={0: ["A", "B", "C"]},
    )
    return env, fake


def test_reset_applies_init_state_and_warmup():
    env, fake = _make_env(num_steps_wait=1)
    env.set_task(0)
    assert env.task_description == "desc"

    obs, info = env.reset(episode_id=1, task_id=0)

    assert fake.applied == "B"  # init state index 1
    assert info["init_state_index"] == 1
    assert fake.steps == 1  # one warmup dummy step consumed inside reset
    assert obs is not None


def test_warmup_runs_exactly_num_steps_wait():
    env, fake = _make_env(num_steps_wait=3)
    env.set_task(0)
    env.reset(episode_id=0)
    assert fake.steps == 3


def test_full_record_shapes_match_sequential_inputs():
    env, _fake = _make_env(num_steps_wait=1)
    env.set_task(0)
    env.reset(episode_id=0)

    record = env.full_record()

    assert set(record) == {"third_image", "wrist_image", "state", "raw_obs"}
    assert record["third_image"].shape == (8, 8, 3)
    assert record["wrist_image"].shape == (8, 8, 3)
    # eef_pos(3) + axisangle(3) + gripper_qpos(2)
    assert record["state"].shape == (8,)
    # raw_obs threads the current LIBERO obs dict so the parallel OFT base eval
    # can reproduce the sequential path (which reads self._libero_current_raw_obs).
    assert isinstance(record["raw_obs"], dict)
    assert "agentview_image" in record["raw_obs"]
    assert "robot0_eye_in_hand_image" in record["raw_obs"]


def test_step_returns_five_tuple_with_success():
    env, fake = _make_env(num_steps_wait=1)
    env.set_task(0)
    env.reset(episode_id=0)  # warmup consumes 1 step -> fake.steps == 1

    obs, reward, terminated, truncated, info = env.step([0.0] * 7)

    assert fake.steps == 2
    assert terminated is True  # fake ends at steps >= 2
    assert truncated is False
    assert info["success"] is True
    assert obs is not None
    assert reward == 0.0


def test_context_manager_closes_env():
    env, fake = _make_env(num_steps_wait=1)
    with env as e:
        e.set_task(0)
        e.reset(episode_id=0)
    assert fake.closed is True


class _CountingFactory:
    """make_env seam that builds a fresh fake env per call and counts builds."""

    def __init__(self) -> None:
        self.built: list[_FakeLibero] = []

    def __call__(self, task_id: int) -> tuple[_FakeLibero, str]:
        fake = _FakeLibero()
        self.built.append(fake)
        return fake, "desc"


def _make_env_counting(reconfigure: bool) -> tuple[LiberoEvalEnv, _CountingFactory]:
    factory = _CountingFactory()
    env = LiberoEvalEnv(
        task_suite_name="libero_goal",
        resolution=256,
        seed=0,
        num_steps_wait=1,
        max_steps=2,
        make_env=factory,
        init_states={0: ["A", "B", "C"]},
        reconfigure_per_episode=reconfigure,
    )
    return env, factory


def test_reconfigure_per_episode_rebuilds_each_reset():
    env, factory = _make_env_counting(reconfigure=True)
    env.set_task(0)
    assert len(factory.built) == 1  # set_task builds once

    env.reset(episode_id=0)
    assert len(factory.built) == 2  # reset rebuilds a fresh env
    assert factory.built[0].closed is True  # old env closed before rebuild
    assert factory.built[1].applied == "A"  # fresh env got the init state

    env.reset(episode_id=1)
    assert len(factory.built) == 3
    assert factory.built[1].closed is True
    assert factory.built[2].applied == "B"


def test_no_reconfigure_reuses_single_env():
    env, factory = _make_env_counting(reconfigure=False)
    env.set_task(0)
    assert len(factory.built) == 1

    env.reset(episode_id=0)
    env.reset(episode_id=1)
    env.reset(episode_id=2)

    assert len(factory.built) == 1  # built once, reused across resets
    assert factory.built[0].closed is False
    assert factory.built[0].applied == "C"  # last init state applied
