"""Tests for the RLinf-aligned standalone rollout core (no model, no LIBERO).

Locks the contract that made the standalone rollout reach non-zero LIBERO success:
the gripper post-process is applied to every action before ``env.step``, the full
8-action chunk is executed open-loop before the policy is re-queried, and the
LIBERO ``done`` is reported as success.
"""

from __future__ import annotations

import numpy as np

from dreamervla.runners.rlinf_libero_rollout import (
    parse_ids,
    policy_obs_from_env,
    process_action,
    run_episode,
)


def _obs() -> dict:
    return {
        "third_image": np.zeros((4, 4, 3), np.uint8),
        "state": np.zeros(8, np.float32),
        "task_description": "open the drawer",
    }


class FakeEnv:
    """Succeeds (done w/ info.success) exactly at ``succeed_at`` steps."""

    task_description = "open the drawer"

    def __init__(self, succeed_at: int) -> None:
        self.succeed_at = int(succeed_at)
        self.steps = 0
        self.received: list[np.ndarray] = []

    def reset(self, episode_id: int = 0) -> dict:
        self.steps = 0
        self.received = []
        return _obs()

    def step(self, action):
        self.received.append(np.asarray(action, dtype=np.float32))
        self.steps += 1
        done = self.steps >= self.succeed_at
        info = {"success": True} if done else {}
        return _obs(), 0.0, done, info


class FakePolicy:
    """Returns an 8-action chunk; gripper raw=0.9 -> process_action -> -1."""

    def __init__(self) -> None:
        self.calls = 0
        self.obs_seen: list[dict] = []

    def __call__(self, obs: dict, task_description: str):
        self.calls += 1
        self.obs_seen.append(obs)
        return [np.array([0.1 * j, 0, 0, 0, 0, 0, 0.9], np.float32) for j in range(8)]


def test_parse_ids() -> None:
    assert parse_ids("0,1,2") == [0, 1, 2]
    assert parse_ids("0-3") == [0, 1, 2, 3]
    assert parse_ids("5") == [5]


def test_policy_obs_from_env_maps_third_image_to_full_image() -> None:
    obs = _obs()
    p = policy_obs_from_env(obs)
    assert p["full_image"] is obs["third_image"]
    assert p["state"] is obs["state"]


def test_run_episode_reports_success() -> None:
    env = FakeEnv(succeed_at=10)
    assert run_episode(FakePolicy(), env, episode_id=0) is True
    assert env.steps == 10


def test_run_episode_applies_gripper_postprocess_before_step() -> None:
    env = FakeEnv(succeed_at=5)
    run_episode(FakePolicy(), env, episode_id=0)
    # every executed action's gripper must be the binarized/inverted value (raw 0.9 -> -1),
    # never the raw [0,1] model output.
    assert all(float(a[-1]) == -1.0 for a in env.received)


def test_run_episode_executes_full_chunk_open_loop() -> None:
    # 20 steps with chunk=8 => model queried at steps 0, 8, 16 => 3 calls (open-loop).
    env = FakeEnv(succeed_at=20)
    policy = FakePolicy()
    run_episode(policy, env, episode_id=0)
    assert policy.calls == 3
    # policy is fed the adapted obs (full_image present, not third_image).
    assert "full_image" in policy.obs_seen[0]


def test_process_action_is_shared_single_source() -> None:
    from dreamervla.runners import oft_collect_common

    assert process_action is oft_collect_common.process_action
