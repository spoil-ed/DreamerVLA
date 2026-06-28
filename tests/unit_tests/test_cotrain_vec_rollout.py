"""Unit tests for the vectorized (multi-env egl) online cotrain rollout (Option 1).

Pure-CPU: no LIBERO env, no GPU. Helpers and the continuous loop are exercised with
synthetic records and fake VecRolloutEnv/extractors.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rec(h: int = 120, w: int = 160) -> dict:
    rng = np.random.default_rng(0)
    return {
        "agentview_rgb": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
        "eye_in_hand_rgb": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
    }


def _full_record() -> dict:
    rng = np.random.default_rng(1)
    return {
        "agentview_rgb": rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
        "eye_in_hand_rgb": rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
        "ee_pos": rng.standard_normal(3).astype(np.float64),
        "ee_ori": rng.standard_normal(3).astype(np.float64),
        "gripper_states": rng.standard_normal(2).astype(np.float64),
    }


# --------------------------------------------------------------- Task 1
def test_dreamer_image_from_record_matches_format_obs_formula():
    from dreamervla.envs.train_env import DreamerVLAOnlineTrainEnv as Env
    from dreamervla.runners.vectorized_collect import dreamer_image_from_record

    rec, size = _rec(), 64
    out = dreamer_image_from_record(rec, size)
    assert out.shape == (6, size, size) and out.dtype == np.uint8
    third = Env._resize_hwc_uint8(rec["agentview_rgb"], size).transpose(2, 0, 1)
    wrist = Env._resize_hwc_uint8(rec["eye_in_hand_rgb"], size).transpose(2, 0, 1)
    expected = np.concatenate([third, wrist], axis=0).astype(np.uint8)
    np.testing.assert_array_equal(out, expected)


# --------------------------------------------------------------- Task 2
def test_build_transition_has_replay_keys_and_dtypes():
    from dreamervla.runners.online_cotrain_runner import build_cotrain_replay_transition
    from dreamervla.runners.vectorized_collect import (
        dreamer_image_from_record,
        proprio_from_record,
    )

    rec = _full_record()
    emb = np.arange(229376, dtype=np.float32)
    wm = np.arange(7, dtype=np.float32)
    lang_emb = np.arange(6, dtype=np.float32)
    tr = build_cotrain_replay_transition(
        rec, emb, wm, reward=1.5, terminated=True, truncated=False,
        task_id=3, task_description="pick up the bowl", step=4, is_first=False, image_size=64,
        lang_emb=lang_emb,
    )
    # keys OnlineReplay.sample / add_episode require
    for k in (
        "image", "obs_embedding", "reward", "done", "is_terminal", "is_last",
        "wm_action", "task_id",
    ):
        assert k in tr
    np.testing.assert_array_equal(tr["image"], dreamer_image_from_record(rec, 64))
    np.testing.assert_array_equal(tr["state"], proprio_from_record(rec))
    np.testing.assert_array_equal(tr["proprio"], proprio_from_record(rec))
    np.testing.assert_array_equal(tr["lang_emb"], lang_emb)
    np.testing.assert_array_equal(tr["wm_action"], wm)
    assert tr["reward"].dtype == np.float32 and float(tr["reward"]) == 1.5
    assert float(tr["done"]) == 1.0 and bool(tr["is_terminal"]) is True
    assert float(tr["discount"]) == 0.0
    assert tr["task_id"] == 3 and tr["task_description"] == "pick up the bowl"
    assert tr["obs_embedding"].dtype == np.float32


def test_build_transition_truncated_keeps_discount_one():
    from dreamervla.runners.online_cotrain_runner import build_cotrain_replay_transition

    tr = build_cotrain_replay_transition(
        _full_record(), np.zeros(8, np.float32), np.zeros(7, np.float32),
        reward=0.0, terminated=False, truncated=True,
        task_id=0, task_description="t", step=10, is_first=False, image_size=64,
    )
    assert float(tr["done"]) == 1.0
    assert bool(tr["is_terminal"]) is False
    assert float(tr["discount"]) == 1.0


# --------------------------------------------------------------- Task 3
def test_validate_rollout_cfg_rejects_bad_backend():
    from dreamervla.runners.online_cotrain_runner import validate_rollout_cfg

    with pytest.raises(ValueError, match="render_backend"):
        validate_rollout_cfg(num_envs=4, render_backend="vulkan", latent_type="action_hidden")


def test_validate_rollout_cfg_rejects_multienv_backbone_latent():
    from dreamervla.runners.online_cotrain_runner import validate_rollout_cfg

    with pytest.raises(ValueError, match="backbone_latent"):
        validate_rollout_cfg(num_envs=4, render_backend="egl", latent_type="backbone_latent")


def test_validate_rollout_cfg_accepts_singleenv_anything():
    from dreamervla.runners.online_cotrain_runner import validate_rollout_cfg

    validate_rollout_cfg(num_envs=1, render_backend="osmesa", latent_type="backbone_latent")


def test_validate_rollout_cfg_rejects_zero_envs():
    from dreamervla.runners.online_cotrain_runner import validate_rollout_cfg

    with pytest.raises(ValueError, match="num_envs"):
        validate_rollout_cfg(num_envs=0, render_backend="egl", latent_type="action_hidden")


def test_validate_rollout_cfg_requires_render_devices_for_multienv_egl():
    from dreamervla.runners.online_cotrain_runner import validate_rollout_cfg

    with pytest.raises(ValueError, match="render_devices.*osmesa"):
        validate_rollout_cfg(
            num_envs=4,
            render_backend="egl",
            latent_type="action_hidden",
            render_devices=[],
            compute_devices=[0, 1],
        )


def test_validate_rollout_cfg_rejects_egl_render_compute_overlap():
    from dreamervla.runners.online_cotrain_runner import validate_rollout_cfg

    with pytest.raises(ValueError, match="not overlap.*render_backend=osmesa"):
        validate_rollout_cfg(
            num_envs=4,
            render_backend="egl",
            latent_type="action_hidden",
            render_devices=[1, 2],
            compute_devices=[0, 1],
        )


def test_validate_rollout_cfg_accepts_disjoint_egl_render_devices():
    from dreamervla.runners.online_cotrain_runner import validate_rollout_cfg

    validate_rollout_cfg(
        num_envs=4,
        render_backend="egl",
        latent_type="action_hidden",
        render_devices=[2, 3],
        compute_devices=[0, 1],
    )


def test_rollout_progress_metrics_exposes_empty_denominator_and_active_steps():
    from dreamervla.runners.online_cotrain_runner import build_rollout_progress_metrics

    metrics = build_rollout_progress_metrics(
        counters={"n_episodes": 0, "n_success": 0},
        env_step=328,
        num_envs=4,
        episode_horizon=300,
        active_episode_steps=[82, 82, 82, 82],
    )

    assert metrics["rollout/success_rate"] == 0.0
    assert metrics["rollout/success_rate_valid"] == 0.0
    assert metrics["rollout/episodes"] == 0.0
    assert metrics["rollout/successes"] == 0.0
    assert metrics["rollout/env_steps"] == 328.0
    assert metrics["rollout/active_episode_step_min"] == 82.0
    assert metrics["rollout/active_episode_step_mean"] == 82.0
    assert metrics["rollout/active_episode_step_max"] == 82.0
    assert metrics["rollout/episode_progress_max"] == pytest.approx(82.0 / 300.0)


def test_rollout_progress_metrics_marks_success_rate_valid_after_episode():
    from dreamervla.runners.online_cotrain_runner import build_rollout_progress_metrics

    metrics = build_rollout_progress_metrics(
        counters={"n_episodes": 4, "n_success": 1},
        env_step=1200,
        num_envs=4,
        episode_horizon=300,
        active_episode_steps=[0, 0, 0, 0],
    )

    assert metrics["rollout/success_rate"] == 0.25
    assert metrics["rollout/success_rate_valid"] == 1.0
    assert metrics["rollout/recent_success_rate"] == 0.0
    assert metrics["rollout/recent_success_rate_valid"] == 0.0
    assert metrics["rollout/episodes"] == 4.0
    assert metrics["rollout/successes"] == 1.0


def test_rollout_progress_metrics_reports_recent_success_rate():
    from dreamervla.runners.online_cotrain_runner import build_rollout_progress_metrics

    metrics = build_rollout_progress_metrics(
        counters={
            "n_episodes": 4,
            "n_success": 1,
            "recent_success_window": 3,
            "recent_successes": [0, 1, 0, 1],
        },
        env_step=1208,
        num_envs=4,
        episode_horizon=300,
        active_episode_steps=[2, 2, 2, 2],
    )

    assert metrics["rollout/recent_success_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["rollout/recent_success_rate_valid"] == 1.0
    assert "rollout/current_success_rate" not in metrics
    assert "rollout/avg_success_rate" not in metrics
    assert "LUMOS/success_rate" not in metrics


# --------------------------------------------------------------- Task 4
class _FakeVec:
    """Stand-in for VecRolloutEnv: canned full_records, done after `horizon` steps."""

    def __init__(self, num_envs, horizon, full_record_fn):
        self.num_envs = num_envs
        self.h = horizon
        self._fr = full_record_fn
        self._step = [0] * num_envs
        self.actions_seen = []

    def set_task(self, task_ids, env_ids=None):
        return [f"task {t}" for t in task_ids]

    def reset(self, task_ids, episode_ids, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        for k in ids:
            self._step[k] = 0
        return [self._fr() for _ in ids]

    def step(self, actions, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        self.actions_seen.extend(np.asarray(action, dtype=np.float32).copy() for action in actions)
        out = []
        for k in ids:
            self._step[k] += 1
            term = self._step[k] >= self.h
            out.append((1.0, term, False, {}, self._fr()))
        return out


class _FakeExtractor:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def step(self, obs, desc):
        import torch

        return ([np.zeros(7, np.float32)], torch.zeros(8))


class _FakeWorldModel:
    """Minimal WM stub for the actor-driven rollout: carries the action_hidden as the
    'latent' and returns an actor-input feature; the actor produces the env action."""

    def __call__(self, batch):
        import torch

        mode = batch["mode"]
        if mode in ("encode_latent", "observe_next"):
            return {"latent": batch["hidden"]}
        if mode == "actor_input":
            return torch.zeros(1, 6)
        raise AssertionError(mode)


class _FakePolicy:
    def __call__(self, batch):
        import torch

        # sample mode -> (action_chunk[1, chunk, 7], logp, extra)
        first = torch.full((1, 1, 7), 0.25)
        second = torch.full((1, 1, 7), 0.75)
        return torch.cat([first, second], dim=1), None, None


def _make_min_runner():
    import torch

    from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner

    r = OnlineCotrainRunner.__new__(OnlineCotrainRunner)
    r.console_progress = lambda *a, **k: None
    r.console_record_success = lambda *a, **k: None
    # The rollout is now driven by the trained actor (WM latent -> policy sample), so the
    # runner needs a world_model + policy + device (was: frozen OFT base via the extractor).
    r.world_model = _FakeWorldModel()
    r.policy = _FakePolicy()
    r.device = torch.device("cpu")
    return r


def test_vectorized_rollout_isolated_queues_and_episode_grouping():
    captured = []

    class _Replay:
        sequence_length = 2

        def add_episode(self, ep, *, source="online"):
            assert source == "online"
            captured.append(list(ep))
            return None  # mimic too-short / no-success-record path

    runner = _make_min_runner()
    vec = _FakeVec(num_envs=2, horizon=3, full_record_fn=_full_record)
    extractors = [_FakeExtractor(), _FakeExtractor()]
    runner._vectorized_cotrain_rollout(
        vec=vec, extractors=extractors, replay=_Replay(),
        num_envs=2, total_env_steps=12, episode_horizon=3,
        action_steps=1, image_size=64, task_ids=[0, 1],
    )
    # 12 env-steps / (2 envs * 3 horizon) = 2 episodes per slot = 4 episodes
    assert len(captured) == 4
    for ep in captured:
        assert len(ep) == 3
        assert {
            "image", "obs_embedding", "wm_action", "reward", "done", "is_terminal",
        } <= set(ep[0])
        assert float(ep[-1]["is_terminal"]) == 1.0
        assert all(float(s["is_terminal"]) == 0.0 for s in ep[:-1])
    # extractor.reset on every slot start (initial + each refill): 1 initial +
    # 2 refills (after each of the 2 finished episodes) = 3 per slot.
    assert extractors[0].resets == 3 and extractors[1].resets == 3
    assert vec.actions_seen
    assert len(vec.actions_seen) == 12
    for slot in range(2):
        slot_actions = vec.actions_seen[slot::2]
        assert len(slot_actions) == 6
        expected_scalars = [0.25, 0.75, 0.25, 0.25, 0.75, 0.25]
        for action, expected in zip(slot_actions, expected_scalars, strict=True):
            np.testing.assert_array_equal(action, np.full(7, expected, dtype=np.float32))


def test_vectorized_rollout_train_hook_can_stop():
    class _Replay:
        sequence_length = 2

        def add_episode(self, ep, *, source="online"):
            assert source == "online"
            return None

    runner = _make_min_runner()
    vec = _FakeVec(num_envs=2, horizon=3, full_record_fn=_full_record)
    extractors = [_FakeExtractor(), _FakeExtractor()]
    calls = {"n": 0}

    def hook(env_step, active_episode_steps=None, *, episode_added=False):
        assert active_episode_steps is not None
        assert episode_added is False
        calls["n"] += 1
        return calls["n"] >= 3  # stop after 3 env-steps

    runner._vectorized_cotrain_rollout(
        vec=vec, extractors=extractors, replay=_Replay(),
        num_envs=2, total_env_steps=1000, episode_horizon=3,
        action_steps=1, image_size=64, task_ids=[0], train_hook=hook,
    )
    assert calls["n"] == 3  # returned promptly on the stop signal


def test_vectorized_rollout_train_hook_runs_with_grad_enabled():
    """Rollout is ``@torch.no_grad()``, but the training burst (the hook) must run
    with grad ENABLED so the WM/classifier/RL backward builds a graph. Regression
    for the no_grad-wrapped-burst bug (loss did not require grad)."""
    import torch

    class _Replay:
        sequence_length = 2

        def add_episode(self, ep, *, source="online"):
            assert source == "online"
            return None

    runner = _make_min_runner()
    vec = _FakeVec(num_envs=2, horizon=3, full_record_fn=_full_record)
    extractors = [_FakeExtractor(), _FakeExtractor()]
    seen = {"grad": None}

    def hook(env_step, active_episode_steps=None, *, episode_added=False):
        assert active_episode_steps is not None
        assert episode_added is False
        seen["grad"] = torch.is_grad_enabled()
        return True  # stop immediately

    runner._vectorized_cotrain_rollout(
        vec=vec, extractors=extractors, replay=_Replay(),
        num_envs=2, total_env_steps=1000, episode_horizon=3,
        action_steps=1, image_size=64, task_ids=[0], train_hook=hook,
    )
    assert seen["grad"] is True


def test_vectorized_rollout_marks_train_hook_when_full_episode_enters_replay():
    class _Replay:
        sequence_length = 2

        def add_episode(self, ep, *, source="online"):
            assert source == "online"
            return {"success": bool(ep[-1]["reward"] > 0.0)}

    runner = _make_min_runner()
    vec = _FakeVec(num_envs=2, horizon=3, full_record_fn=_full_record)
    extractors = [_FakeExtractor(), _FakeExtractor()]
    events = []

    def hook(env_step, active_episode_steps=None, *, episode_added=False):
        assert active_episode_steps is not None
        events.append((int(env_step), bool(episode_added)))
        return False

    runner._vectorized_cotrain_rollout(
        vec=vec, extractors=extractors, replay=_Replay(),
        num_envs=2, total_env_steps=6, episode_horizon=3,
        action_steps=1, image_size=64, task_ids=[0], train_hook=hook,
    )

    assert len(events) == 6
    assert [flag for _step, flag in events].count(True) == 2
    assert events[-2:] == [(5, True), (6, True)]
