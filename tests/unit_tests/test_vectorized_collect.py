"""Tests for the continuous-stepping vectorized collection loop.

``collect_vectorized`` drives K env slots through a finite (task, episode) work-list:
each tick it gathers K observations, runs ONE batched VLA forward, scatters one action
per slot, steps all active slots in parallel, accumulates per-slot trajectories, and on
a slot's done finalizes+writes that demo and refills the slot with the next episode.

Batching constraint: ``batched_forward`` needs all preps to share a prompt length, so the
loop batches PER TASK (all active slots run the same task).  ``fake_infer`` below asserts
the same-length invariant, so a regression that mixes tasks in one batch fails loudly.

These tests use in-process fakes (no model, no LIBERO) for speed and determinism.
"""

from __future__ import annotations

import numpy as np
import torch

from dreamervla.runners.vectorized_collect import collect_vectorized

# ── fakes ─────────────────────────────────────────────────────────────────────


def _fake_full_record(task_id: int, episode_id: int, t: int) -> dict:
    return {
        "agentview_rgb": np.full((4, 4, 3), t % 255, np.uint8),
        "eye_in_hand_rgb": np.full((4, 4, 3), (t + 1) % 255, np.uint8),
        "ee_pos": np.array([task_id, episode_id, t], np.float64),
        "ee_ori": np.zeros(3, np.float64),
        "ee_states": np.zeros(6, np.float64),
        "gripper_states": np.zeros(2, np.float64),
        "joint_states": np.zeros(7, np.float64),
        "robot_states": np.zeros(9, np.float64),
        "states": np.zeros(11, np.float64),
    }


class FakeVecEnv:
    """In-process stand-in for VecRolloutEnv with the same scatter/gather API."""

    def __init__(self, num_envs: int, term_at_fn):
        self.num_envs = num_envs
        self._term_at = term_at_fn  # (task_id, episode_id) -> step count at which it terminates
        self._slots = [dict(task_id=-1, episode_id=-1, t=0, term_at=10**9) for _ in range(num_envs)]
        self.step_calls = 0
        self.received_actions = []

    def set_task(self, task_ids, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        out = []
        for eid, tid in zip(ids, task_ids, strict=True):
            self._slots[eid]["task_id"] = int(tid)
            out.append(f"task{int(tid)}")
        return out

    def reset(self, task_ids, episode_ids, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        out = []
        for eid, tid, ep in zip(ids, task_ids, episode_ids, strict=True):
            self._slots[eid].update(
                task_id=int(tid), episode_id=int(ep), t=0, term_at=self._term_at(int(tid), int(ep))
            )
            out.append(_fake_full_record(int(tid), int(ep), 0))
        return out

    def step(self, actions, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        self.step_calls += 1
        out = []
        for eid, a in zip(ids, actions, strict=True):
            self.received_actions.append(np.asarray(a, dtype=np.float64).copy())
            s = self._slots[eid]
            s["t"] += 1
            terminated = s["t"] >= s["term_at"]
            info = {"wm_action": np.asarray(a, np.float64).reshape(-1), "success": bool(terminated)}
            rec = _fake_full_record(s["task_id"], s["episode_id"], s["t"])
            out.append((0.0, terminated, False, info, rec))
        return out

    def close(self):
        pass


class FakeExtractor:
    def __init__(self, base_len: int = 10):
        self.base_len = base_len
        self.resets = 0

    def reset(self):
        self.resets += 1

    def prepare(self, obs, task_description):
        # Different tasks -> different prompt length, so mixed-task batches are exercised.
        tid = int(task_description.replace("task", "")) if task_description.startswith("task") else 0
        seq = self.base_len + tid
        return {
            "input_ids": torch.zeros(1, seq, dtype=torch.long),
            "attention_mask": torch.ones(1, seq, dtype=torch.long),
            "pixel_values": None,
            "proprio": None,
        }


def fake_infer(preps):
    """Deterministic batched inference (mixed prompt lengths allowed; left-pad handles them)."""
    out = []
    for i in range(len(preps)):
        action_chunk = [np.full(7, float(i), np.float64)]  # one action per slot
        flat_hidden = torch.zeros(229376, dtype=torch.float16)
        out.append((action_chunk, flat_hidden))
    return out


class FakeWriter:
    def __init__(self):
        self.demos = []
        self.config_writes = 0
        self.attr_writes = 0

    def write_demo(self, index, steps, preprocess_config=None, data_attrs=None, **kwargs):
        if preprocess_config is not None:
            self.config_writes += 1
        if data_attrs is not None:
            self.attr_writes += 1
        first = steps[0]["obs"]["ee_pos"]
        self.demos.append(
            dict(
                index=index,
                n_steps=len(steps),
                last_done=int(steps[-1]["dones"]),
                last_sparse=int(steps[-1]["sparse_rewards"]),
                task=int(first[0]),
                episode=int(first[1]),
                first_t=int(first[2]),
            )
        )


# ── tests ─────────────────────────────────────────────────────────────────────


def test_covers_all_work_with_correct_episode_lengths_and_success():
    K = 3
    work_list = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    horizon = 5
    term_at = {(0, 0): 3, (0, 1): 4, (0, 2): 10, (1, 0): 2, (1, 1): 6}
    vec = FakeVecEnv(K, lambda t, e: term_at[(t, e)])
    exts = [FakeExtractor() for _ in range(K)]
    w = FakeWriter()

    n = collect_vectorized(vec, exts, fake_infer, w, work_list, episode_horizon=horizon)

    assert n == len(work_list) == 5
    assert len(w.demos) == 5
    # coverage: exactly the work-list (order-independent)
    assert sorted((d["task"], d["episode"]) for d in w.demos) == sorted(work_list)

    by_te = {(d["task"], d["episode"]): d for d in w.demos}
    # episode length = min(term_at, horizon)
    assert by_te[(0, 0)]["n_steps"] == 3
    assert by_te[(0, 1)]["n_steps"] == 4
    assert by_te[(0, 2)]["n_steps"] == 5  # horizon-truncated
    assert by_te[(1, 0)]["n_steps"] == 2
    assert by_te[(1, 1)]["n_steps"] == 5  # horizon-truncated
    # success (terminated within horizon) -> sparse_rewards[-1]=1; truncated -> 0
    assert by_te[(0, 0)]["last_sparse"] == 1
    assert by_te[(0, 2)]["last_sparse"] == 0
    assert by_te[(1, 1)]["last_sparse"] == 0
    # every demo ends with dones=1, and step 0 records the post-reset frame (t=0)
    assert all(d["last_done"] == 1 for d in w.demos)
    assert all(d["first_t"] == 0 for d in w.demos)


def test_preprocess_config_and_data_attrs_written_once_on_first_demo():
    K = 2
    work_list = [(0, 0), (0, 1), (1, 0)]
    vec = FakeVecEnv(K, lambda t, e: 2)
    exts = [FakeExtractor() for _ in range(K)]
    w = FakeWriter()

    collect_vectorized(
        vec, exts, fake_infer, w, work_list, episode_horizon=5,
        preprocess_config={"k": "v"}, data_attrs={"a": "b"},
    )
    assert w.config_writes == 1, "preprocess_config must be written exactly once"
    assert w.attr_writes == 1, "data_attrs must be written exactly once"


def test_no_preprocess_config_when_none():
    """Non-rank-0 callers pass preprocess_config=None -> never written."""
    K = 2
    vec = FakeVecEnv(K, lambda t, e: 2)
    exts = [FakeExtractor() for _ in range(K)]
    w = FakeWriter()
    collect_vectorized(vec, exts, fake_infer, w, [(0, 0), (0, 1)], episode_horizon=5)
    assert w.config_writes == 0 and w.attr_writes == 0


def test_more_slots_than_episodes_runs_only_needed_slots():
    """K=4 but a task has 2 episodes: only 2 slots used, both episodes collected."""
    K = 4
    work_list = [(0, 0), (0, 1)]
    vec = FakeVecEnv(K, lambda t, e: 3)
    exts = [FakeExtractor() for _ in range(K)]
    w = FakeWriter()
    n = collect_vectorized(vec, exts, fake_infer, w, work_list, episode_horizon=5)
    assert n == 2
    assert sorted((d["task"], d["episode"]) for d in w.demos) == [(0, 0), (0, 1)]


def test_continuous_loop_produces_mixed_task_batches():
    """At a task boundary, slots run different tasks in one batch (different prompt lengths).

    Proves the per-task barrier is gone: a slot that exhausts task 0 starts task 1 while
    another slot is still finishing task 0, so a mixed-length batch reaches infer_fn.
    """
    K = 2
    work_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    term_at = {(0, 0): 2, (0, 1): 4, (1, 0): 3, (1, 1): 3}
    vec = FakeVecEnv(K, lambda t, e: term_at[(t, e)])
    exts = [FakeExtractor() for _ in range(K)]

    saw_mixed = {"v": False}

    def recording_infer(preps):
        if len({p["input_ids"].shape[-1] for p in preps}) > 1:
            saw_mixed["v"] = True
        return fake_infer(preps)

    w = FakeWriter()
    n = collect_vectorized(vec, exts, recording_infer, w, work_list, episode_horizon=10)
    assert n == 4
    assert sorted((d["task"], d["episode"]) for d in w.demos) == sorted(work_list)
    assert saw_mixed["v"], "continuous loop should produce at least one mixed-task batch"


def test_records_pair_pre_step_state_with_its_embedding():
    """Step record i must hold the state at tick i (pre-step), matching _run_episode."""
    K = 1
    vec = FakeVecEnv(K, lambda t, e: 4)
    exts = [FakeExtractor() for _ in range(K)]

    captured = {}

    class CaptureWriter(FakeWriter):
        def write_demo(self, index, steps, preprocess_config=None, data_attrs=None, **kwargs):
            captured["steps"] = steps
            super().write_demo(index, steps, preprocess_config, data_attrs, **kwargs)

    collect_vectorized(vec, exts, fake_infer, CaptureWriter(), [(0, 0)], episode_horizon=10)
    steps = captured["steps"]
    # term_at=4 -> 4 steps; record t holds ee_pos[2]==t (the pre-step frame)
    assert [int(s["obs"]["ee_pos"][2]) for s in steps] == [0, 1, 2, 3]


def test_executes_action_chunk_open_loop_before_using_next_chunk():
    """OpenVLA-OFT rollout must execute a full chunk before using a new chunk."""
    vec = FakeVecEnv(num_envs=1, term_at_fn=lambda _t, _e: 5)
    exts = [FakeExtractor()]
    writer = FakeWriter()
    call = {"idx": 0}

    def infer_chunks(preps):
        base = call["idx"] * 10
        call["idx"] += 1
        return [
            (
                [np.array([base + j, 0, 0, 0, 0, 0, 0.9], np.float64) for j in range(3)],
                torch.zeros(229376, dtype=torch.float16),
            )
            for _ in preps
        ]

    collect_vectorized(
        vec,
        exts,
        infer_chunks,
        writer,
        [(0, 0)],
        episode_horizon=5,
        action_steps=3,
    )

    assert [int(action[0]) for action in vec.received_actions] == [0, 1, 2, 30, 31]
