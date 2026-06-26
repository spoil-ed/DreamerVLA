"""Q6: InferenceWorker.forward_batch must do ONE D2H per output tensor, not per env.

The batched D2H must be numerically identical (atol 0) to the per-row reference,
and `.cpu()` must be called a fixed number of times (twice: actions + obs_embedding)
regardless of the number of envs. CPU-only, no Ray/GPU; stub encoder/WM/policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dreamervla.workers.inference.inference_worker import InferenceWorker

HIDDEN = 5
ACTION_DIM = 7


class _StubEncoder:
    """Return a deterministic [N, HIDDEN] embedding (one row per obs)."""

    def encode_obs_batch(self, obs_batch: list[dict[str, Any]]) -> torch.Tensor:
        rows = [
            torch.full((HIDDEN,), float(obs["seed"]), dtype=torch.float32)
            for obs in obs_batch
        ]
        return torch.stack(rows, dim=0)


class _StubWM(torch.nn.Module):
    def forward(self, batch: dict[str, Any]) -> Any:
        mode = batch["mode"]
        if mode == "encode_latent":
            return batch["hidden"]
        if mode == "observe_next":
            return batch["hidden"]
        if mode == "actor_input":
            return batch["latent"]
        raise ValueError(mode)


class _StubPolicy(torch.nn.Module):
    def forward(self, batch: dict[str, Any]):
        feat = batch["hidden"]  # [N, HIDDEN]
        n = feat.shape[0]
        # action_chunk [N, 1, ACTION_DIM]; first row col 0 derived from feat so per-env distinct
        chunk = torch.arange(n * ACTION_DIM, dtype=torch.float32).reshape(n, 1, ACTION_DIM)
        chunk = chunk + feat[:, :1].reshape(n, 1, 1)
        return chunk, None, None


class _ChunkPolicy(torch.nn.Module):
    def __init__(self, chunk_size: int = 3) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.calls = 0

    def forward(self, batch: dict[str, Any]):
        self.calls += 1
        feat = batch["hidden"]  # [N, HIDDEN]
        rows = []
        for base in feat[:, 0].detach().cpu().tolist():
            chunk = [
                torch.full((ACTION_DIM,), float(base) + float(j), dtype=torch.float32)
                for j in range(self.chunk_size)
            ]
            rows.append(torch.stack(chunk, dim=0))
        return torch.stack(rows, dim=0), None, None


def _make_worker(num_envs: int, *, action_steps: int | None = None) -> InferenceWorker:
    cfg = {"device": "cpu", "encoder": {}, "world_model": {}, "policy": {}}
    if action_steps is not None:
        cfg["action_steps"] = int(action_steps)
    w = InferenceWorker(cfg, {}, num_envs=num_envs)
    w.encoder = _StubEncoder()
    w.world_model = _StubWM()
    w.policy = _StubPolicy()
    w.state = [w._empty_state() for _ in range(num_envs)]
    return w


def test_batched_d2h_matches_per_row_reference() -> None:
    num_envs = 4
    w = _make_worker(num_envs)
    obs = [{"seed": float(i + 1), "is_first": True} for i in range(num_envs)]
    out = w.forward_batch(obs, list(range(num_envs)))

    assert len(out["actions"]) == num_envs
    assert len(out["obs_embedding"]) == num_envs
    for i in range(num_envs):
        assert out["actions"][i].shape == (ACTION_DIM,)
        assert out["actions"][i].dtype == np.float32
        assert out["obs_embedding"][i].shape == (HIDDEN,)
        assert out["obs_embedding"][i].dtype == np.float32
        # obs_embedding row i is the encoder output for obs i (all == seed)
        assert np.array_equal(out["obs_embedding"][i], np.full((HIDDEN,), float(i + 1), np.float32))


def test_forward_batch_calls_cpu_exactly_twice(monkeypatch) -> None:
    """RED driver: per-row `.cpu()` calls it 2*N times; batched calls it 2."""
    num_envs = 4
    w = _make_worker(num_envs)
    obs = [{"seed": float(i + 1), "is_first": True} for i in range(num_envs)]

    real_cpu = torch.Tensor.cpu
    calls = {"n": 0}

    def counting_cpu(self):
        calls["n"] += 1
        return real_cpu(self)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    w.forward_batch(obs, list(range(num_envs)))
    monkeypatch.setattr(torch.Tensor, "cpu", real_cpu)

    # exactly two batch-level D2H transfers (actions + obs_embedding), independent of num_envs
    assert calls["n"] == 2, f"expected 2 .cpu() calls, got {calls['n']}"


def test_forward_batch_executes_full_action_chunk_before_resampling() -> None:
    w = _make_worker(num_envs=1, action_steps=3)
    policy = _ChunkPolicy(chunk_size=3)
    w.policy = policy

    outs = [
        w.forward_batch([{"seed": seed, "is_first": index == 0}], [0])
        for index, seed in enumerate([0, 10, 20, 30])
    ]

    assert [int(out["actions"][0][0]) for out in outs] == [0, 1, 2, 30]
    assert policy.calls == 2


def test_reset_states_drops_pending_action_chunk() -> None:
    w = _make_worker(num_envs=1, action_steps=3)
    policy = _ChunkPolicy(chunk_size=3)
    w.policy = policy

    first = w.forward_batch([{"seed": 0, "is_first": True}], [0])
    w.reset_states([0])
    second = w.forward_batch([{"seed": 10, "is_first": True}], [0])

    assert int(first["actions"][0][0]) == 0
    assert int(second["actions"][0][0]) == 10
    assert policy.calls == 2
