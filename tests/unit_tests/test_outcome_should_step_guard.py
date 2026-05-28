"""Regression tests for the optimizer-step guard in dino_wmpo_outcome_step.

Pins the contract that the actor optimizer does NOT step when there is no
gradient signal (all-fail batch + zero-variance filter ON + BC disabled),
and that ``metrics["ppo_step_applied"] == 0.0`` exposes the skip to the
caller.  Without the guard, Adam decays its momentum/velocity on the
zero-gradient step and silently drifts the actor in cold-start.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from dreamer_vla.algorithms.ppo.outcome import dino_wmpo_outcome_step


class _TinyChunkWM(torch.nn.Module):
    def forward(self, batch):
        mode = batch["mode"]
        if mode == "observe_sequence":
            return {"latent": batch["obs_embedding"]}
        if mode == "actor_input":
            latent = batch["latent"]
            return latent["hidden"] if isinstance(latent, dict) else latent
        if mode == "predict_next_chunk":
            latent = batch["latent"]
            hidden = latent["hidden"] if isinstance(latent, dict) else latent
            chunk = hidden.unsqueeze(1).repeat(1, batch["actions"].shape[1], 1)
            return {
                "hidden_seq": chunk,
                "history": hidden,
                "actions": batch["actions"],
                "hidden": hidden,
            }
        raise ValueError(f"Unknown mode: {mode}")


class _AlwaysFailClassifier(torch.nn.Module):
    def predict_success(self, video, threshold, stride=1, min_steps=1):
        del threshold, stride, min_steps
        batch = video.shape[0]
        # complete=False, finish_step pinned to T_max-1 (= num_chunks*K - 1).
        # T_max=4, K=2 in the test config below ⇒ finish_step=3.
        return {
            "complete": torch.zeros(batch, dtype=torch.bool, device=video.device),
            "finish_step": torch.full(
                (batch,), 3, dtype=torch.long, device=video.device,
            ),
        }


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Adam needs at least one prior optimizer step with non-zero grad to
        # accumulate state; we use SGD so any failure to skip step() is
        # immediately visible as a parameter delta.
        self.action_value = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, batch):
        hidden = batch["hidden"]
        batch_size = int(hidden.shape[0])
        action_chunk = self.action_value.expand(batch_size, 2, 1)
        if batch["mode"] == "sample":
            log_prob = torch.zeros(batch_size, device=hidden.device)
            if bool(batch.get("return_chunk", False)):
                return action_chunk, log_prob, {"action_chunk": action_chunk}
            first = action_chunk[:, 0, :]
            return first, log_prob, {"action_chunk": action_chunk}
        if batch["mode"] == "evaluate":
            action = batch["action"]
            target = action_chunk if action.ndim == 3 else action_chunk[:, 0, :]
            log_prob = -((action - target) ** 2).reshape(batch_size, -1).sum(dim=-1)
            entropy = torch.zeros_like(log_prob)
            return log_prob, entropy, {"action_chunk": action_chunk}
        raise ValueError(batch["mode"])


def _base_cfg():
    return OmegaConf.create({
        "wmpo": {
            "chunk_size": 2,
            "episode_max_steps": 4,
            "classifier_min_steps": 1,
            "filter_zero_variance_groups": True,
        },
        "ppo_rollouts_per_start": 1,
        "ppo_update_epochs": 1,
        "kl_coef": 0.0,
        "actor_bc_to_ref_scale": 0.0,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.28,
        "advantage_eps": 1.0e-6,
    })


def _optim_cfg():
    return OmegaConf.create({"grad_clip_norm": 10.0, "zero_grad_set_to_none": True})


def test_outcome_step_skips_optimizer_when_all_groups_zero_variance():
    policy = _TinyPolicy()
    before = float(policy.action_value.detach())
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    metrics = dino_wmpo_outcome_step(
        policy=policy,
        chunk_world_model=_TinyChunkWM(),
        classifier=_AlwaysFailClassifier(),
        classifier_threshold=0.5,
        actor_optimizer=optimizer,
        obs={"obs_embedding": torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        algorithm_cfg=_base_cfg(),
        optim_cfg=_optim_cfg(),
        ref_policy=None,
    )
    after = float(policy.action_value.detach())

    assert metrics["ppo_step_applied"] == 0.0, (
        "Expected optimizer to skip the step when all groups are zero-variance "
        "and BC is disabled — got ppo_step_applied=1.0."
    )
    assert after == before, (
        f"Actor parameter changed despite zero gradient signal: {before} → {after}"
    )


def test_outcome_step_reports_finite_finish_step_when_no_rollouts_complete():
    """``wmpo/mean_finish_step`` must be finite (no NaN) to survive
    ``reduce_mean_dict``'s all_reduce(SUM) across DDP ranks."""
    policy = _TinyPolicy()
    metrics = dino_wmpo_outcome_step(
        policy=policy,
        chunk_world_model=_TinyChunkWM(),
        classifier=_AlwaysFailClassifier(),
        classifier_threshold=0.5,
        actor_optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
        obs={"obs_embedding": torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        algorithm_cfg=_base_cfg(),
        optim_cfg=_optim_cfg(),
        ref_policy=None,
    )

    finish = metrics["wmpo/mean_finish_step"]
    import math
    assert math.isfinite(finish), f"mean_finish_step must be finite, got {finish}"
    assert finish == -1.0, (
        f"Expected sentinel -1.0 when no rollouts complete, got {finish}"
    )
    assert metrics["wmpo/success_rate"] == 0.0


def test_outcome_step_reports_real_ratio_stats_in_dict():
    """``ppo_ratio_*`` / ``ppo_clipfrac`` must appear in the returned dict
    even when ``ppo_update_epochs=1`` (ratio≡1) — they used to be silent
    1.0/0.0 stubs synthesized by the workspace via ``.get(..., 1.0)``."""
    policy = _TinyPolicy()
    metrics = dino_wmpo_outcome_step(
        policy=policy,
        chunk_world_model=_TinyChunkWM(),
        classifier=_AlwaysFailClassifier(),
        classifier_threshold=0.5,
        actor_optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
        obs={"obs_embedding": torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        algorithm_cfg=_base_cfg(),
        optim_cfg=_optim_cfg(),
        ref_policy=None,
    )

    for key in ("ppo_ratio_mean", "ppo_ratio_min", "ppo_ratio_max", "ppo_clipfrac"):
        assert key in metrics, f"Missing PPO ratio diagnostic: {key!r}"
