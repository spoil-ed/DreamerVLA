from __future__ import annotations

import torch
from omegaconf import OmegaConf

from dreamervla.algorithms.ppo.outcome import dino_lumos_step


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
            hidden_seq = hidden.unsqueeze(1).repeat(1, batch["actions"].shape[1], 1)
            return {
                "hidden_seq": hidden_seq,
                "history": hidden,
                "actions": batch["actions"],
                "hidden": hidden,
            }
        raise ValueError(f"Unknown mode: {mode}")


class _DiscriminativeClassifier(torch.nn.Module):
    """One success and one failure in each GRPO group of size 2."""

    def predict_success(self, video, threshold, stride=1, min_steps=1):
        del threshold, stride, min_steps
        batch = int(video.shape[0])
        idx = torch.arange(batch, device=video.device)
        return {
            "complete": (idx % 2 == 0),
            "finish_step": torch.zeros(batch, dtype=torch.long, device=video.device),
        }


class _DegenerateClassifier(torch.nn.Module):
    def predict_success(self, video, threshold, stride=1, min_steps=1):
        del threshold, stride, min_steps
        batch = int(video.shape[0])
        return {
            "complete": torch.zeros(batch, dtype=torch.bool, device=video.device),
            "finish_step": torch.full(
                (batch,),
                3,
                dtype=torch.long,
                device=video.device,
            ),
        }


class _ProbabilityClassifier(torch.nn.Module):
    """No threshold success, but verifier probability varies within each group."""

    def predict_success(self, video, threshold, stride=1, min_steps=1):
        del threshold, stride, min_steps
        batch = int(video.shape[0])
        idx = torch.arange(batch, device=video.device)
        score = torch.where(
            idx % 2 == 0,
            torch.full((batch,), 0.8, device=video.device),
            torch.full((batch,), 0.2, device=video.device),
        )
        return {
            "complete": torch.zeros(batch, dtype=torch.bool, device=video.device),
            "finish_step": torch.full(
                (batch,),
                3,
                dtype=torch.long,
                device=video.device,
            ),
            "score": score,
            "score_step": torch.full(
                (batch,),
                3,
                dtype=torch.long,
                device=video.device,
            ),
        }


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_value = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, batch):
        hidden = batch["hidden"]
        batch_size = int(hidden.shape[0])
        mean_chunk = self.action_value.expand(batch_size, 2, 1)
        if batch["mode"] == "sample":
            parity = torch.arange(batch_size, device=hidden.device) % 2
            offset = torch.where(
                parity == 0,
                torch.full_like(parity, -0.05, dtype=mean_chunk.dtype),
                torch.full_like(parity, 0.15, dtype=mean_chunk.dtype),
            ).reshape(batch_size, 1, 1)
            action_chunk = mean_chunk + offset
            return (
                action_chunk,
                torch.zeros(batch_size, device=hidden.device),
                {"action_chunk": action_chunk},
            )
        if batch["mode"] == "evaluate":
            action = batch["action"]
            target = mean_chunk if action.ndim == 3 else mean_chunk[:, 0, :]
            log_prob = -((action - target) ** 2).reshape(batch_size, -1).sum(dim=-1)
            return log_prob, torch.zeros_like(log_prob), {"action_chunk": mean_chunk}
        raise ValueError(batch["mode"])


class _ActionScoringRefPolicy(torch.nn.Module):
    """Reference policy whose log-prob varies with the sampled action.

    Used to prove that KL-derived return variance must not make an all-fail
    classifier group eligible for PPO; the adaptive keep/skip decision is based
    on classifier score/outcome signal, not KL alone.
    """

    def forward(self, batch):
        hidden = batch["hidden"]
        batch_size = int(hidden.shape[0])
        if batch["mode"] == "evaluate":
            action = batch["action"]
            log_prob = -(action.float().reshape(batch_size, -1).square().sum(dim=-1))
            return log_prob, torch.zeros_like(log_prob), {}
        if batch["mode"] == "sample":
            action_chunk = torch.zeros(
                batch_size,
                2,
                1,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            return (
                action_chunk,
                torch.zeros(batch_size, dtype=hidden.dtype, device=hidden.device),
                {"action_chunk": action_chunk},
            )
        raise ValueError(batch["mode"])


def _cfg(reward_model: str = "sparse_outcome"):
    return OmegaConf.create(
        {
            "lumos": {
                "chunk_size": 2,
                "episode_max_steps": 4,
                "reward_model": reward_model,
                "classifier_min_steps": 1,
                "classifier_granularity": "action",
                "filter_zero_variance_groups": True,
            },
            "ppo_rollouts_per_start": 2,
            "ppo_update_epochs": 1,
            "kl_coef": 0.0,
            "actor_bc_to_ref_scale": 0.0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "advantage_eps": 1.0e-6,
            "imag_last": 1,
        }
    )


def _run_step(classifier: torch.nn.Module, *, reward_model: str = "sparse_outcome"):
    policy = _TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    metrics = dino_lumos_step(
        policy=policy,
        chunk_world_model=_TinyChunkWM(),
        classifier=classifier,
        classifier_threshold=0.5,
        actor_optimizer=optimizer,
        obs={"obs_embedding": torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        algorithm_cfg=_cfg(reward_model=reward_model),
        optim_cfg=OmegaConf.create(
            {"grad_clip_norm": 10.0, "zero_grad_set_to_none": True}
        ),
        ref_policy=None,
    )
    return policy, metrics


def test_discriminative_classifier_gives_nonzero_actor_gradient():
    policy, metrics = _run_step(_DiscriminativeClassifier())

    assert abs(metrics["actor_loss"]) > 0.0
    assert metrics["actor_grad_norm"] > 0.0
    assert metrics["ppo_step_applied"] == 1.0
    assert 0.0 < metrics["returns_mean"] < 1.0
    assert policy.action_value.grad is not None
    assert policy.action_value.grad.abs().item() > 0.0


def test_degenerate_classifier_gives_zero_signal():
    policy, metrics = _run_step(_DegenerateClassifier())

    assert metrics["actor_loss"] == 0.0
    assert metrics["actor_grad_norm"] == 0.0
    assert metrics["ppo_step_applied"] == 0.0
    assert metrics["returns_mean"] == 0.0
    assert metrics["LUMOS/skipped_zero_variance_groups"] == 1.0
    assert policy.action_value.grad is None or policy.action_value.grad.abs().item() == 0.0


def test_probability_reward_updates_when_threshold_outcomes_are_constant():
    policy, metrics = _run_step(
        _ProbabilityClassifier(),
        reward_model="probability_outcome",
    )

    assert abs(metrics["actor_loss"]) > 0.0
    assert metrics["actor_grad_norm"] > 0.0
    assert metrics["ppo_step_applied"] == 1.0
    assert metrics["LUMOS/success_rate"] == 0.0
    assert metrics["returns_std"] > 0.0
    assert metrics["advantage_std"] > 0.0
    assert metrics["LUMOS/group_var_keep_frac"] == 1.0
    assert policy.action_value.grad is not None
    assert policy.action_value.grad.abs().item() > 0.0


def test_lumos_rollout_bounds_use_configured_max_group_size_with_adaptive_prefix():
    cfg = _cfg()
    cfg.lumos.ppo_rollouts_per_start_min = 2
    cfg.lumos.ppo_rollouts_per_start_max = 4
    policy = _TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    metrics = dino_lumos_step(
        policy=policy,
        chunk_world_model=_TinyChunkWM(),
        classifier=_DiscriminativeClassifier(),
        classifier_threshold=0.5,
        actor_optimizer=optimizer,
        obs={"obs_embedding": torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        algorithm_cfg=cfg,
        optim_cfg=OmegaConf.create(
            {"grad_clip_norm": 10.0, "zero_grad_set_to_none": True}
        ),
        ref_policy=None,
    )

    assert metrics["LUMOS/group_size_min"] == 2.0
    assert metrics["LUMOS/group_size_max"] == 4.0
    assert metrics["LUMOS/group_size"] == 4.0
    assert metrics["LUMOS/effective_group_size_mean"] == 2.0
    assert metrics["LUMOS/skipped_zero_variance_groups"] == 0.0


def test_zero_classifier_variance_skips_even_when_ref_kl_varies():
    cfg = _cfg()
    cfg.lumos.ppo_rollouts_per_start_min = 2
    cfg.lumos.ppo_rollouts_per_start_max = 2
    cfg.kl_coef = 1.0
    policy = _TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    metrics = dino_lumos_step(
        policy=policy,
        chunk_world_model=_TinyChunkWM(),
        classifier=_DegenerateClassifier(),
        classifier_threshold=0.5,
        actor_optimizer=optimizer,
        obs={"obs_embedding": torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        algorithm_cfg=cfg,
        optim_cfg=OmegaConf.create(
            {"grad_clip_norm": 10.0, "zero_grad_set_to_none": True}
        ),
        ref_policy=_ActionScoringRefPolicy(),
    )

    assert metrics["LUMOS/skipped_zero_variance_groups"] == 1.0
    assert metrics["LUMOS/group_var_keep_frac"] == 0.0
    assert metrics["ppo_step_applied"] == 0.0


def test_group_var_keep_frac_reports_signal_health_when_filter_disabled():
    cfg = _cfg()
    cfg.lumos.filter_zero_variance_groups = False
    policy = _TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    metrics = dino_lumos_step(
        policy=policy,
        chunk_world_model=_TinyChunkWM(),
        classifier=_DegenerateClassifier(),
        classifier_threshold=0.5,
        actor_optimizer=optimizer,
        obs={"obs_embedding": torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        algorithm_cfg=cfg,
        optim_cfg=OmegaConf.create(
            {"grad_clip_norm": 10.0, "zero_grad_set_to_none": True}
        ),
        ref_policy=None,
    )

    assert metrics["LUMOS/skipped_zero_variance_groups"] == 1.0
    assert metrics["LUMOS/group_var_keep_frac"] == 0.0
