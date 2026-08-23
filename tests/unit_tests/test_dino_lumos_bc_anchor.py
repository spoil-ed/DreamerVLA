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
            chunk = hidden.unsqueeze(1).repeat(1, batch["actions"].shape[1], 1)
            return {
                "hidden_seq": chunk,
                "history": hidden,
                "actions": batch["actions"],
                "hidden": hidden,
            }
        raise ValueError(f"Unknown mode: {mode}")


class _AlwaysSuccessClassifier(torch.nn.Module):
    def predict_success(self, video, threshold, stride=1, min_steps=1):
        del threshold, stride, min_steps
        batch = video.shape[0]
        return {
            "complete": torch.ones(batch, dtype=torch.bool, device=video.device),
            "finish_step": torch.zeros(batch, dtype=torch.long, device=video.device),
        }


class _TinyPolicy(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.action_value = torch.nn.Parameter(torch.tensor(float(value)))

    def _action_chunk(self, hidden):
        del hidden
        return self.action_value.expand(1, 2, 1)

    def forward(self, batch):
        hidden = batch["hidden"]
        action_chunk = self._action_chunk(hidden)
        mean = action_chunk[:, 0, :]
        if batch["mode"] == "sample":
            log_prob = torch.zeros(mean.shape[0], device=mean.device, dtype=mean.dtype)
            if bool(batch.get("return_chunk", False)):
                return action_chunk, log_prob, {"action_chunk": action_chunk}
            return mean, log_prob, {"action_chunk": action_chunk}
        if batch["mode"] == "evaluate":
            action = batch["action"]
            target = action_chunk if action.ndim == 3 else mean
            log_prob = -((action - target) ** 2).reshape(action.shape[0], -1).sum(dim=-1)
            entropy = torch.zeros_like(log_prob)
            return log_prob, entropy, {"action_chunk": action_chunk}
        raise ValueError(batch["mode"])


class _TinyChunkPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_value = torch.nn.Parameter(torch.tensor(0.0))
        self.evaluated_shapes: list[tuple[int, ...]] = []

    def forward(self, batch):
        hidden = batch["hidden"]
        batch_size = int(hidden.shape[0])
        action_chunk = self.action_value.expand(batch_size, 2, 1)
        if batch["mode"] == "sample":
            if bool(batch.get("return_chunk", False)):
                log_prob = torch.zeros(batch_size, device=hidden.device)
                return action_chunk, log_prob, {"action_chunk": action_chunk}
            first = action_chunk[:, 0, :]
            log_prob = torch.zeros(batch_size, device=hidden.device)
            return first, log_prob, {"action_chunk": action_chunk}
        if batch["mode"] == "evaluate":
            action = batch["action"]
            self.evaluated_shapes.append(tuple(action.shape))
            target = action_chunk if action.ndim == 3 else action_chunk[:, 0, :]
            log_prob = -((action - target) ** 2).reshape(batch_size, -1).sum(dim=-1)
            entropy = torch.zeros_like(log_prob)
            return log_prob, entropy, {"action_chunk": action_chunk}
        raise ValueError(batch["mode"])


def test_outcome_step_applies_bc_anchor_to_reference_policy():
    policy = _TinyPolicy(value=1.0)
    ref_policy = _TinyPolicy(value=0.0)
    for param in ref_policy.parameters():
        param.requires_grad = False

    cfg = OmegaConf.create(
        {
            "lumos": {
                "chunk_size": 2,
                "episode_max_steps": 2,
                "classifier_min_steps": 1,
                "filter_zero_variance_groups": True,
            },
            "ppo_rollouts_per_start": 1,
            "ppo_update_epochs": 1,
            "kl_coef": 0.0,
            "actor_bc_to_ref_scale": 1.0,
            "rssm_action_scale": "policy",
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "advantage_eps": 1.0e-6,
        }
    )
    optim_cfg = OmegaConf.create({"grad_clip_norm": 10.0, "zero_grad_set_to_none": True})
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    before = float(policy.action_value.detach())
    metrics = dino_lumos_step(
        policy=policy,
        chunk_world_model=_TinyChunkWM(),
        classifier=_AlwaysSuccessClassifier(),
        classifier_threshold=0.5,
        actor_optimizer=optimizer,
        obs={"obs_embedding": torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        algorithm_cfg=cfg,
        optim_cfg=optim_cfg,
        ref_policy=ref_policy,
    )
    after = float(policy.action_value.detach())

    assert metrics["actor_bc_ref_scale"] == 1.0
    assert metrics["actor_bc_ref_loss"] > 0.0
    assert after < before


def test_outcome_step_optimizes_full_action_chunks_not_only_first_action():
    policy = _TinyChunkPolicy()
    cfg = OmegaConf.create(
        {
            "lumos": {
                "chunk_size": 2,
                "episode_max_steps": 2,
                "classifier_min_steps": 1,
                "filter_zero_variance_groups": False,
            },
            "ppo_rollouts_per_start": 1,
            "ppo_update_epochs": 1,
            "kl_coef": 0.0,
            "actor_bc_to_ref_scale": 0.0,
            "rssm_action_scale": "policy",
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "advantage_eps": 1.0e-6,
        }
    )
    optim_cfg = OmegaConf.create({"grad_clip_norm": 10.0, "zero_grad_set_to_none": True})

    dino_lumos_step(
        policy=policy,
        chunk_world_model=_TinyChunkWM(),
        classifier=_AlwaysSuccessClassifier(),
        classifier_threshold=0.5,
        actor_optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
        obs={"obs_embedding": torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        algorithm_cfg=cfg,
        optim_cfg=optim_cfg,
        ref_policy=None,
    )

    assert policy.evaluated_shapes
    assert all(shape == (1, 2, 1) for shape in policy.evaluated_shapes)
