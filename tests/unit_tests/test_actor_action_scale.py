from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from dreamervla.algorithms.dreamervla import _actor_action_for_world_model
from dreamervla.algorithms.ppo.outcome import dino_lumos_step
from dreamervla.envs.libero.libero_env import unnormalize_libero_action


def test_env_and_world_model_action_scale_mapping_match():
    actor_action = torch.tensor([[0.2, -0.4, 0.0, 0.6, -0.8, 0.1, 1.0]])
    cfg = OmegaConf.create({"rssm_action_scale": "env", "rssm_action_clip": True})

    wm_action = _actor_action_for_world_model(actor_action, cfg)
    env_action = unnormalize_libero_action(actor_action.squeeze(0).numpy())

    np.testing.assert_allclose(wm_action.squeeze(0).numpy(), env_action, atol=1e-6)


class _RecordingChunkWM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actions_seen: list[torch.Tensor] = []

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
            actions = batch["actions"]
            self.actions_seen.append(actions.detach().clone())
            hidden_seq = hidden.unsqueeze(1).repeat(1, actions.shape[1], 1)
            return {
                "hidden_seq": hidden_seq,
                "history": hidden,
                "actions": actions,
                "hidden": hidden,
            }
        raise ValueError(f"Unknown mode: {mode}")


class _FixedChunkPolicy(torch.nn.Module):
    def __init__(self, action: torch.Tensor):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("fixed_action", action.float())

    def forward(self, batch):
        hidden = batch["hidden"]
        batch_size = int(hidden.shape[0])
        chunk = self.fixed_action.to(hidden.device).unsqueeze(0).repeat(batch_size, 1, 1)
        chunk = chunk + self.bias * 0.0
        if batch["mode"] == "sample":
            return chunk, torch.zeros(batch_size, device=hidden.device), {"action_chunk": chunk}
        if batch["mode"] == "evaluate":
            action = batch["action"]
            target = chunk if action.ndim == 3 else chunk[:, 0, :]
            log_prob = -((action - target) ** 2).reshape(batch_size, -1).sum(dim=-1)
            return log_prob, torch.zeros_like(log_prob), {"action_chunk": chunk}
        raise ValueError(batch["mode"])


class _AlwaysFailClassifier(torch.nn.Module):
    def predict_success(self, video, threshold, stride=1, min_steps=1):
        del threshold, stride, min_steps
        batch_size = int(video.shape[0])
        return {
            "complete": torch.zeros(batch_size, dtype=torch.bool, device=video.device),
            "finish_step": torch.ones(batch_size, dtype=torch.long, device=video.device),
        }


def test_lumos_imagination_feeds_world_model_env_scale_actions():
    actor_chunk = torch.tensor(
        [
            [0.2, -0.4, 0.0, 0.6, -0.8, 0.1, 1.0],
            [-0.5, 0.3, 0.7, -0.2, 0.4, -1.0, -0.6],
        ]
    )
    world_model = _RecordingChunkWM()
    policy = _FixedChunkPolicy(actor_chunk)
    cfg = OmegaConf.create(
        {
            "lumos": {
                "chunk_size": 2,
                "episode_max_steps": 2,
                "classifier_min_steps": 1,
                "classifier_granularity": "action",
                "filter_zero_variance_groups": True,
            },
            "ppo_rollouts_per_start": 1,
            "ppo_update_epochs": 1,
            "kl_coef": 0.0,
            "actor_bc_to_ref_scale": 0.0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "advantage_eps": 1.0e-6,
            "rssm_action_scale": "env",
            "rssm_action_clip": True,
            "imag_last": 1,
        }
    )

    dino_lumos_step(
        policy=policy,
        chunk_world_model=world_model,
        classifier=_AlwaysFailClassifier(),
        classifier_threshold=0.5,
        actor_optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
        obs={"obs_embedding": torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        algorithm_cfg=cfg,
        optim_cfg=OmegaConf.create(
            {"grad_clip_norm": 10.0, "zero_grad_set_to_none": True}
        ),
        ref_policy=None,
    )

    assert world_model.actions_seen
    expected = _actor_action_for_world_model(actor_chunk.unsqueeze(0), cfg)
    assert torch.allclose(world_model.actions_seen[0], expected, atol=1e-6)
