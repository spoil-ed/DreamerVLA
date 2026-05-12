from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch import nn

from src.algorithms.dreamer_vla import (
    normalize_returns_for_actor_critic,
    imagine_actor_critic_step,
    world_model_pretrain_step,
)
from src.models.critic.twohot_critic import ReturnPercentileTracker, TwohotCritic
from src.models.vla_policy import VLAPolicy
from src.models.world_model.semantic_bottleneck import (
    SemanticBottleneckLatent,
    SemanticBottleneckRSSMWorldModel,
    StochasticBottleneck,
)


def test_stochastic_bottleneck_returns_latent_and_kl() -> None:
    bottleneck = StochasticBottleneck(input_dim=12, latent_dim=4)
    z_sem = torch.randn(3, 5, 12)

    out = bottleneck(z_sem)

    assert out.z.shape == (3, 5, 4)
    assert out.mu.shape == (3, 5, 4)
    assert out.logvar.shape == (3, 5, 4)
    assert out.kl.shape == (3, 5)
    assert torch.all(out.kl >= 0)


def test_semantic_bottleneck_world_model_loss_has_no_image_reconstruction() -> None:
    model = SemanticBottleneckRSSMWorldModel(
        sem_dim=12,
        latent_dim=4,
        deter=8,
        action_dim=2,
        hidden=16,
    )
    batch = {
        "obs_embedding": torch.randn(3, 6, 12),
        "actions": torch.randn(3, 6, 2),
        "rewards": torch.zeros(3, 6),
        "dones": torch.zeros(3, 6),
        "is_first": torch.zeros(3, 6, dtype=torch.bool),
    }
    batch["is_first"][:, 0] = True

    out = model(batch)

    assert out["_loss"].ndim == 0
    assert out["rec_loss"].item() == 0.0
    assert out["image_mse"].item() == 0.0
    assert out["bottleneck_kl_loss"].item() >= 0.0
    assert out["dyn_loss"].item() >= 0.0
    assert out["reward_loss"].item() >= 0.0
    assert out["continue_loss"].item() >= 0.0


def test_semantic_bottleneck_world_model_accepts_workspace_hidden_dim_alias() -> None:
    model = SemanticBottleneckRSSMWorldModel(
        hidden_dim=12,
        latent_dim=4,
        deter=8,
        action_dim=2,
        hidden=16,
    )

    assert model.sem_dim == 12


def test_semantic_bottleneck_world_model_actor_adapter_modes() -> None:
    model = SemanticBottleneckRSSMWorldModel(
        sem_dim=12,
        latent_dim=4,
        deter=8,
        action_dim=2,
        hidden=16,
    )
    latent = SemanticBottleneckLatent(
        deter=torch.zeros(5, 8),
        stoch=torch.zeros(5, 4),
        mu=torch.zeros(5, 4),
        logvar=torch.zeros(5, 4),
    )
    actions = torch.randn(5, 2)

    next_latent = model({"mode": "predict_next", "latent": latent, "actions": actions})
    actor_input = model({"mode": "actor_input", "latent": next_latent})
    reward = model({"mode": "reward", "latent": latent, "actions": actions, "next_latent": next_latent})
    cont = model({"mode": "continue", "latent": next_latent})

    assert isinstance(next_latent, SemanticBottleneckLatent)
    assert actor_input.shape == (5, 12)
    assert reward.shape == (5,)
    assert cont.shape == (5,)


def test_world_model_pretrain_step_backprops_through_live_loss() -> None:
    model = SemanticBottleneckRSSMWorldModel(
        sem_dim=12,
        latent_dim=4,
        deter=8,
        action_dim=2,
        hidden=16,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "obs_embedding": torch.randn(3, 6, 12),
        "actions": torch.randn(3, 6, 2),
        "rewards": torch.zeros(3, 6),
        "dones": torch.zeros(3, 6),
        "is_first": torch.zeros(3, 6, dtype=torch.bool),
    }
    batch["is_first"][:, 0] = True
    before = next(model.parameters()).detach().clone()

    metrics = world_model_pretrain_step(
        policy=nn.Identity(),
        world_model=model,
        optimizer=optimizer,
        batch=batch,
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create({"grad_clip_norm": 100.0, "zero_grad_set_to_none": True}),
    )

    after = next(model.parameters()).detach()
    assert metrics["loss"] > 0.0
    assert not torch.equal(before, after)


def test_semantic_bottleneck_world_model_runs_dreamer_actor_critic_step() -> None:
    model = SemanticBottleneckRSSMWorldModel(
        sem_dim=12,
        latent_dim=4,
        deter=8,
        action_dim=2,
        hidden=16,
        actor_input_dim=12,
    )
    policy = VLAPolicy(action_dim=2, hidden_dim=12, policy_head_hidden_dim=16)
    critic = TwohotCritic(hidden_dim=12, critic_hidden_dim=16, num_bins=31)
    target_critic = TwohotCritic(hidden_dim=12, critic_hidden_dim=16, num_bins=31)
    target_critic.load_state_dict(critic.state_dict())

    metrics = imagine_actor_critic_step(
        policy=policy,
        world_model=model,
        critic=critic,
        target_critic=target_critic,
        actor_optimizer=torch.optim.Adam(policy.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
        return_tracker=ReturnPercentileTracker(),
        obs={
            "obs_embedding": torch.randn(3, 6, 12),
            "actions": torch.randn(3, 6, 2),
            "is_first": torch.zeros(3, 6, dtype=torch.bool),
        },
        device=torch.device("cpu"),
        algorithm_cfg=OmegaConf.create({
            "actor_input_mode": "pooled",
            "imag_last": 2,
            "imagination_horizon": 3,
            "horizon": 333,
            "contdisc": True,
            "lam": 0.95,
            "actent": 3.0e-4,
            "slowtar": False,
            "slowreg": 1.0,
            "target_critic_tau": 0.02,
        }),
        optim_cfg=OmegaConf.create({"grad_clip_norm": 100.0, "zero_grad_set_to_none": True}),
    )

    assert metrics["actor_loss"] == metrics["actor_loss"]
    assert metrics["critic_loss"] > 0.0
    assert 0.0 <= metrics["continue_mean"] <= 1.0


def test_minmax01_return_normalization_maps_returns_and_values_to_unit_interval() -> None:
    returns = torch.tensor([[0.0, 5.0, 10.0], [2.5, 7.5, 20.0]])
    values = torch.tensor([[0.0, 3.0, 12.0], [1.0, 9.0, 30.0]])
    cfg = OmegaConf.create({
        "return_normalization": {
            "mode": "minmax01",
            "low": 0.0,
            "high": 1.0,
            "eps": 1.0e-6,
        }
    })

    out = normalize_returns_for_actor_critic(returns, values, cfg)

    assert torch.all(out.returns >= 0.0)
    assert torch.all(out.returns <= 1.0)
    assert torch.all(out.values >= 0.0)
    assert torch.all(out.values <= 1.0)
    assert torch.isclose(out.returns.min(), torch.tensor(0.0))
    assert torch.isclose(out.returns.max(), torch.tensor(1.0))
    assert out.enabled
