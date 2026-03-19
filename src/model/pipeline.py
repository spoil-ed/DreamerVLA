from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from dreamer_vla.bottleneck import LinearBottleneck
from dreamer_vla.config import DreamerVLAConfig
from dreamer_vla.planner import ActorCriticPlanner
from dreamer_vla.trainer.ppo.core_algos import lambda_return
from dreamer_vla.vla_encoder import build_frozen_vla_encoder
from dreamer_vla.world_model import LatentState, WorldModelAdapter


class DreamerVLAPipeline(nn.Module):
    """Minimal Dreamer-VLA research pipeline.

    The implementation keeps the core design decisions:
    - frozen semantic encoder
    - trainable linear bottleneck
    - simple latent world model
    - imagination-based actor/critic updates
    """

    def __init__(self, config: DreamerVLAConfig | None = None) -> None:
        super().__init__()
        self.config = config or DreamerVLAConfig()
        model_cfg = self.config.model
        self.device_name = self.config.trainer.device

        self.encoder = build_frozen_vla_encoder(self.config)
        self.bottleneck = LinearBottleneck(
            input_dim=model_cfg.semantic_dim,
            latent_dim=model_cfg.bottleneck_dim,
        )
        self.world_model = WorldModelAdapter(
            latent_dim=model_cfg.bottleneck_dim,
            hidden_dim=model_cfg.rssm_hidden_dim,
            action_dim=model_cfg.action_dim,
            reward_hidden_dim=model_cfg.reward_hidden_dim,
        )
        feature_dim = model_cfg.bottleneck_dim
        self.planner = ActorCriticPlanner(
            feature_dim=feature_dim,
            action_dim=model_cfg.action_dim,
            actor_hidden_dim=model_cfg.actor_hidden_dim,
            critic_hidden_dim=model_cfg.critic_hidden_dim,
        )

        self.world_model_optimizer = torch.optim.Adam(
            list(self.bottleneck.parameters()) + list(self.world_model.parameters()),
            lr=self.config.algorithm.world_model_lr,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.planner.actor.parameters(),
            lr=self.config.algorithm.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.planner.critic.parameters(),
            lr=self.config.algorithm.critic_lr,
        )

        self.to(self.device_name)
        self.encoder.eval()

    @property
    def device(self) -> torch.device:
        return next(self.bottleneck.parameters()).device

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _encode_semantics(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder.encode(
            {
                "image": batch["image"],
                "proprio": batch["proprio"],
                "text": batch["text"],
            }
        )

    def _observe(self, batch: dict[str, torch.Tensor]):
        semantics = self._encode_semantics(batch)
        bottleneck_output = self.bottleneck(semantics)
        observed = self.world_model.observe(
            latents=bottleneck_output.latent,
            actions=batch["action"],
            done=batch["done"],
        )
        return semantics, bottleneck_output, observed

    def _final_state(self, latents: torch.Tensor) -> LatentState:
        return LatentState(latent=latents[:, -1])

    def world_model_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], LatentState]:
        _, bottleneck_output, observed = self._observe(batch)
        dynamics_loss = F.mse_loss(observed.next_latent_pred, observed.latents[:, 1:])
        reward_loss = F.mse_loss(observed.reward_pred, batch["reward"])
        continue_loss = self.world_model.continue_loss(observed.continue_logit, batch["done"])
        bottleneck_penalty = bottleneck_output.penalty.mean()
        total_loss = (
            reward_loss
            + self.config.algorithm.continue_loss_scale * continue_loss
            + self.config.algorithm.dynamics_kl_scale * dynamics_loss
            + self.config.algorithm.bottleneck_kl_scale * bottleneck_penalty
        )
        metrics = {
            "world_model/loss": total_loss.detach().item(),
            "world_model/dynamics_loss": dynamics_loss.detach().item(),
            "world_model/reward_loss": reward_loss.detach().item(),
            "world_model/continue_loss": continue_loss.detach().item(),
            "world_model/bottleneck_penalty": bottleneck_penalty.detach().item(),
            "world_model/reward_mse": (observed.reward_pred - batch["reward"]).pow(2).mean().detach().item(),
        }
        return total_loss, metrics, self._final_state(observed.latents)

    def _set_grad_enabled(self, module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def imagination_rollout(self, start_state: LatentState, deterministic: bool = False):
        return self.world_model.imagine(
            start_state=start_state,
            actor=self.planner.actor,
            horizon=self.config.algorithm.imagination_horizon,
            deterministic=deterministic,
        )

    def actor_and_critic_loss(self, start_state: LatentState) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        rollout = self.imagination_rollout(start_state, deterministic=False)
        values = self.planner.value(rollout.latents)
        bootstrap = values[:, -1].detach()
        returns = lambda_return(
            rewards=rollout.rewards,
            values=values,
            continues=rollout.continues,
            bootstrap=bootstrap,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lambda_,
        )

        actor_loss = -returns.mean()
        critic_loss = F.mse_loss(values, returns.detach())
        metrics = {
            "actor/loss": actor_loss.detach().item(),
            "critic/loss": critic_loss.detach().item(),
            "imagination/reward_mean": rollout.rewards.mean().detach().item(),
            "imagination/continue_mean": rollout.continues.mean().detach().item(),
            "imagination/value_mean": values.mean().detach().item(),
            "imagination/entropy_mean": rollout.entropies.mean().detach().item(),
            "imagination/return_mean": returns.mean().detach().item(),
        }
        return actor_loss, critic_loss, metrics

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        batch = self._move_batch(batch)

        self.world_model_optimizer.zero_grad(set_to_none=True)
        world_model_loss, metrics, start_state = self.world_model_loss(batch)
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.bottleneck.parameters()) + list(self.world_model.parameters()),
            self.config.algorithm.grad_clip_norm,
        )
        self.world_model_optimizer.step()

        with torch.no_grad():
            _, _, observed = self._observe(batch)
            start_state = self._final_state(observed.latents).detach()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self._set_grad_enabled(self.world_model, False)
        self._set_grad_enabled(self.planner.critic, False)
        actor_loss, _, actor_metrics = self.actor_and_critic_loss(start_state)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.planner.actor.parameters(), self.config.algorithm.grad_clip_norm)
        self.actor_optimizer.step()
        self._set_grad_enabled(self.world_model, True)
        self._set_grad_enabled(self.planner.critic, True)

        self.critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            rollout = self.imagination_rollout(start_state, deterministic=True)
            imagined_values = self.planner.value(rollout.latents)
            targets = lambda_return(
                rewards=rollout.rewards,
                values=imagined_values,
                continues=rollout.continues,
                bootstrap=imagined_values[:, -1],
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lambda_,
            )
            latents = rollout.latents.detach()
        critic_values = self.planner.value(latents)
        critic_loss = F.mse_loss(critic_values, targets.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.planner.critic.parameters(), self.config.algorithm.grad_clip_norm)
        self.critic_optimizer.step()

        metrics.update(actor_metrics)
        metrics["critic/loss"] = critic_loss.detach().item()
        metrics["data/reward_mean"] = batch["reward"].mean().detach().item()
        metrics["data/done_rate"] = batch["done"].float().mean().detach().item()
        return metrics

    @torch.no_grad()
    def evaluate_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        batch = self._move_batch(batch)
        world_model_loss, metrics, start_state = self.world_model_loss(batch)
        rollout = self.imagination_rollout(start_state, deterministic=True)
        values = self.planner.value(rollout.latents)
        returns = lambda_return(
            rewards=rollout.rewards,
            values=values,
            continues=rollout.continues,
            bootstrap=values[:, -1],
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lambda_,
        )
        metrics.update(
            {
                "eval/world_model_loss": world_model_loss.detach().item(),
                "eval/imagination_reward_mean": rollout.rewards.mean().detach().item(),
                "eval/imagination_return_mean": returns.mean().detach().item(),
            }
        )
        return metrics

    def summary(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "modules": {
                "encoder": self.encoder.__class__.__name__,
                "bottleneck": self.bottleneck.__class__.__name__,
                "world_model": self.world_model.__class__.__name__,
                "planner": self.planner.__class__.__name__,
            },
        }
