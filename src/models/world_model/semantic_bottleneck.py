from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str) -> nn.Module:
    name = str(name).lower()
    if name in {"elu", "ELU".lower()}:
        return nn.ELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int = 2, act: str = "elu") -> nn.Sequential:
    mods: list[nn.Module] = []
    cur = int(in_dim)
    for _ in range(max(int(layers) - 1, 0)):
        mods.extend([nn.Linear(cur, int(hidden)), nn.LayerNorm(int(hidden)), _act(act)])
        cur = int(hidden)
    mods.append(nn.Linear(cur, int(out_dim)))
    return nn.Sequential(*mods)


def gaussian_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor | None = None,
    logvar_p: torch.Tensor | None = None,
) -> torch.Tensor:
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)
    return 0.5 * (
        logvar_p
        - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p).square()) / logvar_p.exp().clamp_min(1e-8)
        - 1.0
    ).sum(dim=-1)


@dataclass
class BottleneckOutput:
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    kl: torch.Tensor


@dataclass
class SemanticBottleneckLatent:
    deter: torch.Tensor
    stoch: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor

    def feature(self) -> torch.Tensor:
        return torch.cat([self.deter, self.stoch], dim=-1)


class StochasticBottleneck(nn.Module):
    """Gaussian VIB bottleneck: z_sem -> z_phys."""

    def __init__(
        self,
        input_dim: int = 4096,
        latent_dim: int = 32,
        hidden: int = 512,
        layers: int = 2,
        min_logvar: float = -10.0,
        max_logvar: float = 4.0,
        act: str = "elu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.net = _mlp(self.input_dim, 2 * self.latent_dim, int(hidden), layers=int(layers), act=act)

    def forward(self, z_sem: torch.Tensor, deterministic: bool = False) -> BottleneckOutput:
        if z_sem.shape[-1] != self.input_dim:
            raise ValueError(f"z_sem dim mismatch: got {z_sem.shape[-1]}, expected {self.input_dim}")
        stats = self.net(z_sem.float())
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(self.min_logvar, self.max_logvar)
        if deterministic:
            z = mu
        else:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return BottleneckOutput(z=z, mu=mu, logvar=logvar, kl=gaussian_kl(mu, logvar))


class GaussianRSSM(nn.Module):
    """Small Gaussian RSSM over z_phys, aligned with the writeup."""

    def __init__(
        self,
        latent_dim: int = 32,
        deter: int = 256,
        action_dim: int = 7,
        hidden: int = 256,
        layers: int = 2,
        min_logvar: float = -10.0,
        max_logvar: float = 4.0,
        act: str = "elu",
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.deter = int(deter)
        self.action_dim = int(action_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.gru = nn.GRUCell(self.latent_dim + self.action_dim, self.deter)
        self.prior = _mlp(self.deter, 2 * self.latent_dim, int(hidden), layers=int(layers), act=act)
        self.posterior = _mlp(self.deter + self.latent_dim, 2 * self.latent_dim, int(hidden), layers=int(layers), act=act)

    def _stats(self, net: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = net(x).chunk(2, dim=-1)
        return mu, logvar.clamp(self.min_logvar, self.max_logvar)

    @staticmethod
    def _sample(mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def initial(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> SemanticBottleneckLatent:
        return SemanticBottleneckLatent(
            deter=torch.zeros(batch_size, self.deter, device=device, dtype=dtype),
            stoch=torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype),
            mu=torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype),
            logvar=torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype),
        )

    def prior_step(
        self,
        latent: SemanticBottleneckLatent,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> SemanticBottleneckLatent:
        x = torch.cat([latent.stoch, action.to(dtype=latent.stoch.dtype)], dim=-1)
        deter = self.gru(x, latent.deter)
        mu, logvar = self._stats(self.prior, deter)
        z = self._sample(mu, logvar, deterministic=deterministic)
        return SemanticBottleneckLatent(deter=deter, stoch=z, mu=mu, logvar=logvar)

    def observe(
        self,
        z_obs: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        if z_obs.ndim != 3:
            raise ValueError(f"z_obs must be [B,T,D], got {tuple(z_obs.shape)}")
        batch, steps, _ = z_obs.shape
        device, dtype = z_obs.device, z_obs.dtype
        prev = self.initial(batch, device=device, dtype=dtype)
        if is_first is None:
            is_first = torch.zeros(batch, steps, device=device, dtype=torch.bool)
        outputs: dict[str, list[torch.Tensor]] = {
            "deter": [],
            "stoch": [],
            "post_mu": [],
            "post_logvar": [],
            "prior_mu": [],
            "prior_logvar": [],
        }
        zero_action = torch.zeros(batch, self.action_dim, device=device, dtype=dtype)
        for tidx in range(steps):
            reset = is_first[:, tidx].to(device=device).bool()
            if reset.any():
                keep = (~reset).to(dtype=dtype).unsqueeze(-1)
                prev = SemanticBottleneckLatent(
                    deter=prev.deter * keep,
                    stoch=prev.stoch * keep,
                    mu=prev.mu * keep,
                    logvar=prev.logvar * keep,
                )
            action = zero_action if tidx == 0 else actions[:, tidx - 1].to(device=device, dtype=dtype)
            prior_latent = self.prior_step(prev, action, deterministic=deterministic)
            post_mu, post_logvar = self._stats(
                self.posterior,
                torch.cat([prior_latent.deter, z_obs[:, tidx]], dim=-1),
            )
            post_z = self._sample(post_mu, post_logvar, deterministic=deterministic)
            prev = SemanticBottleneckLatent(
                deter=prior_latent.deter,
                stoch=post_z,
                mu=post_mu,
                logvar=post_logvar,
            )
            outputs["deter"].append(prev.deter)
            outputs["stoch"].append(prev.stoch)
            outputs["post_mu"].append(post_mu)
            outputs["post_logvar"].append(post_logvar)
            outputs["prior_mu"].append(prior_latent.mu)
            outputs["prior_logvar"].append(prior_latent.logvar)
        return {key: torch.stack(value, dim=1) for key, value in outputs.items()}


class SemanticBottleneckRSSMWorldModel(nn.Module):
    """Dreamer-VLA writeup implementation: frozen z_sem -> VIB z_phys -> Gaussian RSSM."""

    def __init__(
        self,
        sem_dim: int = 4096,
        hidden_dim: int | None = None,
        latent_dim: int = 32,
        deter: int = 256,
        action_dim: int = 7,
        hidden: int = 256,
        bottleneck_hidden: int = 512,
        bottleneck_layers: int = 2,
        rssm_layers: int = 2,
        reward_loss: str = "mse",
        reward_scale: float = 1.0,
        continue_scale: float = 1.0,
        dyn_scale: float = 1.0,
        bottleneck_kl_scale: float = 1.0e-3,
        actor_input_dim: int | None = None,
        act: str = "elu",
    ) -> None:
        super().__init__()
        if hidden_dim is not None:
            sem_dim = int(hidden_dim)
        self.sem_dim = int(sem_dim)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.reward_loss_type = str(reward_loss).lower()
        self.reward_scale = float(reward_scale)
        self.continue_scale = float(continue_scale)
        self.dyn_scale = float(dyn_scale)
        self.bottleneck_kl_scale = float(bottleneck_kl_scale)
        self.bottleneck = StochasticBottleneck(
            input_dim=self.sem_dim,
            latent_dim=self.latent_dim,
            hidden=int(bottleneck_hidden),
            layers=int(bottleneck_layers),
            act=act,
        )
        self.rssm = GaussianRSSM(
            latent_dim=self.latent_dim,
            deter=int(deter),
            action_dim=self.action_dim,
            hidden=int(hidden),
            layers=int(rssm_layers),
            act=act,
        )
        feature_dim = int(deter) + self.latent_dim
        self.feature_dim = feature_dim
        self.reward_head = _mlp(feature_dim, 1, int(hidden), layers=2, act=act)
        self.continue_head = _mlp(feature_dim, 1, int(hidden), layers=2, act=act)
        self.actor_adapter = _mlp(
            feature_dim,
            int(actor_input_dim if actor_input_dim is not None else feature_dim),
            int(hidden),
            layers=2,
            act=act,
        )

    @staticmethod
    def _latent_from_seq(seq: dict[str, torch.Tensor], index: int | slice = -1) -> SemanticBottleneckLatent:
        return SemanticBottleneckLatent(
            deter=seq["deter"][:, index],
            stoch=seq["stoch"][:, index],
            mu=seq["post_mu"][:, index],
            logvar=seq["post_logvar"][:, index],
        )

    def encode_semantic(self, z_sem: torch.Tensor, deterministic: bool = False) -> BottleneckOutput:
        return self.bottleneck(z_sem, deterministic=deterministic)

    def observe_sequence(self, batch: dict[str, torch.Tensor], deterministic: bool = False) -> dict[str, Any]:
        z_sem = batch["obs_embedding"].float()
        actions = batch["actions"].to(device=z_sem.device, dtype=z_sem.dtype)
        is_first = batch.get("is_first")
        if isinstance(is_first, torch.Tensor):
            is_first = is_first.to(device=z_sem.device)
        bottleneck = self.encode_semantic(z_sem, deterministic=deterministic)
        seq = self.rssm.observe(bottleneck.z, actions, is_first=is_first, deterministic=deterministic)
        latent = SemanticBottleneckLatent(
            deter=seq["deter"],
            stoch=seq["stoch"],
            mu=seq["post_mu"],
            logvar=seq["post_logvar"],
        )
        return {"latent": latent, "seq": seq, "bottleneck": bottleneck, "feat": latent.feature()}

    def predict_next(
        self,
        latent: SemanticBottleneckLatent,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> SemanticBottleneckLatent:
        return self.rssm.prior_step(latent, actions, deterministic=deterministic)

    def actor_input(self, latent: SemanticBottleneckLatent) -> torch.Tensor:
        return self.actor_adapter(latent.feature())

    def critic_input(self, latent: SemanticBottleneckLatent) -> torch.Tensor:
        return latent.feature()

    def state_reward(self, latent: SemanticBottleneckLatent) -> torch.Tensor:
        return self.reward_head(latent.feature()).squeeze(-1)

    def continue_prob(self, latent: SemanticBottleneckLatent) -> torch.Tensor:
        return torch.sigmoid(self.continue_head(latent.feature()).squeeze(-1))

    def reward(
        self,
        latent: SemanticBottleneckLatent,
        actions: torch.Tensor | None,
        next_latent: SemanticBottleneckLatent,
    ) -> torch.Tensor:
        del latent, actions
        return self.state_reward(next_latent)

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        obs_embedding = batch["obs_embedding"].to(device=device, dtype=torch.float32)
        actions = batch["actions"].to(device=device, dtype=torch.float32)
        rewards = batch["rewards"].to(device=device, dtype=torch.float32)
        dones = batch.get("is_terminal", batch.get("dones"))
        if dones is None:
            dones = torch.zeros_like(rewards)
        dones = dones.to(device=device, dtype=torch.float32)
        is_first = batch.get("is_first")
        if isinstance(is_first, torch.Tensor):
            is_first = is_first.to(device=device)

        observed = self.observe_sequence(
            {
                "obs_embedding": obs_embedding,
                "actions": actions,
                "is_first": is_first if isinstance(is_first, torch.Tensor) else torch.zeros_like(rewards, dtype=torch.bool),
            }
        )
        seq = observed["seq"]
        bottleneck: BottleneckOutput = observed["bottleneck"]
        feat = observed["feat"]
        reward_logits = self.reward_head(feat).squeeze(-1)
        cont_logits = self.continue_head(feat).squeeze(-1)
        if self.reward_loss_type in {"bce", "binary"}:
            reward_loss = F.binary_cross_entropy_with_logits(reward_logits, rewards.clamp(0.0, 1.0))
            reward_pred = torch.sigmoid(reward_logits)
        else:
            reward_loss = F.mse_loss(reward_logits, rewards)
            reward_pred = reward_logits
        cont_target = 1.0 - dones
        continue_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
        dyn_kl = gaussian_kl(
            seq["post_mu"],
            seq["post_logvar"],
            seq["prior_mu"].detach(),
            seq["prior_logvar"].detach(),
        ).mean()
        rep_kl = gaussian_kl(
            seq["post_mu"].detach(),
            seq["post_logvar"].detach(),
            seq["prior_mu"],
            seq["prior_logvar"],
        ).mean()
        bottleneck_kl = bottleneck.kl.mean()
        loss = (
            self.reward_scale * reward_loss
            + self.continue_scale * continue_loss
            + self.dyn_scale * dyn_kl
            + 0.1 * self.dyn_scale * rep_kl
            + self.bottleneck_kl_scale * bottleneck_kl
        )
        zero = loss.new_zeros(())
        return {
            "_loss": loss,
            "loss": loss.detach(),
            "reward_loss": reward_loss.detach(),
            "continue_loss": continue_loss.detach(),
            "dyn_loss": dyn_kl.detach(),
            "rep_loss": rep_kl.detach(),
            "bottleneck_kl_loss": bottleneck_kl.detach(),
            "bottleneck_kl_scaled_loss": (self.bottleneck_kl_scale * bottleneck_kl).detach(),
            "reward_pred_mean": reward_pred.detach().mean(),
            "continue_pred_mean": torch.sigmoid(cont_logits.detach()).mean(),
            "z_phys_mean": bottleneck.z.detach().mean(),
            "z_phys_std": bottleneck.z.detach().std(),
            # Compatibility with the generic WM workspace logging.
            "rec_loss": zero.detach(),
            "image_mse": zero.detach(),
            "image_psnr": zero.detach(),
        }

    def _forward_adapter(self, batch: dict[str, Any]) -> Any:
        mode = batch.get("mode")
        if mode == "observe_sequence":
            return self.observe_sequence(batch)
        if mode == "predict_next":
            return self.predict_next(batch["latent"], batch["actions"])
        if mode == "reward":
            if "next_latent" in batch:
                return self.reward(batch["latent"], batch.get("actions"), batch["next_latent"])
            return self.state_reward(batch["latent"])
        if mode == "continue":
            return self.continue_prob(batch["latent"])
        if mode == "actor_input":
            return self.actor_input(batch["latent"])
        if mode == "critic_input":
            return self.critic_input(batch["latent"])
        raise ValueError(f"Unknown SemanticBottleneckRSSMWorldModel mode: {mode!r}")

    def forward(self, batch: dict[str, Any]) -> Any:
        if isinstance(batch, dict) and batch.get("mode") is not None:
            return self._forward_adapter(batch)
        return self.loss(batch)


__all__ = [
    "BottleneckOutput",
    "GaussianRSSM",
    "SemanticBottleneckLatent",
    "SemanticBottleneckRSSMWorldModel",
    "StochasticBottleneck",
    "gaussian_kl",
]
