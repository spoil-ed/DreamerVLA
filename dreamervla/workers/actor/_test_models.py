"""Tiny trainable actor models for Ray learner e2e tests."""

from __future__ import annotations

from types import SimpleNamespace

import ray
import torch
from torch import nn


class TinyTrainablePolicy(nn.Linear):
    """Linear policy with a stable constructor signature for tests."""

    def __init__(self, hidden_dim: int = 4, action_dim: int = 7) -> None:
        super().__init__(int(hidden_dim), int(action_dim))
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        return self(hidden.float())


class TinySharedPolicy(TinyTrainablePolicy):
    """Policy usable by both LearnerWorker and InferenceWorker."""

    def forward(self, batch):  # type: ignore[override]
        if isinstance(batch, dict):
            hidden = batch["hidden"].float()
            action = super().forward(hidden).unsqueeze(1)
            return action, None, None
        return super().forward(batch.float())


class TinyLumosPolicy(nn.Module):
    """Tiny policy implementing the LUMOS sample/evaluate protocol."""

    def __init__(
        self,
        hidden_dim: int = 4,
        action_dim: int = 7,
        chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.linear = nn.Linear(self.hidden_dim, self.action_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, batch):  # type: ignore[override]
        if not isinstance(batch, dict):
            return self.linear(batch.float())
        hidden = batch["hidden"].float()
        mean = self.linear(hidden).unsqueeze(1).expand(
            -1,
            self.chunk_size,
            -1,
        )
        mode = str(batch.get("mode", "sample"))
        if mode == "sample":
            action = mean
            log_prob = -((action - mean) ** 2).mean(dim=(1, 2))
            return action, log_prob, {"action_chunk": action}
        if mode == "evaluate":
            action = batch["action"].float()
            if action.ndim == 2:
                mean_for_action = mean[:, 0, :]
                reduce_dims = 1
            else:
                mean_for_action = mean
                reduce_dims = (1, 2)
            log_prob = -((action - mean_for_action) ** 2).mean(dim=reduce_dims)
            entropy = torch.ones_like(log_prob) * 0.5
            return log_prob, entropy, {}
        raise ValueError(f"unknown TinyLumosPolicy mode {mode!r}")


class TinyCheckpointPolicy(TinyTrainablePolicy):
    """Policy exposing the gradient-checkpointing hook used by FSDP tests."""

    def __init__(self, hidden_dim: int = 4, action_dim: int = 7) -> None:
        super().__init__(hidden_dim=hidden_dim, action_dim=action_dim)
        self.register_buffer("checkpoint_flag", torch.zeros((), dtype=torch.long))

    def gradient_checkpointing_enable(self) -> None:
        self.checkpoint_flag.fill_(1)


class TinyScalarModel(nn.Module):
    """Small trainable component for phase-updater tests."""

    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden.float()).squeeze(-1)


class TinyTrainableWorldModel(TinyScalarModel):
    """Small trainable world-model stand-in for learner routing tests."""


class TinyLumosWorldModel(nn.Module):
    """Tiny world model implementing the production DreamerVLA step protocol."""

    def __init__(self, hidden_dim: int = 4, action_dim: int = 7) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.obs_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.action_proj = nn.Linear(self.action_dim, self.hidden_dim)
        nn.init.eye_(self.obs_proj.weight)
        nn.init.zeros_(self.obs_proj.bias)
        nn.init.zeros_(self.action_proj.weight)
        nn.init.zeros_(self.action_proj.bias)

    def forward(self, batch: dict) -> dict[str, torch.Tensor] | torch.Tensor:  # type: ignore[override]
        mode = batch.get("mode")
        if mode is None:
            if "latent" in batch and "action" in batch:
                latent = self._latent_hidden(batch["latent"])
                action_signal = self.action_proj(batch["action"].float())
                return latent + action_signal
            obs = batch["obs_embedding"].float()
            pred = self.obs_proj(obs)
            target = batch["current_actions"].float().mean(dim=-1, keepdim=True)
            target = target.expand_as(pred)
            loss = torch.mean((pred - target) ** 2)
            return {"loss": loss, "_loss": loss}
        if mode == "encode_latent":
            return self.obs_proj(batch["hidden"].float())
        if mode == "observe_next":
            hidden = self.obs_proj(batch["hidden"].float())
            latent = self._latent_hidden(batch["latent"])
            action_signal = self.action_proj(batch["actions"].float())
            return hidden + 0.1 * latent + 0.01 * action_signal
        if mode == "observe_sequence":
            return {"latent": self.obs_proj(batch["obs_embedding"].float())}
        if mode == "actor_input":
            return self._latent_hidden(batch["latent"])
        if mode == "predict_next_chunk":
            latent = self._latent_hidden(batch["latent"])
            actions = batch["actions"].float()
            action_signal = self.action_proj(actions)
            hidden_seq = latent.unsqueeze(1) + action_signal
            return {
                "hidden_seq": hidden_seq,
                "history": hidden_seq,
                "actions": actions,
                "hidden": hidden_seq[:, -1],
            }
        raise ValueError(f"unknown TinyLumosWorldModel mode {mode!r}")

    @staticmethod
    def _latent_hidden(latent) -> torch.Tensor:
        if isinstance(latent, dict):
            return latent["hidden"].float()
        return latent.float()


class TinyValueCritic(TinyScalarModel):
    """Small trainable critic stand-in for learner routing tests."""


class TinySuccessClassifier(nn.Module):
    """Tiny classifier with the cfg attributes used by online classifier updates."""

    def __init__(self, hidden_dim: int = 4, window: int = 3) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(
            window=int(window),
            chunk_size=1,
            chunk_pool="last",
            granularity="action",
        )
        self.linear = nn.Linear(int(hidden_dim), 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim == 2:
            hidden = windows.float()
        else:
            hidden = windows.float().mean(dim=1)
        return self.linear(hidden)

    def predict_success(
        self,
        video: torch.Tensor,
        *,
        threshold: float,
        stride: int,
        min_steps: int,
    ) -> dict[str, torch.Tensor]:
        del stride
        logits = self.linear(video.float())
        probs = torch.softmax(logits, dim=-1)[..., 1]
        scan_start = max(0, int(min_steps) - 1)
        scanned = probs[:, scan_start:]
        above = scanned >= float(threshold)
        has_success = above.any(dim=1)
        first = above.float().argmax(dim=1) + scan_start
        fallback = torch.full_like(first, int(video.shape[1]) - 1)
        finish_step = torch.where(has_success, first, fallback)
        return {"complete": has_success, "finish_step": finish_step}



class TinyWorldModelPhaseUpdater:
    """Configurable phase updater matching LearnerWorker's real-update boundary."""

    def update(
        self,
        *,
        phase: str,
        num_steps: int,
        modules: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        replay,
        device: torch.device,
        train_cfg: dict,
        precision,
    ) -> dict[str, float]:
        if phase != "wm":
            return {f"train/{phase}_loss": 0.0}
        world_model = modules["world_model"]
        optimizer = optimizers["world_model"]
        batch_size = int(train_cfg.get("batch_size", 2))
        last_loss = 0.0
        for _ in range(int(num_steps)):
            batch = ray.get(replay.sample.remote(batch_size))
            hidden = batch["obs_embedding"].to(device).float().mean(dim=1)
            target = batch["current_actions"].to(device).float().mean(dim=(1, 2))
            optimizer.zero_grad(set_to_none=True)
            with precision.context():
                pred = world_model(hidden)
                loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        return {"train/wm_loss": last_loss}
