"""Latent-space world-model environment backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class LatentWorldModelEnv:
    """Inference-only Gymnasium-style env backed by a world model and classifier."""

    def __init__(
        self,
        world_model: nn.Module,
        classifier: nn.Module | None = None,
        *,
        latent_dim: int,
        action_dim: int,
        success_threshold: float = 0.5,
        max_episode_steps: int = 64,
        device: str | torch.device = "cpu",
        initial_latent: Any | None = None,
    ) -> None:
        self.world_model = world_model
        self.classifier = classifier
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.success_threshold = float(success_threshold)
        self.max_episode_steps = int(max_episode_steps)
        self.device = torch.device(device)
        self.wm_version = 0
        self.classifier_version = 0
        self._initial_latent = initial_latent
        self._latent = torch.zeros(self.latent_dim, dtype=torch.float32, device=self.device)
        self._elapsed_steps = 0
        self._task_id = 0
        self._episode_id = 0
        self.world_model.to(self.device).eval()
        if self.classifier is not None:
            self.classifier.to(self.device).eval()

    def reset(
        self,
        *,
        task_id: int = 0,
        episode_id: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._task_id = int(task_id)
        self._episode_id = int(episode_id)
        self._elapsed_steps = 0
        if self._initial_latent is None:
            latent = torch.zeros(self.latent_dim, dtype=torch.float32, device=self.device)
        else:
            latent = torch.as_tensor(
                self._initial_latent,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
            if latent.numel() != self.latent_dim:
                raise ValueError(
                    f"initial_latent has {latent.numel()} values; expected {self.latent_dim}"
                )
        self._latent = latent.detach()
        return self._obs(), {"task_id": self._task_id, "episode_id": self._episode_id}

    @torch.no_grad()
    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(-1)
        if action_t.numel() != self.action_dim:
            raise ValueError(f"action has {action_t.numel()} values; expected {self.action_dim}")
        batch = {
            "latent": self._latent.reshape(1, self.latent_dim),
            "action": action_t.reshape(1, self.action_dim),
        }
        next_latent = self._extract_latent(self.world_model(batch)).reshape(-1)
        if next_latent.numel() != self.latent_dim:
            raise ValueError(
                f"world_model returned {next_latent.numel()} latent values; "
                f"expected {self.latent_dim}"
            )
        self._latent = next_latent.detach()
        self._elapsed_steps += 1
        score = self._score(self._latent)
        reward = float(score)
        terminated = bool(score >= self.success_threshold)
        truncated = bool(self._elapsed_steps >= self.max_episode_steps and not terminated)
        info = {
            "task_id": self._task_id,
            "episode_id": self._episode_id,
            "elapsed_steps": self._elapsed_steps,
            "success_score": float(score),
            "wm_version": self.wm_version,
            "classifier_version": self.classifier_version,
        }
        return self._obs(), reward, terminated, truncated, info

    def chunk_step(
        self,
        action_chunk: Any,
    ) -> tuple[
        list[dict[str, Any]],
        list[float],
        list[bool],
        list[bool],
        list[dict[str, Any]],
    ]:
        observations: list[dict[str, Any]] = []
        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        for action in np.asarray(action_chunk, dtype=np.float32).reshape(-1, self.action_dim):
            obs, reward, terminated, truncated, info = self.step(action)
            observations.append(obs)
            rewards.append(float(reward))
            terminations.append(bool(terminated))
            truncations.append(bool(truncated))
            infos.append(info)
            if terminated or truncated:
                break
        return observations, rewards, terminations, truncations, infos

    def load_world_model_state(self, state_dict: dict[str, Any], version: int) -> None:
        if state_dict:
            self.world_model.load_state_dict(state_dict)
        self.world_model.to(self.device).eval()
        self.wm_version = int(version)

    def load_classifier_state(self, state_dict: dict[str, Any], version: int) -> None:
        if self.classifier is None:
            if state_dict:
                raise RuntimeError("cannot load classifier state without a classifier module")
        elif state_dict:
            self.classifier.load_state_dict(state_dict)
            self.classifier.to(self.device).eval()
        self.classifier_version = int(version)

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "latent": self._latent.detach().cpu().numpy().astype(np.float32, copy=False)
        }

    def _score(self, latent: torch.Tensor) -> float:
        if self.classifier is None:
            return 0.0
        raw = self.classifier(latent.reshape(1, self.latent_dim))
        if isinstance(raw, dict):
            raw = raw.get("success", raw.get("logits", raw.get("score", 0.0)))
        return float(torch.as_tensor(raw, dtype=torch.float32, device=self.device).reshape(-1)[0])

    def _extract_latent(self, value: Any) -> torch.Tensor:
        if isinstance(value, dict):
            for key in ("next_latent", "latent", "state"):
                if key in value:
                    return torch.as_tensor(value[key], dtype=torch.float32, device=self.device)
            raise ValueError("world_model output dict must include next_latent, latent, or state")
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)
