"""Latent-space world-model environment backend."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch
from torch import nn


class LatentWorldModelEnv:
    """Inference-only Gymnasium-style env backed by a world model and classifier."""

    def __init__(
        self,
        world_model: nn.Module | dict[str, Any],
        classifier: nn.Module | dict[str, Any] | None = None,
        *,
        latent_dim: int,
        action_dim: int,
        success_threshold: float = 0.5,
        max_episode_steps: int = 64,
        image_shape: tuple[int, int, int] = (4, 4, 3),
        device: str | torch.device = "cpu",
        initial_latent: Any | None = None,
    ) -> None:
        self.world_model = _build_component(world_model)
        self.classifier = None if classifier is None else _build_component(classifier)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.success_threshold = float(success_threshold)
        self.max_episode_steps = int(max_episode_steps)
        self.image_shape = tuple(int(value) for value in image_shape)
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
        action_np = np.asarray(action, dtype=np.float32).copy()
        action_t = torch.as_tensor(
            action_np, dtype=torch.float32, device=self.device
        ).reshape(-1)
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

    def make_transition(
        self,
        obs: dict[str, Any],
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the replay record consumed by OnlineReplay/LearnerWorker."""

        done = bool(terminated or truncated)
        latent = np.asarray(obs["latent"], dtype=np.float32).reshape(self.latent_dim)
        action_arr = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        info = dict(info or {})
        return {
            "image": np.zeros(self.image_shape, dtype=np.uint8),
            "state": latent.copy(),
            "action": action_arr.copy(),
            "wm_action": action_arr.copy(),
            "policy_action": action_arr.copy(),
            "obs_embedding": latent.copy(),
            "reward": np.float32(reward),
            "done": np.float32(done),
            "discount": np.float32(0.0 if terminated else 1.0),
            "is_first": bool(obs.get("is_first", False)),
            "is_terminal": bool(terminated),
            "is_last": bool(done),
            "task_id": int(obs.get("task_id", self._task_id)),
            "episode_id": int(obs.get("episode_id", self._episode_id)),
            "step": int(obs.get("step", self._elapsed_steps)),
            "task_description": str(obs.get("task_description", f"task {self._task_id}")),
            "success": bool(info.get("success", terminated)),
            "wm_version": int(info.get("wm_version", self.wm_version)),
            "classifier_version": int(
                info.get("classifier_version", self.classifier_version)
            ),
        }

    def _obs(self) -> dict[str, Any]:
        return {
            "latent": self._latent.detach().cpu().numpy().astype(np.float32, copy=False),
            "task_id": int(self._task_id),
            "episode_id": int(self._episode_id),
            "step": int(self._elapsed_steps),
            "task_description": f"task {self._task_id}",
            "is_first": bool(self._elapsed_steps == 0),
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


def _build_component(cfg_or_module: nn.Module | dict[str, Any]) -> nn.Module:
    if isinstance(cfg_or_module, nn.Module):
        return cfg_or_module
    cfg = dict(cfg_or_module)
    target = cfg.get("target") or cfg.get("_target_") or cfg.get("class_path")
    if not target:
        raise ValueError("component config must include target/_target_/class_path")
    kwargs = dict(cfg.get("kwargs", {}))
    if ":" in str(target):
        module_name, class_name = str(target).split(":", 1)
    else:
        module_name, class_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)
