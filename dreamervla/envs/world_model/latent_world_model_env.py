"""Latent-space world-model environment backend."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
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
        num_envs: int = 1,
    ) -> None:
        self.world_model = _build_component(world_model)
        self.classifier = None if classifier is None else _build_component(classifier)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.num_envs = int(num_envs)
        self.success_threshold = float(success_threshold)
        self.max_episode_steps = int(max_episode_steps)
        self.image_shape = tuple(int(value) for value in image_shape)
        self.device = torch.device(device)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.wm_version = 0
        self.classifier_version = 0
        self._initial_latent = initial_latent
        self._latent = torch.zeros(
            (self.num_envs, self.latent_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._elapsed_steps = np.zeros((self.num_envs,), dtype=np.int64)
        self._task_ids = np.zeros((self.num_envs,), dtype=np.int64)
        self._episode_ids = np.zeros((self.num_envs,), dtype=np.int64)
        self.world_model.to(self.device).eval()
        if self.classifier is not None:
            self.classifier.to(self.device).eval()

    def reset(
        self,
        *,
        task_id: int = 0,
        episode_id: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.reset_slot(0, task_id=task_id, episode_id=episode_id)

    @torch.no_grad()
    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self.step_slot(0, action)

    def reset_slot(
        self,
        slot_id: int,
        *,
        task_id: int = 0,
        episode_id: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._validate_slot(slot_id)
        slot = int(slot_id)
        self._task_ids[slot] = int(task_id)
        self._episode_ids[slot] = int(episode_id)
        self._elapsed_steps[slot] = 0
        self._latent[slot] = self._initial_latent_for_slot(slot).detach()
        return self._obs(slot), {
            "slot_id": slot,
            "task_id": int(self._task_ids[slot]),
            "episode_id": int(self._episode_ids[slot]),
        }

    @torch.no_grad()
    def step_slot(
        self,
        slot_id: int,
        action: Any,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observations, rewards, terminations, truncations, infos = self.step_batch(
            np.asarray(action, dtype=np.float32).reshape(1, -1),
            env_ids=[int(slot_id)],
        )
        return (
            observations[0],
            rewards[0],
            terminations[0],
            truncations[0],
            infos[0],
        )

    def reset_batch(
        self,
        task_ids: Sequence[int],
        episode_ids: Sequence[int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if len(task_ids) != self.num_envs or len(episode_ids) != self.num_envs:
            raise ValueError(
                "reset_batch task_ids and episode_ids must match num_envs "
                f"({self.num_envs})"
            )
        observations: list[dict[str, Any]] = []
        infos: list[dict[str, Any]] = []
        for slot_id, (task_id, episode_id) in enumerate(zip(task_ids, episode_ids)):
            obs, info = self.reset_slot(
                slot_id,
                task_id=int(task_id),
                episode_id=int(episode_id),
            )
            observations.append(obs)
            infos.append(info)
        return observations, infos

    def set_initial_latents(self, latents: Any) -> None:
        """Set the latent state used by future slot resets."""

        latent = torch.as_tensor(latents, dtype=torch.float32, device=self.device)
        if latent.numel() == self.latent_dim:
            self._initial_latent = (
                latent.reshape(self.latent_dim).detach().cpu().numpy()
            )
            return
        if latent.numel() == self.num_envs * self.latent_dim:
            self._initial_latent = (
                latent.reshape(self.num_envs, self.latent_dim).detach().cpu().numpy()
            )
            return
        raise ValueError(
            f"initial latents have {latent.numel()} values; expected "
            f"{self.latent_dim} or {self.num_envs * self.latent_dim}"
        )

    @torch.no_grad()
    def step_batch(
        self,
        actions: Any,
        env_ids: Sequence[int] | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        list[float],
        list[bool],
        list[bool],
        list[dict[str, Any]],
    ]:
        slots = (
            list(range(self.num_envs))
            if env_ids is None
            else [int(v) for v in env_ids]
        )
        if not slots:
            return [], [], [], [], []
        for slot_id in slots:
            self._validate_slot(slot_id)
        action_t = torch.as_tensor(
            np.asarray(actions, dtype=np.float32).copy(),
            dtype=torch.float32,
            device=self.device,
        ).reshape(len(slots), -1)
        if action_t.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions have {action_t.shape[-1]} values; expected {self.action_dim}"
            )
        batch = {
            "latent": self._latent[slots].reshape(len(slots), self.latent_dim),
            "action": action_t.reshape(len(slots), self.action_dim),
        }
        next_latent = self._extract_latent(self.world_model(batch)).reshape(
            len(slots), -1
        )
        if next_latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"world_model returned {next_latent.shape[-1]} latent values; "
                f"expected {self.latent_dim}"
            )
        self._latent[slots] = next_latent.detach()
        self._elapsed_steps[slots] += 1
        scores = self._score_batch(self._latent[slots])

        observations: list[dict[str, Any]] = []
        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        for index, slot_id in enumerate(slots):
            score = float(scores[index])
            terminated = bool(score >= self.success_threshold)
            truncated = bool(
                self._elapsed_steps[slot_id] >= self.max_episode_steps and not terminated
            )
            info = {
                "slot_id": int(slot_id),
                "task_id": int(self._task_ids[slot_id]),
                "episode_id": int(self._episode_ids[slot_id]),
                "elapsed_steps": int(self._elapsed_steps[slot_id]),
                "success_score": score,
                "wm_version": self.wm_version,
                "classifier_version": self.classifier_version,
            }
            observations.append(self._obs(slot_id))
            rewards.append(score)
            terminations.append(terminated)
            truncations.append(truncated)
            infos.append(info)
        return observations, rewards, terminations, truncations, infos

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
            "task_id": int(obs.get("task_id", self._task_ids[0])),
            "episode_id": int(obs.get("episode_id", self._episode_ids[0])),
            "step": int(obs.get("step", self._elapsed_steps[0])),
            "task_description": str(
                obs.get("task_description", f"task {self._task_ids[0]}")
            ),
            "success": bool(info.get("success", terminated)),
            "wm_version": int(info.get("wm_version", self.wm_version)),
            "classifier_version": int(
                info.get("classifier_version", self.classifier_version)
            ),
        }

    def _obs(self, slot_id: int = 0) -> dict[str, Any]:
        self._validate_slot(slot_id)
        return {
            "latent": self._latent[slot_id]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False),
            "task_id": int(self._task_ids[slot_id]),
            "episode_id": int(self._episode_ids[slot_id]),
            "step": int(self._elapsed_steps[slot_id]),
            "task_description": f"task {self._task_ids[slot_id]}",
            "is_first": bool(self._elapsed_steps[slot_id] == 0),
        }

    def _score(self, latent: torch.Tensor) -> float:
        return float(self._score_batch(latent.reshape(1, self.latent_dim))[0])

    def _score_batch(self, latent: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            return torch.zeros(latent.shape[0], dtype=torch.float32, device=self.device)
        raw = self.classifier(latent.reshape(latent.shape[0], self.latent_dim))
        raw_is_logits = not isinstance(raw, dict)
        if isinstance(raw, dict):
            if "success" in raw:
                raw = raw["success"]
            elif "score" in raw:
                raw = raw["score"]
            elif "logits" in raw:
                raw = raw["logits"]
                raw_is_logits = True
            else:
                raise ValueError(
                    "classifier output dict must include success, score, or logits"
                )
        score_tensor = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
        if (
            raw_is_logits
            and score_tensor.ndim >= 2
            and score_tensor.shape[0] == latent.shape[0]
            and score_tensor.shape[-1] == 2
        ):
            score_tensor = torch.sigmoid(score_tensor[..., 1])
        elif score_tensor.ndim > 1 and score_tensor.shape[0] == latent.shape[0]:
            score_tensor = score_tensor[..., -1]
        scores = score_tensor.reshape(-1)
        if scores.numel() != latent.shape[0]:
            raise ValueError(
                f"classifier returned {scores.numel()} scores; expected {latent.shape[0]}"
            )
        return scores

    def _extract_latent(self, value: Any) -> torch.Tensor:
        if isinstance(value, dict):
            for key in ("next_latent", "latent", "state"):
                if key in value:
                    return torch.as_tensor(value[key], dtype=torch.float32, device=self.device)
            raise ValueError(
                "world_model output dict must include next_latent, latent, or state"
            )
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _initial_latent_for_slot(self, slot_id: int) -> torch.Tensor:
        if self._initial_latent is None:
            return torch.zeros(self.latent_dim, dtype=torch.float32, device=self.device)
        latent = torch.as_tensor(
            self._initial_latent,
            dtype=torch.float32,
            device=self.device,
        )
        if latent.numel() == self.latent_dim:
            return latent.reshape(self.latent_dim)
        if latent.numel() == self.num_envs * self.latent_dim:
            return latent.reshape(self.num_envs, self.latent_dim)[slot_id]
        raise ValueError(
            f"initial_latent has {latent.numel()} values; expected {self.latent_dim} "
            f"or {self.num_envs * self.latent_dim}"
        )

    def _validate_slot(self, slot_id: int) -> None:
        if not 0 <= int(slot_id) < self.num_envs:
            raise ValueError(f"slot_id {slot_id} is out of range")


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
