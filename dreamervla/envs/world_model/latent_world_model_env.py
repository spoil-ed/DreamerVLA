"""Latent-space world-model environment backend."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn

from dreamervla.utils.frozen_components import (
    assert_module_frozen,
    module_state_sha256,
)

_LOGGER = logging.getLogger(__name__)


def _float32_contiguous_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _cpu_tensor_snapshot(
    value: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=dtype, copy=True)


def _resolve_inference_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        if value in {torch.bfloat16, torch.float16}:
            return value
        if value == torch.float32:
            return None
        raise ValueError(f"inference_dtype must be fp32, bf16, or fp16; got {value}")
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "false", "off", "fp32", "float32"}:
        return None
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(
        "inference_dtype must be one of fp32, bf16, or fp16; "
        f"got {value!r}"
    )


class LatentWorldModelEnv:
    """Inference-only Gymnasium-style env backed by a world model and classifier."""

    def __init__(
        self,
        world_model: nn.Module | dict[str, Any],
        classifier: nn.Module | dict[str, Any] | None = None,
        *,
        latent_dim: int,
        token_count: int | None = None,
        token_dim: int | None = None,
        action_dim: int,
        success_threshold: float = 0.5,
        max_episode_steps: int = 64,
        image_shape: tuple[int, int, int] = (4, 4, 3),
        device: str | torch.device = "cpu",
        initial_latent: Any | None = None,
        lang_dim: int = 0,
        initial_lang_emb: Any | None = None,
        proprio_dim: int = 0,
        initial_proprio: Any | None = None,
        num_envs: int = 1,
        inference_dtype: str | torch.dtype | None = None,
        observation_format: str = "numpy",
        freeze_components: bool = False,
    ) -> None:
        self.world_model = _build_component(world_model)
        self.classifier = None if classifier is None else _build_component(classifier)
        self.latent_dim = int(latent_dim)
        self.token_count = None if token_count is None else int(token_count)
        self.token_dim = None if token_dim is None else int(token_dim)
        self.action_dim = int(action_dim)
        self.lang_dim = int(lang_dim)
        self.proprio_dim = int(proprio_dim)
        self.num_envs = int(num_envs)
        self.success_threshold = float(success_threshold)
        self.max_episode_steps = int(max_episode_steps)
        self.image_shape = tuple(int(value) for value in image_shape)
        self.device = torch.device(device)
        self._autocast_dtype = _resolve_inference_dtype(inference_dtype)
        normalized_observation_format = str(observation_format).strip().lower()
        if normalized_observation_format in {"np", "array"}:
            normalized_observation_format = "numpy"
        if normalized_observation_format == "torch":
            normalized_observation_format = "tensor"
        if normalized_observation_format not in {"numpy", "tensor"}:
            raise ValueError(
                "observation_format must be either 'numpy' or 'tensor', "
                f"got {observation_format!r}"
            )
        self.observation_format = normalized_observation_format
        self.freeze_components = bool(freeze_components)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if (self.token_count is None) != (self.token_dim is None):
            raise ValueError("token_count and token_dim must be configured together")
        if self.token_count is not None and self.token_dim is not None:
            if self.token_count <= 0 or self.token_dim <= 0:
                raise ValueError("token_count and token_dim must be positive")
            if self.latent_dim != self.token_count * self.token_dim:
                raise ValueError(
                    "latent_dim must equal token_count * token_dim; "
                    f"{self.latent_dim} != {self.token_count} * {self.token_dim}"
                )
        if self.lang_dim < 0:
            raise ValueError("lang_dim must be non-negative")
        if self.proprio_dim < 0:
            raise ValueError("proprio_dim must be non-negative")
        self.wm_version = 0
        self.classifier_version = 0
        self._initial_latent = initial_latent
        self._initial_lang_emb = initial_lang_emb
        self._initial_proprio = initial_proprio
        self._latent = torch.zeros(
            (self.num_envs, self.latent_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._lang_emb = torch.zeros(
            (self.num_envs, self.lang_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._proprio = torch.zeros(
            (self.num_envs, self.proprio_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._elapsed_steps = np.zeros((self.num_envs,), dtype=np.int64)
        self._task_ids = np.zeros((self.num_envs,), dtype=np.int64)
        self._episode_ids = np.zeros((self.num_envs,), dtype=np.int64)
        self._wm_forward_calls = 0
        self._classifier_forward_calls = 0
        self._wm_forward_time_s = 0.0
        self._classifier_forward_time_s = 0.0
        self._batch_size_sum = 0
        self._batch_size_min: int | None = None
        self._batch_size_max = 0
        self._score_samples: list[np.ndarray] = []
        self._chunk_fallback_warned = False
        self.world_model.to(self.device).eval()
        if self.classifier is not None:
            self.classifier.to(self.device).eval()
        if self.freeze_components:
            self.world_model.requires_grad_(False)
            if self.classifier is not None:
                self.classifier.requires_grad_(False)
        classifier_window = self._classifier_window_size()
        classifier_history_dtype = self._observation_tensor_dtype()
        self._classifier_latent_history = torch.zeros(
            (self.num_envs, classifier_window, self.latent_dim),
            dtype=classifier_history_dtype,
            device=self.device,
        )
        self._classifier_proprio_history = torch.zeros(
            (self.num_envs, classifier_window, self.proprio_dim),
            dtype=classifier_history_dtype,
            device=self.device,
        )

    def _world_model_autocast(self):
        if self._autocast_dtype is None or self.device.type not in {"cuda", "cpu"}:
            return nullcontext()
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self._autocast_dtype,
        )

    def _observation_tensor_dtype(self) -> torch.dtype:
        return torch.float32 if self._autocast_dtype is None else self._autocast_dtype

    def set_success_threshold(self, threshold: float) -> None:
        self.success_threshold = float(threshold)

    def component_state_hashes(self) -> dict[str, str]:
        """Hash immutable inference components for causal-audit boundaries."""

        if self.freeze_components:
            assert_module_frozen(self.world_model, name="world_model")
            if self.classifier is not None:
                assert_module_frozen(self.classifier, name="classifier")
        hashes = {"world_model": module_state_sha256(self.world_model)}
        if self.classifier is not None:
            hashes["classifier"] = module_state_sha256(self.classifier)
        return hashes

    def reset(
        self,
        *,
        task_id: int = 0,
        episode_id: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.reset_slot(0, task_id=task_id, episode_id=episode_id)

    @torch.inference_mode()
    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self.step_slot(0, action)

    def reset_slot(
        self,
        slot_id: int,
        *,
        task_id: int = 0,
        episode_id: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        info = self._reset_slot_state(
            slot_id,
            task_id=int(task_id),
            episode_id=int(episode_id),
        )
        return self._obs(int(slot_id)), info

    def _reset_slot_state(
        self,
        slot_id: int,
        *,
        task_id: int,
        episode_id: int,
    ) -> dict[str, Any]:
        self._validate_slot(slot_id)
        slot = int(slot_id)
        self._task_ids[slot] = int(task_id)
        self._episode_ids[slot] = int(episode_id)
        self._elapsed_steps[slot] = 0
        self._latent[slot] = self._initial_latent_for_slot(slot).detach()
        if self.lang_dim > 0:
            self._lang_emb[slot] = self._initial_lang_for_slot(slot).detach()
        if self.proprio_dim > 0:
            self._proprio[slot] = self._initial_proprio_for_slot(slot).detach()
        self._reset_classifier_history(slot)
        return {
            "slot_id": slot,
            "task_id": int(self._task_ids[slot]),
            "episode_id": int(self._episode_ids[slot]),
        }

    @torch.inference_mode()
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
        infos: list[dict[str, Any]] = []
        for slot_id, (task_id, episode_id) in enumerate(
            zip(task_ids, episode_ids, strict=True)
        ):
            info = self._reset_slot_state(
                slot_id,
                task_id=int(task_id),
                episode_id=int(episode_id),
            )
            infos.append(info)
        return self._obs_batch(list(range(self.num_envs))), infos

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

    def set_initial_lang_embs(self, lang_embs: Any) -> None:
        """Set the language embedding sidecar used by future slot resets."""

        if self.lang_dim <= 0:
            return
        lang = torch.as_tensor(lang_embs, dtype=torch.float32, device=self.device)
        if lang.numel() == self.lang_dim:
            self._initial_lang_emb = (
                lang.reshape(self.lang_dim).detach().cpu().numpy()
            )
            return
        if lang.numel() == self.num_envs * self.lang_dim:
            self._initial_lang_emb = (
                lang.reshape(self.num_envs, self.lang_dim).detach().cpu().numpy()
            )
            return
        raise ValueError(
            f"initial lang_embs have {lang.numel()} values; expected "
            f"{self.lang_dim} or {self.num_envs * self.lang_dim}"
        )

    def set_initial_proprios(self, proprios: Any) -> None:
        """Set the raw proprio state sidecar used by future slot resets."""

        if self.proprio_dim <= 0:
            return
        proprio = torch.as_tensor(proprios, dtype=torch.float32, device=self.device)
        if proprio.numel() == self.proprio_dim:
            self._initial_proprio = (
                proprio.reshape(self.proprio_dim).detach().cpu().numpy()
            )
            return
        if proprio.numel() == self.num_envs * self.proprio_dim:
            self._initial_proprio = (
                proprio.reshape(self.num_envs, self.proprio_dim)
                .detach()
                .cpu()
                .numpy()
            )
            return
        raise ValueError(
            f"initial proprios have {proprio.numel()} values; expected "
            f"{self.proprio_dim} or {self.num_envs * self.proprio_dim}"
        )

    @torch.inference_mode()
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
        batch_size = len(slots)
        self._batch_size_sum += int(batch_size)
        self._batch_size_max = max(self._batch_size_max, int(batch_size))
        self._batch_size_min = (
            int(batch_size)
            if self._batch_size_min is None
            else min(self._batch_size_min, int(batch_size))
        )
        action_t = torch.as_tensor(
            _float32_contiguous_array(actions),
            dtype=torch.float32,
            device=self.device,
        ).reshape(batch_size, -1)
        if action_t.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions have {action_t.shape[-1]} values; expected {self.action_dim}"
            )
        batch = {
            "mode": "predict_next",
            "latent": self._token_grid(
                self._latent[slots].reshape(batch_size, self.latent_dim)
            ),
            "action": action_t.reshape(batch_size, self.action_dim),
            "actions": action_t.reshape(batch_size, 1, self.action_dim),
        }
        if self.lang_dim > 0:
            batch["lang_emb"] = self._lang_emb[slots].reshape(
                batch_size, self.lang_dim
            )
        if self.proprio_dim > 0:
            batch["proprio"] = self._proprio[slots].reshape(
                batch_size, self.proprio_dim
            )
        wm_start = time.perf_counter()
        with self._world_model_autocast():
            wm_out = self.world_model(batch)
        self._wm_forward_calls += 1
        self._wm_forward_time_s += float(time.perf_counter() - wm_start)
        next_latent = self._coerce_latent_shape(
            self._extract_latent(wm_out),
            batch_size=batch_size,
        )
        self._latent[slots] = next_latent.detach()
        next_lang = self._extract_lang_emb(wm_out)
        if next_lang is not None:
            self._lang_emb[slots] = next_lang.reshape(batch_size, self.lang_dim).detach()
        next_proprio = self._extract_proprio(wm_out)
        if next_proprio is not None:
            self._proprio[slots] = next_proprio.reshape(
                batch_size, self.proprio_dim
            ).detach()
        self._elapsed_steps[slots] += 1
        score_start = time.perf_counter()
        with self._world_model_autocast():
            scores = self._score_batch(self._latent[slots], slots=slots)
        if self.classifier is not None:
            self._classifier_forward_calls += 1
            self._classifier_forward_time_s += float(time.perf_counter() - score_start)
        scores_cpu = scores.detach().float().cpu().numpy()
        if self.classifier is not None:
            self._record_scores(scores_cpu)

        observations = self._obs_batch(slots)
        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        for index, slot_id in enumerate(slots):
            score = float(scores_cpu[index])
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
            rewards.append(score)
            terminations.append(terminated)
            truncations.append(truncated)
            infos.append(info)
        return observations, rewards, terminations, truncations, infos

    @torch.inference_mode()
    def chunk_step_batch(
        self,
        actions: Any,
        env_ids: Sequence[int] | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        """Advance a batch of slots by one policy action chunk.

        This is the chunk-batched world-model env boundary: one call accepts
        ``[B, K, A]`` actions and performs one chunk world-model forward.
        """

        slots = (
            list(range(self.num_envs))
            if env_ids is None
            else [int(v) for v in env_ids]
        )
        if not slots:
            empty_rewards = np.zeros((0, 0), dtype=np.float32)
            empty_dones = np.zeros((0, 0), dtype=np.bool_)
            return [], empty_rewards, empty_dones, empty_dones.copy(), []
        for slot_id in slots:
            self._validate_slot(slot_id)
        batch_size = len(slots)
        action_arr = _float32_contiguous_array(actions).reshape(
            batch_size,
            -1,
            self.action_dim,
        )
        action_t = torch.as_tensor(
            action_arr,
            dtype=torch.float32,
            device=self.device,
        )
        chunk_len = int(action_t.shape[1])
        if chunk_len <= 0:
            raise ValueError("actions must include at least one chunk step")
        batch = {
            "mode": "predict_next_chunk",
            "latent": self._token_grid(
                self._latent[slots].reshape(batch_size, self.latent_dim)
            ),
            "actions": action_t,
        }
        if self.lang_dim > 0:
            batch["lang_emb"] = self._lang_emb[slots].reshape(
                batch_size,
                self.lang_dim,
            )
        if self.proprio_dim > 0:
            batch["proprio"] = self._proprio[slots].reshape(
                batch_size,
                self.proprio_dim,
            )
        wm_start = time.perf_counter()
        try:
            with self._world_model_autocast():
                wm_out = self.world_model(batch)
        except (ValueError, NotImplementedError) as exc:
            if not _looks_like_missing_chunk_mode(exc):
                raise
            return self._chunk_step_batch_fallback(action_arr, slots)
        wm_elapsed_s = float(time.perf_counter() - wm_start)
        if not isinstance(wm_out, dict) or "hidden_seq" not in wm_out:
            return self._chunk_step_batch_fallback(action_arr, slots)

        hidden_seq = torch.as_tensor(
            wm_out["hidden_seq"],
            dtype=torch.float32,
            device=self.device,
        )
        if hidden_seq.ndim < 3 or int(hidden_seq.shape[0]) != batch_size:
            raise ValueError(
                "world_model hidden_seq must be shaped [B,K,...], got "
                f"{tuple(hidden_seq.shape)}"
            )
        if int(hidden_seq.shape[1]) != chunk_len:
            raise ValueError(
                f"world_model hidden_seq time dim {int(hidden_seq.shape[1])} "
                f"!= action chunk length {chunk_len}"
            )
        flat_latents = self._coerce_latent_shape(
            hidden_seq.reshape(batch_size * chunk_len, *hidden_seq.shape[2:]),
            batch_size=batch_size * chunk_len,
        )
        latent_seq = flat_latents.reshape(batch_size, chunk_len, self.latent_dim)
        final_latent = (
            self._coerce_latent_shape(
                self._extract_latent(wm_out),
                batch_size=batch_size,
            )
            if any(key in wm_out for key in ("next_latent", "hidden", "latent", "state"))
            else latent_seq[:, -1]
        )
        self._latent[slots] = final_latent.detach()

        lang_seq = self._chunk_sidecar_sequence(
            wm_out,
            keys=("lang_emb", "lang"),
            dim=self.lang_dim,
            batch_size=batch_size,
            chunk_len=chunk_len,
        )
        if lang_seq is not None:
            self._lang_emb[slots] = lang_seq[:, -1].detach()
        proprio_seq = self._chunk_sidecar_sequence(
            wm_out,
            keys=("proprio_seq", "proprio", "state"),
            dim=self.proprio_dim,
            batch_size=batch_size,
            chunk_len=chunk_len,
        )
        if proprio_seq is not None:
            self._proprio[slots] = proprio_seq[:, -1].detach()

        self._wm_forward_calls += 1
        self._wm_forward_time_s += wm_elapsed_s
        self._batch_size_sum += int(batch_size)
        self._batch_size_max = max(self._batch_size_max, int(batch_size))
        self._batch_size_min = (
            int(batch_size)
            if self._batch_size_min is None
            else min(self._batch_size_min, int(batch_size))
        )

        score_start = time.perf_counter()
        recorded_scores: np.ndarray | None = None
        classifier_granularity = (
            self._classifier_granularity()
            if self.classifier is not None
            else None
        )
        if self.classifier is None:
            rewards = np.zeros((batch_size, chunk_len), dtype=np.float32)
        elif classifier_granularity == "chunk":
            classifier_latent = self._pool_classifier_chunk_sequence(latent_seq)
            classifier_proprio = (
                self._pool_classifier_chunk_sequence(proprio_seq)
                if proprio_seq is not None
                else None
            )
            classifier_lang = lang_seq[:, -1] if lang_seq is not None else None
            with self._world_model_autocast():
                scores = self._score_batch(
                    classifier_latent,
                    slots=slots,
                    proprio=classifier_proprio,
                    lang_emb=classifier_lang,
                )
            recorded_scores = scores.detach().float().cpu().numpy()
            rewards = np.zeros((batch_size, chunk_len), dtype=np.float32)
            rewards[:, -1] = recorded_scores
        else:
            repeated_slots = [slot_id for slot_id in slots for _ in range(chunk_len)]
            with self._world_model_autocast():
                scores = self._score_batch(
                    flat_latents,
                    slots=repeated_slots,
                    proprio=(
                        proprio_seq.reshape(batch_size * chunk_len, self.proprio_dim)
                        if proprio_seq is not None
                        else None
                    ),
                    lang_emb=(
                        lang_seq.reshape(batch_size * chunk_len, self.lang_dim)
                        if lang_seq is not None
                        else None
                    ),
                )
            rewards = scores.reshape(batch_size, chunk_len).detach().cpu().numpy()
            rewards = rewards.astype(np.float32, copy=False)
            recorded_scores = rewards
        if self.classifier is not None:
            self._classifier_forward_calls += 1
            self._classifier_forward_time_s += float(time.perf_counter() - score_start)
            if recorded_scores is not None:
                self._record_scores(recorded_scores)
        slot_index = np.asarray(slots, dtype=np.int64)
        old_elapsed = self._elapsed_steps[slot_index].copy()
        success_by_slot = (rewards >= self.success_threshold).any(axis=1)
        if self.classifier is None:
            classifier_evaluations = np.zeros((batch_size,), dtype=np.int64)
            classifier_success_evaluations = np.zeros(
                (batch_size,), dtype=np.int64
            )
        elif classifier_granularity == "chunk":
            classifier_evaluations = np.ones((batch_size,), dtype=np.int64)
            classifier_success_evaluations = success_by_slot.astype(np.int64)
        else:
            classifier_evaluations = np.full(
                (batch_size,), chunk_len, dtype=np.int64
            )
            classifier_success_evaluations = (
                rewards >= self.success_threshold
            ).sum(axis=1, dtype=np.int64)
        terminations = np.zeros((batch_size, chunk_len), dtype=np.bool_)
        terminations[:, -1] = success_by_slot
        timeout_by_slot = old_elapsed + chunk_len >= self.max_episode_steps
        truncations = np.zeros((batch_size, chunk_len), dtype=np.bool_)
        truncations[:, -1] = timeout_by_slot
        self._elapsed_steps[slot_index] = old_elapsed + chunk_len

        observations = self._obs_batch(slots)
        infos: list[dict[str, Any]] = []
        for index, slot_id in enumerate(slots):
            info = {
                "slot_id": int(slot_id),
                "task_id": int(self._task_ids[slot_id]),
                "episode_id": int(self._episode_ids[slot_id]),
                "elapsed_steps": int(self._elapsed_steps[slot_id]),
                "success": bool(success_by_slot[index]),
                "success_score": float(rewards[index, -1]),
                "classifier_evaluations": int(classifier_evaluations[index]),
                "classifier_success_evaluations": int(
                    classifier_success_evaluations[index]
                ),
                "wm_action": np.asarray(
                    action_arr[index, -1],
                    dtype=np.float32,
                ).reshape(self.action_dim),
                "wm_version": self.wm_version,
                "classifier_version": self.classifier_version,
            }
            infos.append(info)
        return observations, rewards, terminations, truncations, infos

    def _chunk_step_batch_fallback(
        self,
        actions_np: np.ndarray,
        slots: Sequence[int],
    ) -> tuple[
        list[dict[str, Any]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        """Preserve behavior for world models without chunk prediction support."""

        if not self._chunk_fallback_warned:
            self._chunk_fallback_warned = True
            _LOGGER.warning(
                "world model lacks chunk mode (predict_next_chunk); "
                "chunk_step_batch is falling back to per-step step_batch"
            )
        batch_size = len(slots)
        actions_np = np.asarray(actions_np, dtype=np.float32).reshape(
            batch_size,
            -1,
            self.action_dim,
        )
        chunk_len = int(actions_np.shape[1])
        rewards = np.zeros((batch_size, chunk_len), dtype=np.float32)
        terminations = np.zeros((batch_size, chunk_len), dtype=np.bool_)
        truncations = np.zeros((batch_size, chunk_len), dtype=np.bool_)
        observations = self._obs_batch(slots)
        infos: list[dict[str, Any]] = [
            {
                "slot_id": int(slot_id),
                "task_id": int(self._task_ids[slot_id]),
                "episode_id": int(self._episode_ids[slot_id]),
            }
            for slot_id in slots
        ]
        active = np.ones((batch_size,), dtype=np.bool_)
        for action_index in range(chunk_len):
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                break
            active_slots = [int(slots[index]) for index in active_indices]
            step_out = self.step_batch(
                actions_np[active_indices, action_index],
                env_ids=active_slots,
            )
            next_obs_list, step_rewards, step_terms, step_truncs, step_infos = step_out
            for local_index, batch_index in enumerate(active_indices):
                rewards[batch_index, action_index] = float(step_rewards[local_index])
                terminated = bool(step_terms[local_index])
                truncated = bool(step_truncs[local_index])
                terminations[batch_index, action_index] = terminated
                truncations[batch_index, action_index] = truncated
                observations[batch_index] = dict(next_obs_list[local_index])
                infos[batch_index] = dict(step_infos[local_index] or {})
                if terminated or truncated:
                    active[batch_index] = False
                    if action_index + 1 < chunk_len:
                        if terminated:
                            terminations[batch_index, action_index + 1 :] = True
                        else:
                            truncations[batch_index, action_index + 1 :] = True
        return observations, rewards, terminations, truncations, infos

    def get_metrics(self, *, reset: bool = False) -> dict[str, float]:
        """Return world-model env inference counters for runtime validation."""

        score_values = (
            np.concatenate(self._score_samples)
            if self._score_samples
            else np.asarray([], dtype=np.float32)
        )
        score_count = int(score_values.size)
        metrics = {
            "model_forwards": float(self._wm_forward_calls),
            "wm_forward_calls": float(self._wm_forward_calls),
            "classifier_forward_calls": float(self._classifier_forward_calls),
            "wm_forward_time_s": float(self._wm_forward_time_s),
            "classifier_forward_time_s": float(self._classifier_forward_time_s),
            "batch_size_sum": float(self._batch_size_sum),
            "batch_size_avg": float(
                self._batch_size_sum / max(1, self._wm_forward_calls)
            ),
            "batch_size_min": float(
                0 if self._batch_size_min is None else self._batch_size_min
            ),
            "batch_size_max": float(self._batch_size_max),
            "score_sum": float(score_values.sum()) if score_count else 0.0,
            "score_count": float(score_count),
            "score_mean": float(score_values.mean()) if score_count else 0.0,
            "score_p50": (
                float(np.percentile(score_values, 50)) if score_count else 0.0
            ),
            "score_p90": (
                float(np.percentile(score_values, 90)) if score_count else 0.0
            ),
            "score_max": float(score_values.max()) if score_count else 0.0,
        }
        if reset:
            self._wm_forward_calls = 0
            self._classifier_forward_calls = 0
            self._wm_forward_time_s = 0.0
            self._classifier_forward_time_s = 0.0
            self._batch_size_sum = 0
            self._batch_size_min = None
            self._batch_size_max = 0
            self._score_samples.clear()
        return metrics

    def _record_scores(self, scores: torch.Tensor | np.ndarray) -> None:
        values = (
            scores.detach().float().reshape(-1).cpu().numpy()
            if isinstance(scores, torch.Tensor)
            else np.asarray(scores, dtype=np.float32).reshape(-1)
        )
        if values.size:
            self._score_samples.append(values.astype(np.float32, copy=True))

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
            self.world_model.load_state_dict(state_dict, assign=True)
        self.world_model.to(self.device).eval()
        if self.freeze_components:
            self.world_model.requires_grad_(False)
        self.wm_version = int(version)

    def load_classifier_state(self, state_dict: dict[str, Any], version: int) -> None:
        if self.classifier is None:
            if state_dict:
                raise RuntimeError("cannot load classifier state without a classifier module")
        elif state_dict:
            self.classifier.load_state_dict(state_dict, assign=True)
            self.classifier.to(self.device).eval()
            if self.freeze_components:
                self.classifier.requires_grad_(False)
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
        transition = {
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
        if "lang_emb" in obs:
            transition["lang_emb"] = np.asarray(
                obs["lang_emb"],
                dtype=np.float32,
            ).reshape(self.lang_dim)
        if "proprio" in obs:
            transition["proprio"] = np.asarray(
                obs["proprio"],
                dtype=np.float32,
            ).reshape(self.proprio_dim)
        return transition

    def _obs(self, slot_id: int = 0) -> dict[str, Any]:
        return self._obs_batch([int(slot_id)])[0]

    def _obs_batch(self, slot_ids: Sequence[int]) -> list[dict[str, Any]]:
        slots = [int(slot_id) for slot_id in slot_ids]
        if not slots:
            return []
        for slot_id in slots:
            self._validate_slot(slot_id)
        if self.observation_format == "tensor":
            obs_dtype = self._observation_tensor_dtype()
            latent_batch: Any = _cpu_tensor_snapshot(
                self._latent[slots],
                dtype=obs_dtype,
            )
        else:
            latent_batch = (
                self._latent[slots]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )
        latent_batch = self._token_grid(latent_batch)

        lang_batch: Any | None = None
        if self.lang_dim > 0:
            if self.observation_format == "tensor":
                lang_batch = _cpu_tensor_snapshot(
                    self._lang_emb[slots],
                    dtype=self._observation_tensor_dtype(),
                )
            else:
                lang_batch = (
                    self._lang_emb[slots]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )

        proprio_batch: Any | None = None
        if self.proprio_dim > 0:
            if self.observation_format == "tensor":
                proprio_batch = _cpu_tensor_snapshot(
                    self._proprio[slots],
                    dtype=self._observation_tensor_dtype(),
                )
            else:
                proprio_batch = (
                    self._proprio[slots]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )

        observations: list[dict[str, Any]] = []
        for index, slot_id in enumerate(slots):
            obs = {
                "latent": latent_batch[index],
                "task_id": int(self._task_ids[slot_id]),
                "episode_id": int(self._episode_ids[slot_id]),
                "step": int(self._elapsed_steps[slot_id]),
                "task_description": f"task {self._task_ids[slot_id]}",
                "is_first": bool(self._elapsed_steps[slot_id] == 0),
            }
            if lang_batch is not None:
                obs["lang_emb"] = lang_batch[index]
            if proprio_batch is not None:
                proprio = proprio_batch[index]
                obs["proprio"] = proprio
                obs["state"] = proprio
            observations.append(obs)
        return observations

    def _score(self, latent: torch.Tensor) -> float:
        return float(self._score_batch(latent.reshape(1, self.latent_dim))[0])

    def _score_batch(
        self,
        latent: torch.Tensor,
        *,
        slots: Sequence[int] | None = None,
        proprio: torch.Tensor | None = None,
        lang_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.classifier is None:
            return torch.zeros(latent.shape[0], dtype=torch.float32, device=self.device)
        classifier_cfg = getattr(self.classifier, "cfg", None)
        window = getattr(classifier_cfg, "window", None)
        if window is None:
            raw = self.classifier(
                self._token_grid(latent.reshape(latent.shape[0], self.latent_dim))
            )
        else:
            (
                latent_window,
                proprio_window,
                latent_updates,
                proprio_updates,
            ) = self._classifier_temporal_windows(
                latent,
                window=int(window),
                slots=slots,
                proprio=proprio,
            )
            raw = self.classifier(
                self._token_grid(latent_window),
                **self._classifier_sidecars(
                    latent.shape[0],
                    int(window),
                    slots,
                    proprio_window=proprio_window,
                    lang_emb=lang_emb,
                ),
            )
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
        ):
            if score_tensor.shape[-1] == 1:
                score_tensor = torch.sigmoid(score_tensor[..., 0])
            elif score_tensor.shape[-1] == 2:
                score_tensor = torch.softmax(score_tensor, dim=-1)[..., 1]
            else:
                score_tensor = score_tensor[..., -1]
        elif score_tensor.ndim > 1 and score_tensor.shape[0] == latent.shape[0]:
            score_tensor = score_tensor[..., -1]
        scores = score_tensor.reshape(-1)
        if scores.numel() != latent.shape[0]:
            raise ValueError(
                f"classifier returned {scores.numel()} scores; expected {latent.shape[0]}"
            )
        if window is not None:
            self._commit_classifier_history(latent_updates, proprio_updates)
        return scores

    def _token_grid(
        self,
        latent: torch.Tensor | np.ndarray,
    ) -> torch.Tensor | np.ndarray:
        if self.token_count is None or self.token_dim is None:
            return latent
        if int(latent.shape[-1]) != self.latent_dim:
            raise ValueError(
                "latent tokenization expects trailing latent_dim "
                f"{self.latent_dim}, got {tuple(latent.shape)}"
            )
        return latent.reshape(
            *tuple(int(dim) for dim in latent.shape[:-1]),
            self.token_count,
            self.token_dim,
        )

    def _classifier_window_size(self) -> int:
        classifier_cfg = getattr(self.classifier, "cfg", None)
        window = getattr(classifier_cfg, "window", 1)
        return max(1, int(window or 1))

    def _classifier_granularity(self) -> str:
        classifier_cfg = getattr(self.classifier, "cfg", None)
        granularity = str(getattr(classifier_cfg, "granularity", "action"))
        if granularity not in {"action", "chunk"}:
            raise ValueError(
                "classifier granularity must be 'action' or 'chunk', got "
                f"{granularity!r}"
            )
        return granularity

    def _pool_classifier_chunk_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        classifier_cfg = getattr(self.classifier, "cfg", None)
        pool = str(getattr(classifier_cfg, "chunk_pool", "last"))
        if pool == "last":
            return sequence[:, -1]
        if pool == "first":
            return sequence[:, 0]
        if pool == "mean":
            return sequence.mean(dim=1)
        raise ValueError(
            "chunk classifier pool must be 'last', 'first', or 'mean', got "
            f"{pool!r}"
        )

    def _ensure_classifier_history_window(self, window: int) -> None:
        window = max(1, int(window))
        if int(self._classifier_latent_history.shape[1]) == window:
            return
        self._classifier_latent_history = (
            self._latent[:, None, :]
            .expand(self.num_envs, window, self.latent_dim)
            .to(dtype=self._observation_tensor_dtype())
            .contiguous()
        )
        self._classifier_proprio_history = (
            self._proprio[:, None, :]
            .expand(self.num_envs, window, self.proprio_dim)
            .to(dtype=self._observation_tensor_dtype())
            .contiguous()
        )

    def _reset_classifier_history(self, slot_id: int) -> None:
        window = self._classifier_window_size()
        self._ensure_classifier_history_window(window)
        slot = int(slot_id)
        self._classifier_latent_history[slot] = (
            self._latent[slot]
            .reshape(1, self.latent_dim)
            .expand(window, self.latent_dim)
        )
        if self.proprio_dim > 0:
            self._classifier_proprio_history[slot] = (
                self._proprio[slot]
                .reshape(1, self.proprio_dim)
                .expand(window, self.proprio_dim)
            )

    def _classifier_temporal_windows(
        self,
        latent: torch.Tensor,
        *,
        window: int,
        slots: Sequence[int] | None,
        proprio: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
    ]:
        if int(window) <= 0:
            raise ValueError(f"classifier window must be positive, got {window}")
        self._ensure_classifier_history_window(int(window))
        batch_size = int(latent.shape[0])
        slot_ids = self._slot_ids_for_classifier_batch(batch_size, slots)
        latent_rows = latent.reshape(batch_size, self.latent_dim).to(
            dtype=self._classifier_latent_history.dtype
        )
        proprio_rows = None
        if self.proprio_dim > 0:
            proprio_rows = (
                torch.as_tensor(proprio, dtype=torch.float32, device=self.device)
                if proprio is not None
                else self._proprio[slot_ids]
            ).reshape(batch_size, self.proprio_dim).to(
                dtype=self._classifier_proprio_history.dtype
            )

        local_latent: dict[int, torch.Tensor] = {}
        local_proprio: dict[int, torch.Tensor] = {}
        latent_windows = torch.empty(
            (batch_size, int(window), self.latent_dim),
            dtype=self._classifier_latent_history.dtype,
            device=self.device,
        )
        proprio_windows = (
            torch.empty(
                (batch_size, int(window), self.proprio_dim),
                dtype=self._classifier_proprio_history.dtype,
                device=self.device,
            )
            if proprio_rows is not None
            else None
        )
        for index, slot_id in enumerate(slot_ids):
            slot = int(slot_id)
            hist = local_latent.get(slot)
            if hist is None:
                hist = self._classifier_latent_history[slot]
            next_hist = latent_windows[index]
            if int(window) > 1:
                next_hist[:-1].copy_(hist[1:])
            next_hist[-1].copy_(latent_rows[index])
            local_latent[slot] = next_hist

            if proprio_rows is not None and proprio_windows is not None:
                phist = local_proprio.get(slot)
                if phist is None:
                    phist = self._classifier_proprio_history[slot]
                next_phist = proprio_windows[index]
                if int(window) > 1:
                    next_phist[:-1].copy_(phist[1:])
                next_phist[-1].copy_(proprio_rows[index])
                local_proprio[slot] = next_phist

        return (
            latent_windows,
            proprio_windows,
            local_latent,
            local_proprio,
        )

    def _commit_classifier_history(
        self,
        latent_updates: dict[int, torch.Tensor],
        proprio_updates: dict[int, torch.Tensor],
    ) -> None:
        for slot_id, history in latent_updates.items():
            self._classifier_latent_history[int(slot_id)] = history.detach()
        for slot_id, history in proprio_updates.items():
            self._classifier_proprio_history[int(slot_id)] = history.detach()

    def _slot_ids_for_classifier_batch(
        self,
        batch_size: int,
        slots: Sequence[int] | None,
    ) -> list[int]:
        if slots is None:
            slot_ids = list(range(int(batch_size)))
        else:
            slot_ids = [int(slot) for slot in slots]
        if len(slot_ids) != int(batch_size):
            raise ValueError(
                f"classifier slots length {len(slot_ids)} != batch size {batch_size}"
            )
        for slot_id in slot_ids:
            self._validate_slot(slot_id)
        return slot_ids

    def _classifier_sidecars(
        self,
        batch_size: int,
        window: int,
        slots: Sequence[int] | None,
        *,
        proprio_window: torch.Tensor | None = None,
        lang_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        slot_ids = self._slot_ids_for_classifier_batch(int(batch_size), slots)
        sidecars: dict[str, torch.Tensor] = {
            "task_ids": torch.as_tensor(
                [int(self._task_ids[slot]) for slot in slot_ids],
                dtype=torch.long,
                device=self.device,
            )
        }
        if self.proprio_dim > 0:
            if proprio_window is None:
                raise ValueError("classifier proprio history was not built")
            sidecars["proprio"] = proprio_window.reshape(
                int(batch_size),
                int(window),
                self.proprio_dim,
            )
        if self.lang_dim > 0:
            lang_t = (
                torch.as_tensor(lang_emb, dtype=torch.float32, device=self.device)
                if lang_emb is not None
                else self._lang_emb[slot_ids]
            )
            sidecars["lang_emb"] = lang_t.reshape(int(batch_size), self.lang_dim)
        return sidecars

    def _chunk_sidecar_sequence(
        self,
        value: Any,
        *,
        keys: Sequence[str],
        dim: int,
        batch_size: int,
        chunk_len: int,
    ) -> torch.Tensor | None:
        if dim <= 0 or not isinstance(value, dict):
            return None
        for key in keys:
            if key not in value or value[key] is None:
                continue
            tensor = torch.as_tensor(
                value[key],
                dtype=torch.float32,
                device=self.device,
            )
            if tensor.numel() == int(batch_size) * int(chunk_len) * int(dim):
                return tensor.reshape(int(batch_size), int(chunk_len), int(dim))
            if tensor.numel() == int(batch_size) * int(dim):
                return tensor.reshape(int(batch_size), 1, int(dim)).expand(
                    -1,
                    int(chunk_len),
                    -1,
                )
            raise ValueError(
                f"world_model returned {key} with shape {tuple(tensor.shape)}; "
                f"expected [B,K,{dim}] or [B,{dim}]"
            )
        return None

    def _extract_latent(self, value: Any) -> torch.Tensor:
        if isinstance(value, dict):
            for key in ("next_latent", "hidden", "latent", "state"):
                if key in value:
                    return torch.as_tensor(value[key], dtype=torch.float32, device=self.device)
            raise ValueError(
                "world_model output dict must include next_latent, hidden, latent, or state"
            )
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _coerce_latent_shape(
        self,
        latent: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        batch = int(batch_size)
        if latent.numel() == batch * self.latent_dim:
            return latent.reshape(batch, self.latent_dim)
        if latent.ndim >= 3 and int(latent.shape[0]) == batch:
            token_count = int(np.prod(tuple(int(dim) for dim in latent.shape[1:-1])))
            if (
                token_count > 0
                and self.latent_dim % token_count == 0
                and self.latent_dim < int(np.prod(tuple(int(dim) for dim in latent.shape[1:])))
            ):
                visual_dim = self.latent_dim // token_count
                if 0 < visual_dim < int(latent.shape[-1]):
                    return latent[..., :visual_dim].reshape(batch, self.latent_dim)
        returned = int(latent.reshape(batch, -1).shape[-1]) if latent.numel() % batch == 0 else int(latent.numel())
        raise ValueError(
            f"world_model returned {returned} latent values; expected {self.latent_dim}"
        )

    def _extract_lang_emb(self, value: Any) -> torch.Tensor | None:
        if self.lang_dim <= 0 or not isinstance(value, dict):
            return None
        for key in ("lang_emb", "lang"):
            if key in value:
                lang = torch.as_tensor(
                    value[key],
                    dtype=torch.float32,
                    device=self.device,
                )
                if lang.ndim == 0 or int(lang.shape[-1]) != self.lang_dim:
                    raise ValueError(
                        f"world_model returned lang embedding with shape {tuple(lang.shape)}; "
                        f"expected trailing dim {self.lang_dim}"
                    )
                if lang.reshape(-1).numel() % self.lang_dim != 0:
                    raise ValueError(
                        f"world_model returned {lang.numel()} lang values; "
                        f"expected a multiple of {self.lang_dim}"
                    )
                return lang
        return None

    def _extract_proprio(self, value: Any) -> torch.Tensor | None:
        if self.proprio_dim <= 0 or not isinstance(value, dict):
            return None
        for key in ("proprio", "state"):
            if key in value:
                proprio = torch.as_tensor(
                    value[key],
                    dtype=torch.float32,
                    device=self.device,
                )
                if proprio.ndim == 0 or int(proprio.shape[-1]) != self.proprio_dim:
                    raise ValueError(
                        f"world_model returned proprio with shape {tuple(proprio.shape)}; "
                        f"expected trailing dim {self.proprio_dim}"
                    )
                if proprio.reshape(-1).numel() % self.proprio_dim != 0:
                    raise ValueError(
                        f"world_model returned {proprio.numel()} proprio values; "
                        f"expected a multiple of {self.proprio_dim}"
                    )
                return proprio
        return None

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

    def _initial_lang_for_slot(self, slot_id: int) -> torch.Tensor:
        if self.lang_dim <= 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        if self._initial_lang_emb is None:
            return torch.zeros(self.lang_dim, dtype=torch.float32, device=self.device)
        lang = torch.as_tensor(
            self._initial_lang_emb,
            dtype=torch.float32,
            device=self.device,
        )
        if lang.numel() == self.lang_dim:
            return lang.reshape(self.lang_dim)
        if lang.numel() == self.num_envs * self.lang_dim:
            return lang.reshape(self.num_envs, self.lang_dim)[slot_id]
        raise ValueError(
            f"initial_lang_emb has {lang.numel()} values; expected {self.lang_dim} "
            f"or {self.num_envs * self.lang_dim}"
        )

    def _initial_proprio_for_slot(self, slot_id: int) -> torch.Tensor:
        if self.proprio_dim <= 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        if self._initial_proprio is None:
            return torch.zeros(
                self.proprio_dim,
                dtype=torch.float32,
                device=self.device,
            )
        proprio = torch.as_tensor(
            self._initial_proprio,
            dtype=torch.float32,
            device=self.device,
        )
        if proprio.numel() == self.proprio_dim:
            return proprio.reshape(self.proprio_dim)
        if proprio.numel() == self.num_envs * self.proprio_dim:
            return proprio.reshape(self.num_envs, self.proprio_dim)[slot_id]
        raise ValueError(
            f"initial_proprio has {proprio.numel()} values; expected {self.proprio_dim} "
            f"or {self.num_envs * self.proprio_dim}"
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
    reserved = {
        "target",
        "_target_",
        "class_path",
        "kwargs",
        "init_args",
        "_recursive_",
        "_convert_",
        "_partial_",
    }
    kwargs = {str(key): value for key, value in cfg.items() if key not in reserved}
    kwargs.update(dict(cfg.get("init_args", {}) or {}))
    kwargs.update(dict(cfg.get("kwargs", {}) or {}))
    if ":" in str(target):
        module_name, class_name = str(target).split(":", 1)
    else:
        module_name, class_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def _looks_like_missing_chunk_mode(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "predict_next_chunk" in message
        or "unknown" in message
        and "mode" in message
        or "notimplemented" in type(exc).__name__.lower()
    )
