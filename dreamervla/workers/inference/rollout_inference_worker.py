"""Model-agnostic Ray inference worker for cold-start rollout collection.

Runs a config-injected rollout bundle. One batched forward yields an action and
obs_embedding per env plus optional lang_emb sidecars, with isolated per-env
extractor history.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch

from dreamervla.runtime.rollout.action_chunk_queue import ActionChunkQueue
from dreamervla.runtime.rollout.oft_collect import process_action
from dreamervla.scheduler.worker import Worker
from dreamervla.workers.inference.rollout_contract import RolloutBatchOutput


def _build_from_cfg(cfg: dict[str, Any]) -> Any:
    target = cfg.get("target") or cfg.get("_target_") or cfg.get("class_path")
    if not target:
        raise ValueError("component config must include target/_target_/class_path")
    kwargs = dict(cfg.get("kwargs", {}))
    if ":" in str(target):
        module_name, class_name = str(target).split(":", 1)
    else:
        module_name, class_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)


class RolloutInferenceWorker(Worker):
    """Run a config-injected rollout bundle for cold-start collection."""

    def __init__(self, model_cfg: dict[str, Any], init_ckpt: dict[str, Any], num_envs: int) -> None:
        super().__init__()
        self._cfg = dict(model_cfg)
        self._init_ckpt = dict(init_ckpt)
        self._num_envs = int(num_envs)
        self._action_dim = int(self._cfg.get("action_dim", 7))
        self._action_steps = max(1, int(self._cfg.get("action_steps", 1)))
        self._emit_hidden_sidecar = bool(self._cfg.get("emit_hidden_sidecar", True))
        self._bundle: Any | None = None
        self._actions_are_env_ready = False
        self._extractors: list[Any] = []
        self._action_queues = [
            ActionChunkQueue(action_dim=self._action_dim, action_steps=self._action_steps)
            for _ in range(self._num_envs)
        ]

    def init(self) -> None:
        decoder_cfg = dict(self._cfg["decoder"])
        decoder_kwargs = dict(decoder_cfg.get("kwargs", {}))
        target = str(decoder_cfg.get("target") or decoder_cfg.get("_target_") or "")
        if target.endswith(
            (
                "oft_rollout:OFTRolloutBundle",
                "oft_rollout.OFTRolloutBundle",
            )
        ):
            decoder_kwargs.setdefault("device", self.device)
        decoder_cfg["kwargs"] = decoder_kwargs
        self._bundle = _build_from_cfg(decoder_cfg)
        self._actions_are_env_ready = bool(getattr(self._bundle, "actions_are_env_ready", False))
        if hasattr(self._bundle, "to"):
            self._bundle.to(self.device)
        self._extractors = [self._bundle.make_extractor() for _ in range(self._num_envs)]

    @torch.no_grad()
    def forward_batch(
        self,
        obs_batch: list[dict[str, Any]],
        env_ids: list[int],
    ) -> dict[str, list[Any]]:
        bundle = self._require_bundle()
        if len(obs_batch) != len(env_ids):
            raise ValueError("obs_batch and env_ids must have the same length")

        refill_positions = [
            position
            for position, env_id in enumerate(env_ids)
            if not self._action_queues[int(env_id)].has_pending
        ]
        # Hidden sidecars describe every environment step, so that route still
        # needs one prediction per input observation.  Action-only collection can
        # execute a cached open-loop chunk without rerunning the policy until the
        # corresponding queue needs a refill.
        prediction_positions = (
            list(range(len(env_ids))) if self._emit_hidden_sidecar else refill_positions
        )
        preps = [
            self._extractors[int(env_ids[position])].prepare(
                obs_batch[position],
                str(obs_batch[position].get("task_description", "")),
            )
            for position in prediction_positions
        ]
        results_by_position: dict[int, Any] = {}
        if preps:
            if not self._emit_hidden_sidecar and hasattr(bundle, "predict_actions_batch"):
                action_chunks = bundle.predict_actions_batch(preps)
                results = [(action_chunk, None) for action_chunk in action_chunks]
            else:
                results = bundle.predict_batch(preps)
            results_by_position = dict(zip(prediction_positions, results, strict=True))

        actions: list[np.ndarray] = []
        hidden: list[np.ndarray] = []
        lang: list[np.ndarray | None] = []
        has_lang = False
        for position, env_id in enumerate(env_ids):
            # Gripper post-process here (single point for the ray path); the EnvWorker
            # must NOT re-apply it. Without it grasping/success fails.
            env_index = int(env_id)
            queue = self._action_queues[env_index]
            if not queue.has_pending:
                result = results_by_position.get(position)
                if result is None:
                    raise RuntimeError(
                        f"missing action-chunk prediction for refill env_id={env_index}"
                    )
                action_chunk, _flat_hidden = result
                queue.refill(np.asarray(action_chunk, dtype=np.float32))
            queued_action = queue.pop()
            action = (
                np.asarray(queued_action, dtype=np.float32)
                if self._actions_are_env_ready
                else process_action(queued_action)
            )[: self._action_dim]
            actions.append(action)
            if self._emit_hidden_sidecar:
                result = results_by_position[position]
                _action_chunk, flat_hidden = result
                obs_embedding = (
                    flat_hidden.numpy()
                    if hasattr(flat_hidden, "numpy")
                    else np.asarray(flat_hidden)
                )
                hidden.append(obs_embedding.astype(np.float16, copy=False))
                lang_emb = _optional_lang_emb(result)
                if lang_emb is None:
                    lang.append(None)
                else:
                    has_lang = True
                    lang.append(np.asarray(lang_emb, dtype=np.float16).reshape(-1))
        sidecars = {"obs_embedding": hidden} if self._emit_hidden_sidecar else {}
        if self._emit_hidden_sidecar and has_lang:
            sidecars["lang_emb"] = lang
        return RolloutBatchOutput(actions=actions, sidecars=sidecars).to_compat_dict()

    def reset_states(self, env_ids: list[int]) -> None:
        bundle = self._require_bundle()
        for env_id in env_ids:
            extractor = self._extractors[int(env_id)]
            if hasattr(extractor, "reset"):
                extractor.reset()
            else:
                self._extractors[int(env_id)] = bundle.make_extractor()
            self._action_queues[int(env_id)].clear()

    def pull_weights(self, store_name: str, key: str, local_version: int) -> int | None:
        """No-op weight sync for the async overlap loop.

        OFT online cotrain drives the env with the fixed OFT base policy (open-loop
        action chunk); the learned actor is trained only in imagination, so the rollout
        policy is never updated and there is nothing to pull. Returning None leaves the
        caller's local version unchanged.
        """
        return None

    def _require_bundle(self) -> Any:
        if self._bundle is None:
            raise RuntimeError("RolloutInferenceWorker.init() has not been called")
        return self._bundle


def _optional_lang_emb(result: Any) -> Any | None:
    if hasattr(result, "lang_emb"):
        return result.lang_emb
    try:
        if len(result) > 2:
            return result[2]
    except TypeError:
        return None
    return None
