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

from dreamervla.runtime.action_chunk_queue import ActionChunkQueue
from dreamervla.runtime.oft_collect import process_action
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
        self._extractors: list[Any] = []
        self._action_queues = [
            ActionChunkQueue(action_dim=self._action_dim, action_steps=self._action_steps)
            for _ in range(self._num_envs)
        ]

    def init(self) -> None:
        decoder_cfg = dict(self._cfg["decoder"])
        decoder_kwargs = dict(decoder_cfg.get("kwargs", {}))
        target = str(decoder_cfg.get("target") or decoder_cfg.get("_target_") or "")
        if target.endswith("oft_rollout:OFTRolloutBundle") or target.endswith(
            "oft_rollout.OFTRolloutBundle"
        ):
            decoder_kwargs.setdefault("device", self.device)
        decoder_cfg["kwargs"] = decoder_kwargs
        self._bundle = _build_from_cfg(decoder_cfg)
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
        preps = [
            self._extractors[int(env_id)].prepare(obs, str(obs.get("task_description", "")))
            for env_id, obs in zip(env_ids, obs_batch, strict=True)
        ]
        results = bundle.predict_batch(preps)
        actions: list[np.ndarray] = []
        hidden: list[np.ndarray] = []
        lang: list[np.ndarray | None] = []
        has_lang = False
        for env_id, result in zip(env_ids, results, strict=True):
            action_chunk, flat_hidden = result
            # Gripper post-process here (single point for the ray path); the EnvWorker
            # must NOT re-apply it. Without it grasping/success fails.
            env_index = int(env_id)
            queue = self._action_queues[env_index]
            if not queue.has_pending:
                queue.refill(np.asarray(action_chunk, dtype=np.float32))
            action = process_action(queue.pop())[: self._action_dim]
            obs_embedding = (
                flat_hidden.numpy() if hasattr(flat_hidden, "numpy") else np.asarray(flat_hidden)
            )
            actions.append(action)
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
