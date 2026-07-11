"""Policy-only imagined RL with immutable pretrained WM and classifier."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.algorithms.registry import get_actor_update_route
from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.offline_seed import seed_replay_from_offline
from dreamervla.runners.online_replay import OnlineReplay
from dreamervla.utils.frozen_components import (
    assert_module_frozen,
    load_frozen_component,
    module_state_sha256,
)
from dreamervla.utils.hf_checkpoint import load_runner_payload
from dreamervla.utils.optim import build_optimizer
from dreamervla.utils.seed import capture_rng_state, restore_rng_state, set_seed
from dreamervla.utils.torch_utils import freeze_module, resolve_device

_REPLAY_BATCH_KEYS = (
    "obs_embedding",
    "actions",
    "rewards",
    "dones",
    "is_first",
    "is_terminal",
    "is_last",
    "proprio",
    "lang_emb",
)


class FrozenModelPolicyRunner(BaseRunner):
    """Train only the actor from official replay through frozen imagination."""

    runner_name = "frozen_model_policy"
    runner_status = "pre_mainline"
    runner_family = "actor"
    include_keys = (
        "global_step",
        "classifier_threshold",
        "frozen_state_hashes",
        "policy_initial_hash",
        "policy_final_hash",
        "applied_policy_steps",
        "source_checkpoints",
        "reference_policy_hash",
        "resume_contract_hash",
        "replay_sample_cursor",
        "rng_state",
    )
    exclude_keys = ("ref_policy", "replay")

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir=output_dir)
        self.device = torch.device("cpu")
        self.world_model: torch.nn.Module | None = None
        self.classifier: torch.nn.Module | None = None
        self.policy: torch.nn.Module | None = None
        self.ref_policy: torch.nn.Module | None = None
        self.policy_optimizer: torch.optim.Optimizer | None = None
        self.replay: OnlineReplay | None = None
        self.classifier_threshold = 0.5
        self.frozen_state_hashes: dict[str, str] = {}
        self.policy_initial_hash = ""
        self.policy_final_hash = ""
        self.applied_policy_steps = 0
        self.source_checkpoints: dict[str, str] = {}
        self.reference_policy_hash = ""
        self.resume_contract_hash = ""
        self.replay_sample_cursor = 0
        self.rng_state: dict[str, Any] = {}

    def setup(self) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
        if world_size != 1:
            raise RuntimeError(
                f"FrozenModelPolicyRunner requires a single process; got WORLD_SIZE={world_size}"
            )
        super().setup()
        set_seed(int(OmegaConf.select(self.cfg, "seed", default=0) or 0))
        self.device = resolve_device(
            str(OmegaConf.select(self.cfg, "training.device", default="auto"))
        )

        precision = str(OmegaConf.select(self.cfg, "optim.precision", default="fp32")).lower()
        dtype_by_precision = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if precision not in dtype_by_precision:
            raise ValueError("optim.precision must be one of: bf16, fp16, fp32")
        self.world_model = hydra.utils.instantiate(self.cfg.world_model).to(
            device=self.device,
            dtype=dtype_by_precision[precision],
        )
        self.classifier = hydra.utils.instantiate(self.cfg.classifier).to(self.device)
        self.policy = hydra.utils.instantiate(self.cfg.policy).to(self.device)

        wm_path = str(OmegaConf.select(self.cfg, "init.world_model_state_ckpt"))
        classifier_path = str(OmegaConf.select(self.cfg, "init.classifier_state_ckpt"))
        loaded_wm = load_frozen_component(wm_path, "world_model")
        loaded_classifier = load_frozen_component(classifier_path, "classifier")
        self._require_component_config_match(
            loaded_wm.metadata,
            component="world_model",
            active_cfg=self.cfg.world_model,
        )
        self._require_component_config_match(
            loaded_classifier.metadata,
            component="classifier",
            active_cfg=self.cfg.classifier,
        )
        self.world_model.load_state_dict(loaded_wm.state_dict, strict=True)
        self.classifier.load_state_dict(loaded_classifier.state_dict, strict=True)
        self.source_checkpoints = {
            "world_model": str(Path(wm_path).expanduser().resolve()),
            "classifier": str(Path(classifier_path).expanduser().resolve()),
        }
        self.classifier_threshold = self._resolve_classifier_threshold(loaded_classifier.metadata)

        freeze_module(self.world_model)
        freeze_module(self.classifier)
        assert_module_frozen(self.world_model, name="world_model")
        assert_module_frozen(self.classifier, name="classifier")
        self.frozen_state_hashes = {
            "world_model": module_state_sha256(self.world_model),
            "classifier": module_state_sha256(self.classifier),
        }
        expected_frozen_hashes = dict(self.frozen_state_hashes)
        expected_source_checkpoints = dict(self.source_checkpoints)
        expected_classifier_threshold = float(self.classifier_threshold)

        self.policy_initial_hash = module_state_sha256(self.policy)
        self.policy_final_hash = self.policy_initial_hash
        self.policy_optimizer = build_optimizer(self.policy, self.cfg.optim.policy)
        self.ref_policy = self._build_reference_policy()
        self.reference_policy_hash = (
            module_state_sha256(self.ref_policy) if self.ref_policy is not None else ""
        )
        self.resume_contract_hash = self._current_resume_contract_hash()
        self.replay = self._build_official_replay()
        if bool(OmegaConf.select(self.cfg, "training.resume", default=False)):
            resume_path = self._resolve_resume_checkpoint()
            payload = load_runner_payload(resume_path)
            self._validate_resume_payload(payload, resume_path)
            expected_contract_hash = str(self.resume_contract_hash)
            expected_reference_hash = str(self.reference_policy_hash)
            expected_policy_initial_hash = str(self.policy_initial_hash)
            self.load_payload(payload)
            freeze_module(self.world_model)
            freeze_module(self.classifier)
            current_hashes = {
                "world_model": module_state_sha256(self.world_model),
                "classifier": module_state_sha256(self.classifier),
            }
            if current_hashes != expected_frozen_hashes:
                raise RuntimeError(
                    "resume checkpoint frozen states differ from the explicit "
                    "WM/CLS source checkpoints"
                )
            if self.frozen_state_hashes != expected_frozen_hashes:
                raise RuntimeError(
                    "resume checkpoint frozen hash manifest does not match its states"
                )
            if float(self.classifier_threshold) != expected_classifier_threshold:
                raise RuntimeError(
                    "resume checkpoint classifier threshold differs from the "
                    "explicit classifier checkpoint"
                )
            if not self.policy_initial_hash:
                raise RuntimeError("resume checkpoint has no initial policy hash")
            if self.resume_contract_hash != expected_contract_hash:
                raise RuntimeError(
                    "resume contract differs from the current frozen-model RL objective"
                )
            if self.reference_policy_hash != expected_reference_hash:
                raise RuntimeError(
                    "resume reference policy hash differs from the reconstructed reference"
                )
            if self.policy_initial_hash != expected_policy_initial_hash:
                raise RuntimeError(
                    "resume initial policy differs from the current Hydra construction"
                )
            if self.source_checkpoints != expected_source_checkpoints:
                raise RuntimeError(
                    "resume source checkpoints differ from the explicit WM/CLS checkpoints"
                )
            self.source_checkpoints = expected_source_checkpoints
            self.policy_final_hash = module_state_sha256(self.policy)
            self.replay.set_task_sample_cursor(int(self.replay_sample_cursor))
            restore_rng_state(self.rng_state, strict=True)
        else:
            self.save_checkpoint(tag="baseline")

    @staticmethod
    def _resolved_container(value: Any) -> Any:
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return copy.deepcopy(value)

    @classmethod
    def _require_component_config_match(
        cls,
        metadata: Mapping[str, Any],
        *,
        component: str,
        active_cfg: Any,
    ) -> None:
        raw_config = metadata.get("config")
        config = cls._resolved_container(raw_config)
        if not isinstance(config, Mapping) or component not in config:
            raise ValueError(
                f"{component} checkpoint must contain config.{component} metadata"
            )
        checkpoint_cfg = cls._resolved_container(config[component])
        resolved_active = cls._resolved_container(active_cfg)
        if checkpoint_cfg != resolved_active:
            raise ValueError(
                f"{component} checkpoint config does not match the active Hydra config"
            )

    def _current_resume_contract_hash(self) -> str:
        payload = {
            "seed": int(OmegaConf.select(self.cfg, "seed", default=0) or 0),
            "algorithm": self._resolved_container(self.cfg.algorithm),
            "optim": self._resolved_container(self.cfg.optim),
            "world_model": self._resolved_container(self.cfg.world_model),
            "classifier": self._resolved_container(self.cfg.classifier),
            "policy": self._resolved_container(self.cfg.policy),
            "official_replay": self._resolved_container(self.cfg.official_replay),
            "batch_size": int(self.cfg.dataloader.batch_size),
            "device": str(
                OmegaConf.select(
                    self.cfg,
                    "training.device",
                    default="auto",
                )
            ),
            "require_policy_update": bool(
                OmegaConf.select(
                    self.cfg,
                    "training.require_policy_update",
                    default=True,
                )
            ),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _resolve_resume_checkpoint(self) -> Path:
        raw = OmegaConf.select(self.cfg, "training.resume_dir", default=None)
        if raw in (None, ""):
            candidates = (self.get_checkpoint_path(prefer_existing=True),)
        else:
            source = Path(str(raw)).expanduser().resolve()
            candidates = (
                source,
                source / "checkpoints" / "latest.ckpt",
                source / "ckpt" / "latest.ckpt",
                source / "latest.ckpt",
            )
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is None:
            rendered = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(
                f"frozen-model RL resume checkpoint does not exist: {rendered}"
            )
        return selected.resolve()

    @staticmethod
    def _validate_resume_payload(payload: Any, path: Path) -> None:
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"resume checkpoint payload must be a mapping: {path}")
        state_dicts = payload.get("state_dicts")
        pickles = payload.get("pickles")
        if not isinstance(state_dicts, Mapping):
            raise RuntimeError(f"resume checkpoint has no state_dicts mapping: {path}")
        if not isinstance(pickles, Mapping):
            raise RuntimeError(f"resume checkpoint has no pickles mapping: {path}")
        required_states = {
            "world_model",
            "classifier",
            "policy",
            "policy_optimizer",
        }
        required_pickles = {
            "global_step",
            "classifier_threshold",
            "frozen_state_hashes",
            "policy_initial_hash",
            "applied_policy_steps",
            "source_checkpoints",
            "reference_policy_hash",
            "resume_contract_hash",
            "replay_sample_cursor",
            "rng_state",
        }
        missing_states = sorted(required_states.difference(state_dicts))
        missing_pickles = sorted(required_pickles.difference(pickles))
        if missing_states:
            raise RuntimeError(
                f"resume checkpoint is missing required state_dicts: {missing_states}"
            )
        if missing_pickles:
            raise RuntimeError(
                f"resume checkpoint is missing required pickles: {missing_pickles}"
            )

    def _resolve_classifier_threshold(self, metadata: dict[str, Any]) -> float:
        checkpoint_value = metadata.get("threshold")
        configured = OmegaConf.select(
            self.cfg,
            "algorithm.lumos.classifier_threshold",
            default=None,
        )
        if configured is None:
            if checkpoint_value is None:
                raise ValueError("classifier checkpoint must provide a validation threshold")
            resolved = float(checkpoint_value)
        else:
            if checkpoint_value is not None and float(configured) != float(checkpoint_value):
                raise ValueError(
                    "configured classifier threshold must equal the selected checkpoint "
                    f"threshold ({float(configured)} != {float(checkpoint_value)})"
                )
            resolved = float(configured)
        if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
            raise ValueError(
                f"classifier threshold must be finite and within [0,1], got {resolved}"
            )
        return resolved

    def _build_reference_policy(self) -> torch.nn.Module | None:
        assert self.policy is not None
        use_reference = any(
            float(OmegaConf.select(self.cfg, key, default=0.0) or 0.0) > 0.0
            for key in (
                "algorithm.kl_coef",
                "algorithm.actor_bc_to_ref_scale",
            )
        )
        if not use_reference:
            return None
        reference = copy.deepcopy(self.policy).to(self.device)
        freeze_module(reference)
        return reference

    def _build_official_replay(self) -> OnlineReplay:
        replay_cfg = self.cfg.official_replay
        raw_task_ids = OmegaConf.select(replay_cfg, "task_ids", default=[0])
        if isinstance(raw_task_ids, (ListConfig, list, tuple)):
            task_ids = tuple(int(task_id) for task_id in raw_task_ids)
        else:
            task_ids = (int(raw_task_ids),)
        replay_sampling = OmegaConf.select(
            replay_cfg,
            "replay_sampling",
            default={},
        )
        if OmegaConf.is_config(replay_sampling):
            replay_sampling = OmegaConf.to_container(replay_sampling, resolve=True)
        if not isinstance(replay_sampling, Mapping):
            raise TypeError("official_replay.replay_sampling must be a mapping")
        replay = OnlineReplay(
            capacity=int(replay_cfg.capacity),
            sequence_length=int(replay_cfg.sequence_length),
            task_ids=task_ids,
            capacity_mode=str(replay_cfg.capacity_mode),
            task_balanced=bool(replay_cfg.task_balanced),
            rank=int(replay_cfg.rank),
            replay_sampling=replay_sampling,
        )
        default_task_id = OmegaConf.select(replay_cfg, "task_id", default=None)
        added = seed_replay_from_offline(
            replay,
            data_dir=str(replay_cfg.data_dir),
            hidden_dir=str(replay_cfg.hidden_dir),
            default_task_id=(None if default_task_id is None else int(default_task_id)),
            infer_task_id_from_shard=bool(
                OmegaConf.select(
                    replay_cfg,
                    "infer_task_id_from_shard",
                    default=False,
                )
            ),
            max_episodes_per_task=OmegaConf.select(
                replay_cfg,
                "max_episodes_per_task",
                default=None,
            ),
            require_reference_complete=bool(
                OmegaConf.select(
                    replay_cfg,
                    "require_reference_complete",
                    default=True,
                )
            ),
        )
        if added <= 0 or replay.sampleable_window_count() <= 0:
            raise RuntimeError("official replay has no sampleable sequences for frozen-model RL")
        retained = len(replay.episodes)
        if retained != added:
            raise RuntimeError(
                "official replay capacity did not retain every sampleable episode: "
                f"added={added}, retained={retained}; increase "
                "official_replay.capacity in Hydra"
            )
        missing_tasks = sorted(set(task_ids).difference(replay.task_episode_counts()))
        if missing_tasks:
            raise RuntimeError(
                "official replay has no sampleable sequence for task IDs "
                f"{missing_tasks}"
            )
        return replay

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> str:
        """Capture RNG state before delegating to the common checkpoint writer."""

        if self.replay is not None:
            self.replay_sample_cursor = self.replay.task_sample_cursor
        self.rng_state = capture_rng_state()
        return super().save_checkpoint(*args, **kwargs)

    def _assert_frozen_unchanged(self) -> None:
        assert self.world_model is not None and self.classifier is not None
        for name, module in (
            ("world_model", self.world_model),
            ("classifier", self.classifier),
        ):
            assert_module_frozen(module, name=name)
            current = module_state_sha256(module)
            expected = self.frozen_state_hashes[name]
            if current != expected:
                raise RuntimeError(
                    f"{name} state changed during frozen-model policy training: "
                    f"{current} != {expected}"
                )

    def _assert_frozen_training_disabled(self) -> None:
        assert self.world_model is not None and self.classifier is not None
        assert_module_frozen(self.world_model, name="world_model")
        assert_module_frozen(self.classifier, name="classifier")

    @staticmethod
    def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, numbers.Number) and not isinstance(value, bool)
        }

    def run(self) -> dict[str, Any]:
        assert self.world_model is not None
        assert self.classifier is not None
        assert self.policy is not None
        assert self.policy_optimizer is not None
        assert self.replay is not None

        route = get_actor_update_route(str(self.cfg.algorithm.update_type))
        if route.world_model_arg != "chunk_world_model" or not route.requires_classifier:
            raise ValueError(
                "frozen-model RL requires a chunk-world-model actor route with classifier"
            )
        total_updates = int(self.cfg.training.num_updates)
        batch_size = int(self.cfg.dataloader.batch_size)
        checkpoint_every = int(
            OmegaConf.select(self.cfg, "training.checkpoint_every", default=0) or 0
        )
        last_metrics: dict[str, float] = {}

        for update_index in range(int(self.global_step), total_updates):
            self._assert_frozen_training_disabled()
            replay_batch = self.replay.sample(batch_size, include_images=False)
            observations = {
                key: replay_batch[key] for key in _REPLAY_BATCH_KEYS if key in replay_batch
            }
            raw_metrics = route.step_fn(
                policy=self.policy,
                chunk_world_model=self.world_model,
                classifier=self.classifier,
                classifier_threshold=float(self.classifier_threshold),
                actor_optimizer=self.policy_optimizer,
                obs=observations,
                device=self.device,
                algorithm_cfg=self.cfg.algorithm,
                optim_cfg=self.cfg.optim,
                ref_policy=self.ref_policy,
            )
            last_metrics = self._scalar_metrics(raw_metrics)
            self.applied_policy_steps += int(float(last_metrics.get("ppo_step_applied", 0.0)) > 0.0)
            self.global_step = update_index + 1
            self.log_metrics(last_metrics, step=self.global_step, prefix="train/rl")
            if checkpoint_every > 0 and self.global_step % checkpoint_every == 0:
                self._assert_frozen_unchanged()
                self.save_checkpoint(tag="latest")

        self._assert_frozen_unchanged()
        self.policy_final_hash = module_state_sha256(self.policy)
        if bool(
            OmegaConf.select(
                self.cfg,
                "training.require_policy_update",
                default=True,
            )
        ):
            if self.applied_policy_steps < 1:
                raise RuntimeError("no policy optimizer step was applied")
            if self.policy_final_hash == self.policy_initial_hash:
                raise RuntimeError("policy state did not change during imagined RL")

        self.save_checkpoint(tag="latest")
        final_checkpoint = self.save_checkpoint(tag="final")
        summary: dict[str, Any] = {
            "schema_version": 1,
            "official_data_dir": str(self.cfg.official_replay.data_dir),
            "official_hidden_dir": str(self.cfg.official_replay.hidden_dir),
            "source_checkpoints": dict(self.source_checkpoints),
            "frozen_hashes_before": dict(self.frozen_state_hashes),
            "frozen_hashes_after": {
                "world_model": module_state_sha256(self.world_model),
                "classifier": module_state_sha256(self.classifier),
            },
            "policy_hash_before": self.policy_initial_hash,
            "policy_hash_after": self.policy_final_hash,
            "policy_changed": self.policy_final_hash != self.policy_initial_hash,
            "applied_policy_steps": int(self.applied_policy_steps),
            "total_updates": int(self.global_step),
            "replay_episodes": len(self.replay.episodes),
            "replay_transitions": int(self.replay.num_transitions),
            "classifier_threshold": float(self.classifier_threshold),
            "reference_policy_hash": self.reference_policy_hash or None,
            "resume_contract_hash": self.resume_contract_hash,
            "replay_sample_cursor": int(self.replay.task_sample_cursor),
            "final_checkpoint": str(final_checkpoint),
            "last_metrics": last_metrics,
        }
        summary_path = Path(self.output_dir) / "frozen_rl_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary


__all__ = ["FrozenModelPolicyRunner"]
