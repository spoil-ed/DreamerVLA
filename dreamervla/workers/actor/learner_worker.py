"""Ray LearnerWorker for optional online cotrain backend."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import OmegaConf
from torch import nn

from dreamervla.hybrid_engines.fsdp import FSDPModelManager
from dreamervla.hybrid_engines.weight_syncer.objectstore import (
    ObjectStoreWeightSyncer,
    _independent_cpu,
)
from dreamervla.scheduler.worker import Worker


class LearnerWorker(Worker):
    """Single-GPU learner actor.

    The first implementation includes a lightweight ``synthetic_ppo`` update
    path used by Ray e2e tests. Real DreamerVLA WM/CLS/RL phases use the same
    public ``update`` and ``sync_weights`` boundaries.
    """

    def __init__(
        self,
        model_cfg: dict[str, Any],
        init_ckpt: dict[str, Any],
        train_cfg: dict[str, Any],
        replay: Any,
    ) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.init_ckpt = dict(init_ckpt)
        self.train_cfg = dict(train_cfg)
        self.replay = replay
        configured_device = str(self.train_cfg.get("device", self.device))
        if configured_device == "auto":
            configured_device = self.device
        self.torch_device = torch.device(configured_device)
        self.components: dict[str, nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.policy: nn.Module | None = None
        self.world_model: nn.Module | None = None
        self.classifier: nn.Module | None = None
        self.critic: nn.Module | None = None
        self.ref_policy: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.policy_optimizer: torch.optim.Optimizer | None = None
        self.world_model_optimizer: torch.optim.Optimizer | None = None
        self.classifier_optimizer: torch.optim.Optimizer | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self.syncer: ObjectStoreWeightSyncer | None = None
        self.precision: PrecisionConfig | None = None
        self.grad_scaler: torch.amp.GradScaler | None = None
        self.fsdp_manager: FSDPModelManager | None = None
        self.phase_updater: Any | None = None
        self.replay_client: ReplayClient | None = None
        self._cotrain_classifier_updates = 0
        self._cotrain_last_classifier_f1 = 0.0
        self._train_progress_path = self._resolve_progress_path()
        self._train_progress: dict[str, Any] = {
            "active": False,
            "phase": "idle",
            "train_step": 0,
            "total_train_steps": 0,
            "wm_step": 0,
            "wm_total_steps": 0,
            "cls_step": 0,
            "cls_total_steps": 0,
            "vlarl_step": 0,
            "vlarl_total_steps": 0,
        }

    def init(self) -> None:
        self.precision = _resolve_precision(self.train_cfg, self.torch_device)
        mode = str(self.train_cfg.get("mode", "synthetic_ppo"))
        if mode == "wm_classifier_only" and self.train_cfg.get("fsdp") is not None:
            raise ValueError("wm_classifier_only LearnerWorker must not enable FSDP")
        self.fsdp_manager = _build_fsdp_manager(self.train_cfg)
        self.components = self._build_components()
        self.policy = self.components.get("policy")
        self.world_model = self.components.get("world_model")
        self.classifier = self.components.get("classifier")
        self.critic = self.components.get("critic")
        self.ref_policy = self.components.get("ref_policy")
        self.optimizers = self._build_optimizers()
        self.optimizer = self.optimizers.get("policy")
        self.policy_optimizer = self.optimizers.get("policy")
        self.world_model_optimizer = self.optimizers.get("world_model")
        self.classifier_optimizer = self.optimizers.get("classifier")
        self.critic_optimizer = self.optimizers.get("critic")
        self.grad_scaler = torch.amp.GradScaler(
            device=self.precision.device_type,
            enabled=self.precision.use_grad_scaler,
        )
        self.replay_client = ReplayClient(self.replay)
        if str(self.train_cfg.get("mode", "synthetic_ppo")) == "dreamervla_cotrain":
            self._validate_dreamervla_cotrain_components()
        phase_updater_cfg = self.train_cfg.get("phase_updater")
        if phase_updater_cfg is not None:
            self.phase_updater = _build_from_cfg(dict(phase_updater_cfg))
        syncer_cfg = dict(self.train_cfg.get("syncer", {}))
        self.syncer = ObjectStoreWeightSyncer(**syncer_cfg)
        self._set_train_progress(
            active=False,
            phase="idle",
            train_step=0,
            total_train_steps=0,
            wm_step=0,
            wm_total_steps=0,
            cls_step=0,
            cls_total_steps=0,
            vlarl_step=0,
            vlarl_total_steps=0,
        )

    def update(self, phase: str, num_steps: int) -> dict[str, float]:
        mode = str(self.train_cfg.get("mode", "synthetic_ppo"))
        if mode == "dreamervla_cotrain":
            return self._dreamervla_cotrain_update(str(phase), int(num_steps))
        if mode == "wm_classifier_only":
            return self._wm_classifier_only_update(str(phase), int(num_steps))
        if mode in {"phase_updater", "dreamervla_phase"}:
            return self._phase_update(str(phase), int(num_steps))
        if mode != "synthetic_ppo":
            raise NotImplementedError(f"unknown LearnerWorker mode: {mode!r}")
        if str(phase) != "rl":
            return {f"train/{phase}_loss": 0.0}
        return self._synthetic_ppo_update(int(num_steps))

    def sync_weights(self, what: str, version: int) -> None:
        name = str(what)
        if name not in self.components:
            raise ValueError(f"unknown component for weight sync: {name!r}")
        self._syncer().push(name, self.components[name].state_dict(), int(version))

    def state_dicts(self) -> dict[str, Any]:
        state_dicts: dict[str, Any] = {
            name: _cpu_state_dict(module)
            for name, module in self.components.items()
        }
        state_dicts["classifier_threshold"] = float(self._classifier_threshold())
        return state_dicts

    def _build_components(self) -> dict[str, nn.Module]:
        components: dict[str, nn.Module] = {}
        for name, cfg in self.model_cfg.items():
            if not _is_component_cfg(cfg):
                continue
            module = _build_from_cfg(_as_plain_dict(cfg)).to(self.torch_device)
            if name in self.init_ckpt:
                module.load_state_dict(
                    _to_device_state(self.init_ckpt[name], self.torch_device)
                )
            if self.fsdp_manager is not None:
                module = self.fsdp_manager.prepare_model(module)
            components[str(name)] = module
        mode = str(self.train_cfg.get("mode", "synthetic_ppo"))
        if mode == "wm_classifier_only":
            missing = [
                name
                for name in ("world_model", "classifier")
                if name not in components
            ]
            if missing:
                raise ValueError(
                    "wm_classifier_only requires components: "
                    f"{', '.join(missing)}"
                )
            unexpected = sorted(
                name for name in components if name not in {"world_model", "classifier"}
            )
            if unexpected:
                raise ValueError(
                    "wm_classifier_only only accepts world_model and classifier "
                    f"components; got {', '.join(unexpected)}"
                )
        elif "policy" not in components:
            raise ValueError("LearnerWorker model_cfg must include a policy component")
        return components

    def _build_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        optimizers: dict[str, torch.optim.Optimizer] = {}
        optimizer_cfgs = dict(self.train_cfg.get("optimizers", {}))
        for name, module in self.components.items():
            params = [param for param in module.parameters() if param.requires_grad]
            if not params:
                continue
            cfg = dict(optimizer_cfgs.get(name, {}))
            lr = float(cfg.get("lr", self.train_cfg.get(f"{name}_lr", self.train_cfg.get("lr", 1e-3))))
            weight_decay = float(cfg.get("weight_decay", 0.0))
            optimizers[name] = torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
            )
        return optimizers

    def _phase_update(self, phase: str, num_steps: int) -> dict[str, float]:
        updater = self._phase_updater()
        kwargs = {
            "phase": phase,
            "num_steps": max(1, int(num_steps)),
            "modules": self.components,
            "optimizers": self.optimizers,
            "replay": self.replay,
            "device": self.torch_device,
            "train_cfg": self.train_cfg,
            "precision": self._precision(),
        }
        if hasattr(updater, "update"):
            metrics = updater.update(**kwargs)
        elif callable(updater):
            metrics = updater(**kwargs)
        else:
            raise TypeError("phase_updater must be callable or expose update(...)")
        return {str(key): float(value) for key, value in dict(metrics).items()}

    def _dreamervla_cotrain_update(self, phase: str, num_steps: int) -> dict[str, float]:
        phases = ("wm", "classifier", "rl") if phase == "cotrain" else (phase,)
        allowed = {"wm", "classifier", "rl"}
        unknown = [item for item in phases if item not in allowed]
        if unknown:
            raise ValueError(
                "dreamervla_cotrain phase must be one of "
                f"{sorted(allowed | {'cotrain'})}; got {phase!r}"
            )

        steps_per_phase = max(1, int(num_steps))
        wm_total_steps = steps_per_phase if "wm" in phases else 0
        cls_total_steps = steps_per_phase if "classifier" in phases else 0
        vlarl_steps_per_call = self._planned_vlarl_optimizer_steps()
        vlarl_total_steps = (
            steps_per_phase * vlarl_steps_per_call if "rl" in phases else 0
        )
        total_train_steps = wm_total_steps + cls_total_steps + vlarl_total_steps
        metrics: dict[str, float] = {}
        train_step = 0
        wm_step = 0
        cls_step = 0
        vlarl_step = 0
        self._set_train_progress(
            active=True,
            phase=str(phase),
            train_step=0,
            total_train_steps=total_train_steps,
            wm_step=wm_step,
            wm_total_steps=wm_total_steps,
            cls_step=cls_step,
            cls_total_steps=cls_total_steps,
            vlarl_step=vlarl_step,
            vlarl_total_steps=vlarl_total_steps,
        )
        try:
            for _ in range(steps_per_phase):
                for item in phases:
                    self._set_train_progress(
                        active=True,
                        phase=item,
                        train_step=train_step,
                        total_train_steps=total_train_steps,
                        wm_step=wm_step,
                        wm_total_steps=wm_total_steps,
                        cls_step=cls_step,
                        cls_total_steps=cls_total_steps,
                        vlarl_step=vlarl_step,
                        vlarl_total_steps=vlarl_total_steps,
                    )
                    if item == "wm":
                        metrics.update(self._dreamervla_wm_update_once())
                        wm_step += 1
                        train_step += 1
                    elif item == "classifier":
                        metrics.update(self._dreamervla_classifier_update_once())
                        cls_step += 1
                        train_step += 1
                    else:
                        rl_metrics = self._dreamervla_rl_update_once()
                        metrics.update(rl_metrics)
                        applied = bool(
                            float(rl_metrics.get("rl/ppo_step_applied", 0.0)) > 0.0
                        )
                        completed = (
                            int(float(rl_metrics.get("rl/ppo_update_epochs", 0.0)))
                            if applied
                            else 0
                        )
                        vlarl_step += max(0, completed)
                        train_step += max(0, completed)
                    self._set_train_progress(
                        active=True,
                        phase=item,
                        train_step=train_step,
                        total_train_steps=total_train_steps,
                        wm_step=wm_step,
                        wm_total_steps=wm_total_steps,
                        cls_step=cls_step,
                        cls_total_steps=cls_total_steps,
                        vlarl_step=vlarl_step,
                        vlarl_total_steps=vlarl_total_steps,
                    )
        except Exception:
            self._set_train_progress(
                active=False,
                phase="failed",
                train_step=train_step,
                total_train_steps=total_train_steps,
                wm_step=wm_step,
                wm_total_steps=wm_total_steps,
                cls_step=cls_step,
                cls_total_steps=cls_total_steps,
                vlarl_step=vlarl_step,
                vlarl_total_steps=vlarl_total_steps,
            )
            raise
        self._set_train_progress(
            active=False,
            phase="done",
            train_step=train_step,
            total_train_steps=total_train_steps,
            wm_step=wm_step,
            wm_total_steps=wm_total_steps,
            cls_step=cls_step,
            cls_total_steps=cls_total_steps,
            vlarl_step=vlarl_step,
            vlarl_total_steps=vlarl_total_steps,
        )
        return metrics

    def _wm_classifier_only_update(self, phase: str, num_steps: int) -> dict[str, float]:
        if phase == "cotrain":
            phases = ("wm", "classifier")
        elif phase in {"wm", "classifier"}:
            phases = (phase,)
        else:
            raise ValueError(
                "wm_classifier_only supports only wm, classifier, cotrain; "
                f"got {phase!r}"
            )

        metrics: dict[str, float] = {}
        for _ in range(max(1, int(num_steps))):
            for item in phases:
                if item == "wm":
                    metrics.update(self._dreamervla_wm_update_once())
                else:
                    metrics.update(self._dreamervla_classifier_update_once())
        return metrics

    def _planned_vlarl_optimizer_steps(self) -> int:
        cfg = self.train_cfg.get("algorithm_cfg", {}) or {}
        if hasattr(cfg, "get"):
            return max(1, int(cfg.get("ppo_update_epochs", 1)))
        return 1

    def _resolve_progress_path(self) -> Path | None:
        raw = self.train_cfg.get("progress_path")
        if raw in (None, ""):
            return None
        return Path(str(raw)).expanduser()

    def _set_train_progress(
        self,
        *,
        active: bool,
        phase: str,
        train_step: int,
        total_train_steps: int,
        wm_step: int,
        wm_total_steps: int,
        cls_step: int,
        cls_total_steps: int,
        vlarl_step: int,
        vlarl_total_steps: int,
    ) -> None:
        payload = {
            "active": bool(active),
            "phase": str(phase),
            "train_step": int(train_step),
            "total_train_steps": int(total_train_steps),
            "wm_step": int(wm_step),
            "wm_total_steps": int(wm_total_steps),
            "cls_step": int(cls_step),
            "cls_total_steps": int(cls_total_steps),
            "vlarl_step": int(vlarl_step),
            "vlarl_total_steps": int(vlarl_total_steps),
            "time": float(time.time()),
        }
        self._train_progress = payload
        path = self._train_progress_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, path)

    def _dreamervla_wm_update_once(self) -> dict[str, float]:
        batch = self._replay_client().sample(
            int(self.train_cfg.get("batch_size", 2)),
            include_images=False,
        )
        with self._precision().context():
            raw = world_model_pretrain_step(
                policy=self.components.get("policy"),
                world_model=self._required_module("world_model"),
                optimizer=self._required_optimizer("world_model"),
                batch=batch,
                device=self.torch_device,
                optim_cfg=self._optim_cfg(),
            )
        return namespaced_world_model_metrics(raw)

    def _dreamervla_classifier_update_once(self) -> dict[str, float]:
        with self._precision().context():
            raw = online_classifier_update_step(
                classifier=self._required_module("classifier"),
                optimizer=self._required_optimizer("classifier"),
                replay=self._replay_client(),
                device=self.torch_device,
                batch_size=int(
                    self.train_cfg.get(
                        "classifier_batch_size",
                        self.train_cfg.get("batch_size", 2),
                    )
                ),
                early_neg_stride=int(self.train_cfg.get("classifier_early_neg_stride", 8)),
                grad_clip=float(self._optim_cfg().get("grad_clip_norm", 1.0)),
                loss_type=self.train_cfg.get("classifier_loss_type", None),
                sampling_protocol=str(
                    self.train_cfg.get("classifier_sampling_protocol", "lumos")
                ),
                balance_batches=bool(
                    self.train_cfg.get("classifier_balance_batches", False)
                ),
            )
        if float(raw.get("updated", 1.0)) > 0.5:
            self._cotrain_classifier_updates += 1
            self._cotrain_last_classifier_f1 = float(raw.get("f1", 0.0))
        return {
            "cls/loss": float(raw.get("loss", 0.0)),
            "cls/acc": float(raw.get("acc", 0.0)),
            "cls/f1": float(raw.get("f1", 0.0)),
            "cls/updated": float(raw.get("updated", 1.0)),
            "cls/updates": float(self._cotrain_classifier_updates),
            "cls/skipped_single_class_batch": float(
                raw.get("skipped_single_class_batch", 0.0)
            ),
        }

    def _dreamervla_rl_update_once(self) -> dict[str, float]:
        gate_cfg = self._actor_signal_gate_cfg()
        if not self._actor_signal_ready(gate_cfg):
            return self._skipped_rl_metrics()

        # Phase 4 off-policy gating: drop replay samples older than
        # ``staleness_threshold`` rollout-policy versions (None / <0 disables → the
        # default, behaviour-preserving path). Graceful for fixed-base rollouts
        # (OFT), where the replay falls back to all valid samples.
        staleness_threshold = self.train_cfg.get("staleness_threshold", None)
        replay_batch_size = int(self.train_cfg.get("batch_size", 2))
        rollout_epoch = self._vlarl_rollout_epoch()
        batch = self._sample_vlarl_batch(
            replay_batch_size,
            rollout_epoch=rollout_epoch,
            staleness_threshold=(
                None if staleness_threshold is None else int(staleness_threshold)
            ),
        )
        obs_for_update = {
            key: batch[key]
            for key in (
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
            if key in batch
        }
        with self._precision().context():
            raw = dino_lumos_step(
                policy=self._required_module("policy"),
                chunk_world_model=self._required_module("world_model"),
                classifier=self._required_module("classifier"),
                classifier_threshold=self._classifier_threshold(),
                actor_optimizer=self._required_optimizer("policy"),
                obs=obs_for_update,
                device=self.torch_device,
                algorithm_cfg=self._algorithm_cfg(),
                optim_cfg=self._optim_cfg(),
                ref_policy=self.components.get("ref_policy"),
            )
        return {
            "rl/actor_loss": float(raw.get("actor_loss", 0.0)),
            "rl/returns_mean": float(raw.get("returns_mean", 0.0)),
            "rl/returns_std": float(raw.get("returns_std", 0.0)),
            "rl/policy_grad_norm": float(raw.get("actor_grad_norm", 0.0)),
            "rl/skipped_zero_variance_groups": float(
                raw.get("LUMOS/skipped_zero_variance_groups", 0.0)
            ),
            "rl/ppo_step_applied": float(raw.get("ppo_step_applied", 0.0)),
            "rl/ppo_update_epochs": float(raw.get("ppo_update_epochs", 1.0)),
            "rl/actor_signal_ready": 1.0,
            "rl/skipped_no_signal": 0.0,
            "rl/classifier_f1_gate": float(self._cotrain_last_classifier_f1),
            "rl/classifier_updates": float(self._cotrain_classifier_updates),
            "LUMOS/score_mean": float(raw.get("LUMOS/score_mean", raw.get("score_mean", 0.0))),
            "LUMOS/score_std": float(raw.get("LUMOS/score_std", raw.get("score_std", 0.0))),
            "LUMOS/group_var_keep_frac": float(raw.get("LUMOS/group_var_keep_frac", 0.0)),
            "LUMOS/num_mixed_groups": float(raw.get("LUMOS/num_mixed_groups", 0.0)),
            "rl/rollout_epoch": float(rollout_epoch),
            "rl/replay_batch_size": float(replay_batch_size),
            "rl/ppo_start_batch_size": float(
                _first_batch_dim(obs_for_update, default=0)
            ),
            "rl/imagined_rollouts": float(
                _estimated_imagined_rollouts(raw, obs_for_update, self._algorithm_cfg())
            ),
        }

    def _vlarl_rollout_epoch(self) -> int:
        cfg = self.train_cfg.get("algorithm_cfg", {}) or {}
        if hasattr(cfg, "get"):
            return max(1, int(cfg.get("rollout_epoch", 1)))
        return 1

    def _sample_vlarl_batch(
        self,
        batch_size: int,
        *,
        rollout_epoch: int,
        staleness_threshold: int | None,
    ) -> dict[str, Any]:
        replay = self._replay_client()
        epoch_count = max(1, int(rollout_epoch))
        batches = [
            replay.sample(
                int(batch_size),
                staleness_threshold=staleness_threshold,
                include_images=False,
            )
            for _ in range(epoch_count)
        ]
        if len(batches) == 1:
            return batches[0]
        return _concat_replay_batches(batches)

    def _actor_signal_gate_cfg(self) -> dict[str, Any]:
        raw = self.train_cfg.get("actor_signal_gate", {})
        if isinstance(raw, bool):
            return {"enabled": bool(raw)}
        cfg = _as_plain_dict(raw) if raw is not None else {}
        cfg.setdefault("enabled", False)
        cfg.setdefault("min_classifier_f1", 0.0)
        cfg.setdefault("min_classifier_updates", 0)
        return cfg

    def _actor_signal_ready(self, gate_cfg: dict[str, Any]) -> bool:
        if not bool(gate_cfg.get("enabled", False)):
            return True
        min_updates = int(gate_cfg.get("min_classifier_updates", 0))
        min_f1 = float(gate_cfg.get("min_classifier_f1", 0.0))
        return (
            int(self._cotrain_classifier_updates) >= min_updates
            and float(self._cotrain_last_classifier_f1) >= min_f1
        )

    def _skipped_rl_metrics(self) -> dict[str, float]:
        return {
            "rl/actor_loss": 0.0,
            "rl/returns_mean": 0.0,
            "rl/returns_std": 0.0,
            "rl/policy_grad_norm": 0.0,
            "rl/skipped_zero_variance_groups": 0.0,
            "rl/ppo_step_applied": 0.0,
            "rl/ppo_update_epochs": 0.0,
            "rl/actor_signal_ready": 0.0,
            "rl/skipped_no_signal": 1.0,
            "rl/classifier_f1_gate": float(self._cotrain_last_classifier_f1),
            "rl/classifier_updates": float(self._cotrain_classifier_updates),
            "LUMOS/score_mean": 0.0,
            "LUMOS/score_std": 0.0,
            "LUMOS/group_var_keep_frac": 0.0,
            "LUMOS/num_mixed_groups": 0.0,
        }

    def _synthetic_ppo_update(self, num_steps: int) -> dict[str, float]:
        policy = self._policy()
        optimizer = self._optimizer()
        batch_size = int(self.train_cfg.get("batch_size", 2))
        last_loss = 0.0
        for _ in range(max(1, int(num_steps))):
            batch = ray.get(self.replay.sample.remote(batch_size))
            obs_embedding = batch["obs_embedding"].to(self.torch_device).float()
            target = batch["current_actions"].to(self.torch_device).float().mean(dim=1)
            hidden = obs_embedding.mean(dim=1)
            optimizer.zero_grad(set_to_none=True)
            with self._precision().context():
                pred = _predict(policy, hidden)
                loss = torch.mean((pred - target) ** 2)
            scaler = self._grad_scaler()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        return {"train/rl_loss": last_loss}

    def _policy(self) -> nn.Module:
        if self.policy is None:
            raise RuntimeError("LearnerWorker.init() has not been called")
        return self.policy

    def _optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            raise RuntimeError("LearnerWorker.init() has not been called")
        return self.optimizer

    def _syncer(self) -> ObjectStoreWeightSyncer:
        if self.syncer is None:
            raise RuntimeError("LearnerWorker.init() has not been called")
        return self.syncer

    def _precision(self) -> PrecisionConfig:
        if self.precision is None:
            raise RuntimeError("LearnerWorker.init() has not been called")
        return self.precision

    def _grad_scaler(self) -> torch.amp.GradScaler:
        if self.grad_scaler is None:
            raise RuntimeError("LearnerWorker.init() has not been called")
        return self.grad_scaler

    def _replay_client(self) -> ReplayClient:
        if self.replay_client is None:
            raise RuntimeError("LearnerWorker.init() has not been called")
        return self.replay_client

    def _phase_updater(self) -> Any:
        if self.phase_updater is None:
            raise RuntimeError("LearnerWorker phase_updater has not been configured")
        return self.phase_updater

    def _required_module(self, name: str) -> nn.Module:
        module = self.components.get(str(name))
        if module is None:
            raise RuntimeError(f"LearnerWorker component {name!r} has not been initialized")
        return module

    def _required_optimizer(self, name: str) -> torch.optim.Optimizer:
        optimizer = self.optimizers.get(str(name))
        if optimizer is None:
            raise RuntimeError(f"LearnerWorker optimizer {name!r} has not been initialized")
        return optimizer

    def _classifier_threshold(self) -> float:
        value = self.train_cfg.get("classifier_threshold", None)
        if value is None:
            value = self.init_ckpt.get("classifier_threshold", 0.5)
        return float(value)

    def _validate_dreamervla_cotrain_components(self) -> None:
        missing = [
            name
            for name in ("policy", "world_model", "classifier")
            if name not in self.components
        ]
        if missing:
            raise ValueError(
                "dreamervla_cotrain LearnerWorker requires components: "
                f"{', '.join(missing)}"
            )
        missing_optimizers = [
            name
            for name in ("policy", "world_model", "classifier")
            if name not in self.optimizers
        ]
        if missing_optimizers:
            raise ValueError(
                "dreamervla_cotrain LearnerWorker requires optimizers for: "
                f"{', '.join(missing_optimizers)}"
            )

    def _optim_cfg(self):
        return OmegaConf.create(
            {
                "grad_clip_norm": 1.0,
                "zero_grad_set_to_none": True,
                **dict(self.train_cfg.get("optim_cfg", {})),
            }
        )

    def _algorithm_cfg(self):
        return OmegaConf.create(dict(self.train_cfg.get("algorithm_cfg", {})))


class ReplayClient:
    """Local facade over direct OnlineReplay objects or Ray ReplayWorker actors."""

    def __init__(self, replay: Any) -> None:
        self.replay = replay

    def sample(
        self,
        batch_size: int,
        *,
        staleness_threshold: int | None = None,
        include_images: bool | None = None,
    ) -> dict[str, Any]:
        # Forward the Phase 4 staleness gate only when it is set, so the default
        # path stays a byte-identical 1-arg call and minimal replay backends
        # (e.g. test doubles) that don't know about staleness still work.
        kwargs = (
            {}
            if staleness_threshold is None
            else {"staleness_threshold": int(staleness_threshold)}
        )
        if include_images is not None:
            kwargs["include_images"] = bool(include_images)
        return self._call("sample", int(batch_size), **kwargs)

    def sample_classifier_windows(
        self,
        batch_size: int,
        *,
        window: int,
        chunk_size: int,
        chunk_pool: str,
        early_neg_stride: int,
    ) -> dict[str, Any]:
        return self._call(
            "sample_classifier_windows",
            int(batch_size),
            window=int(window),
            chunk_size=int(chunk_size),
            chunk_pool=str(chunk_pool),
            early_neg_stride=int(early_neg_stride),
        )

    def classifier_window_count(self, *, window: int, chunk_size: int) -> int:
        return int(
            self._call(
                "classifier_window_count",
                window=int(window),
                chunk_size=int(chunk_size),
            )
        )

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self.replay, name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return ray.get(remote(*args, **kwargs))
        return method(*args, **kwargs)


def _concat_replay_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate repeated rollout-epoch replay samples along batch dim."""

    if not batches:
        raise ValueError("cannot concatenate an empty replay batch list")
    common_keys = [
        key for key in batches[0].keys() if all(key in batch for batch in batches)
    ]
    return {
        str(key): _concat_replay_values([batch[key] for batch in batches], path=str(key))
        for key in common_keys
    }


def _concat_replay_values(values: list[Any], *, path: str) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"cannot concatenate mixed replay values at {path!r}")
        if first.ndim == 0:
            return torch.stack(values, dim=0)
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        common_keys = [
            key for key in first.keys() if all(isinstance(value, dict) and key in value for value in values)
        ]
        return {
            str(key): _concat_replay_values(
                [value[key] for value in values],
                path=f"{path}.{key}",
            )
            for key in common_keys
        }
    if isinstance(first, list):
        merged: list[Any] = []
        for value in values:
            if not isinstance(value, list):
                raise TypeError(f"cannot concatenate mixed replay values at {path!r}")
            merged.extend(value)
        return merged
    if isinstance(first, tuple):
        merged_tuple: list[Any] = []
        for value in values:
            if not isinstance(value, tuple):
                raise TypeError(f"cannot concatenate mixed replay values at {path!r}")
            merged_tuple.extend(value)
        return tuple(merged_tuple)
    return first


def _first_batch_dim(batch: dict[str, Any], *, default: int) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, dict):
            nested = _first_batch_dim(value, default=-1)
            if nested >= 0:
                return nested
    return int(default)


def _estimated_imagined_rollouts(
    raw_metrics: dict[str, Any],
    obs_batch: dict[str, Any],
    algorithm_cfg: Any,
) -> int:
    group_size = int(
        float(
            raw_metrics.get(
                "LUMOS/group_size",
                OmegaConf.select(
                    algorithm_cfg,
                    "lumos.ppo_rollouts_per_start_max",
                    default=OmegaConf.select(
                        algorithm_cfg,
                        "ppo_rollouts_per_start",
                        default=1,
                    ),
                ),
            )
        )
    )
    num_groups = raw_metrics.get("LUMOS/num_groups", None)
    if num_groups is not None:
        return int(float(num_groups) * max(1, group_size))

    batch_dim = _first_batch_dim(obs_batch, default=0)
    obs_embedding = obs_batch.get("obs_embedding")
    seq_len = (
        int(obs_embedding.shape[1])
        if isinstance(obs_embedding, torch.Tensor) and obs_embedding.ndim >= 2
        else 1
    )
    imag_last = int(OmegaConf.select(algorithm_cfg, "imag_last", default=4))
    start_count = min(seq_len, imag_last if imag_last > 0 else seq_len)
    return int(batch_dim * max(1, start_count) * max(1, group_size))


@dataclass(frozen=True)
class PrecisionConfig:
    """Manual AMP selection for Ray learner updates."""

    name: str
    device_type: str
    autocast_dtype: torch.dtype | None
    use_grad_scaler: bool = False

    def context(self) -> contextlib.AbstractContextManager[None]:
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.amp.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
        )


def _resolve_precision(train_cfg: dict[str, Any], device: torch.device) -> PrecisionConfig:
    raw = str(train_cfg.get("precision", train_cfg.get("dtype", "fp32"))).strip().lower()
    aliases = {
        "": "fp32",
        "none": "fp32",
        "float32": "fp32",
        "fp32": "fp32",
        "bfloat16": "bf16",
        "bf16": "bf16",
        "float16": "fp16",
        "fp16": "fp16",
    }
    if raw not in aliases:
        raise ValueError(
            "learner.train_cfg.precision must be one of "
            "fp32, bf16, or fp16; got "
            f"{train_cfg.get('precision', train_cfg.get('dtype'))!r}"
        )
    name = aliases[raw]
    device_type = device.type
    if name == "fp32":
        return PrecisionConfig(
            name="fp32",
            device_type=device_type,
            autocast_dtype=None,
        )
    if name == "bf16":
        return PrecisionConfig(
            name="bf16",
            device_type=device_type,
            autocast_dtype=torch.bfloat16,
        )
    return PrecisionConfig(
        name="fp16",
        device_type=device_type,
        autocast_dtype=torch.float16,
        use_grad_scaler=(device_type == "cuda"),
    )


def _build_from_cfg(cfg: dict[str, Any]) -> Any:
    target = cfg.get("target") or cfg.get("_target_") or cfg.get("class_path")
    if not target:
        raise ValueError("component config must include target/_target_/class_path")
    kwargs = _as_plain_dict(cfg.get("kwargs", {}))
    if ":" in str(target):
        module_name, class_name = str(target).split(":", 1)
    else:
        module_name, class_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)


def _build_fsdp_manager(train_cfg: dict[str, Any]) -> FSDPModelManager | None:
    fsdp_cfg = train_cfg.get("fsdp")
    if fsdp_cfg is None:
        return None
    return FSDPModelManager(**_as_plain_dict(fsdp_cfg))


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=True) or {})
    return dict(value)


def _is_component_cfg(value: Any) -> bool:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return isinstance(value, dict) and any(
        key in value for key in ("target", "_target_", "class_path")
    )


def _predict(policy: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(policy, "predict"):
        return policy.predict(hidden)
    return policy(hidden)


def world_model_pretrain_step(**kwargs: Any) -> dict[str, float]:
    """Lazy import wrapper preserving optional-Ray import isolation."""

    from dreamervla.algorithms.dreamervla import world_model_pretrain_step as _impl

    return _impl(**kwargs)


def namespaced_world_model_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Return world-model metrics without importing the heavy algorithm package."""

    out: dict[str, float] = {}
    for key, value in dict(metrics).items():
        name = str(key)
        if name.startswith("_"):
            continue
        out[name if name.startswith("wm/") else f"wm/{name}"] = float(value)
    return out


def online_classifier_update_step(**kwargs: Any) -> dict[str, Any]:
    """Lazy import wrapper preserving optional-Ray import isolation."""

    from dreamervla.runners.online_dreamervla import online_classifier_update_step as _impl

    return _impl(**kwargs)


def dino_lumos_step(**kwargs: Any) -> dict[str, float]:
    """Lazy import wrapper preserving optional-Ray import isolation."""

    from dreamervla.algorithms.ppo import dino_lumos_step as _impl

    return _impl(**kwargs)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: _independent_cpu(value) for key, value in module.state_dict().items()}


def _to_device_state(state_dict: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: (value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)).to(device)
        for key, value in state_dict.items()
    }
