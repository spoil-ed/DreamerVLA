"""Ray LearnerWorker for optional online cotrain backend."""

from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass
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

    def init(self) -> None:
        self.precision = _resolve_precision(self.train_cfg, self.torch_device)
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

    def update(self, phase: str, num_steps: int) -> dict[str, float]:
        mode = str(self.train_cfg.get("mode", "synthetic_ppo"))
        if mode == "dreamervla_cotrain":
            return self._dreamervla_cotrain_update(str(phase), int(num_steps))
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

    def state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            name: _cpu_state_dict(module)
            for name, module in self.components.items()
        }

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
        if "policy" not in components:
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

        metrics: dict[str, float] = {}
        for _ in range(max(1, int(num_steps))):
            for item in phases:
                if item == "wm":
                    metrics.update(self._dreamervla_wm_update_once())
                elif item == "classifier":
                    metrics.update(self._dreamervla_classifier_update_once())
                else:
                    metrics.update(self._dreamervla_rl_update_once())
        return metrics

    def _dreamervla_wm_update_once(self) -> dict[str, float]:
        batch = self._replay_client().sample(int(self.train_cfg.get("batch_size", 2)))
        with self._precision().context():
            raw = world_model_pretrain_step(
                policy=self._required_module("policy"),
                world_model=self._required_module("world_model"),
                optimizer=self._required_optimizer("world_model"),
                batch=batch,
                device=self.torch_device,
                optim_cfg=self._optim_cfg(),
            )
        return {
            "wm/loss": float(raw.get("loss", 0.0)),
        }

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
            )
        return {
            "cls/loss": float(raw.get("loss", 0.0)),
            "cls/acc": float(raw.get("acc", 0.0)),
            "cls/f1": float(raw.get("f1", 0.0)),
        }

    def _dreamervla_rl_update_once(self) -> dict[str, float]:
        # Phase 4 off-policy gating: drop replay samples older than
        # ``staleness_threshold`` rollout-policy versions (None / <0 disables → the
        # default, behaviour-preserving path). Graceful for fixed-base rollouts
        # (OFT), where the replay falls back to all valid samples.
        staleness_threshold = self.train_cfg.get("staleness_threshold", None)
        batch = self._replay_client().sample(
            int(self.train_cfg.get("batch_size", 2)),
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
            )
            if key in batch
        }
        with self._precision().context():
            raw = dino_wmpo_outcome_step(
                policy=self._required_module("policy"),
                chunk_world_model=self._required_module("world_model"),
                classifier=self._required_module("classifier"),
                classifier_threshold=float(self.train_cfg.get("classifier_threshold", 0.5)),
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
            "rl/policy_grad_norm": float(raw.get("actor_grad_norm", 0.0)),
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
        self, batch_size: int, *, staleness_threshold: int | None = None
    ) -> dict[str, Any]:
        # Forward the Phase 4 staleness gate only when it is set, so the default
        # path stays a byte-identical 1-arg call and minimal replay backends
        # (e.g. test doubles) that don't know about staleness still work.
        kwargs = (
            {}
            if staleness_threshold is None
            else {"staleness_threshold": int(staleness_threshold)}
        )
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


def online_classifier_update_step(**kwargs: Any) -> dict[str, Any]:
    """Lazy import wrapper preserving optional-Ray import isolation."""

    from dreamervla.runners.online_dreamervla import online_classifier_update_step as _impl

    return _impl(**kwargs)


def dino_wmpo_outcome_step(**kwargs: Any) -> dict[str, float]:
    """Lazy import wrapper preserving optional-Ray import isolation."""

    from dreamervla.algorithms.ppo import dino_wmpo_outcome_step as _impl

    return _impl(**kwargs)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: _independent_cpu(value) for key, value in module.state_dict().items()}


def _to_device_state(state_dict: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: (value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)).to(device)
        for key, value in state_dict.items()
    }
