"""Shared construction helpers for offline WM/classifier warmup."""

from __future__ import annotations

from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from dreamervla.algorithms.critic import build_classifier
from dreamervla.runtime.distributed import unwrap_module as _unwrap
from dreamervla.runtime.world_model_training_base import WorldModelTrainingBase
from dreamervla.utils.hf_checkpoint import is_hf_checkpoint
from dreamervla.utils.hf_module import load_module_pretrained
from dreamervla.utils.optim import build_optimizer
from dreamervla.utils.torch_utils import precision_dtype

_WORLD_MODEL_DDP_DEFAULTS: dict[str, bool] = {
    "find_unused_parameters": True,
    "broadcast_buffers": True,
}
_WORLD_MODEL_DDP_OPTION_KEYS = frozenset(
    {
        "find_unused_parameters",
        "broadcast_buffers",
        "static_graph",
        "gradient_as_bucket_view",
    }
)


def _component_hydra_cfg(
    cfg: DictConfig,
    *,
    component_path: str,
    worker_component_path: str,
) -> DictConfig:
    """Resolve a normal Hydra component or a worker ``target`` config."""

    component = OmegaConf.select(cfg, component_path, default=None)
    if component is not None and OmegaConf.select(component, "_target_", default=None) is not None:
        return component

    worker_component = OmegaConf.select(cfg, worker_component_path, default=None)
    target = OmegaConf.select(worker_component, "target", default=None)
    if target is None:
        raise ValueError(
            f"{component_path} requires either _target_ or {worker_component_path}.target"
        )
    kwargs = OmegaConf.select(worker_component, "kwargs", default={}) or {}
    resolved = dict(OmegaConf.to_container(kwargs, resolve=True))
    if component is not None:
        overrides = dict(OmegaConf.to_container(component, resolve=True))
        overrides.pop("_target_", None)
        overrides.pop("target", None)
        overrides.pop("kwargs", None)
        resolved.update(overrides)
    return OmegaConf.create({"_target_": str(target), **resolved})


def _world_model_ddp_wrap_kwargs(cfg: DictConfig) -> dict[str, bool]:
    """Resolve optional WM-only DDP knobs for offline replay training."""

    raw = OmegaConf.select(cfg, "training.world_model_ddp", default=None)
    if raw is None:
        return dict(_WORLD_MODEL_DDP_DEFAULTS)
    plain = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(plain, dict):
        raise TypeError("training.world_model_ddp must be a mapping")
    unknown = sorted(set(plain) - _WORLD_MODEL_DDP_OPTION_KEYS)
    if unknown:
        raise ValueError(
            "training.world_model_ddp contains unknown option(s): "
            + ", ".join(str(item) for item in unknown)
        )
    resolved = dict(_WORLD_MODEL_DDP_DEFAULTS)
    for key, value in plain.items():
        if not isinstance(value, bool):
            raise TypeError(f"training.world_model_ddp.{key} must be boolean")
        resolved[str(key)] = value
    return resolved


def validate_task_conditioning_cfg(
    cfg: DictConfig,
    *,
    world_model: Any,
    classifier: Any,
) -> None:
    """Validate optional explicit task conditioning without binding implementations."""

    tc = OmegaConf.select(cfg, "task_conditioning", default={}) or {}
    if OmegaConf.is_config(tc):
        tc = OmegaConf.to_container(tc, resolve=True)
    tc = dict(tc)
    if not bool(tc.get("enabled", False)):
        return
    num_tasks = int(tc.get("num_tasks", 0) or 0)
    embedding_dim = int(tc.get("embedding_dim", 0) or 0)
    if num_tasks <= 0 or embedding_dim <= 0:
        raise ValueError("task_conditioning.enabled requires positive num_tasks and embedding_dim")
    missing: list[str] = []
    for name, module in (
        ("world_model", _unwrap(world_model)),
        ("classifier", _unwrap(classifier)),
    ):
        if not bool(getattr(module, "supports_task_conditioning", False)):
            missing.append(name)
    if missing:
        raise ValueError(
            "task_conditioning.enabled=true but selected implementation(s) lack "
            f"task-conditioning support: {', '.join(missing)}"
        )


class _WorldModelTrainingCommon(WorldModelTrainingBase):
    """Build the trainable modules required by offline warmup."""

    runner_name = "world_model_training"
    runner_status = "current"
    runner_family = "world_model"

    include_keys = (*WorldModelTrainingBase.include_keys, "classifier_threshold")

    @property
    def _rank(self) -> int:
        return int(getattr(self.distributed, "rank", 0) or 0)

    @property
    def _world_size(self) -> int:
        return int(getattr(self.distributed, "world_size", 1) or 1)

    def _build_trainable_classifier(self, cfg: DictConfig) -> None:
        """Build the warmup classifier and its optimizer from Hydra config."""

        self.classifier_threshold = float(
            OmegaConf.select(cfg, "algorithm.lumos.classifier_threshold", default=0.5) or 0.5
        )
        cls_blob = OmegaConf.select(cfg, "classifier", default=None)
        cls_kwargs: dict[str, Any] = (
            dict(OmegaConf.to_container(cls_blob, resolve=True)) if cls_blob is not None else {}
        )
        if (
            cls_kwargs.get("latent_dim") is None
            and str(cls_kwargs.get("token_pool", "flat")) != "mean"
        ):
            cls_kwargs["latent_dim"] = int(OmegaConf.select(cfg, "world_model.obs_dim"))
        self._classifier_target = str(
            cls_kwargs.get("_target_") or "dreamervla.algorithms.critic.LatentSuccessClassifier"
        )
        self._classifier_cls_kwargs = {
            key: value for key, value in cls_kwargs.items() if key != "_target_"
        }
        classifier = build_classifier(cls_kwargs).to(self.device)
        warm = OmegaConf.select(cfg, "init.classifier_state_ckpt", default=None)
        if warm:
            if is_hf_checkpoint(str(warm)):
                src = load_module_pretrained(str(warm))
                classifier.load_state_dict(src.state_dict())
            else:
                payload = torch.load(str(warm), map_location="cpu", weights_only=False)
                model_sd = payload.get("model", payload.get("state_dicts", {}).get("model"))
                classifier.load_state_dict(model_sd)
                self.classifier_threshold = float(
                    payload.get("threshold", self.classifier_threshold)
                )
            if self.distributed.is_main_process:
                print(f"[warmup] classifier warm-started from {warm}", flush=True)
        for parameter in classifier.parameters():
            parameter.requires_grad_(True)
        classifier.train()
        self.classifier = self.distributed.wrap_trainable_module(classifier)
        cls_optim_cfg = OmegaConf.select(cfg, "optim.classifier")
        if cls_optim_cfg is None:
            raise ValueError("classifier warmup requires `optim.classifier`.")
        self.classifier_optimizer = build_optimizer(self.classifier, cls_optim_cfg)
        self._cls_window = int(cls_kwargs.get("window", 8))

    def _load_world_model_init_ckpt(self, ckpt_path: str) -> None:
        """Load either a portable HF module or a normal runner checkpoint."""

        if is_hf_checkpoint(ckpt_path):
            src = load_module_pretrained(ckpt_path)
            self._unwrapped_world_model.load_state_dict(src.state_dict())
            if self.distributed.is_main_process:
                print(f"[init] world_model loaded from HF dir: {ckpt_path}", flush=True)
        else:
            super()._load_world_model_init_ckpt(ckpt_path)

    def _assert_optimizers_disjoint(self) -> None:
        seen: set[int] = set()
        for name, optimizer in (
            ("world_model", self.world_model_optimizer),
            ("classifier", self.classifier_optimizer),
        ):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if id(parameter) in seen:
                        raise RuntimeError(
                            f"optimizer parameter sets overlap at {name} — phase "
                            "isolation would be violated."
                        )
                    seen.add(id(parameter))
        if self.distributed.is_main_process:
            print("[ok] warmup optimizer parameter sets are disjoint", flush=True)

    def _build_components(self, cfg: DictConfig) -> None:
        self.encoder = None
        self.processor = None
        self._oft_hidden_token_extractor = None
        self.policy = None
        self.policy_optimizer = None
        self.ref_policy = None
        self.critic = None
        self.critic_optimizer = None
        self.classifier = None
        self.classifier_optimizer = None

        world_model_cfg = _component_hydra_cfg(
            cfg,
            component_path="world_model",
            worker_component_path="ray_components.world_model",
        )
        OmegaConf.update(cfg, "world_model", world_model_cfg, merge=False)
        parameter_precision = OmegaConf.select(cfg, "optim.param_precision", default=None)
        if parameter_precision is None:
            raise ValueError(
                "optim.param_precision is required to select world-model parameter dtype"
            )
        self.world_model = hydra.utils.instantiate(world_model_cfg).to(
            device=self.device,
            dtype=precision_dtype(str(parameter_precision)),
        )
        self._unwrapped_world_model = self.world_model
        wm_ckpt = OmegaConf.select(cfg, "init.world_model_state_ckpt", default=None)
        if wm_ckpt:
            self._load_world_model_init_ckpt(str(wm_ckpt))
        self.world_model = self.distributed.wrap_trainable_module(
            self.world_model,
            **_world_model_ddp_wrap_kwargs(cfg),
        )
        self.world_model_optimizer = build_optimizer(
            self.world_model, OmegaConf.select(cfg, "optim.world_model")
        )

        classifier_steps = int(
            OmegaConf.select(cfg, "training.classifier_warmup_steps", default=0) or 0
        )
        if classifier_steps <= 0:
            return

        self._build_trainable_classifier(cfg)
        validate_task_conditioning_cfg(
            cfg,
            world_model=self.world_model,
            classifier=self.classifier,
        )
        self._assert_optimizers_disjoint()
