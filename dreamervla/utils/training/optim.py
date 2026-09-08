from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn


def build_optimizer(module: nn.Module, optim_cfg: DictConfig) -> torch.optim.Optimizer:
    # Optimizer type
    name = str(optim_cfg.name).lower()
    if name not in ("adam", "adamw"):
        raise ValueError(f"Unsupported optimizer: {optim_cfg.name}")
    # Optional args
    betas = optim_cfg.get("betas")
    eps = optim_cfg.get("eps")
    # Base args
    optimizer_kwargs: dict[str, Any] = {
        "lr": float(optim_cfg.lr),
        "weight_decay": float(optim_cfg.weight_decay),
    }
    if betas is not None:
        optimizer_kwargs["betas"] = tuple(float(beta) for beta in betas)
    if eps is not None:
        optimizer_kwargs["eps"] = float(eps)
    # Optimizer build
    trainable_parameters = [
        parameter for parameter in module.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError(
            f"Module `{module.__class__.__name__}` does not expose any trainable parameters."
        )
    parameter_group_lrs = optim_cfg.get("parameter_group_lrs")
    optimizer_parameters: Any = trainable_parameters
    if parameter_group_lrs is not None:
        provider = module
        if not callable(getattr(provider, "optimizer_parameter_groups", None)):
            wrapped = getattr(module, "module", None)
            if callable(getattr(wrapped, "optimizer_parameter_groups", None)):
                provider = wrapped
        group_fn = getattr(provider, "optimizer_parameter_groups", None)
        if not callable(group_fn):
            raise TypeError(
                f"Module `{provider.__class__.__name__}` does not implement "
                "optimizer_parameter_groups required by parameter_group_lrs."
            )
        raw_groups = list(group_fn())
        grouped_ids: set[int] = set()
        optimizer_parameters = []
        for raw_group in raw_groups:
            if not isinstance(raw_group, Mapping):
                raise TypeError("optimizer_parameter_groups entries must be mappings")
            group_name = str(raw_group.get("group_name", "")).strip()
            if not group_name:
                raise ValueError("optimizer parameter groups require a non-empty group_name")
            parameters = [
                parameter for parameter in raw_group.get("params", ()) if parameter.requires_grad
            ]
            if not parameters:
                continue
            duplicates = [parameter for parameter in parameters if id(parameter) in grouped_ids]
            if duplicates:
                raise ValueError(f"optimizer parameter group {group_name!r} overlaps another group")
            grouped_ids.update(id(parameter) for parameter in parameters)
            group_lr = float(parameter_group_lrs.get(group_name, optim_cfg.lr))
            optimizer_parameters.append(
                {
                    "params": parameters,
                    "group_name": group_name,
                    "lr": group_lr,
                    "initial_lr": group_lr,
                }
            )
        expected_ids = {id(parameter) for parameter in trainable_parameters}
        if grouped_ids != expected_ids:
            raise ValueError(
                "optimizer_parameter_groups must cover every trainable parameter exactly once"
            )
    if name == "adamw":
        return torch.optim.AdamW(optimizer_parameters, **optimizer_kwargs)
    return torch.optim.Adam(optimizer_parameters, **optimizer_kwargs)


def _warmup_cosine_scale(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """Return a one-based linear warmup followed by cosine decay."""

    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be within [0, 1]")
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(1, int(total_steps) - int(warmup_steps) - 1)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def apply_optimizer_lr_schedule(
    optimizer: torch.optim.Optimizer,
    optim_cfg: DictConfig,
    *,
    step: int,
    total_steps: int,
) -> dict[str, float]:
    """Apply the configured per-group warmup/cosine schedule for one update.

    The ``pretrained_backbone`` group is disabled during adapter alignment, so
    its Adam moments are not populated until its own warmup begins.
    """

    schedule_name = str(optim_cfg.get("lr_scheduler", "constant")).strip().lower()
    if schedule_name not in {"constant", "cosine"}:
        raise ValueError(f"unsupported optimizer lr_scheduler: {schedule_name}")
    warmup_steps = int(optim_cfg.get("lr_warmup_steps", 0) or 0)
    backbone_warmup_steps = int(
        optim_cfg.get("pretrained_backbone_warmup_steps", warmup_steps) or 0
    )
    alignment_steps = int(optim_cfg.get("adapter_alignment_steps", 0) or 0)
    min_lr_ratio = float(optim_cfg.get("min_lr_ratio", 0.0) or 0.0)
    metrics: dict[str, float] = {}
    has_pretrained_group = any(
        str(group.get("group_name", "")) == "pretrained_backbone"
        for group in optimizer.param_groups
    )
    alignment_active = has_pretrained_group and int(step) < alignment_steps
    for index, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("group_name", f"group_{index}"))
        initial_lr = float(group.get("initial_lr", group["lr"]))
        is_pretrained = group_name == "pretrained_backbone"
        if is_pretrained and alignment_active:
            enabled = False
            scale = 0.0
        else:
            enabled = True
            local_step = int(step) - (alignment_steps if is_pretrained else 0)
            local_total = int(total_steps) - (alignment_steps if is_pretrained else 0)
            local_warmup = backbone_warmup_steps if is_pretrained else warmup_steps
            scale = (
                1.0
                if schedule_name == "constant"
                else _warmup_cosine_scale(
                    step=max(0, local_step),
                    total_steps=max(1, local_total),
                    warmup_steps=local_warmup,
                    min_lr_ratio=min_lr_ratio,
                )
            )
        group["lr"] = initial_lr * scale
        group["update_enabled"] = enabled
        metrics[f"{group_name}_learning_rate"] = float(group["lr"])
    metrics["adapter_alignment_active"] = float(alignment_active)
    return metrics
