from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn


def build_optimizer(module: nn.Module, optim_cfg: DictConfig) -> torch.optim.Optimizer:
    # Optimizer type
    if str(optim_cfg.name).lower() != "adam":
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
    trainable_parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError(f"Module `{module.__class__.__name__}` does not expose any trainable parameters.")
    return torch.optim.Adam(trainable_parameters, **optimizer_kwargs)
