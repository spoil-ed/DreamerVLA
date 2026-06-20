"""Save/load any nn.Module as a HF-style dir (config.json + model.safetensors).

Wraps a plain module + its Hydra init-args in a PreTrainedModel so HF's
save_pretrained/from_pretrained machinery produces a portable checkpoint;
load rebuilds the inner module via hydra.utils.instantiate and loads weights.
"""

from __future__ import annotations

from typing import Any

import hydra
import torch
from transformers import PretrainedConfig, PreTrainedModel


class HFModuleConfig(PretrainedConfig):
    model_type = "dreamervla_module"

    def __init__(self, target: str = "", init_args: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.target = target
        self.init_args = init_args or {}
        super().__init__(**kwargs)


class HFModuleWrapper(PreTrainedModel):
    config_class = HFModuleConfig

    def __init__(self, config: HFModuleConfig, module: torch.nn.Module | None = None) -> None:
        super().__init__(config)
        if module is None:
            module = hydra.utils.instantiate({"_target_": config.target, **config.init_args})
        self.module = module
        self.post_init()

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - not used for save/load
        return self.module(*args, **kwargs)


def save_module_pretrained(
    module: torch.nn.Module, save_dir: str, *, target: str, init_args: dict[str, Any]
) -> None:
    cfg = HFModuleConfig(target=target, init_args=dict(init_args))
    wrapper = HFModuleWrapper(cfg, module=module)
    wrapper.save_pretrained(save_dir, safe_serialization=True)


def load_module_pretrained(save_dir: str, *, map_location: str = "cpu") -> torch.nn.Module:
    wrapper = HFModuleWrapper.from_pretrained(save_dir)
    return wrapper.module.to(map_location)
