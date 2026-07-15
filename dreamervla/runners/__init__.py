from __future__ import annotations

import importlib

from dreamervla.runners.base_runner import BaseRunner


_RUNNER_MODULES = {
    "DinoTokenWorldModelTrainingRunner": (
        "dreamervla.runners.dino_token_world_model_training_runner"
    ),
    "RolloutCollectionRunner": "dreamervla.runners.rollout_collection_runner",
    "WorldModelTrainingRunner": "dreamervla.runners.world_model_training_runner",
    "SuccessClassifierTrainingRunner": (
        "dreamervla.runners.success_classifier_training_runner"
    ),
    "CotrainRunner": "dreamervla.runners.cotrain_runner",
    "DreamerRunner": "dreamervla.runners.dreamer_runner",
    "LIBEROVLAEvaluationRunner": (
        "dreamervla.runners.libero_vla_evaluation_runner"
    ),
}

PUBLIC_RUNNERS = list(_RUNNER_MODULES)


__all__ = [
    "BaseRunner",
    "PUBLIC_RUNNERS",
    *PUBLIC_RUNNERS,
]


def __getattr__(name: str) -> object:
    module_name = _RUNNER_MODULES.get(name)
    if module_name is not None:
        runner = getattr(importlib.import_module(module_name), name)
        globals()[name] = runner
        return runner
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted([*globals(), *PUBLIC_RUNNERS])
