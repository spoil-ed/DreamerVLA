"""Environment package exports with lazy optional dependency imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ACTION_HIGH": "dreamervla.envs.train_env",
    "ACTION_LOW": "dreamervla.envs.train_env",
    "DreamerVLAOnlineEvalEnv": "dreamervla.envs.eval_env",
    "DreamerVLAOnlineEvalEnvConfig": "dreamervla.envs.eval_env",
    "DreamerVLAOnlineTrainEnv": "dreamervla.envs.train_env",
    "DreamerVLAOnlineTrainEnvConfig": "dreamervla.envs.train_env",
    "EvalEnv": "dreamervla.envs.eval_env",
    "LIBERODreamerEnv": "dreamervla.envs.libero_env",
    "LIBEROOnlineEnv": "dreamervla.envs.libero_online_env",
    "LIBEROOnlineEnvConfig": "dreamervla.envs.libero_online_env",
    "TASK_MAX_STEPS": "dreamervla.envs.libero_env",
    "TrainEnv": "dreamervla.envs.train_env",
    "build_dreamervla_online_train_envs": "dreamervla.envs.train_env",
    "build_libero_online_envs": "dreamervla.envs.libero_online_env",
    "get_libero_dummy_action": "dreamervla.envs.libero_env",
    "get_libero_env": "dreamervla.envs.libero_env",
    "get_libero_image": "dreamervla.envs.libero_env",
    "normalize_libero_action": "dreamervla.envs.train_env",
    "quat2axisangle": "dreamervla.envs.libero_env",
    "resolve_libero_eval_protocol": "dreamervla.envs.libero_env",
    "save_rollout_video": "dreamervla.envs.libero_env",
    "select_libero_action_chunk": "dreamervla.envs.libero_env",
    "unnormalize_libero_action": "dreamervla.envs.train_env",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
