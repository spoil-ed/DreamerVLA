"""LIBERO env package, aligned with the RLinf env layout."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ACTION_HIGH": "dreamervla.envs.libero.libero_env",
    "ACTION_LOW": "dreamervla.envs.libero.libero_env",
    "DreamerVLAOnlineTrainEnv": "dreamervla.envs.libero.libero_env",
    "DreamerVLAOnlineTrainEnvConfig": "dreamervla.envs.libero.libero_env",
    "LIBERODreamerEnv": "dreamervla.envs.libero.utils",
    "LiberoEnv": "dreamervla.envs.libero.libero_env",
    "OnlineEglVecEnv": "dreamervla.envs.libero.venv",
    "ReconfigureSubprocEnv": "dreamervla.envs.libero.venv",
    "TASK_MAX_STEPS": "dreamervla.envs.libero.utils",
    "build_dreamervla_online_train_envs": "dreamervla.envs.libero.libero_env",
    "get_libero_dummy_action": "dreamervla.envs.libero.utils",
    "get_libero_env": "dreamervla.envs.libero.utils",
    "get_libero_image": "dreamervla.envs.libero.utils",
    "normalize_libero_action": "dreamervla.envs.libero.libero_env",
    "quat2axisangle": "dreamervla.envs.libero.utils",
    "resolve_libero_eval_protocol": "dreamervla.envs.libero.utils",
    "save_rollout_video": "dreamervla.envs.libero.utils",
    "select_libero_action_chunk": "dreamervla.envs.libero.utils",
    "unnormalize_libero_action": "dreamervla.envs.libero.libero_env",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
