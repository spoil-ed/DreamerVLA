"""Fail-fast hyperparameter contracts shared by Hydra and direct APIs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _finite_float(config: Mapping[str, Any], key: str, prefix: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{_path(prefix, key)} must be a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{_path(prefix, key)} must be a finite number, got {value!r}")
    return result


def _integer(config: Mapping[str, Any], key: str, prefix: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{_path(prefix, key)} must be an integer, got {value!r}")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{_path(prefix, key)} must be an integer, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric != float(result):
        raise ValueError(f"{_path(prefix, key)} must be an integer, got {value!r}")
    return result


def _require_unit_interval(config: Mapping[str, Any], key: str, prefix: str) -> None:
    value = _finite_float(config, key, prefix)
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{_path(prefix, key)} must be in [0, 1], got {value!r}")


def _require_non_negative(config: Mapping[str, Any], key: str, prefix: str) -> None:
    value = _finite_float(config, key, prefix)
    if value is not None and value < 0.0:
        raise ValueError(f"{_path(prefix, key)} must be >= 0, got {value!r}")


def _require_positive(config: Mapping[str, Any], key: str, prefix: str) -> None:
    value = _finite_float(config, key, prefix)
    if value is not None and value <= 0.0:
        raise ValueError(f"{_path(prefix, key)} must be > 0, got {value!r}")


def _require_positive_integer(config: Mapping[str, Any], key: str, prefix: str) -> None:
    value = _integer(config, key, prefix)
    if value is not None and value <= 0:
        raise ValueError(f"{_path(prefix, key)} must be a positive integer, got {value!r}")


def validate_ppo_hyperparameters(
    config: Mapping[str, Any] | None,
    *,
    prefix: str,
) -> None:
    """Validate configured PPO/Dreamer ranges without supplying defaults."""

    if config is None:
        return
    for key in ("gamma", "gae_lambda", "ppo_gamma", "success_return_shaping_discount"):
        _require_unit_interval(config, key, prefix)
    for key in (
        "kl_beta",
        "kl_coef",
        "prev_kl_coef",
        "entropy_bonus",
        "entropy_coef",
        "actent",
        "actor_bc_to_ref_scale",
        "actor_bc_to_vla_scale",
        "actor_bc_scale",
        "reward_coef",
    ):
        _require_non_negative(config, key, prefix)
    for key in ("advantage_eps", "clip_log_ratio"):
        _require_positive(config, key, prefix)
    for key in (
        "group_size",
        "ppo_rollouts_per_start",
        "ppo_update_epochs",
        "imagination_horizon",
        "horizon",
        "imag_last",
    ):
        _require_positive_integer(config, key, prefix)

    clip_low = _finite_float(config, "clip_ratio_low", prefix)
    if clip_low is not None and not 0.0 <= clip_low < 1.0:
        raise ValueError(f"{_path(prefix, 'clip_ratio_low')} must be in [0, 1), got {clip_low!r}")
    _require_non_negative(config, "clip_ratio_high", prefix)
    clip_c = _finite_float(config, "clip_ratio_c", prefix)
    if clip_c is not None and clip_c <= 1.0:
        raise ValueError(f"{_path(prefix, 'clip_ratio_c')} must be > 1, got {clip_c!r}")

    reward_low = _finite_float(config, "rewards_lower_bound", prefix)
    reward_high = _finite_float(config, "rewards_upper_bound", prefix)
    if reward_low is not None and reward_high is not None and reward_low > reward_high:
        raise ValueError(
            f"{_path(prefix, 'rewards_lower_bound')} must be <= "
            f"{_path(prefix, 'rewards_upper_bound')} ({reward_low!r} > {reward_high!r})"
        )

    lumos = config.get("lumos")
    if lumos is None:
        return
    if not isinstance(lumos, Mapping):
        raise ValueError(f"{_path(prefix, 'lumos')} must be a mapping")
    lumos_prefix = _path(prefix, "lumos")
    _require_unit_interval(lumos, "classifier_threshold", lumos_prefix)
    for key in (
        "chunk_size",
        "episode_max_steps",
        "classifier_min_steps",
        "ppo_rollouts_per_start_min",
        "ppo_rollouts_per_start_max",
        "update_micro_batch_starts",
        "eval_micro_batch",
    ):
        _require_positive_integer(lumos, key, lumos_prefix)
    group_min = _integer(lumos, "ppo_rollouts_per_start_min", lumos_prefix)
    group_max = _integer(lumos, "ppo_rollouts_per_start_max", lumos_prefix)
    if group_min is not None and group_max is not None and group_min > group_max:
        raise ValueError(
            f"{_path(lumos_prefix, 'ppo_rollouts_per_start_min')} must be <= "
            f"{_path(lumos_prefix, 'ppo_rollouts_per_start_max')} "
            f"({group_min} > {group_max})"
        )


def validate_tdmpc_hyperparameters(
    config: Mapping[str, Any] | None,
    *,
    prefix: str,
) -> None:
    """Validate TD-MPC CEM/MPPI geometry without silently clamping it."""

    if config is None:
        return
    for key in (
        "horizon",
        "iterations",
        "num_samples",
        "num_elites",
        "action_dim",
        "execute_steps",
    ):
        _require_positive_integer(config, key, prefix)
    num_pi_trajs = _integer(config, "num_pi_trajs", prefix)
    if num_pi_trajs is not None and num_pi_trajs < 0:
        raise ValueError(
            f"{_path(prefix, 'num_pi_trajs')} must be a non-negative integer, got {num_pi_trajs!r}"
        )
    for key in ("min_std", "max_std", "temperature"):
        _require_positive(config, key, prefix)
    _require_unit_interval(config, "gamma", prefix)

    num_samples = _integer(config, "num_samples", prefix)
    num_elites = _integer(config, "num_elites", prefix)
    if num_elites is not None and num_samples is not None and num_elites > num_samples:
        raise ValueError(
            f"{_path(prefix, 'num_elites')} must be <= {_path(prefix, 'num_samples')} "
            f"({num_elites} > {num_samples})"
        )
    if num_pi_trajs is not None and num_samples is not None and num_pi_trajs > num_samples:
        raise ValueError(
            f"{_path(prefix, 'num_pi_trajs')} must be <= {_path(prefix, 'num_samples')} "
            f"({num_pi_trajs} > {num_samples})"
        )
    min_std = _finite_float(config, "min_std", prefix)
    max_std = _finite_float(config, "max_std", prefix)
    if min_std is not None and max_std is not None and min_std > max_std:
        raise ValueError(
            f"{_path(prefix, 'min_std')} must be <= {_path(prefix, 'max_std')} "
            f"({min_std!r} > {max_std!r})"
        )
    execute_steps = _integer(config, "execute_steps", prefix)
    horizon = _integer(config, "horizon", prefix)
    if execute_steps is not None and horizon is not None and execute_steps > horizon:
        raise ValueError(
            f"{_path(prefix, 'execute_steps')} must be <= {_path(prefix, 'horizon')} "
            f"({execute_steps} > {horizon})"
        )


__all__ = ["validate_ppo_hyperparameters", "validate_tdmpc_hyperparameters"]
