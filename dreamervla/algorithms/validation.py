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


def _require_non_negative_integer(config: Mapping[str, Any], key: str, prefix: str) -> None:
    value = _integer(config, key, prefix)
    if value is not None and value < 0:
        raise ValueError(f"{_path(prefix, key)} must be a non-negative integer, got {value!r}")


def _require_positive_unit_interval(config: Mapping[str, Any], key: str, prefix: str) -> None:
    value = _finite_float(config, key, prefix)
    if value is not None and not 0.0 < value <= 1.0:
        raise ValueError(f"{_path(prefix, key)} must be in (0, 1], got {value!r}")


def validate_ppo_hyperparameters(
    config: Mapping[str, Any] | None,
    *,
    prefix: str,
) -> None:
    """Validate configured PPO/Dreamer ranges without supplying defaults."""

    if config is None:
        return
    for key in (
        "gamma",
        "gae_lambda",
        "lam",
        "ppo_gamma",
        "success_return_shaping_discount",
    ):
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
    _require_positive_unit_interval(config, "target_critic_tau", prefix)
    for key in (
        "group_size",
        "ppo_rollouts_per_start",
        "ppo_update_epochs",
        "imagination_horizon",
        "horizon",
        "rssm_action_dim",
    ):
        _require_positive_integer(config, key, prefix)
    _require_non_negative_integer(config, "imag_last", prefix)

    action_scale = config.get("rssm_action_scale")
    if action_scale is not None and str(action_scale).lower() not in {
        "policy",
        "raw",
        "identity",
        "",
        "env",
        "libero_env",
        "libero",
    }:
        raise ValueError(
            f"{_path(prefix, 'rssm_action_scale')} must be policy or env, got {action_scale!r}"
        )
    action_clip = config.get("rssm_action_clip")
    if action_clip is not None and not isinstance(action_clip, bool):
        raise ValueError(
            f"{_path(prefix, 'rssm_action_clip')} must be a boolean, got {action_clip!r}"
        )
    action_low_raw = config.get("rssm_action_low")
    action_high_raw = config.get("rssm_action_high")
    if (action_low_raw is None) != (action_high_raw is None):
        raise ValueError(
            f"{_path(prefix, 'rssm_action_low')} and "
            f"{_path(prefix, 'rssm_action_high')} must be configured together"
        )
    if action_low_raw is not None:
        try:
            if isinstance(action_low_raw, (str, bytes)) or isinstance(
                action_high_raw, (str, bytes)
            ):
                raise TypeError
            action_low = [float(value) for value in action_low_raw]
            action_high = [float(value) for value in action_high_raw]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{_path(prefix, 'rssm_action_low')} and "
                f"{_path(prefix, 'rssm_action_high')} must be finite numeric vectors"
            ) from exc
        if not action_low:
            raise ValueError(f"{_path(prefix, 'rssm_action_low')} must be non-empty")
        if len(action_low) != len(action_high):
            raise ValueError(
                f"{_path(prefix, 'rssm_action_high')} must have length {len(action_low)}, "
                f"got {len(action_high)}"
            )
        if not all(math.isfinite(value) for value in (*action_low, *action_high)):
            raise ValueError(
                f"{_path(prefix, 'rssm_action_low')} and "
                f"{_path(prefix, 'rssm_action_high')} must contain only finite values"
            )
        if any(low >= high for low, high in zip(action_low, action_high, strict=True)):
            raise ValueError(
                f"{_path(prefix, 'rssm_action_low')} must be elementwise smaller than "
                f"{_path(prefix, 'rssm_action_high')}"
            )
        action_dim = _integer(config, "rssm_action_dim", prefix)
        if action_dim is not None and action_dim != len(action_low):
            raise ValueError(
                f"{_path(prefix, 'rssm_action_dim')}={action_dim} does not match "
                f"configured bound width {len(action_low)}"
            )

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
    if lumos is not None:
        if not isinstance(lumos, Mapping):
            raise ValueError(f"{_path(prefix, 'lumos')} must be a mapping")
        lumos_prefix = _path(prefix, "lumos")
        _require_unit_interval(lumos, "classifier_threshold", lumos_prefix)
        for key in (
            "chunk_size",
            "episode_max_steps",
            "ppo_rollouts_per_start_min",
            "ppo_rollouts_per_start_max",
            "eval_micro_batch",
        ):
            _require_positive_integer(lumos, key, lumos_prefix)
        for key in (
            "classifier_min_steps",
            "update_micro_batch_starts",
            "imagine_micro_batch",
        ):
            _require_non_negative_integer(lumos, key, lumos_prefix)
        group_min = _integer(lumos, "ppo_rollouts_per_start_min", lumos_prefix)
        group_max = _integer(lumos, "ppo_rollouts_per_start_max", lumos_prefix)
        if group_min is not None and group_max is not None and group_min > group_max:
            raise ValueError(
                f"{_path(lumos_prefix, 'ppo_rollouts_per_start_min')} must be <= "
                f"{_path(lumos_prefix, 'ppo_rollouts_per_start_max')} "
                f"({group_min} > {group_max})"
            )

    tdmpc_ac = config.get("tdmpc_ac")
    if tdmpc_ac is not None:
        if not isinstance(tdmpc_ac, Mapping):
            raise ValueError(f"{_path(prefix, 'tdmpc_ac')} must be a mapping")
        tdmpc_prefix = _path(prefix, "tdmpc_ac")
        _require_positive_integer(tdmpc_ac, "action_dim", tdmpc_prefix)
        _require_positive_unit_interval(tdmpc_ac, "target_critic_tau", tdmpc_prefix)
        for key in (
            "critic_loss_scale",
            "imagined_critic_loss_scale",
            "replay_critic_loss_scale",
        ):
            _require_non_negative(tdmpc_ac, key, tdmpc_prefix)
        _finite_float(tdmpc_ac, "terminal_value_scale", tdmpc_prefix)
        value_mode = tdmpc_ac.get("value_mode")
        if value_mode is not None and str(value_mode).lower() not in {
            "state",
            "v",
            "v_z",
            "state_action",
            "q",
            "q_za",
            "q(z,a)",
        }:
            raise ValueError(
                f"{_path(tdmpc_prefix, 'value_mode')} must be state or state_action, "
                f"got {value_mode!r}"
            )

    relabel = config.get("real_rollout_relabel")
    if relabel is not None:
        if not isinstance(relabel, Mapping):
            raise ValueError(f"{_path(prefix, 'real_rollout_relabel')} must be a mapping")
        relabel_prefix = _path(prefix, "real_rollout_relabel")
        for key in ("loss_scale", "positive_weight", "negative_weight"):
            _require_non_negative(relabel, key, relabel_prefix)
        _require_positive_integer(relabel, "batch_size", relabel_prefix)
        _require_non_negative_integer(relabel, "max_steps_per_trajectory", relabel_prefix)
        _require_unit_interval(relabel, "outcome_baseline", relabel_prefix)

    normalization = config.get("return_normalization")
    if normalization is not None:
        if not isinstance(normalization, Mapping):
            raise ValueError(f"{_path(prefix, 'return_normalization')} must be a mapping")
        normalization_prefix = _path(prefix, "return_normalization")
        mode = str(normalization.get("mode", "none")).lower()
        if mode not in {
            "none",
            "identity",
            "off",
            "false",
            "0",
            "dreamerv3",
            "perc",
            "percentile",
            "percentile_scale",
        }:
            raise ValueError(
                f"{_path(normalization_prefix, 'mode')} must be none or dreamerv3, got {mode!r}"
            )
        _require_unit_interval(normalization, "low", normalization_prefix)
        _require_unit_interval(normalization, "high", normalization_prefix)
        _require_positive(normalization, "eps", normalization_prefix)
        low = _finite_float(normalization, "low", normalization_prefix)
        high = _finite_float(normalization, "high", normalization_prefix)
        if low is not None and high is not None and low > high:
            raise ValueError(
                f"{_path(normalization_prefix, 'low')} must be <= "
                f"{_path(normalization_prefix, 'high')} ({low!r} > {high!r})"
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
    for key in ("terminal_value_scale", "reward_scale"):
        _finite_float(config, key, prefix)
    _require_unit_interval(config, "gamma", prefix)

    value_mode = config.get("value_mode")
    if value_mode is not None and str(value_mode).lower() not in {
        "state",
        "v",
        "v_z",
        "state_action",
        "q",
        "q_za",
        "q(z,a)",
    }:
        raise ValueError(
            f"{_path(prefix, 'value_mode')} must be state or state_action, got {value_mode!r}"
        )

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
