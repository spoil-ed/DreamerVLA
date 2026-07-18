from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"gamma": -0.01}, "algorithm.gamma"),
        ({"gamma": 1.01}, "algorithm.gamma"),
        ({"gae_lambda": -0.01}, "algorithm.gae_lambda"),
        ({"lam": 1.01}, "algorithm.lam"),
        ({"ppo_gamma": 1.01}, "algorithm.ppo_gamma"),
        ({"group_size": 0}, "algorithm.group_size"),
        ({"ppo_rollouts_per_start": 1.5}, "algorithm.ppo_rollouts_per_start"),
        ({"ppo_update_epochs": 0}, "algorithm.ppo_update_epochs"),
        ({"clip_ratio_low": 1.0}, "algorithm.clip_ratio_low"),
        ({"clip_ratio_high": -0.1}, "algorithm.clip_ratio_high"),
        ({"clip_ratio_c": 1.0}, "algorithm.clip_ratio_c"),
        ({"clip_log_ratio": 0.0}, "algorithm.clip_log_ratio"),
        ({"advantage_eps": 0.0}, "algorithm.advantage_eps"),
        ({"kl_coef": -0.1}, "algorithm.kl_coef"),
        ({"entropy_coef": -0.1}, "algorithm.entropy_coef"),
        ({"target_critic_tau": 0.0}, "algorithm.target_critic_tau"),
        ({"rssm_action_dim": 0}, "algorithm.rssm_action_dim"),
        ({"rssm_action_scale": "mystery"}, "algorithm.rssm_action_scale"),
        ({"rssm_action_low": [0.0]}, "algorithm.rssm_action_low"),
        (
            {"rssm_action_low": [0.0, -1.0], "rssm_action_high": [1.0]},
            "algorithm.rssm_action_high",
        ),
        (
            {"rssm_action_low": [0.0], "rssm_action_high": [0.0]},
            "algorithm.rssm_action_low",
        ),
        (
            {
                "rssm_action_dim": 2,
                "rssm_action_low": [-1.0],
                "rssm_action_high": [1.0],
            },
            "algorithm.rssm_action_dim",
        ),
        (
            {"rewards_lower_bound": 2.0, "rewards_upper_bound": 1.0},
            "algorithm.rewards_lower_bound",
        ),
        ({"lumos": {"classifier_threshold": 1.1}}, "algorithm.lumos.classifier_threshold"),
        ({"lumos": {"chunk_size": 0}}, "algorithm.lumos.chunk_size"),
        (
            {"lumos": {"update_micro_batch_starts": -1}},
            "algorithm.lumos.update_micro_batch_starts",
        ),
        (
            {
                "lumos": {
                    "ppo_rollouts_per_start_min": 8,
                    "ppo_rollouts_per_start_max": 4,
                }
            },
            "algorithm.lumos.ppo_rollouts_per_start_min",
        ),
    ],
)
def test_validate_cfg_rejects_invalid_algorithm_hyperparameters(
    section: dict[str, object], message: str
) -> None:
    cfg = OmegaConf.create({"algorithm": section})

    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    "base",
    ["actor.train_cfg.algorithm_cfg", "learner.train_cfg.algorithm_cfg"],
)
def test_validate_cfg_checks_nested_worker_algorithm_blocks(base: str) -> None:
    cfg = OmegaConf.create({})
    OmegaConf.update(cfg, base, {"clip_ratio_low": -0.1}, merge=False)

    with pytest.raises(ValueError, match=base.replace(".", r"\.") + r"\.clip_ratio_low"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("tdmpc_ac", "message"),
    [
        ({"action_dim": 0}, "action_dim"),
        ({"target_critic_tau": 0.0}, "target_critic_tau"),
        ({"critic_loss_scale": -1.0}, "critic_loss_scale"),
        ({"imagined_critic_loss_scale": -1.0}, "imagined_critic_loss_scale"),
        ({"replay_critic_loss_scale": -1.0}, "replay_critic_loss_scale"),
        ({"terminal_value_scale": float("nan")}, "terminal_value_scale"),
        ({"value_mode": "mystery"}, "value_mode"),
    ],
)
def test_validate_cfg_rejects_invalid_tdmpc_actor_critic_block(
    tdmpc_ac: dict[str, object], message: str
) -> None:
    cfg = OmegaConf.create({"algorithm": {"tdmpc_ac": tdmpc_ac}})

    with pytest.raises(ValueError, match=rf"algorithm\.tdmpc_ac\.{message}"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("real_relabel", "message"),
    [
        ({"loss_scale": -1.0}, "loss_scale"),
        ({"batch_size": 0}, "batch_size"),
        ({"positive_weight": -1.0}, "positive_weight"),
        ({"negative_weight": -1.0}, "negative_weight"),
        ({"outcome_baseline": float("nan")}, "outcome_baseline"),
        ({"max_steps_per_trajectory": -1}, "max_steps_per_trajectory"),
    ],
)
def test_validate_cfg_rejects_invalid_real_relabel_block(
    real_relabel: dict[str, object], message: str
) -> None:
    cfg = OmegaConf.create({"algorithm": {"real_rollout_relabel": real_relabel}})

    with pytest.raises(ValueError, match=rf"algorithm\.real_rollout_relabel\.{message}"):
        validate_cfg(cfg)


def test_validate_cfg_accepts_zero_sentinel_batch_limits() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "imag_last": 0,
                "lumos": {
                    "classifier_min_steps": 0,
                    "update_micro_batch_starts": 0,
                    "imagine_micro_batch": 0,
                },
            }
        }
    )

    assert validate_cfg(cfg) is cfg


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizon", 0),
        ("iterations", 0),
        ("num_samples", 0),
        ("num_elites", 0),
        ("num_pi_trajs", -1),
        ("action_dim", 0),
        ("min_std", 0.0),
        ("max_std", 0.0),
        ("temperature", 0.0),
        ("gamma", 1.1),
        ("execute_steps", 0),
    ],
)
def test_validate_cfg_rejects_invalid_enabled_tdmpc_mpc(field: str, value: float) -> None:
    section = {
        "enabled": True,
        "horizon": 3,
        "iterations": 2,
        "num_samples": 8,
        "num_elites": 2,
        "num_pi_trajs": 1,
        "action_dim": 7,
        "min_std": 0.05,
        "max_std": 2.0,
        "temperature": 0.5,
        "gamma": 0.99,
        "execute_steps": 1,
    }
    section[field] = value
    cfg = OmegaConf.create({"eval": {"tdmpc_mpc": section}})

    with pytest.raises(ValueError, match=rf"eval\.tdmpc_mpc\.{field}"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_cross_field_tdmpc_mpc_geometry() -> None:
    cfg = OmegaConf.create(
        {
            "eval": {
                "tdmpc_mpc": {
                    "enabled": True,
                    "horizon": 2,
                    "iterations": 1,
                    "num_samples": 4,
                    "num_elites": 5,
                    "num_pi_trajs": 5,
                    "action_dim": 7,
                    "min_std": 2.0,
                    "max_std": 1.0,
                    "temperature": 0.5,
                    "gamma": 0.99,
                    "execute_steps": 3,
                }
            }
        }
    )

    with pytest.raises(ValueError, match=r"eval\.tdmpc_mpc\.num_elites"):
        validate_cfg(cfg)


def test_validate_cfg_accepts_valid_algorithm_hyperparameters() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ppo_gamma": 1.0,
                "group_size": 8,
                "ppo_rollouts_per_start": 8,
                "ppo_update_epochs": 1,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "clip_ratio_c": 3.0,
                "clip_log_ratio": 20.0,
                "advantage_eps": 1.0e-6,
                "kl_coef": 0.0,
                "entropy_coef": 0.0,
                "rewards_lower_bound": 0.5,
                "rewards_upper_bound": 4.5,
                "lumos": {
                    "classifier_threshold": 0.5,
                    "chunk_size": 8,
                    "episode_max_steps": 512,
                    "ppo_rollouts_per_start_min": 4,
                    "ppo_rollouts_per_start_max": 8,
                    "update_micro_batch_starts": 1,
                },
            }
        }
    )

    assert validate_cfg(cfg) is cfg
