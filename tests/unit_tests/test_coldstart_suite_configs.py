"""Each per-suite cold-start task config must compose and bind ITS OWN traj1
discrete ckpt + suite + unnorm key, with the shared discrete (h1/no-state) and
single-view (num_images=1) settings. One VLA ckpt per LIBERO suite — explicit
``task=`` selection, no silent defaulting.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg

# (task name, ckpt dir suffix, unnorm key, suite, artifact substring)
SUITES = [
    (
        "openvla_onetraj_coldstart_libero",
        "Openvla-oft-SFT-libero-goal-traj1",
        "libero_goal_no_noops",
        "libero_goal",
        "OpenVLA_Onetraj_LIBERO_libero_goal",
    ),
    (
        "openvla_onetraj_coldstart_libero_10",
        "Openvla-oft-SFT-libero10-traj1",
        "libero_10_no_noops",
        "libero_10",
        "OpenVLA_Onetraj_LIBERO_libero_10",
    ),
    (
        "openvla_onetraj_coldstart_libero_object",
        "Openvla-oft-SFT-libero-object-traj1",
        "libero_object_no_noops",
        "libero_object",
        "OpenVLA_Onetraj_LIBERO_libero_object",
    ),
    (
        "openvla_onetraj_coldstart_libero_spatial",
        "Openvla-oft-SFT-libero-spatial-traj1",
        "libero_spatial_no_noops",
        "libero_spatial",
        "OpenVLA_Onetraj_LIBERO_libero_spatial",
    ),
]


@pytest.mark.parametrize("task,ckpt_suffix,unnorm_key,suite,artifact", SUITES)
def test_coldstart_suite_binds_own_ckpt_and_suite(task, ckpt_suffix, unnorm_key, suite, artifact):
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=collect_rollouts", f"task={task}"],
        )
    OmegaConf.resolve(cfg)
    validate_cfg(cfg)

    assert cfg._target_ == "dreamervla.runners.RolloutCollectionRunner"
    assert cfg.task.suite == suite
    oft = cfg.task.openvla_oft

    # VLA ckpt + unnorm key are bound to THIS suite (not goal's)
    assert str(oft.ckpt_path).endswith(ckpt_suffix)
    assert oft.dataset_statistics_key == unnorm_key
    assert artifact in str(oft.hdf5_reward_dir)
    assert ("input_" + "tokens") not in oft
    assert str(oft.hidden_token_dir).endswith(
        "_oft_hidden_token_vla_policy_h1"
    )
    # generated rollouts live under a marked root, never the offline processed_data/
    assert "/collected_rollouts/" in str(oft.hdf5_reward_dir)
    assert "/collected_rollouts/" in str(oft.hidden_token_dir)
    assert "/processed_data/" not in str(oft.hdf5_reward_dir)

    # shared discrete one-traj (h1 / no-state) settings
    hidden_token = oft.hidden_token
    assert hidden_token.expected_action_head_type == "oft_discrete_token"
    assert hidden_token.expected_include_state is False
    assert int(hidden_token.expected_history) == 1
    assert int(oft.time_horizon) == 8
    assert int(hidden_token.chunk_size) == 8
    assert hidden_token.expected_obs_hidden_source == "hidden_token"
    assert hidden_token.expected_prompt_style == "vla_policy"
    assert hidden_token.expected_rotate_images_180 is True
    assert int(hidden_token.num_images_in_input) == 1
    assert int(hidden_token.patches_per_image) == 256
    assert int(hidden_token.token_count) == 256
    assert int(hidden_token.token_dim) == 4096
    assert int(hidden_token.wm_obs_dim) == 1_048_576

    # single-view central default (the cold-start fix): VLA sees 1 agentview image
    assert int(cfg.collect.num_images_in_input) == 1
