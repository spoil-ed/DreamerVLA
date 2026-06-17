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
        "OpenVLA_Onetraj_ColdStart_LIBERO",
        "Openvla-oft-SFT-libero-goal-traj1",
        "libero_goal_no_noops",
        "libero_goal",
        "OpenVLA_Onetraj_LIBERO_libero_goal",
    ),
    (
        "OpenVLA_Onetraj_ColdStart_LIBERO_10",
        "Openvla-oft-SFT-libero10-traj1",
        "libero_10_no_noops",
        "libero_10",
        "OpenVLA_Onetraj_LIBERO_libero_10",
    ),
    (
        "OpenVLA_Onetraj_ColdStart_LIBERO_Object",
        "Openvla-oft-SFT-libero-object-traj1",
        "libero_object_no_noops",
        "libero_object",
        "OpenVLA_Onetraj_LIBERO_libero_object",
    ),
    (
        "OpenVLA_Onetraj_ColdStart_LIBERO_Spatial",
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
            overrides=["experiment=collect_rollouts_onetraj", f"task={task}"],
        )
    OmegaConf.resolve(cfg)
    validate_cfg(cfg)

    assert cfg._target_ == "dreamervla.runners.CollectRolloutsRunner"
    assert cfg.task.suite == suite
    oft = cfg.task.openvla_oft

    # VLA ckpt + unnorm key are bound to THIS suite (not goal's)
    assert str(oft.ckpt_path).endswith(ckpt_suffix)
    assert oft.dataset_statistics_key == unnorm_key
    assert artifact in str(oft.hdf5_reward_dir)
    assert str(oft.action_hidden_dir).endswith("_oft_legacy_action_hidden_vla_policy_h1")
    # generated rollouts live under a marked root, never the offline processed_data/
    assert "/collected_rollouts/" in str(oft.hdf5_reward_dir)
    assert "/collected_rollouts/" in str(oft.action_hidden_dir)
    assert "/processed_data/" not in str(oft.hdf5_reward_dir)

    # shared discrete one-traj (h1 / no-state) settings
    assert oft.expected_action_head_type == "oft_discrete_token"
    assert oft.expected_include_state is False
    assert int(oft.expected_history) == 1
    assert int(oft.time_horizon) == 8
    assert int(oft.chunk_size) == 8
    # inherited from the suite base
    assert oft.expected_obs_hidden_source == "action_query"
    assert oft.expected_prompt_style == "vla_policy"
    assert oft.expected_rotate_images_180 is True

    # single-view central default (the cold-start fix): VLA sees 1 agentview image
    assert int(cfg.collect.num_images_in_input) == 1
