"""num_images_in_input alignment for the cold-start collector.

OFT checkpoints do NOT persist num_images_in_input; it is a deployment param.
The collector must take it from the central ``collect.*`` config (default = OFT
single-view 1), NOT derive it from ``len(task.image_keys)`` (which counts the
stored camera views, 2 for libero). Deriving from image_keys fed the discrete
one-traj VLA 2 images when it expects 1 -> ~0% rollouts.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dreamervla.runners.collect_rollouts_runner import CollectRolloutsRunner
from dreamervla.runners.oft_collect_common import select_vla_image_keys


def _make_cfg(num_images=None):
    oft = {
        "ckpt_path": "/tmp/ckpt",
        "dataset_statistics_key": "libero_goal_no_noops",
        "hdf5_reward_dir": "/tmp/rw",
        "input_token_dir": "/tmp/h1",
        "input_tokens": {
            "expected_history": 1,
            "expected_action_head_type": "oft_discrete_token",
            "expected_include_state": False,
            "expected_obs_hidden_source": "input_token_embedding",
            "expected_prompt_style": "vla_policy",
            "expected_rotate_images_180": True,
            "time_horizon": 8,
            "patches_per_image": 256,
            "token_count": 256,
            "token_dim": 4096,
            "wm_obs_dim": 1_048_576,
            "chunk_size": 8,
        },
    }
    collect = {
        "policy_mode": "discrete",
        "task_ids": "all",
        "episodes_per_task": 2,
        "episode_horizon": 64,
        "envs_per_gpu": 1,
        "memory_fraction": 0.8,
        "gpu_id": 0,
    }
    if num_images is not None:
        collect["num_images_in_input"] = num_images
    return OmegaConf.create(
        {
            "task": {
                "openvla_oft": oft,
                "image_keys": ["agentview_rgb"],
                "suite": "libero_goal",
                "action_dim": 7,
                "image_resolution": 256,
            },
            "collect": collect,
        }
    )


def _build(cfg):
    runner = CollectRolloutsRunner.__new__(CollectRolloutsRunner)
    runner.cfg = cfg
    return runner._build_collect_cfg()


def test_num_images_from_central_config_not_image_keys():
    # central config says 1; the 2 image_keys must be ignored for VLA input
    assert _build(_make_cfg(num_images=1))["num_images_in_input"] == 1


def test_num_images_defaults_to_one_when_unset():
    # no central value -> OFT single-view default 1 (not len(image_keys)=2)
    assert _build(_make_cfg(num_images=None))["num_images_in_input"] == 1


def test_num_images_rejects_non_mainline_central_value():
    with pytest.raises(ValueError, match="num_images_in_input=1"):
        _build(_make_cfg(num_images=2))


def test_single_image_policy_uses_single_rollout_camera():
    cc = _build(_make_cfg(num_images=1))

    assert cc["image_keys"] == ["agentview_rgb"]


def test_two_image_policy_is_closed():
    with pytest.raises(ValueError, match="num_images_in_input=1"):
        _build(_make_cfg(num_images=2))


def test_image_selection_rejects_non_mainline_history():
    with pytest.raises(ValueError, match="expected_history=1"):
        select_vla_image_keys(["agentview_rgb"], history=2, num_images_in_input=1)


def test_image_selection_rejects_unavailable_views():
    with pytest.raises(ValueError, match="num_images_in_input=1"):
        select_vla_image_keys(["agentview_rgb"], history=1, num_images_in_input=2)


def test_image_selection_rejects_extra_stored_view() -> None:
    with pytest.raises(ValueError, match="exactly one primary camera"):
        select_vla_image_keys(
            ["agentview_rgb", "eye_in_hand_rgb"],
            history=1,
            num_images_in_input=1,
        )
