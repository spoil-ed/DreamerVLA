"""num_images_in_input alignment for the cold-start collector.

OFT checkpoints do NOT persist num_images_in_input; it is a deployment param.
The collector must take it from the central ``collect.*`` config (default = OFT
single-view 1), NOT derive it from ``len(task.image_keys)`` (which counts the
stored camera views, 2 for libero). Deriving from image_keys fed the discrete
one-traj VLA 2 images when it expects 1 -> ~0% rollouts.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from dreamervla.runners.collect_rollouts_runner import CollectRolloutsRunner


def _make_cfg(num_images=None):
    oft = {
        "ckpt_path": "/tmp/ckpt",
        "dataset_statistics_key": "libero_goal_no_noops",
        "expected_history": 1,
        "hdf5_reward_dir": "/tmp/rw",
        "action_hidden_dir": "/tmp/h1",
        "expected_action_head_type": "oft_discrete_token",
        "expected_include_state": False,
        "expected_obs_hidden_source": "action_query",
        "expected_prompt_style": "vla_policy",
        "expected_rotate_images_180": True,
        "time_horizon": 8,
        "token_dim": 4096,
        "chunk_size": 8,
    }
    collect = {
        "policy_mode": "auto",
        "task_ids": "all",
        "episodes_per_task": 2,
        "episode_horizon": 64,
        "envs_per_gpu": 1,
        "gpu_id": 0,
    }
    if num_images is not None:
        collect["num_images_in_input"] = num_images
    return OmegaConf.create(
        {
            "task": {
                "openvla_oft": oft,
                # two stored camera views — must NOT dictate VLA num_images
                "image_keys": ["agentview_rgb", "eye_in_hand_rgb"],
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


def test_num_images_reads_central_value_verbatim():
    # proves the value is READ from the central config (not hardcoded to 1):
    # a 2-view+wrist VLA can set 2, independent of len(image_keys).
    assert _build(_make_cfg(num_images=2))["num_images_in_input"] == 2
