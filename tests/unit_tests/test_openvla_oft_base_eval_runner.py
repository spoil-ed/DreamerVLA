from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from dreamervla.runners.embodied_eval_runner import EmbodiedEvalRunner


def test_openvla_oft_base_eval_policy_cfg_uses_task_metadata() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "image_keys": ["agentview_rgb"],
                "openvla_oft": {
                    "num_images_in_input": 1,
                    "dataset_statistics_key": "libero_goal_no_noops",
                    "hidden_token": {
                        "expected_action_head_type": "oft_discrete_token",
                        "expected_include_state": False,
                    },
                },
            }
        }
    )

    policy_cfg = EmbodiedEvalRunner._oft_base_policy_cfg(cfg, "/tmp/oft")

    assert policy_cfg == {
        "model_path": "/tmp/oft",
        "num_images_in_input": 1,
        "policy_mode": "discrete",
        "unnorm_key": "libero_goal_no_noops",
        "expected_action_head_type": "oft_discrete_token",
        "expected_include_state": False,
        "_rank": 0,
    }


def test_openvla_oft_base_eval_generates_postprocessed_actions() -> None:
    runner = object.__new__(EmbodiedEvalRunner)
    runner.cfg = OmegaConf.create({})
    runner._base_oft_extractor = SimpleNamespace(
        reset=lambda: None,
        step=lambda obs, task: SimpleNamespace(
            action_chunk=[
                np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32),
                np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32),
            ],
            obs=obs,
            task=task,
        ),
    )
    runner._libero_current_raw_obs = {
        "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((2, 2, 3), dtype=np.uint8),
    }

    actions = runner._generate_actions(
        backbone=None,
        item_processor=None,
        frame_history=[],
        state=np.arange(8, dtype=np.float32),
        task_description="open the drawer",
        action_steps=2,
    )

    assert len(actions) == 2
    assert actions[0][-1] == 1.0
    assert actions[1][-1] == -1.0
