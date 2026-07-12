from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
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


def test_openvla_oft_vla_policy_eval_uses_hidden_tokens_and_actor_actions() -> None:
    class _Actor:
        def __init__(self) -> None:
            self.last_batch = None

        def __call__(self, batch):
            self.last_batch = batch
            action_chunk = torch.zeros((1, 2, 7), dtype=torch.float32)
            action_chunk[0, 1, -1] = 1.0
            return action_chunk, torch.zeros((1, 2, 7)), {}

    actor = _Actor()
    runner = object.__new__(EmbodiedEvalRunner)
    runner.cfg = OmegaConf.create({})
    runner.device = torch.device("cpu")
    runner._vla_policy_eval_policy = actor
    runner._base_oft_extractor = SimpleNamespace(
        reset=lambda: None,
        step=lambda obs, task: SimpleNamespace(
            action_chunk=[],
            hidden_state=torch.zeros((256, 4096), dtype=torch.float16),
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

    assert actor.last_batch["deterministic"] is True
    assert actor.last_batch["return_chunk"] is True
    assert tuple(actor.last_batch["hidden"].shape) == (1, 256, 4096)
    assert len(actions) == 2
    assert actions[0][-1] == 1.0
    assert actions[1][-1] == -1.0


def test_vla_policy_checkpoint_kind_dispatches_without_world_model(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "policy.ckpt"
    checkpoint.touch()
    runner = EmbodiedEvalRunner(
        OmegaConf.create(
                {
                    "seed": 7,
                    "trainer": {"device": "cpu"},
                    "training": {"out_dir": str(tmp_path / "eval")},
                "eval": {
                    "ckpt_path": str(checkpoint),
                    "ckpt_kind": "vla_policy",
                },
            }
        )
    )
    payload = {
        "state_dicts": {"policy": {"weight": torch.ones(1)}},
        "cfg": {},
    }
    runner._load_checkpoint_payload = lambda _path: payload
    called = []
    runner._run_vla_policy_eval = lambda cfg, path, item: called.append(
        (cfg, path, item)
    ) or [{"eval_success_rate": 0.5}]
    monkeypatch.setattr(
        "dreamervla.runners.embodied_eval_runner.is_hf_checkpoint",
        lambda _path: False,
    )

    metrics = runner.run()

    assert metrics == [{"eval_success_rate": 0.5}]
    assert called[0][1] == str(checkpoint.resolve())
    assert called[0][2] is payload
