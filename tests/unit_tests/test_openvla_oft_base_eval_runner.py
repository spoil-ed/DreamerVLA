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


def test_openvla_oft_vla_policy_eval_uses_full_checkpoint_raw_path() -> None:
    class _Extractor:
        def __init__(self) -> None:
            self.calls = []

        def reset(self) -> None:
            return None

        def step(self, obs, task):
            self.calls.append((obs, task))
            return SimpleNamespace(
                action_chunk=[
                    np.zeros((7,), dtype=np.float32),
                    np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                ],
                hidden_state=torch.zeros((256, 4096), dtype=torch.float16),
            )

    extractor = _Extractor()

    class _Policy:
        def __init__(self) -> None:
            self.make_extractor_calls = 0

        def make_extractor(self):
            self.make_extractor_calls += 1
            return extractor

        def __call__(self, _batch):
            raise AssertionError(
                "VLA-policy eval must not re-decode fixed-base hidden tokens"
            )

    policy = _Policy()
    runner = object.__new__(EmbodiedEvalRunner)
    runner.cfg = OmegaConf.create({})
    runner.device = torch.device("cpu")
    runner._vla_policy_eval_policy = policy
    runner._base_oft_extractor = extractor
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

    assert len(extractor.calls) == 1
    assert extractor.calls[0][1] == "open the drawer"
    assert len(actions) == 2
    assert actions[0][-1] == 1.0
    assert actions[1][-1] == -1.0

    slot_extractor = runner._make_parallel_oft_slot_extractor()
    assert slot_extractor is extractor
    assert policy.make_extractor_calls == 1


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


def test_cotrain_eval_observer_loads_checkpoint_models_and_fixed_threshold() -> None:
    runner = object.__new__(EmbodiedEvalRunner)
    runner.device = torch.device("cpu")
    runner.distributed = SimpleNamespace(is_main_process=False)
    built = [torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)]
    runner._build_from_target_cfg = lambda _cfg: built.pop(0)
    loaded: list[tuple[str, dict]] = []
    runner._load_module_state = (
        lambda _module, state, name: loaded.append((name, state))
    )
    cfg = OmegaConf.create(
        {
            "eval": {
                "cotrain_diagnostics": True,
                "cotrain_expected_trajectories": 100,
                "cotrain_encode_batch_size": 4,
            },
            "learner": {
                "model_cfg": {
                    "world_model": {"target": "test.WorldModel"},
                    "classifier": {"target": "test.Classifier"},
                },
                "train_cfg": {"precision": "fp32"},
            },
        }
    )
    policy = torch.nn.Linear(1, 1)

    runner._setup_cotrain_eval_observer(
        cfg=cfg,
        payload={
            "classifier_threshold": 0.43,
            "state_dicts": {
                "world_model": {"wm": torch.ones(1)},
                "classifier": {"cls": torch.ones(1)},
                "world_model_optimizer": {"state": {}},
            },
        },
        policy=policy,
    )

    observer = runner._cotrain_eval_observer
    assert [name for name, _state in loaded] == ["world_model", "classifier"]
    assert observer.accumulator.classifier_threshold == 0.43
    assert observer.accumulator.threshold_source == "checkpoint"
    assert observer.expected_trajectories == 100
    assert observer.policy is policy
    assert all(
        not parameter.requires_grad
        for module in (observer.world_model, observer.classifier)
        for parameter in module.parameters()
    )
