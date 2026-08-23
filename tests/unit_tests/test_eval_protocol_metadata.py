from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


def test_eval_protocol_metadata_records_comparator_fields() -> None:
    from dreamervla.runners.libero_vla_evaluation_runner import evaluation_protocol_metadata

    cfg = OmegaConf.create(
        {
            "seed": 1,
            "eval": {
                "task_suite_name": "libero_goal",
                "task_ids": [0, 2],
                "task_start": 0,
                "max_tasks": 2,
                "num_episodes_per_task": 10,
                "num_envs": 4,
                "seed": 7,
                "num_steps_wait": 10,
                "action_steps": 8,
                "max_steps": 300,
                "enumerate_all_init_states": False,
                "scheme": "rlinf_chunk",
                "reconfigure_per_episode": True,
                "history_length": 1,
                "action_postprocess": "none",
                "render_backend": "osmesa",
            },
        }
    )

    metadata = evaluation_protocol_metadata(cfg)

    assert metadata == {
        "task_suite": "libero_goal",
        "num_episodes_per_task": 10,
        "num_envs": 4,
        "seed": 7,
        "num_steps_wait": 10,
        "action_steps": 8,
        "task_ids": [0, 2],
        "task_start": 0,
        "max_tasks": 2,
        "max_steps": 300,
        "enumerate_all_init_states": False,
        "scheme": "rlinf_chunk",
        "reconfigure_per_episode": True,
        "history_length": 1,
        "action_postprocess": "none",
        "render_backend": "osmesa",
    }


def test_eval_strict_component_load_rejects_partial_state() -> None:
    from dreamervla.runners.libero_vla_evaluation_runner import LIBEROVLAEvaluationRunner

    runner = LIBEROVLAEvaluationRunner.__new__(LIBEROVLAEvaluationRunner)
    runner.cfg = OmegaConf.create({"eval": {"require_strict_component_load": True}})
    runner.distributed = SimpleNamespace(is_main_process=False)
    model = torch.nn.Linear(2, 1)

    with pytest.raises(RuntimeError, match="strict.*policy"):
        runner._load_module_state(model, {"weight": model.weight.detach().clone()}, "policy")
