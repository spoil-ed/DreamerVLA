from __future__ import annotations

import torch
from omegaconf import OmegaConf

import dreamervla.runtime.world_model_training_base as training_base
from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner
from dreamervla.runtime.world_model_training_common import (
    _component_hydra_cfg,
    _WorldModelTrainingCommon,
)
from dreamervla.utils.torch_utils import precision_dtype


class _FakeDistributed:
    rank = 0
    local_rank = 0
    world_size = 1
    is_main_process = True

    def __init__(self) -> None:
        self.configured_device: str | None = None

    def resolve_device(self, configured_device: str) -> torch.device:
        self.configured_device = configured_device
        return torch.device(configured_device)


def test_world_model_training_base_reads_training_device(monkeypatch, tmp_path) -> None:
    distributed = _FakeDistributed()

    class _FakeDistributedFactory:
        @staticmethod
        def initialize(**_kwargs):
            return distributed

    monkeypatch.setattr(
        training_base,
        "NopretokenizeSFTDistributedHelper",
        _FakeDistributedFactory,
    )
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "training": {
                "device": "cpu",
                "distributed_strategy": "ddp",
                "out_dir": str(tmp_path),
            },
        }
    )

    runner = training_base.WorldModelTrainingBase(cfg)

    assert runner.device == torch.device("cpu")
    assert distributed.configured_device == "cpu"


def test_component_hydra_cfg_merges_worker_kwargs_and_local_overrides() -> None:
    cfg = OmegaConf.create(
        {
            "ray_components": {
                "world_model": {
                    "target": "torch.nn.Linear",
                    "kwargs": {"in_features": 3, "out_features": 5, "bias": True},
                }
            },
            "world_model": {"bias": False},
        }
    )

    resolved = _component_hydra_cfg(
        cfg,
        component_path="world_model",
        worker_component_path="ray_components.world_model",
    )

    assert OmegaConf.to_container(resolved, resolve=True) == {
        "_target_": "torch.nn.Linear",
        "in_features": 3,
        "out_features": 5,
        "bias": False,
    }


def test_replay_epoch_budget_uses_effective_ddp_batch_size() -> None:
    global_batch_size = WorldModelTrainingRunner._global_batch_size(
        per_rank_batch_size=16,
        world_size=8,
    )

    assert global_batch_size == 128


def test_reproduction_global_batch_resolves_to_per_rank_batch() -> None:
    assert (
        WorldModelTrainingRunner._per_rank_batch_size(
            configured_batch_size=32,
            global_batch_size=32,
            world_size=8,
            gradient_accumulate_every=1,
        )
        == 4
    )
    assert (
        WorldModelTrainingRunner._per_rank_batch_size(
            configured_batch_size=16,
            global_batch_size=None,
            world_size=8,
            gradient_accumulate_every=1,
        )
        == 16
    )


def test_precision_dtype_maps_hydra_precision() -> None:
    assert precision_dtype("fp32") is torch.float32
    assert precision_dtype("bf16") is torch.bfloat16
    assert precision_dtype("fp16") is torch.float16


def test_wm_only_build_skips_policy_critic_and_classifier() -> None:
    class _LocalDistributed:
        @staticmethod
        def wrap_trainable_module(module, **_kwargs):
            return module

    cfg = OmegaConf.create(
        {
            "training": {"classifier_warmup_steps": 0},
            "online_rollout": {"total_env_steps": 0},
            "world_model": {
                "_target_": "torch.nn.Linear",
                "in_features": 3,
                "out_features": 5,
            },
            "optim": {
                "precision": "fp32",
                "param_precision": "fp32",
                "world_model": {
                    "name": "adamw",
                    "lr": 1.0e-4,
                    "weight_decay": 0.0,
                },
            },
            "init": {"world_model_state_ckpt": None},
        }
    )
    runner = object.__new__(_WorldModelTrainingCommon)
    runner.device = torch.device("cpu")
    runner.distributed = _LocalDistributed()

    runner._build_components(cfg)

    assert isinstance(runner.world_model, torch.nn.Linear)
    assert runner.world_model.weight.dtype is torch.float32
    assert runner.world_model_optimizer is not None
    assert runner.policy is None
    assert runner.critic is None
    assert runner.classifier is None


def test_wm_only_build_keeps_fp32_parameters_for_bf16_compute() -> None:
    class _LocalDistributed:
        @staticmethod
        def wrap_trainable_module(module, **_kwargs):
            return module

    cfg = OmegaConf.create(
        {
            "training": {"classifier_warmup_steps": 0},
            "online_rollout": {"total_env_steps": 0},
            "world_model": {
                "_target_": "torch.nn.Linear",
                "in_features": 3,
                "out_features": 5,
            },
            "optim": {
                "precision": "bf16",
                "param_precision": "fp32",
                "world_model": {
                    "name": "adamw",
                    "lr": 1.0e-4,
                    "weight_decay": 0.0,
                },
            },
            "init": {"world_model_state_ckpt": None},
        }
    )
    runner = object.__new__(_WorldModelTrainingCommon)
    runner.device = torch.device("cpu")
    runner.distributed = _LocalDistributed()

    runner._build_components(cfg)

    assert runner.world_model.weight.dtype is torch.float32
    loss = runner.world_model.weight.square().mean()
    loss.backward()
    runner.world_model_optimizer.step()
    state = runner.world_model_optimizer.state[runner.world_model.weight]
    assert state["exp_avg"].dtype is torch.float32
    assert state["exp_avg_sq"].dtype is torch.float32


def test_wm_only_build_allows_bf16_parameters_with_fp32_compute() -> None:
    class _LocalDistributed:
        @staticmethod
        def wrap_trainable_module(module, **_kwargs):
            return module

    cfg = OmegaConf.create(
        {
            "training": {"classifier_warmup_steps": 0},
            "online_rollout": {"total_env_steps": 0},
            "world_model": {
                "_target_": "torch.nn.Linear",
                "in_features": 3,
                "out_features": 5,
            },
            "optim": {
                "precision": "fp32",
                "param_precision": "bf16",
                "world_model": {
                    "name": "adamw",
                    "lr": 1.0e-4,
                    "weight_decay": 0.0,
                },
            },
            "init": {"world_model_state_ckpt": None},
        }
    )
    runner = object.__new__(_WorldModelTrainingCommon)
    runner.device = torch.device("cpu")
    runner.distributed = _LocalDistributed()

    runner._build_components(cfg)

    assert runner.world_model.weight.dtype is torch.bfloat16
