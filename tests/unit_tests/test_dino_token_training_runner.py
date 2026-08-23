from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dreamervla.runners.dino_token_world_model_training_runner import (
    DinoTokenWorldModelTrainingRunner,
)


def _runner_config(tmp_path):
    return OmegaConf.create(
        {
            "seed": 0,
            "training": {
                "out_dir": str(tmp_path),
                "device": "cpu",
                "distributed_strategy": "ddp",
                "fsdp_mixed_precision": "fp32",
                "enable_activation_checkpointing": False,
                "resume": False,
            },
            "checkpoint": {
                "topk": {
                    "monitor_key": "eval/loss",
                    "metric_name": "loss",
                    "mode": "min",
                    "k": 2,
                }
            },
            "optim": {
                "param_precision": "fp32",
                "precision": "fp32",
                "predictor": {
                    "name": "adamw",
                    "lr": 3.0e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1.0e-8,
                    "weight_decay": 0.01,
                },
                "conditioning": {
                    "name": "adamw",
                    "lr": 3.0e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1.0e-8,
                    "weight_decay": 0.01,
                },
            },
            "world_model": {
                "_target_": ("dreamervla.models.embodiment.world_model.DinoTokenWorldModel"),
                "token_count": 2,
                "token_dim": 4,
                "action_dim": 6,
                "proprio_dim": 3,
                "action_emb_dim": 2,
                "proprio_emb_dim": 2,
                "num_hist": 3,
                "num_pred": 1,
                "depth": 1,
                "heads": 2,
                "dim_head": 2,
                "mlp_dim": 8,
                "dropout": 0.0,
                "emb_dropout": 0.0,
            },
        }
    )


def test_dino_runner_matches_dreamer_per_rank_batch_semantics() -> None:
    assert (
        DinoTokenWorldModelTrainingRunner._per_rank_batch_size(
            configured_batch_size=16,
            global_batch_size=None,
            world_size=8,
        )
        == 16
    )
    assert (
        DinoTokenWorldModelTrainingRunner._per_rank_batch_size(
            configured_batch_size=16,
            global_batch_size=32,
            world_size=8,
        )
        == 4
    )


def test_dino_runner_progress_displays_step_loss_and_diagnostic_cosine() -> None:
    status = DinoTokenWorldModelTrainingRunner._progress_status(
        {"loss": 0.25, "one_step_cosine_similarity": 0.9375},
        global_step=123,
    )

    assert status == "global_step=123 loss=0.250000 cos=0.937500"


def test_dino_runner_uses_separate_disjoint_upstream_optimizers(tmp_path) -> None:
    runner = DinoTokenWorldModelTrainingRunner(_runner_config(tmp_path))
    runner._build_model_and_optimizers(runner.cfg)
    model = runner._unwrapped_world_model

    predictor_ids = {
        id(parameter)
        for group in runner.predictor_optimizer.param_groups
        for parameter in group["params"]
    }
    conditioning_ids = {
        id(parameter)
        for group in runner.conditioning_optimizer.param_groups
        for parameter in group["params"]
    }

    assert predictor_ids == {id(parameter) for parameter in model.predictor.parameters()}
    assert conditioning_ids == {
        id(parameter)
        for module in (model.action_encoder, model.proprio_encoder)
        for parameter in module.parameters()
    }
    assert predictor_ids.isdisjoint(conditioning_ids)


def test_dino_runner_requires_fp32_master_parameters(tmp_path) -> None:
    cfg = _runner_config(tmp_path)
    cfg.optim.param_precision = "bf16"
    runner = DinoTokenWorldModelTrainingRunner(cfg)

    with pytest.raises(ValueError, match="param_precision=fp32"):
        runner._build_model_and_optimizers(runner.cfg)


def test_dino_epoch_checkpoint_writes_flat_latest_and_metric_copy(tmp_path) -> None:
    runner = DinoTokenWorldModelTrainingRunner(_runner_config(tmp_path))
    runner.global_step = 12
    runner.epoch = 3
    captured: dict[str, object] = {}
    runner.save_checkpoint = lambda **kwargs: captured.update(kwargs) or ""

    runner._save_epoch_checkpoint(
        {"epoch": 3, "eval/loss": 0.25},
        runner._make_checkpoint_manager(),
    )

    assert captured["path"] == tmp_path / "checkpoints" / "latest.ckpt"
    assert captured["extra_paths"] == (tmp_path / "checkpoints" / "epoch=0003-loss=0.250000.ckpt",)
