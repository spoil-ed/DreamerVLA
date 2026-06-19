from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg


def test_validate_cfg_rejects_unknown_logger_backend() -> None:
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "logger_backends": ["tensorboard", "mlflow"],
                }
            }
        }
    )

    with pytest.raises(ValueError, match="runner.logger.logger_backends"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_unknown_actor_update_route() -> None:
    cfg = OmegaConf.create({"algorithm": {"update_type": "not_a_route"}})

    with pytest.raises(ValueError, match="Unknown actor update route"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_oft_sidecar_mismatch() -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"hidden_dir": "/tmp/wrong-sidecar"},
            "task": {
                "openvla_oft": {
                    "action_hidden_dir": "/tmp/canonical-sidecar",
                }
            },
        }
    )

    with pytest.raises(ValueError, match="dataset.hidden_dir"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_global_batch_not_divisible_by_world_size() -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "global_batch_size": 10,
                "gradient_accumulate_every": 1,
            }
        }
    )

    with pytest.raises(ValueError, match="global_batch_size"):
        validate_cfg(cfg, world_size=4)


def test_validate_cfg_rejects_ray_auto_vram_knobs() -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.online_cotrain_ray_runner.OnlineCotrainRayRunner",
            "training": {"auto_vram_batch": True},
            "collect": {"auto_vram_envs": True},
        }
    )

    with pytest.raises(ValueError, match="auto_vram"):
        validate_cfg(cfg)


def test_validate_cfg_accepts_manual_ray_precision_and_batch_knobs() -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.online_cotrain_ray_runner.OnlineCotrainRayRunner",
            "env": {"num_workers": 2},
            "rollout": {"steps": 4},
            "replay": {"cfg": {"sequence_length": 3}},
            "learner": {
                "train_cfg": {
                    "mode": "synthetic_ppo",
                    "batch_size": 2,
                    "precision": "bf16",
                }
            },
        }
    )

    validate_cfg(cfg)


def test_validate_cfg_rejects_ray_multinode_cluster_request() -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.online_cotrain_ray_runner.OnlineCotrainRayRunner",
            "cluster": {"num_nodes": 2},
        }
    )

    with pytest.raises(ValueError, match="single-node"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_invalid_ray_learner_placement() -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.online_cotrain_ray_runner.OnlineCotrainRayRunner",
            "learner": {
                "num_workers": 2,
                "placement": {
                    "strategy": "packed",
                    "start_gpu": 2,
                    "end_gpu": 1,
                    "num_gpus_per_worker": 1,
                },
            },
        }
    )

    with pytest.raises(ValueError, match="learner.placement"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_unknown_ray_precision() -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.online_cotrain_ray_runner.OnlineCotrainRayRunner",
            "learner": {"train_cfg": {"precision": "auto"}},
        }
    )

    with pytest.raises(ValueError, match="learner.train_cfg.precision"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_unknown_model_type() -> None:
    cfg = OmegaConf.create({"policy": {"model_type": "missing_model"}})

    with pytest.raises(ValueError, match="unknown model_type"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_missing_explicit_resume_path(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "resume": True,
                "resume_path": str(tmp_path / "missing.ckpt"),
            }
        }
    )

    with pytest.raises(ValueError, match="training.resume_path"):
        validate_cfg(cfg)


def test_validate_cfg_can_require_existing_dataset_paths(tmp_path: Path) -> None:
    hdf5_dir = tmp_path / "hdf5"
    hdf5_dir.mkdir()
    cfg = OmegaConf.create(
        {
            "validation": {"require_existing_paths": True},
            "dataset": {
                "hdf5_dir": str(hdf5_dir),
                "hidden_dir": str(tmp_path / "missing-hidden"),
            },
        }
    )

    with pytest.raises(ValueError, match="dataset.hidden_dir"):
        validate_cfg(cfg)


def test_validate_cfg_accepts_mainline_grouped_routes() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    route_names = [
        "world_model_dinowm_chunk",
        "oft_world_model_dinowm_chunk",
        "dreamervla_rynn_dino_wm_wmpo_outcome",
        "dreamervla_oft_dino_wm_wmpo_outcome",
    ]

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfgs = [
            compose(config_name="train", overrides=[f"experiment={name}"])
            for name in route_names
        ]

    for cfg in cfgs:
        validate_cfg(cfg, world_size=1)


def test_tensorboard_wandb_logger_route_composes_and_validates() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=world_model_dinowm_chunk",
                "logger=tensorboard_wandb",
            ],
        )

    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.runner.logger.wandb_mode == "online"
    validate_cfg(cfg)


def test_train_run_validates_config_before_runner_setup(monkeypatch) -> None:
    import dreamervla.train as train

    events: list[str] = []

    class DummyRunner:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg

        def setup(self) -> None:
            assert self.cfg.validated is True
            events.append("setup")

        def execute(self) -> None:
            events.append("execute")

        def teardown(self) -> None:
            events.append("teardown")

    def fake_validate(cfg: Any) -> Any:
        cfg.validated = True
        events.append("validate")
        return cfg

    monkeypatch.setattr(train, "validate_cfg", fake_validate)
    monkeypatch.setattr(train.hydra.utils, "get_class", lambda target: DummyRunner)

    cfg = OmegaConf.create({"_target_": "dummy.Runner", "training": {}})
    train.run(cfg)

    assert events == ["validate", "setup", "execute", "teardown"]
