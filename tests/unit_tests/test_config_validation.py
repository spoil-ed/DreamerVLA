from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg

_REMOVED_UNDERSCORE_WM_ROUTE = "dino" + "_wm"
_REMOVED_COMPACT_WM_ROUTE = "dino" + "wm"
_REMOVED_DASHED_WM_LABEL = "DINO" + "-WM"


def _contains_removed_wm_wording(text: str) -> bool:
    lower = text.lower()
    return (
        _REMOVED_UNDERSCORE_WM_ROUTE in lower
        or _REMOVED_COMPACT_WM_ROUTE in lower
        or _REMOVED_DASHED_WM_LABEL in text
    )


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
        "world_model_chunk",
    ]

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfgs = [
            compose(config_name="train", overrides=[f"experiment={name}"])
            for name in route_names
        ]

    for cfg in cfgs:
        validate_cfg(cfg, world_size=1)


def test_query_before_world_model_routes_use_compact_transformer_budget() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    rynn = [
        "experiment=world_model_chunk",
        "task=libero_goal",
        "worldmodel=rynnvla_input_token_chunk",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        rynn_cfg = compose(config_name="train", overrides=rynn)

    # Rynn query-before stays on the compact budget.
    validate_cfg(rynn_cfg, world_size=1)
    assert rynn_cfg.world_model.latent_stage == "query_before"
    assert rynn_cfg.world_model.depth == 4
    assert rynn_cfg.world_model.heads == 8
    assert rynn_cfg.world_model.dim_head == 32
    assert rynn_cfg.world_model.mlp_dim == 1024


@pytest.mark.parametrize(
    ("task_name", "suite_time_horizon", "legacy_chunk_size", "legacy_tokens", "legacy_obs_dim"),
    [
        ("libero_goal", 5, 5, 35, 35840),
        ("libero_object", 5, 5, 35, 35840),
        ("libero_spatial", 10, 10, 70, 71680),
        ("libero_10", 10, 10, 70, 71680),
    ],
)
def test_task_latent_specs_are_explicit_model_derived_values(
    task_name: str,
    suite_time_horizon: int,
    legacy_chunk_size: int,
    legacy_tokens: int,
    legacy_obs_dim: int,
) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=[f"task={task_name}"])

    assert cfg.task.action_dim == 7
    assert cfg.task.time_horizon == suite_time_horizon

    legacy = cfg.task.legacy_action_hidden
    assert legacy.latent_stage == "query_after"
    assert legacy.token_dim == 1024
    assert legacy.chunk_size == legacy_chunk_size
    assert legacy.token_count == legacy_tokens
    assert legacy.wm_obs_dim == legacy_obs_dim
    assert legacy.token_count == legacy.chunk_size * cfg.task.action_dim
    assert legacy.wm_obs_dim == legacy.token_count * legacy.token_dim

    input_tokens = cfg.task.legacy_input_tokens
    assert input_tokens.latent_stage == "query_before"
    assert input_tokens.chunk_size == legacy.chunk_size
    assert input_tokens.token_count == 2048
    assert input_tokens.token_dim == 4096
    assert input_tokens.wm_obs_dim == 2048 * 4096

    oft = cfg.task.openvla_oft
    assert oft.latent_stage == "query_after"
    assert oft.chunk_size == oft.time_horizon == 8
    assert oft.token_count == oft.chunk_size * cfg.task.action_dim
    assert oft.wm_obs_dim == oft.token_count * oft.token_dim

    oft_input = cfg.task.openvla_oft.input_tokens
    assert oft_input.latent_stage == "query_before"
    assert oft_input.token_count == 512
    assert oft_input.token_dim == 4096
    assert oft_input.wm_obs_dim == 512 * 4096


def test_validate_cfg_rejects_inconsistent_task_latent_spec() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "action_dim": 7,
                "legacy_action_hidden": {
                    "wm_obs_dim": 35840,
                    "token_count": 70,
                    "token_dim": 1024,
                    "chunk_size": 10,
                },
            }
        }
    )

    with pytest.raises(ValueError, match="task.legacy_action_hidden"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_inconsistent_oft_input_token_patch_count() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "openvla_oft": {
                    "num_images_in_input": 1,
                    "input_tokens": {
                        "wm_obs_dim": 2097152,
                        "token_count": 512,
                        "token_dim": 4096,
                        "patches_per_image": 256,
                    },
                }
            }
        }
    )

    with pytest.raises(ValueError, match="num_images_in_input \\* patches_per_image"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_inconsistent_component_latent_tuple() -> None:
    cfg = OmegaConf.create(
        {
            "world_model": {
                "obs_dim": 35840,
                "token_count": 70,
                "token_dim": 1024,
            }
        }
    )

    with pytest.raises(ValueError, match="world_model"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_world_model_latent_stage_mismatch() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "legacy_action_hidden": {
                    "wm_obs_dim": 35840,
                    "token_count": 35,
                    "token_dim": 1024,
                    "chunk_size": 5,
                    "latent_stage": "query_after",
                },
            },
            "world_model": {
                "obs_dim": 35840,
                "token_count": 35,
                "token_dim": 1024,
                "chunk_size": 5,
                "latent_stage": "query_before",
            },
        }
    )

    with pytest.raises(ValueError, match="latent_stage"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_invalid_chunk_world_model_concat_dim() -> None:
    cfg = OmegaConf.create(
        {
            "world_model": {
                "_target_": "dreamervla.models.world_model.wm_chunk.ChunkAwareWorldModel",
                "obs_dim": 229376,
                "token_count": 56,
                "token_dim": 4096,
                "action_emb_dim": 10,
                "num_action_repeat": 1,
                "model_dim": 4096,
                "depth": 4,
                "heads": 8,
                "dim_head": 32,
                "mlp_dim": 1024,
            }
        }
    )

    with pytest.raises(ValueError, match="model_dim.*action_emb_dim"):
        validate_cfg(cfg)


def test_config_validation_messages_use_role_based_wm_wording() -> None:
    config_source = (
        Path(__file__).resolve().parents[2] / "dreamervla" / "config.py"
    ).read_text(encoding="utf-8")
    assert f"{_REMOVED_DASHED_WM_LABEL} concat conditioning" not in config_source


def test_worldmodel_config_comments_use_role_based_wm_wording() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "worldmodel"
    offenders = {
        path.name: path.read_text(encoding="utf-8")
        for path in config_dir.glob("*.yaml")
        if _contains_removed_wm_wording(path.read_text(encoding="utf-8"))
    }
    assert offenders == {}


def test_dreamervla_config_comments_use_role_based_wm_wording() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "dreamervla"
    offenders = {
        path.name: path.read_text(encoding="utf-8")
        for path in config_dir.glob("*.yaml")
        if _contains_removed_wm_wording(path.read_text(encoding="utf-8"))
    }
    assert offenders == {}


def test_classifier_config_comments_use_role_based_wm_wording() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "classifier"
    offenders = {
        path.name: path.read_text(encoding="utf-8")
        for path in config_dir.glob("*.yaml")
        if _contains_removed_wm_wording(path.read_text(encoding="utf-8"))
    }
    assert offenders == {}


def test_openvla_coldstart_task_comment_uses_role_based_wm_alias() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "task"
        / "openvla_onetraj_coldstart_libero.yaml"
    )
    comment_text = "\n".join(
        line for line in config_path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    )

    assert "experiment=oft_discrete_token_world_model_chunk" in comment_text
    assert (
        f"experiment=oft_discrete_token_world_model_{_REMOVED_COMPACT_WM_ROUTE}_chunk"
        not in comment_text
    )


def test_validate_cfg_rejects_chunk_world_model_sequence_length_mismatch() -> None:
    cfg = OmegaConf.create(
        {
            "world_model": {
                "_target_": "dreamervla.models.world_model.wm_chunk.ChunkAwareWorldModel",
                "obs_dim": 229376,
                "token_count": 56,
                "token_dim": 4096,
                "action_emb_dim": 10,
                "num_action_repeat": 1,
                "model_dim": 4106,
                "depth": 6,
                "heads": 16,
                "dim_head": 256,
                "mlp_dim": 4096,
                "num_hist": 3,
                "chunk_size": 8,
                "chunk_rollout_chunks": 4,
            },
            "dataset": {"sequence_length": 35},
            "online_rollout": {"sequence_length": 36},
        }
    )

    with pytest.raises(ValueError, match="sequence_length.*num_hist.*chunk_rollout_chunks"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_nested_chunk_world_model_sequence_length_mismatch() -> None:
    cfg = OmegaConf.create(
        {
            "ray_components": {
                "world_model": {
                    "target": "dreamervla.models.world_model.wm_chunk.ChunkAwareWorldModel",
                    "kwargs": {
                        "obs_dim": 35840,
                        "token_count": 35,
                        "token_dim": 1024,
                        "action_emb_dim": 10,
                        "num_action_repeat": 1,
                        "model_dim": 1034,
                        "depth": 6,
                        "heads": 16,
                        "dim_head": 64,
                        "mlp_dim": 2048,
                        "num_hist": 3,
                        "chunk_size": 5,
                        "chunk_rollout_chunks": 4,
                    },
                }
            },
            "ray_data": {"sequence_length": 23},
            "replay": {"cfg": {"sequence_length": 24}},
        }
    )

    with pytest.raises(ValueError, match="ray_data.sequence_length.*num_hist"):
        validate_cfg(cfg)


def test_tensorboard_wandb_logger_route_composes_and_validates() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=world_model_chunk",
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


def test_input_token_classifier_label_smoothing_uses_plain_ce_by_default() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "classifier=openvla_oft_input_token_chunk",
                "task=openvla_onetraj_libero",
            ],
        )
    assert float(cfg.training.label_smoothing) == 0.0
