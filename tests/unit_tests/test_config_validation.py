from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg
from dreamervla.config_resolvers import register_dreamervla_resolvers

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


def _compose_mainline(*overrides: str):
    register_dreamervla_resolvers()
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    base = [
        "experiment=openvla_onetraj_libero_cotrain",
        "task=openvla_onetraj_libero",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="train", overrides=[*base, *overrides])


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
    cfg = _compose_mainline()
    OmegaConf.update(cfg, "dataset.hidden_dir", "/tmp/wrong-sidecar", force_add=True)

    with pytest.raises(ValueError, match="dataset.hidden_dir"):
        validate_cfg(cfg)


@pytest.mark.parametrize("policy_mode", ["auto", "l1"])
def test_validate_cfg_rejects_non_discrete_mainline_policy_mode(policy_mode: str) -> None:
    cfg = _compose_mainline()
    OmegaConf.update(cfg, "collect.policy_mode", policy_mode, force_add=True)

    with pytest.raises(ValueError, match="collect.policy_mode must be 'discrete'"):
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


def test_dino_token_wm_requires_shifted_one_step_sequence_length() -> None:
    cfg = OmegaConf.create(
        {
            "world_model": {
                "_target_": ("dreamervla.models.embodiment.world_model.DinoTokenWorldModel"),
                "num_hist": 3,
                "num_pred": 1,
            },
            "online_rollout": {"sequence_length": 5},
        }
    )

    with pytest.raises(ValueError, match=r"num_hist \+.*num_pred"):
        validate_cfg(cfg)


def test_dino_token_wm_accepts_four_frame_training_window() -> None:
    cfg = OmegaConf.create(
        {
            "world_model": {
                "_target_": ("dreamervla.models.embodiment.world_model.DinoTokenWorldModel"),
                "num_hist": 3,
                "num_pred": 1,
            },
            "online_rollout": {"sequence_length": 4},
        }
    )

    validate_cfg(cfg)


def test_dino_token_runner_rejects_dataset_model_frameskip_mismatch() -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.DinoTokenWorldModelTrainingRunner",
            "training": {"global_batch_size": 32},
            "optim": {"param_precision": "fp32", "precision": "fp32"},
            "world_model": {
                "_target_": ("dreamervla.models.embodiment.world_model.DinoTokenWorldModel"),
                "action_dim": 35,
                "num_hist": 3,
                "num_pred": 1,
            },
            "task": {"action_dim": 7},
            "dino_wm": {"frameskip": 5},
            "dataset": {
                "train": {"num_hist": 3, "num_pred": 1, "frameskip": 1},
                "valid": {"num_hist": 3, "num_pred": 1, "frameskip": 5},
            },
        }
    )

    with pytest.raises(ValueError, match="dataset.train.frameskip"):
        validate_cfg(cfg)


def test_dino_token_runner_rejects_non_fp32_training() -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.DinoTokenWorldModelTrainingRunner",
            "training": {"global_batch_size": 32},
            "optim": {"precision": "bf16"},
            "world_model": {
                "_target_": ("dreamervla.models.embodiment.world_model.DinoTokenWorldModel"),
                "action_dim": 35,
                "num_hist": 3,
                "num_pred": 1,
            },
            "task": {"action_dim": 7},
            "dino_wm": {"frameskip": 5},
            "dataset": {
                "train": {"num_hist": 3, "num_pred": 1, "frameskip": 5},
                "valid": {"num_hist": 3, "num_pred": 1, "frameskip": 5},
            },
        }
    )

    with pytest.raises(ValueError, match="optim.precision=fp32"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_ray_auto_vram_knobs() -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.CotrainRunner",
            "training": {"auto_vram_batch": True},
            "collect": {"auto_vram_envs": True},
        }
    )

    with pytest.raises(ValueError, match="auto_vram"):
        validate_cfg(cfg)


def test_validate_cfg_accepts_manual_ray_precision_and_batch_knobs() -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.CotrainRunner",
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
            "_target_": "dreamervla.runners.CotrainRunner",
            "cluster": {"num_nodes": 2},
        }
    )

    with pytest.raises(ValueError, match="single-node"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_invalid_ray_learner_placement() -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.CotrainRunner",
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
            "_target_": "dreamervla.runners.CotrainRunner",
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


@pytest.mark.parametrize(
    "task_name",
    ["libero_goal", "libero_object", "libero_spatial", "libero_10"],
)
def test_task_latent_specs_are_canonical_hidden_token(task_name: str) -> None:
    register_dreamervla_resolvers()
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=[f"+task={task_name}"])

    assert cfg.task.action_dim == 7
    assert cfg.task.time_horizon == 8
    assert "hidden_token_dir" not in cfg.task
    assert "hidden_token_tokens" not in cfg.task
    assert "hidden_token_specs" not in cfg.task

    oft_hidden = cfg.task.openvla_oft.hidden_token
    assert oft_hidden.latent_stage == "query_before"
    assert oft_hidden.expected_obs_hidden_source == "hidden_token"
    assert oft_hidden.token_count == 256
    assert oft_hidden.token_dim == 4096
    assert oft_hidden.wm_obs_dim == 256 * 4096


def test_validate_cfg_rejects_removed_task_latent_spec() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "action_dim": 7,
                "hidden_token_tokens": {
                    "wm_obs_dim": 35840,
                    "token_count": 70,
                    "token_dim": 1024,
                    "chunk_size": 10,
                },
            }
        }
    )

    with pytest.raises(ValueError, match="removed action-query/hidden-token"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    "field",
    [
        "action_head_ckpt",
        "proprio_projector_ckpt",
        "component_ckpt_dir",
        "resume_step",
    ],
)
def test_validate_cfg_rejects_removed_oft_component_fields_even_when_null(
    field: str,
) -> None:
    cfg = OmegaConf.create({"task": {"openvla_oft": {field: None}}})

    with pytest.raises(ValueError, match="removed action-query/hidden-token"):
        validate_cfg(cfg)


def test_validate_cfg_accepts_dynamic_oft_hidden_geometry_from_task_metadata() -> None:
    composed = _compose_mainline()
    task = OmegaConf.create(
        {"task": {"openvla_oft": OmegaConf.to_container(composed.task.openvla_oft, resolve=True)}}
    )
    hidden = task.task.openvla_oft.hidden_token
    hidden.patches_per_image = 128
    hidden.token_count = 128
    hidden.token_dim = 1024
    hidden.wm_obs_dim = 128 * 1024

    validate_cfg(task)


def test_validate_cfg_rejects_inconsistent_oft_hidden_geometry() -> None:
    composed = _compose_mainline()
    cfg = OmegaConf.create(
        {"task": {"openvla_oft": OmegaConf.to_container(composed.task.openvla_oft, resolve=True)}}
    )
    cfg.task.openvla_oft.hidden_token.patches_per_image = 128

    with pytest.raises(ValueError, match="token_count must equal"):
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


@pytest.mark.parametrize(
    "component",
    [
        {"classifier": {"latent_dim": 56 * 1024}},
        {
            "world_model": {
                "obs_dim": 56 * 1024,
                "token_count": 14,
                "token_dim": 4096,
            }
        },
        {"policy": {"source_token_count": 56, "source_token_dim": 1024}},
        {
            "actor": {
                "policy_cfg": {
                    "kwargs": {
                        "source_token_count": 56,
                        "source_token_dim": 4096,
                    }
                }
            }
        },
    ],
)
def test_validate_cfg_unconditionally_rejects_removed_56_token_interfaces(
    component: dict[str, Any],
) -> None:
    cfg = OmegaConf.create(component)

    with pytest.raises(ValueError, match="removed 56"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("policy.cfg.kwargs.source_token_count", 56),
        ("env.wm.cfg.kwargs.latent_dim", 56 * 1024),
        ("collect.oft_latent_spec.token_count", 56),
        ("collect.oft_latent_spec.wm_obs_dim", 56 * 1024),
        ("rollout.dump.preprocess_config.token_count", 56),
        ("rollout.dump.preprocess_config.hidden_dim", 56 * 1024),
        ("rollout.dump.preprocess_config.flat_dim", 56 * 1024),
        ("rollout.dump.preprocess_config.obs_embedding_shape", [56, 1024]),
    ],
)
def test_validate_cfg_rejects_removed_geometry_at_any_nested_path(
    path: str,
    value: Any,
) -> None:
    cfg = _compose_mainline(
        "experiment=openvla_onetraj_libero_cotrain",
    )
    OmegaConf.update(cfg, path, value, force_add=True)

    with pytest.raises(ValueError, match="removed 56"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_world_model_latent_stage_mismatch() -> None:
    cfg = _compose_mainline()
    cfg.world_model.latent_stage = "query_after"

    with pytest.raises(ValueError, match="latent_stage"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_wm_env_token_grid_mismatch() -> None:
    cfg = _compose_mainline(
        "experiment=openvla_onetraj_libero_cotrain",
    )
    cfg.env.wm.cfg.kwargs.token_count = 128

    with pytest.raises(ValueError, match=r"env\.wm\.cfg\.kwargs\.token_count"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_invalid_chunk_world_model_concat_dim() -> None:
    cfg = OmegaConf.create(
        {
            "world_model": {
                "_target_": "dreamervla.models.embodiment.world_model.wm_chunk.ChunkAwareWorldModel",
                "obs_dim": 1_048_576,
                "token_count": 256,
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
    config_source = (Path(__file__).resolve().parents[2] / "dreamervla" / "config.py").read_text(
        encoding="utf-8"
    )
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


def test_openvla_coldstart_task_comment_documents_hidden_token_source() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "task"
        / "openvla_onetraj_coldstart_libero.yaml"
    )
    comment_text = "\n".join(
        line
        for line in config_path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    )

    assert "obs_hidden_source=hidden_token" in comment_text
    legacy_source = "input_" + "token_" + "embedding"
    assert f"obs_hidden_source={legacy_source}" not in comment_text


def test_validate_cfg_rejects_chunk_world_model_sequence_length_mismatch() -> None:
    cfg = OmegaConf.create(
        {
            "world_model": {
                "_target_": "dreamervla.models.embodiment.world_model.wm_chunk.ChunkAwareWorldModel",
                "obs_dim": 1_048_576,
                "token_count": 256,
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
                    "target": "dreamervla.models.embodiment.world_model.wm_chunk.ChunkAwareWorldModel",
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
                "experiment=collect_rollouts",
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
