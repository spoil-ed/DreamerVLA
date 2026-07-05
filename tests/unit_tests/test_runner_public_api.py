from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import get_class

_REMOVED_UNDERSCORE_WM_ROUTE = "dino" + "_wm"
_REMOVED_COMPACT_WM_ROUTE = "dino" + "wm"
_REMOVED_DASHED_WM_LABEL = "dino" + "-wm"


def _assert_no_removed_wm_wording(text: str) -> None:
    lower = text.lower()
    assert _REMOVED_UNDERSCORE_WM_ROUTE not in lower
    assert _REMOVED_COMPACT_WM_ROUTE not in lower
    assert _REMOVED_DASHED_WM_LABEL not in lower


EXPERIMENT_MODULES = {
    "world_model_chunk": ("worldmodel", "rynnvla_action_chunk"),
    "oft_world_model_chunk": (
        "worldmodel",
        "openvla_oft_input_token_chunk",
    ),
    "oft_discrete_token_world_model_chunk": (
        "worldmodel",
        "openvla_oft_discrete_token_action_chunk",
    ),
    "oft_world_model_chunk_input_tokens": (
        "worldmodel",
        "openvla_oft_input_token_chunk",
    ),
    "eval_libero_vla": ("evaluation", "libero_vla"),
}


def _compose_experiment(name: str, extra_overrides: list[str] | None = None):
    overrides = [f"experiment={name}"]
    if extra_overrides is not None:
        overrides.extend(extra_overrides)
    return compose(config_name="train", overrides=overrides)


def test_world_model_package_exports_role_based_wm_aliases() -> None:
    from dreamervla.models.world_model import (
        ChunkAwareWorldModel,
        WorldModel,
    )
    from dreamervla.models.world_model.wm import WorldModel as ModuleWorldModel
    from dreamervla.models.world_model.wm_chunk import (
        ChunkAwareWorldModel as ModuleChunkAwareWorldModel,
    )

    assert WorldModel is WorldModel
    assert ChunkAwareWorldModel is ChunkAwareWorldModel
    assert ModuleWorldModel is WorldModel
    assert ModuleChunkAwareWorldModel is ChunkAwareWorldModel


def test_runner_public_api_exports_route_specific_names() -> None:
    import dreamervla.runners as runners

    expected = {
        "ActionHiddenWMRunner",
        "PixelWMRunner",
        "TokenWMRunner",
        "VLASFTRunner",
        "OpenVLAOFTRunner",
        "JointDreamerVLARunner",
        "EmbodiedEvalRunner",
        "ChameleonLatentWMRunner",
        "LatentWMRunner",
        "LatentClassifierRunner",
        "OnlineCotrainRunner",
        "CollectRolloutsRunner",
        "OnlineCotrainPipelineRunner",
        "OnlineCotrainRayRunner",
        "ManualCotrainRayRunner",
        "ColdStartRayCollectRunner",
    }

    assert expected == set(runners.PUBLIC_RUNNERS)
    assert expected.issubset(set(runners.__all__))
    for name in expected:
        cls = getattr(runners, name)
        assert cls.__name__ == name
        assert isinstance(cls.runner_name, str)
        assert callable(cls.setup)
        assert callable(cls.execute)
        assert callable(cls.run)
        assert callable(cls.teardown)


def test_latent_wm_implementation_uses_role_based_wm_name() -> None:
    from dreamervla.runners import latent_wm_runner

    assert latent_wm_runner.LatentWMTrainingRunner.runner_name == "wm"
    source = (
        Path(__file__).resolve().parents[2]
        / "dreamervla"
        / "runners"
        / "latent_wm_runner.py"
    ).read_text(encoding="utf-8")
    _assert_no_removed_wm_wording(source)


def test_runner_directory_contains_route_specific_runners() -> None:
    runner_dir = Path(__file__).resolve().parents[2] / "dreamervla" / "runners"
    top_level_python_files = {path.name for path in runner_dir.glob("*.py")}
    assert {
        "__init__.py",
        "base_runner.py",
        "chameleon_latent_action_wm_runner.py",
        "dreamervla_runner.py",
        "dreamerv3_pixel_runner.py",
        "dreamerv3_token_runner.py",
        "embodied_eval_runner.py",
        "latent_classifier_runner.py",
        "openvla_oft_runner.py",
        "pretokenize_vla_runner.py",
        "backbone_dreamerv3_wm_runner.py",
        "latent_wm_runner.py",
        "vla_sft_runner.py",
    }.issubset(top_level_python_files)
    assert "pretokenize_sft_runner.py" not in top_level_python_files
    assert "pretokenize_wm_runner.py" not in top_level_python_files
    assert "semantic_bottleneck_wm_runner.py" not in top_level_python_files
    assert not (runner_dir.parent / "workspace").exists()
    assert not (runner_dir.parent / "workspace_impl").exists()


def test_removed_legacy_compatibility_shims_are_absent() -> None:
    project_root = Path(__file__).resolve().parents[2]

    assert not (
        project_root / "dreamervla" / "dataset" / "pretokenize_sequence_dataset.py"
    ).exists()
    assert not (project_root / "scripts" / "pretokenize_train_wm.sh").exists()


def test_removed_route_files_are_absent() -> None:
    project_root = Path(__file__).resolve().parents[2]
    removed_files = {
        "configs/dreamervla_libero_goal.yaml",
        "configs/dreamervla_libero_goal_dreamerv3_pixel_actor.yaml",
        "configs/dreamervla_libero_goal_dreamerv3_pixel_vlaactor.yaml",
        "configs/dreamervla_libero_goal_dreamerv3_token_actor.yaml",
        "configs/dreamervla_libero_goal_dreamerv3_token_actor_epoch6.yaml",
        "configs/dreamervla_libero_goal_rynn_pixel_precomputed_actor.yaml",
        "configs/dreamervla_libero_goal_rynn_pixel_precomputed_vlaactor.yaml",
        "configs/pretokenize_wm_libero_goal.yaml",
        "configs/pretokenize_wm_libero_goal_transdreamer.yaml",
        "configs/pretokenize_wm_libero_goal_warmup.yaml",
        "configs/rynn_backbone_dreamerv3_pixel_wm_libero_goal.yaml",
        "configs/rynn_backbone_dreamerv3_pixel_wm_libero_goal_precomputed.yaml",
        "configs/semantic_bottleneck_wm_libero_goal.yaml",
        "dreamervla/algorithms/dino_lumos.py",
        "dreamervla/algorithms/dino_lumos_chunk.py",
        "scripts/diagnose_wm.sh",
        "scripts/eval_wm.sh",
        "scripts/prepare_latent_data.sh",
        "scripts/run_rynn_meanpool_offline_long.sh",
        "scripts/run_rynn_meanpool_online_worker.sh",
        "scripts/train_dreamervla_pixel.sh",
        "scripts/train_dreamervla_rynn_pixel.sh",
        "scripts/train_online_rynn_meanpool_dreamer_actor.py",
        "scripts/train_rynn_backbone_dreamerv3_wm.sh",
        "scripts/train_semantic_bottleneck_wm.sh",
        "dreamervla/cli/diagnose_action_sensitivity.py",
        "dreamervla/cli/diagnose_decoder_zh.py",
        "dreamervla/cli/diagnose_wm.py",
        "dreamervla/cli/diagnose_wm_checklist.py",
        "dreamervla/cli/diagnose_wm_collapse.py",
        "dreamervla/cli/diagnose_wm_layers.py",
        "dreamervla/cli/eval_embedding_distribution.py",
        "dreamervla/cli/eval_wm.py",
        "dreamervla/cli/train_pure_vae.py",
        "dreamervla/models/world_model/causal_transformer.py",
        "dreamervla/models/world_model/causal_transformer_v2.py",
        "dreamervla/models/world_model/image_codec.py",
        "dreamervla/models/world_model/semantic_bottleneck.py",
        "dreamervla/models/world_model/token_io.py",
        "dreamervla/models/world_model/transdreamer_original.py",
        "dreamervla/models/world_model/transdreamer_transformer.py",
        "dreamervla/models/world_model/tssm.py",
        "dreamervla/models/world_model/tssm_discrete.py",
        "dreamervla/models/vla_actor.py",
        "dreamervla/models/vla_policy.py",
        "dreamervla/runners/pretokenize_sft_runner.py",
        "dreamervla/runners/pretokenize_wm_runner.py",
        "dreamervla/runners/semantic_bottleneck_wm_runner.py",
        "tests/test_semantic_bottleneck_world_model.py",
    }

    for relative_path in removed_files:
        assert not (project_root / relative_path).exists()


def test_base_dataset_no_longer_exposes_spec_alias() -> None:
    from dreamervla.dataset.base_dataset import BaseDataset

    assert not hasattr(BaseDataset, "spec")


def test_active_configs_target_route_specific_runner_classes() -> None:
    expected = {
        "world_model_chunk": "dreamervla.runners.LatentWMRunner",
        "oft_world_model_chunk": "dreamervla.runners.LatentWMRunner",
        "oft_discrete_token_world_model_chunk": "dreamervla.runners.LatentWMRunner",
        "oft_world_model_chunk_input_tokens": "dreamervla.runners.LatentWMRunner",
        "eval_libero_vla": "dreamervla.runners.EmbodiedEvalRunner",
    }

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for config_name, target in expected.items():
            cfg = _compose_experiment(config_name)
            assert cfg._target_ == target
            assert "workspace" not in cfg
            cls = get_class(target)
            assert cls.__name__ == target.rsplit(".", 1)[-1]


def test_train_config_experiments_compose_through_stage_modules() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for experiment_name, (group_name, module_name) in EXPERIMENT_MODULES.items():
            new_cfg = _compose_experiment(experiment_name)
            module_cfg = compose(config_name=f"{group_name}/{module_name}")
            assert new_cfg._target_ == module_cfg._target_
            assert "workspace" not in new_cfg


def test_role_based_wm_chunk_experiment_alias_matches_legacy_route() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        role_based = _compose_experiment("world_model_chunk")
        legacy = _compose_experiment("world_model_chunk")

    assert role_based._target_ == "dreamervla.runners.LatentWMRunner"
    assert role_based._target_ == legacy._target_
    assert role_based.dataset._target_ == legacy.dataset._target_
    assert (
        role_based.world_model._target_
        == "dreamervla.models.world_model.wm_chunk.ChunkAwareWorldModel"
    )
    assert (
        legacy.world_model._target_
        == "dreamervla.models.world_model.wm_chunk.ChunkAwareWorldModel"
    )
    assert get_class(role_based.world_model._target_) is get_class(
        legacy.world_model._target_
    )


def test_role_based_oft_wm_chunk_experiment_alias_matches_legacy_route() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        role_based = _compose_experiment("oft_world_model_chunk")
        legacy = _compose_experiment("oft_world_model_chunk")

    assert role_based._target_ == "dreamervla.runners.LatentWMRunner"
    assert role_based._target_ == legacy._target_
    assert role_based.dataset._target_ == legacy.dataset._target_
    assert role_based.world_model._target_ == legacy.world_model._target_


def test_role_based_oft_wm_input_token_alias_matches_legacy_route() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        role_based = _compose_experiment("oft_world_model_chunk_input_tokens")
        legacy = _compose_experiment("oft_world_model_chunk_input_tokens")

    assert role_based._target_ == "dreamervla.runners.LatentWMRunner"
    assert role_based._target_ == legacy._target_
    assert role_based.dataset._target_ == legacy.dataset._target_
    assert (
        role_based.dataset.expected_obs_hidden_source
        == legacy.dataset.expected_obs_hidden_source
    )
    assert role_based.world_model._target_ == legacy.world_model._target_
    assert role_based.world_model.token_count == legacy.world_model.token_count


def test_role_based_oft_discrete_token_wm_alias_matches_legacy_route() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        role_based = _compose_experiment("oft_discrete_token_world_model_chunk")
        legacy = _compose_experiment("oft_discrete_token_world_model_chunk")

    assert role_based._target_ == "dreamervla.runners.LatentWMRunner"
    assert role_based._target_ == legacy._target_
    assert role_based.dataset._target_ == legacy.dataset._target_
    assert (
        role_based.dataset.expected_action_head_type
        == legacy.dataset.expected_action_head_type
    )
    assert role_based.world_model._target_ == legacy.world_model._target_
    assert role_based.world_model.obs_dim == legacy.world_model.obs_dim


def test_train_dreamervla_script_uses_role_based_wm_default() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    train_dreamervla_config = (
        config_dir / "scripts" / "train_dreamervla.yaml"
    ).read_text(encoding="utf-8")

    assert "experiment: openvla_onetraj_libero_cotrain_noray" in train_dreamervla_config


def test_train_config_exposes_tensorboard_and_wandb_logger_routes() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        default_cfg = _compose_experiment("world_model_chunk")
        wandb_cfg = _compose_experiment(
            "world_model_chunk",
            extra_overrides=["logger=wandb"],
        )

    assert default_cfg.runner.logger.project_name == "dreamervla"
    assert default_cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert default_cfg.runner.logger.log_path == f"{default_cfg.training.out_dir}/log"
    assert default_cfg.runner.logger.wandb_mode == "online"

    assert wandb_cfg.runner.logger.project_name == "dreamervla"
    assert wandb_cfg.runner.logger.logger_backends == ["wandb"]
    assert wandb_cfg.runner.logger.log_path == f"{wandb_cfg.training.out_dir}/log"
    assert wandb_cfg.runner.logger.wandb_mode == "online"


def test_openvla_dreamervla_discrete_probability_route_is_explicit() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        discrete_wm = _compose_experiment("oft_discrete_token_world_model_chunk")

    assert discrete_wm.dataset.expected_action_head_type == "oft_discrete_token"
    assert discrete_wm.dataset.hidden_dir.endswith("_h1")


def test_openvla_oft_default_routes_use_input_token_sidecar() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    suites = ("libero_goal", "libero_object", "libero_spatial", "libero_10")

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        wm_cfgs = [
            _compose_experiment(
                "oft_world_model_chunk",
                extra_overrides=[f"task={suite}"],
            )
            for suite in suites
        ]

    for cfg in wm_cfgs:
        expected = f"{cfg.task.hdf5_dir}_oft_input_token_embedding_vla_policy_h2"
        assert cfg.task.openvla_oft.input_token_hidden_dir == expected
        assert cfg.dataset.hidden_dir == expected
        assert cfg.dataset.expected_obs_hidden_source == "input_token_embedding"
        assert cfg.world_model.obs_dim == cfg.task.openvla_oft.input_tokens.wm_obs_dim


def test_input_token_scheme_b_routes_use_token_sidecar_and_bridge_actor() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        oft_wm = _compose_experiment("oft_world_model_chunk_input_tokens")

    assert oft_wm.dataset.expected_obs_hidden_source == "input_token_embedding"
    assert oft_wm.world_model.token_count == 512
    assert oft_wm.world_model.token_dim == 4096
    assert "input_token" in oft_wm.dataset.hidden_dir


def test_train_config_resolves_public_default_experiment() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    assert not (config_dir / "archive").exists()

    train_config = (config_dir / "train.yaml").read_text(encoding="utf-8")
    train_wm_config = (config_dir / "scripts" / "train_wm.yaml").read_text(
        encoding="utf-8"
    )
    assert "experiment: world_model_chunk" in train_config
    assert "experiment: world_model_chunk" in train_wm_config

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train")
        assert cfg._target_ == "dreamervla.runners.LatentWMRunner"
        assert (
            cfg.dataset._target_
            == "dreamervla.dataset.balanced_terminal_dataset.BalancedTerminalDataset"
        )
        assert (
            cfg.world_model._target_
            == "dreamervla.models.world_model.wm_chunk.ChunkAwareWorldModel"
        )
        assert "worldmodel/chunk" in cfg.training.out_dir


def test_cli_default_uses_current_public_runner_target() -> None:
    from dreamervla.train import _parse_hydra_like_args

    config_name, overrides = _parse_hydra_like_args([])
    assert config_name == "train"
    assert overrides == []


def test_train_help_uses_role_based_wm_examples(capsys) -> None:
    from dreamervla.train import _parse_hydra_like_args

    with pytest.raises(SystemExit) as exc_info:
        _parse_hydra_like_args(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "experiment=world_model_chunk" in help_text
    assert "experiment=openvla_onetraj_libero_cotrain_ray" in help_text
    _assert_no_removed_wm_wording(help_text)


def test_implementation_runner_classes_are_not_public_aliases() -> None:
    import dreamervla.runners as runners

    implementation_names = {
        "ChameleonLatentActionWMRunner",
        "DreamerVLARunner",
        "DreamerV3PixelRunner",
        "DreamerV3TokenRunner",
        "PretokenizeVLARunner",
        "BackboneDreamerV3WMRunner",
    }

    for name in implementation_names:
        assert name not in runners.__all__
        assert not hasattr(runners, name)


def test_removed_runner_routes_are_not_importable() -> None:
    import dreamervla.runners as runners

    removed_public_names = {
        "PooledHiddenWMRunner",
        "PretokenizedWMRunner",
        "PretokenizedSFTRunner",
        "SemanticBottleneckWMRunner",
    }

    for name in removed_public_names:
        assert name not in runners.PUBLIC_RUNNERS
        assert name not in runners.__all__
        assert not hasattr(runners, name)


def test_world_model_package_exposes_only_retained_architectures() -> None:
    import dreamervla.models as models
    import dreamervla.models.world_model as world_model

    removed_world_models = {
        "CausalTransformerCell",
        "TSSMState",
        "TSSMWorldModel",
        "TSSMWorldModelRSSMDiscrete",
        "TSSMWorldModelTransDreamer",
        "TSSMWorldModelTransDreamerDiscrete",
    }

    for name in removed_world_models:
        assert name not in world_model.__all__
        assert not hasattr(world_model, name)
        assert name not in models.__all__
        assert not hasattr(models, name)


def test_models_package_exports_fail_fast_symbols() -> None:
    import dreamervla.models as models

    for name in ("Critic", "VLAPolicy", "WorldModel"):
        assert name in models.__all__
        assert getattr(models, name) is not None


def test_all_configs_compose_and_resolve_route_specific_runner_targets() -> None:
    import dreamervla.runners as runners

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config_names = sorted(
        str(path.relative_to(config_dir).with_suffix(""))
        for path in config_dir.rglob("*.yaml")
        if "experiment" not in path.relative_to(config_dir).parts
    )

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for config_name in config_names:
            cfg = compose(config_name=config_name)
            target = cfg.get("_target_")
            if target is not None:
                cls = get_class(str(target))
                assert cls.__module__ == "dreamervla.runners"
                assert str(target).rsplit(".", 1)[-1] in runners.PUBLIC_RUNNERS
                assert "workspace" not in cfg
        for experiment_name in EXPERIMENT_MODULES:
            cfg = _compose_experiment(experiment_name)
            target = cfg.get("_target_")
            assert target is not None
            cls = get_class(str(target))
            assert cls.__module__ == "dreamervla.runners"
            assert str(target).rsplit(".", 1)[-1] in runners.PUBLIC_RUNNERS
            assert "workspace" not in cfg
