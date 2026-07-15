from __future__ import annotations

import inspect
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf

_REMOVED_UNDERSCORE_WM_ROUTE = "dino" + "_wm"
_REMOVED_COMPACT_WM_ROUTE = "dino" + "wm"
_REMOVED_DASHED_WM_LABEL = "dino" + "-wm"


def _assert_no_removed_wm_wording(text: str) -> None:
    lower = text.lower()
    assert _REMOVED_UNDERSCORE_WM_ROUTE not in lower
    assert _REMOVED_COMPACT_WM_ROUTE not in lower
    assert _REMOVED_DASHED_WM_LABEL not in lower


EXPERIMENT_MODULES = {
    "eval_libero_vla": ("evaluation", "libero_vla"),
}


def _compose_experiment(name: str, extra_overrides: list[str] | None = None):
    overrides = [f"experiment={name}"]
    if extra_overrides is not None:
        overrides.extend(extra_overrides)
    return compose(config_name="train", overrides=overrides)


def test_active_experiments_use_named_timestamped_run_roots() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    experiment_dir = config_dir / "experiment"
    experiments = sorted(path.stem for path in experiment_dir.glob("*.yaml"))

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for experiment in experiments:
            cfg = _compose_experiment(experiment)
            OmegaConf.resolve(cfg)
            out_dir = Path(str(cfg.training.out_dir))
            assert str(cfg.run.name) == experiment
            assert out_dir.parent.name == experiment
            assert out_dir.name == str(cfg.run.timestamp)


def test_world_model_package_exports_role_based_wm_aliases() -> None:
    from dreamervla.models.embodiment.world_model import (
        ChunkAwareWorldModel,
        WorldModel,
    )
    from dreamervla.models.embodiment.world_model.wm import WorldModel as ModuleWorldModel
    from dreamervla.models.embodiment.world_model.wm_chunk import (
        ChunkAwareWorldModel as ModuleChunkAwareWorldModel,
    )

    assert WorldModel is WorldModel
    assert ChunkAwareWorldModel is ChunkAwareWorldModel
    assert ModuleWorldModel is WorldModel
    assert ModuleChunkAwareWorldModel is ChunkAwareWorldModel


def test_runner_public_api_exports_only_canonical_mainline_roles() -> None:
    import dreamervla.runners as runners

    expected = [
        "DinoTokenWorldModelTrainingRunner",
        "RolloutCollectionRunner",
        "WorldModelTrainingRunner",
        "SuccessClassifierTrainingRunner",
        "CotrainRunner",
        "LIBEROVLAEvaluationRunner",
    ]

    assert runners.PUBLIC_RUNNERS == expected
    assert set(expected).issubset(set(runners.__all__))
    for name in expected:
        cls = getattr(runners, name)
        assert cls.__name__ == name
        assert isinstance(cls.runner_name, str)
        assert cls.runner_status == "current"
        assert callable(cls.setup)
        assert callable(cls.execute)
        assert callable(cls.run)
        assert callable(cls.teardown)

    for removed in (
        "JointDreamerVLARunner",
        "EmbodiedEvalRunner",
        "LatentClassifierRunner",
        "OnlineCotrainRunner",
        "CollectRolloutsRunner",
        "OnlineCotrainPipelineRunner",
        "OnlineCotrainRayRunner",
        "ManualCotrainRayRunner",
        "ColdStartRayCollectRunner",
        "FrozenModelPolicyRunner",
    ):
        assert not hasattr(runners, removed)


def test_runner_directory_contains_only_canonical_route_runners() -> None:
    runner_dir = Path(__file__).resolve().parents[2] / "dreamervla" / "runners"
    top_level_python_files = {path.name for path in runner_dir.glob("*.py")}
    assert {
        "__init__.py",
        "base_runner.py",
        "rollout_collection_runner.py",
        "world_model_training_runner.py",
        "success_classifier_training_runner.py",
        "cotrain_runner.py",
        "libero_vla_evaluation_runner.py",
    }.issubset(top_level_python_files)
    assert {
        "cold_start_ray_collect_runner.py",
        "collect_rollouts_runner.py",
        "embodied_eval_runner.py",
        "frozen_model_policy_runner.py",
        "latent_classifier_runner.py",
        "manual_cotrain_ray_runner.py",
        "online_cotrain_pipeline_runner.py",
        "online_cotrain_ray_runner.py",
        "online_cotrain_runner.py",
    }.isdisjoint(top_level_python_files)
    assert not (runner_dir.parent / "workspace").exists()
    assert not (runner_dir.parent / "workspace_impl").exists()


def test_canonical_runners_do_not_wrap_legacy_runner_classes() -> None:
    runner_dir = Path(__file__).resolve().parents[2] / "dreamervla" / "runners"
    for filename in (
        "rollout_collection_runner.py",
        "world_model_training_runner.py",
        "success_classifier_training_runner.py",
        "cotrain_runner.py",
        "libero_vla_evaluation_runner.py",
    ):
        text = (runner_dir / filename).read_text(encoding="utf-8")
        assert "ManualCotrainRayRunner" not in text
        assert "OnlineCotrainPipelineRunner" not in text
        assert "OnlineCotrainRunner" not in text
        assert "ColdStartRayCollectRunner" not in text
        assert "CollectRolloutsRunner" not in text
        assert "LatentClassifierRunner" not in text
        assert "EmbodiedEvalRunner" not in text


def test_removed_compatibility_shims_are_absent() -> None:
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
        "configs/experiment/latent_classifier_openvla_onetraj_libero_goal_h1.yaml",
        "configs/parallelism/fsdp.yaml",
        "configs/parallelism/none.yaml",
        "configs/precision/bf16.yaml",
        "configs/precision/fp16.yaml",
        "configs/precision/fp32.yaml",
        "configs/scheduler/local.yaml",
        "configs/scheduler/ray_auto.yaml",
        "configs/scripts/train_dreamervla.yaml",
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
        "dreamervla/models/embodiment/world_model/causal_transformer.py",
        "dreamervla/models/embodiment/world_model/causal_transformer_v2.py",
        "dreamervla/models/embodiment/world_model/image_codec.py",
        "dreamervla/models/embodiment/world_model/semantic_bottleneck.py",
        "dreamervla/models/embodiment/world_model/token_io.py",
        "dreamervla/models/embodiment/world_model/transdreamer_original.py",
        "dreamervla/models/embodiment/world_model/transdreamer_transformer.py",
        "dreamervla/models/embodiment/world_model/tssm.py",
        "dreamervla/models/embodiment/world_model/tssm_discrete.py",
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
        "eval_libero_vla": "dreamervla.runners.LIBEROVLAEvaluationRunner",
        "wm_full_dataset_train": "dreamervla.runners.WorldModelTrainingRunner",
        "wmpo_token_classifier_openvla_onetraj_libero_goal_h1": (
            "dreamervla.runners.SuccessClassifierTrainingRunner"
        ),
        "openvla_libero": "dreamervla.runners.CotrainRunner",
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


def test_train_config_exposes_tensorboard_and_wandb_logger_routes() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        default_cfg = _compose_experiment("collect_rollouts")
        wandb_cfg = _compose_experiment(
            "collect_rollouts",
            extra_overrides=["logger=wandb"],
        )

    assert default_cfg.runner.logger.project_name == "dreamervla"
    assert default_cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert default_cfg.runner.logger.log_path == default_cfg.training.out_dir
    assert default_cfg.runner.logger.wandb_mode == "online"

    assert wandb_cfg.runner.logger.project_name == "dreamervla"
    assert wandb_cfg.runner.logger.logger_backends == ["wandb"]
    assert wandb_cfg.runner.logger.log_path == wandb_cfg.training.out_dir
    assert wandb_cfg.runner.logger.wandb_mode == "online"


def test_train_config_resolves_public_default_experiment() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    train_config = (config_dir / "train.yaml").read_text(encoding="utf-8")
    assert "experiment: openvla_onetraj_libero_cotrain" in train_config

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train")
        assert cfg._target_ == "dreamervla.runners.CotrainRunner"
    assert cfg.run.name == "openvla_onetraj_libero_cotrain"
    assert Path(cfg.training.out_dir).parent.name == cfg.run.name


def test_train_main_is_native_hydra_entrypoint() -> None:
    from dreamervla.train import main

    assert hasattr(main, "__wrapped__")
    assert list(inspect.signature(main.__wrapped__).parameters) == ["cfg"]


def test_implementation_runner_classes_are_not_public_aliases() -> None:
    import dreamervla.runners as runners

    implementation_names = {
        "ChameleonLatentActionWMRunner",
        "WorldModelTrainingBase",
        "DreamerV3PixelRunner",
        "DreamerV3TokenRunner",
        "LIBEROVLAEvaluationBase",
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
    import dreamervla.models.embodiment.world_model as world_model

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


def test_models_package_exports_embodiment_fail_fast_symbols() -> None:
    import dreamervla.models as models

    assert "Critic" not in models.__all__
    assert "VLAPolicy" not in models.__all__
    assert not hasattr(models, "Critic")
    assert not hasattr(models, "VLAPolicy")

    for name in ("WorldModel",):
        assert name in models.__all__
        assert getattr(models, name) is not None


def test_algorithms_package_exports_actor_and_critic_symbols() -> None:
    import dreamervla.algorithms.actor as actor
    import dreamervla.algorithms.critic as critic

    for name in ("VLAPolicy",):
        assert name in actor.__all__
        assert getattr(actor, name) is not None

    for name in ("Critic", "LatentSuccessClassifier", "TwohotCritic"):
        assert name in critic.__all__
        assert getattr(critic, name) is not None


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
                assert cls.__module__.startswith("dreamervla.runners.")
                assert str(target).rsplit(".", 1)[-1] in runners.PUBLIC_RUNNERS
                assert "workspace" not in cfg
        for experiment_name in EXPERIMENT_MODULES:
            cfg = _compose_experiment(experiment_name)
            target = cfg.get("_target_")
            assert target is not None
            cls = get_class(str(target))
            assert cls.__module__.startswith("dreamervla.runners.")
            assert str(target).rsplit(".", 1)[-1] in runners.PUBLIC_RUNNERS
            assert "workspace" not in cfg
