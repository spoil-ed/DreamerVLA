from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class


EXPERIMENT_MODULES = {
    "vla_rynnvla_action_head": ("VLA", "rynnvla_action_head"),
    "vla_sft_one_trajectory": ("VLA", "rynnvla_one_trajectory"),
    "openvla_oft_hdf5": ("VLA", "openvla_oft"),
    "openvla_oft_hdf5_one_trajectory": ("VLA", "openvla_oft_one_trajectory"),
    "openvla_oft_hdf5_one_trajectory_l1": (
        "VLA",
        "openvla_oft_l1_one_trajectory",
    ),
    "world_model_dinowm_step": ("worldmodel", "rynnvla_action_step"),
    "world_model_dinowm_chunk": ("worldmodel", "rynnvla_action_chunk"),
    "world_model_dinowm_chunk_input_tokens": (
        "worldmodel",
        "rynnvla_input_token_chunk",
    ),
    "oft_world_model_dinowm_chunk": ("worldmodel", "openvla_oft_action_chunk"),
    "oft_discrete_token_world_model_dinowm_chunk": (
        "worldmodel",
        "openvla_oft_discrete_token_action_chunk",
    ),
    "oft_world_model_dinowm_chunk_input_tokens": (
        "worldmodel",
        "openvla_oft_input_token_chunk",
    ),
    "latent_classifier_libero_goal_chunk": ("classifier", "rynnvla_action_chunk"),
    "latent_classifier_libero_goal_chunk_input_tokens": (
        "classifier",
        "rynnvla_input_token_chunk",
    ),
    "oft_latent_classifier_chunk": ("classifier", "openvla_oft_action_chunk"),
    "oft_latent_classifier_chunk_input_tokens": (
        "classifier",
        "openvla_oft_input_token_chunk",
    ),
    "dreamervla_rynn_dino_wm_actor_critic": (
        "dreamervla",
        "rynnvla_actor_critic",
    ),
    "dreamervla_rynn_dino_wm_wmpo_outcome": (
        "dreamervla",
        "rynnvla_wmpo_outcome",
    ),
    "dreamervla_rynn_dino_wm_wmpo_outcome_input_tokens": (
        "dreamervla",
        "rynnvla_input_token_wmpo_outcome",
    ),
    "dreamervla_oft_dino_wm_wmpo_outcome": (
        "dreamervla",
        "openvla_oft_wmpo_outcome",
    ),
    "dreamervla_oft_discrete_token_dino_wm_wmpo_outcome": (
        "dreamervla",
        "openvla_oft_discrete_token_wmpo_outcome",
    ),
    "dreamervla_oft_dino_wm_wmpo_outcome_input_tokens": (
        "dreamervla",
        "openvla_oft_input_token_wmpo_outcome",
    ),
    "online_wmpo_outcome_libero_goal": (
        "dreamervla",
        "online_wmpo_outcome_libero_goal",
    ),
    "eval_libero_vla": ("evaluation", "libero_vla"),
}


def _compose_experiment(name: str, extra_overrides: list[str] | None = None):
    overrides = [f"experiment={name}"]
    if extra_overrides is not None:
        overrides.extend(extra_overrides)
    return compose(config_name="train", overrides=overrides)


def test_runner_public_api_exports_route_specific_names() -> None:
    import dreamer_vla.runners as runners

    expected = {
        "ActionHiddenWMRunner",
        "PixelWMRunner",
        "TokenWMRunner",
        "VLASFTRunner",
        "OpenVLAOFTRunner",
        "JointDreamerVLARunner",
        "LiberoEvalRunner",
        "ChameleonLatentWMRunner",
        "RynnDinoWMRunner",
        "OFTDinoWMRunner",
        "LatentClassifierRunner",
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


def test_runner_directory_contains_route_specific_runners() -> None:
    runner_dir = Path(__file__).resolve().parents[2] / "dreamer_vla" / "runners"
    top_level_python_files = {path.name for path in runner_dir.glob("*.py")}
    assert {
        "__init__.py",
        "base_runner.py",
        "chameleon_latent_action_wm_runner.py",
        "dreamer_vla_runner.py",
        "dreamerv3_pixel_runner.py",
        "dreamerv3_token_runner.py",
        "eval_libero_vla_runner.py",
        "latent_classifier_runner.py",
        "openvla_oft_runner.py",
        "pretokenize_vla_runner.py",
        "rynn_backbone_dreamerv3_wm_runner.py",
        "rynn_dino_wm_runner.py",
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
        project_root / "dreamer_vla" / "dataset" / "pretokenize_sequence_dataset.py"
    ).exists()
    assert not (project_root / "scripts" / "pretokenize_train_wm.sh").exists()


def test_removed_route_files_are_absent() -> None:
    project_root = Path(__file__).resolve().parents[2]
    removed_files = {
        "configs/dreamer_vla_libero_goal.yaml",
        "configs/dreamer_vla_libero_goal_dreamerv3_pixel_actor.yaml",
        "configs/dreamer_vla_libero_goal_dreamerv3_pixel_vlaactor.yaml",
        "configs/dreamer_vla_libero_goal_dreamerv3_token_actor.yaml",
        "configs/dreamer_vla_libero_goal_dreamerv3_token_actor_epoch6.yaml",
        "configs/dreamer_vla_libero_goal_rynn_pixel_precomputed_actor.yaml",
        "configs/dreamer_vla_libero_goal_rynn_pixel_precomputed_vlaactor.yaml",
        "configs/pretokenize_wm_libero_goal.yaml",
        "configs/pretokenize_wm_libero_goal_transdreamer.yaml",
        "configs/pretokenize_wm_libero_goal_warmup.yaml",
        "configs/rynn_backbone_dreamerv3_pixel_wm_libero_goal.yaml",
        "configs/rynn_backbone_dreamerv3_pixel_wm_libero_goal_precomputed.yaml",
        "configs/semantic_bottleneck_wm_libero_goal.yaml",
        "dreamer_vla/algorithms/dino_wmpo.py",
        "dreamer_vla/algorithms/dino_wmpo_chunk.py",
        "scripts/diagnose_wm.sh",
        "scripts/eval_wm.sh",
        "scripts/prepare_latent_data.sh",
        "scripts/run_rynn_meanpool_offline_long.sh",
        "scripts/run_rynn_meanpool_online_worker.sh",
        "scripts/train_dreamer_vla_pixel.sh",
        "scripts/train_dreamer_vla_rynn_pixel.sh",
        "scripts/train_online_rynn_meanpool_dreamer_actor.py",
        "scripts/train_rynn_backbone_dreamerv3_wm.sh",
        "scripts/train_semantic_bottleneck_wm.sh",
        "dreamer_vla/cli/diagnose_action_sensitivity.py",
        "dreamer_vla/cli/diagnose_decoder_zh.py",
        "dreamer_vla/cli/diagnose_wm.py",
        "dreamer_vla/cli/diagnose_wm_checklist.py",
        "dreamer_vla/cli/diagnose_wm_collapse.py",
        "dreamer_vla/cli/diagnose_wm_layers.py",
        "dreamer_vla/cli/eval_embedding_distribution.py",
        "dreamer_vla/cli/eval_wm.py",
        "dreamer_vla/cli/train_pure_vae.py",
        "dreamer_vla/models/world_model/causal_transformer.py",
        "dreamer_vla/models/world_model/causal_transformer_v2.py",
        "dreamer_vla/models/world_model/image_codec.py",
        "dreamer_vla/models/world_model/semantic_bottleneck.py",
        "dreamer_vla/models/world_model/token_io.py",
        "dreamer_vla/models/world_model/transdreamer_original.py",
        "dreamer_vla/models/world_model/transdreamer_transformer.py",
        "dreamer_vla/models/world_model/tssm.py",
        "dreamer_vla/models/world_model/tssm_discrete.py",
        "dreamer_vla/models/vla_actor.py",
        "dreamer_vla/models/vla_policy.py",
        "dreamer_vla/runners/pretokenize_sft_runner.py",
        "dreamer_vla/runners/pretokenize_wm_runner.py",
        "dreamer_vla/runners/semantic_bottleneck_wm_runner.py",
        "tests/test_semantic_bottleneck_world_model.py",
    }

    for relative_path in removed_files:
        assert not (project_root / relative_path).exists()


def test_base_dataset_no_longer_exposes_spec_alias() -> None:
    from dreamer_vla.dataset.base_dataset import BaseDataset

    assert not hasattr(BaseDataset, "spec")


def test_active_configs_target_route_specific_runner_classes() -> None:
    expected = {
        "vla_rynnvla_action_head": "dreamer_vla.runners.VLASFTRunner",
        "vla_sft_one_trajectory": "dreamer_vla.runners.VLASFTRunner",
        "world_model_dinowm_chunk": "dreamer_vla.runners.RynnDinoWMRunner",
        "world_model_dinowm_chunk_input_tokens": "dreamer_vla.runners.RynnDinoWMRunner",
        "world_model_dinowm_step": "dreamer_vla.runners.RynnDinoWMRunner",
        "oft_world_model_dinowm_chunk": "dreamer_vla.runners.OFTDinoWMRunner",
        "oft_discrete_token_world_model_dinowm_chunk": "dreamer_vla.runners.OFTDinoWMRunner",
        "oft_world_model_dinowm_chunk_input_tokens": "dreamer_vla.runners.OFTDinoWMRunner",
        "dreamervla_rynn_dino_wm_actor_critic": "dreamer_vla.runners.JointDreamerVLARunner",
        "dreamervla_rynn_dino_wm_wmpo_outcome": "dreamer_vla.runners.JointDreamerVLARunner",
        "dreamervla_rynn_dino_wm_wmpo_outcome_input_tokens": "dreamer_vla.runners.JointDreamerVLARunner",
        "dreamervla_oft_dino_wm_wmpo_outcome": "dreamer_vla.runners.JointDreamerVLARunner",
        "dreamervla_oft_discrete_token_dino_wm_wmpo_outcome": "dreamer_vla.runners.JointDreamerVLARunner",
        "dreamervla_oft_dino_wm_wmpo_outcome_input_tokens": "dreamer_vla.runners.JointDreamerVLARunner",
        "eval_libero_vla": "dreamer_vla.runners.LiberoEvalRunner",
        "openvla_oft_hdf5": "dreamer_vla.runners.OpenVLAOFTRunner",
        "openvla_oft_hdf5_one_trajectory": "dreamer_vla.runners.OpenVLAOFTRunner",
        "openvla_oft_hdf5_one_trajectory_l1": "dreamer_vla.runners.OpenVLAOFTRunner",
        "latent_classifier_libero_goal_chunk": "dreamer_vla.runners.LatentClassifierRunner",
        "latent_classifier_libero_goal_chunk_input_tokens": "dreamer_vla.runners.LatentClassifierRunner",
        "oft_latent_classifier_chunk": "dreamer_vla.runners.LatentClassifierRunner",
        "oft_latent_classifier_chunk_input_tokens": "dreamer_vla.runners.LatentClassifierRunner",
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
        default_cfg = _compose_experiment("world_model_dinowm_chunk")
        wandb_cfg = _compose_experiment(
            "dreamervla_rynn_dino_wm_wmpo_outcome",
            extra_overrides=["logger=wandb"],
        )

    assert default_cfg.runner.logger.project_name == "dreamer_vla"
    assert default_cfg.runner.logger.logger_backends == ["tensorboard"]
    assert default_cfg.runner.logger.log_path == f"{default_cfg.training.out_dir}/log"

    assert wandb_cfg.runner.logger.project_name == "dreamer_vla"
    assert wandb_cfg.runner.logger.logger_backends == ["wandb"]
    assert wandb_cfg.runner.logger.log_path == f"{wandb_cfg.training.out_dir}/log"
    assert wandb_cfg.runner.logger.wandb_mode == "online"


def test_openvla_oft_one_trajectory_routes_distinguish_action_heads() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        lm_head = _compose_experiment("openvla_oft_hdf5_one_trajectory")
        l1 = _compose_experiment("openvla_oft_hdf5_one_trajectory_l1")

    assert lm_head.policy.use_l1_regression is False
    assert l1.policy.use_l1_regression is True
    for cfg in (lm_head, l1):
        assert cfg.dataset.demos_per_task == 1
        assert cfg.dataset.demo_selection_seed == cfg.seed
        assert cfg.dataset.max_demos_per_file is None
    assert "openvla_oft_l1_one_trajectory" in l1.training.out_dir
    assert "openvla_oft_lm_head_one_trajectory" in lm_head.training.out_dir


def test_openvla_dreamervla_discrete_probability_route_is_explicit() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        discrete_wm = _compose_experiment("oft_discrete_token_world_model_dinowm_chunk")
        discrete = _compose_experiment(
            "dreamervla_oft_discrete_token_dino_wm_wmpo_outcome"
        )

    assert discrete.policy._target_ == "dreamer_vla.models.actor.OpenVLADiscreteTokenActor"
    assert discrete.policy.head_type == "oft_discrete_token"
    assert discrete.policy.init_lm_head_ckpt == discrete.task.openvla_oft.ckpt_path
    assert discrete.task.openvla_oft.expected_action_head_type == "oft_discrete_token"
    assert discrete.dataset.expected_include_state is False
    assert discrete.dataset.expected_history == 1
    assert discrete.dataset.hidden_dir.endswith("_h1")
    assert discrete_wm.dataset.expected_action_head_type == "oft_discrete_token"
    assert discrete_wm.dataset.hidden_dir.endswith("_h1")


def test_openvla_oft_action_hidden_defaults_match_preprocess_output() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    suites = ("libero_goal", "libero_object", "libero_spatial", "libero_10")

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfgs = [
            _compose_experiment(
                "oft_world_model_dinowm_chunk",
                extra_overrides=[f"task={suite}"],
            )
            for suite in suites
        ]

    for cfg in cfgs:
        expected = f"{cfg.task.hdf5_dir}_oft_legacy_action_hidden_vla_policy_h2"
        assert cfg.task.openvla_oft.action_hidden_dir == expected
        assert cfg.dataset.hidden_dir == expected


def test_input_token_scheme_b_routes_use_token_sidecar_and_bridge_actor() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        rynn_wm = _compose_experiment("world_model_dinowm_chunk_input_tokens")
        oft_wm = _compose_experiment("oft_world_model_dinowm_chunk_input_tokens")
        rynn_dreamer = _compose_experiment(
            "dreamervla_rynn_dino_wm_wmpo_outcome_input_tokens"
        )
        oft_dreamer = _compose_experiment(
            "dreamervla_oft_dino_wm_wmpo_outcome_input_tokens"
        )

    assert rynn_wm.dataset.expected_obs_hidden_source == "input_token_embedding"
    assert rynn_wm.world_model.token_count == 2048
    assert rynn_wm.world_model.token_dim == 4096
    assert "input_token" in rynn_wm.dataset.hidden_dir

    assert oft_wm.dataset.expected_obs_hidden_source == "input_token_embedding"
    assert oft_wm.world_model.token_count == 512
    assert oft_wm.world_model.token_dim == 4096
    assert "input_token" in oft_wm.dataset.hidden_dir

    assert rynn_dreamer.policy._target_ == "dreamer_vla.models.actor.LatentToActionHiddenActor"
    assert rynn_dreamer.policy.source_token_count == 2048
    assert rynn_dreamer.policy.action_hidden_dim == 1024

    assert oft_dreamer.policy._target_ == "dreamer_vla.models.actor.LatentToActionHiddenActor"
    assert oft_dreamer.policy.source_token_count == 512
    assert oft_dreamer.policy.action_hidden_dim == 4096


def test_train_config_resolves_public_default_experiment() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    assert not (config_dir / "archive").exists()

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train")
        assert cfg._target_ == "dreamer_vla.runners.RynnDinoWMRunner"
        assert (
            cfg.dataset._target_
            == "dreamer_vla.dataset.libero_balanced_terminal_dataset.LIBEROBalancedTerminalDataset"
        )
        assert (
            cfg.world_model._target_
            == "dreamer_vla.models.world_model.rynn_dino_wm_chunk.ChunkAwareRynnDinoWMWorldModel"
        )
        assert "dinowm_chunk" in cfg.training.out_dir


def test_cli_default_uses_current_public_runner_target() -> None:
    from dreamer_vla.train import _parse_hydra_like_args

    config_name, overrides = _parse_hydra_like_args([])
    assert config_name == "train"
    assert overrides == []


def test_implementation_runner_classes_are_not_public_aliases() -> None:
    import dreamer_vla.runners as runners

    implementation_names = {
        "ChameleonLatentActionWMRunner",
        "DreamerVLARunner",
        "DreamerV3PixelRunner",
        "DreamerV3TokenRunner",
        "EvalLiberoVLARunner",
        "PretokenizeVLARunner",
        "RynnBackboneDreamerV3WMRunner",
    }

    for name in implementation_names:
        assert name not in runners.__all__
        assert not hasattr(runners, name)


def test_removed_runner_routes_are_not_importable() -> None:
    import dreamer_vla.runners as runners

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
    import dreamer_vla.models as models
    import dreamer_vla.models.world_model as world_model

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
    import dreamer_vla.models as models

    for name in ("Critic", "VLAPolicy", "OFTDinoWMWorldModel", "RynnDinoWMWorldModel"):
        assert name in models.__all__
        assert getattr(models, name) is not None


def test_all_configs_compose_and_resolve_route_specific_runner_targets() -> None:
    import dreamer_vla.runners as runners

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
                assert cls.__module__ == "dreamer_vla.runners"
                assert str(target).rsplit(".", 1)[-1] in runners.PUBLIC_RUNNERS
                assert "workspace" not in cfg
        for experiment_name in EXPERIMENT_MODULES:
            cfg = _compose_experiment(experiment_name)
            target = cfg.get("_target_")
            assert target is not None
            cls = get_class(str(target))
            assert cls.__module__ == "dreamer_vla.runners"
            assert str(target).rsplit(".", 1)[-1] in runners.PUBLIC_RUNNERS
            assert "workspace" not in cfg
