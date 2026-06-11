from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class


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
        project_root
        / "dreamer_vla"
        / "dataset"
        / "pretokenize_sequence_dataset.py"
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
        "world_model_dinowm_step": "dreamer_vla.runners.RynnDinoWMRunner",
        "oft_world_model_dinowm_chunk": "dreamer_vla.runners.OFTDinoWMRunner",
        "dreamervla_rynn_dino_wm_actor_critic": "dreamer_vla.runners.JointDreamerVLARunner",
        "dreamervla_rynn_dino_wm_wmpo_outcome": "dreamer_vla.runners.JointDreamerVLARunner",
        "dreamervla_oft_dino_wm_wmpo_outcome": "dreamer_vla.runners.JointDreamerVLARunner",
        "eval_libero_vla": "dreamer_vla.runners.LiberoEvalRunner",
        "openvla_oft_hdf5": "dreamer_vla.runners.OpenVLAOFTRunner",
        "openvla_oft_hdf5_one_trajectory": "dreamer_vla.runners.OpenVLAOFTRunner",
        "latent_classifier_libero_goal_chunk": "dreamer_vla.runners.LatentClassifierRunner",
        "oft_latent_classifier_chunk": "dreamer_vla.runners.LatentClassifierRunner",
    }

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for config_name, target in expected.items():
            cfg = compose(config_name=config_name)
            assert cfg._target_ == target
            assert "workspace" not in cfg
            cls = get_class(target)
            assert cls.__name__ == target.rsplit(".", 1)[-1]


def test_root_configs_resolve_public_route_defaults() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    assert not (config_dir / "archive").exists()

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="world_model_dinowm_chunk")
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
    from dreamer_vla.cli.train import _parse_hydra_like_args

    config_name, overrides = _parse_hydra_like_args([])
    assert config_name == "world_model_dinowm_chunk"
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
