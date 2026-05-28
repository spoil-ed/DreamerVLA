from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class


def test_workspace_public_api_exports_route_specific_names() -> None:
    import src.workspace as workspace

    expected = {
        "ActionHiddenWMWorkspace",
        "PixelWMWorkspace",
        "TokenWMWorkspace",
        "VLASFTWorkspace",
        "JointDreamerVLAWorkspace",
        "LiberoEvalWorkspace",
        "ChameleonLatentWMWorkspace",
    }

    assert expected == set(workspace.PUBLIC_WORKSPACES)
    assert expected.issubset(set(workspace.__all__))
    for name in expected:
        cls = getattr(workspace, name)
        assert cls.__name__ == name
        assert isinstance(getattr(cls, "workspace_name"), str)
        assert callable(getattr(cls, "setup"))
        assert callable(getattr(cls, "execute"))
        assert callable(getattr(cls, "run"))
        assert callable(getattr(cls, "teardown"))


def test_workspace_directory_contains_route_specific_workspaces() -> None:
    workspace_dir = Path(__file__).resolve().parents[1] / "src" / "workspace"
    top_level_python_files = {path.name for path in workspace_dir.glob("*.py")}
    assert {
        "__init__.py",
        "base_workspace.py",
        "chameleon_latent_action_wm_workspace.py",
        "dreamer_vla_workspace.py",
        "dreamerv3_pixel_workspace.py",
        "dreamerv3_token_workspace.py",
        "eval_libero_vla_workspace.py",
        "pretokenize_vla_workspace.py",
        "rynn_backbone_dreamerv3_wm_workspace.py",
    }.issubset(top_level_python_files)
    assert "pretokenize_sft_workspace.py" not in top_level_python_files
    assert "pretokenize_wm_workspace.py" not in top_level_python_files
    assert "semantic_bottleneck_wm_workspace.py" not in top_level_python_files
    assert not (workspace_dir.parent / "workspace_impl").exists()


def test_removed_legacy_compatibility_shims_are_absent() -> None:
    project_root = Path(__file__).resolve().parents[1]

    assert not (project_root / "src" / "dataloader" / "pretokenize_sequence_dataset.py").exists()
    assert not (project_root / "scripts" / "pretokenize_train_wm.sh").exists()


def test_removed_route_files_are_absent() -> None:
    project_root = Path(__file__).resolve().parents[1]
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
        "src/cli/diagnose_action_sensitivity.py",
        "src/cli/diagnose_decoder_zh.py",
        "src/cli/diagnose_wm.py",
        "src/cli/diagnose_wm_checklist.py",
        "src/cli/diagnose_wm_collapse.py",
        "src/cli/diagnose_wm_layers.py",
        "src/cli/eval_embedding_distribution.py",
        "src/cli/eval_wm.py",
        "src/cli/train_pure_vae.py",
        "src/models/world_model/causal_transformer.py",
        "src/models/world_model/causal_transformer_v2.py",
        "src/models/world_model/image_codec.py",
        "src/models/world_model/semantic_bottleneck.py",
        "src/models/world_model/token_io.py",
        "src/models/world_model/transdreamer_original.py",
        "src/models/world_model/transdreamer_transformer.py",
        "src/models/world_model/tssm.py",
        "src/models/world_model/tssm_discrete.py",
        "src/workspace/pretokenize_sft_workspace.py",
        "src/workspace/pretokenize_wm_workspace.py",
        "src/workspace/semantic_bottleneck_wm_workspace.py",
        "tests/test_semantic_bottleneck_world_model.py",
    }

    for relative_path in removed_files:
        assert not (project_root / relative_path).exists()


def test_base_dataset_no_longer_exposes_spec_alias() -> None:
    from src.dataloader.base_dataset import BaseDataset

    assert not hasattr(BaseDataset, "spec")


def test_active_configs_target_route_specific_workspace_classes() -> None:
    expected = {
        "pretokenize_vla_libero_goal": "src.workspace.VLASFTWorkspace",
        "pretokenize_vla_libero_goal_pi0_query": "src.workspace.VLASFTWorkspace",
        "rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed": "src.workspace.ActionHiddenWMWorkspace",
        "dreamerv3_pixel_libero_goal": "src.workspace.PixelWMWorkspace",
        "dreamerv3_token_libero_goal": "src.workspace.TokenWMWorkspace",
        "dreamer_vla_libero_goal_pi0_action_hidden_head_actor": "src.workspace.JointDreamerVLAWorkspace",
        "eval_libero_vla": "src.workspace.LiberoEvalWorkspace",
        "chameleon_latent_action_wm_libero_goal": "src.workspace.ChameleonLatentWMWorkspace",
    }

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for config_name, target in expected.items():
            cfg = compose(config_name=config_name)
            assert cfg._target_ == target
            assert "workspace" not in cfg
            cls = get_class(target)
            assert cls.__name__ == target.rsplit(".", 1)[-1]


def test_root_configs_inherit_archive_defaults_at_global_package() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="dreamerv3_pixel_libero_goal")
        assert cfg._target_ == "src.workspace.PixelWMWorkspace"
        assert cfg.dataset._target_ == "src.dataloader.libero_pixel_sequence_dataset.LIBEROPixelSequenceDataset"
        assert cfg.world_model._target_ == "src.models.world_model.dreamerv3_torch.DreamerV3PixelWorldModel"
        assert cfg.training.out_dir.endswith("dreamerv3_pixel_libero_goal")

        archive_cfg = compose(config_name="archive/libero10_legacy/dreamerv3_pixel_libero_10")
        assert archive_cfg._target_ == "src.workspace.PixelWMWorkspace"
        assert archive_cfg.dataset._target_ == "src.dataloader.libero_pixel_sequence_dataset.LIBEROPixelSequenceDataset"


def test_cli_default_uses_current_public_workspace_target() -> None:
    from src.cli.train import _parse_hydra_like_args

    config_name, overrides = _parse_hydra_like_args([])
    assert config_name == "rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed"
    assert overrides == []


def test_implementation_workspace_classes_are_not_public_aliases() -> None:
    import src.workspace as workspace

    implementation_names = {
        "ChameleonLatentActionWMWorkspace",
        "DreamerVLAWorkspace",
        "DreamerV3PixelWorkspace",
        "DreamerV3TokenWorkspace",
        "EvalLiberoVLAWorkspace",
        "PretokenizeVLAWorkspace",
        "RynnBackboneDreamerV3WMWorkspace",
    }

    for name in implementation_names:
        assert name not in workspace.__all__
        assert not hasattr(workspace, name)


def test_removed_workspace_routes_are_not_importable() -> None:
    import src.workspace as workspace

    removed_public_names = {
        "PooledHiddenWMWorkspace",
        "PretokenizedWMWorkspace",
        "PretokenizedSFTWorkspace",
        "SemanticBottleneckWMWorkspace",
    }

    for name in removed_public_names:
        assert name not in workspace.PUBLIC_WORKSPACES
        assert name not in workspace.__all__
        assert not hasattr(workspace, name)


def test_world_model_package_exposes_only_retained_architectures() -> None:
    import src.models as models
    import src.models.world_model as world_model

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


def test_all_configs_compose_and_resolve_route_specific_workspace_targets() -> None:
    import src.workspace as workspace

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    config_names = sorted(str(path.relative_to(config_dir).with_suffix("")) for path in config_dir.rglob("*.yaml"))

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for config_name in config_names:
            cfg = compose(config_name=config_name)
            target = cfg.get("_target_")
            if target is not None:
                cls = get_class(str(target))
                assert cls.__module__ == "src.workspace"
                assert str(target).rsplit(".", 1)[-1] in workspace.PUBLIC_WORKSPACES
                assert "workspace" not in cfg
