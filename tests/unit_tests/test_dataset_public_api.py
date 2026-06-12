from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class


def test_dataset_public_api_exports_only_retained_routes() -> None:
    import dreamer_vla.dataset as dataset

    expected = {
        "BaseDataset",
        "LIBEROPixelRynnHiddenSequenceDataset",
        "LIBEROPixelSequenceDataset",
        "LIBEROPixelSequenceSpec",
        "LIBEROTokenSequenceDataset",
        "LIBEROTokenSequenceSpec",
        "OneTrajectoryPretokenizeActionChunkDataset",
        "OpenVLAOFTHDF5Dataset",
        "OpenVLAOFTHDF5DatasetFactory",
        "OpenVLAOFTHDF5Spec",
        "OpenVLAOFTRLDSDatasetBundle",
        "OpenVLAOFTRLDSDatasetFactory",
        "PretokenizeActionChunkDataset",
        "PretokenizeDataSpec",
        "PretokenizeDataset",
    }

    assert set(dataset.__all__) == expected
    for name in expected:
        assert hasattr(dataset, name)

    removed_names = {
        "LIBERODataSpec",
        "LIBEROTransitionDataset",
        "PretokenizeFlatDataset",
        "TrainingDataSpec",
        "TransitionDataset",
    }
    for name in removed_names:
        assert name not in dataset.__all__
        assert not hasattr(dataset, name)


def test_removed_dataset_files_are_absent() -> None:
    project_root = Path(__file__).resolve().parents[2]
    removed_files = {
        "dreamer_vla/dataset/libero_dataset.py",
        "dreamer_vla/dataset/transition_dataset.py",
    }

    for relative_path in removed_files:
        assert not (project_root / relative_path).exists()


def test_configs_use_importable_dreamer_vla_dataset_targets() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config_names = sorted(
        str(path.relative_to(config_dir).with_suffix(""))
        for path in config_dir.rglob("*.yaml")
        if "experiment" not in path.relative_to(config_dir).parts
    )

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfgs = [compose(config_name=config_name) for config_name in config_names]
        experiment_names = sorted(
            path.stem for path in (config_dir / "experiment").glob("*.yaml")
        )
        cfgs.extend(
            compose(config_name="train", overrides=[f"experiment={experiment_name}"])
            for experiment_name in experiment_names
        )
        for cfg in cfgs:
            for section in ("dataset", "dataset_val_ind", "dataset_val_ood"):
                target = (
                    cfg.get(section, {}).get("_target_")
                    if cfg.get(section) is not None
                    else None
                )
                if target is not None:
                    assert str(target).startswith("dreamer_vla.dataset.")
                    get_class(str(target))
