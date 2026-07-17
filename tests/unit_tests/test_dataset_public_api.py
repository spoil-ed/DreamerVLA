from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class


def test_dataset_public_api_exports_only_retained_routes() -> None:
    import dreamervla.dataset as dataset

    expected = {
        "BaseDataset",
        "PixelHiddenSequenceDataset",
        "PixelSequenceDataset",
        "PixelSequenceSpec",
        "TokenSequenceDataset",
        "TokenSequenceSpec",
        "DinoTokenTrajectoryDataset",
        "OneTrajectoryPretokenizeActionChunkDataset",
        "VLASFTHDF5Dataset",
        "VLASFTHDF5DatasetFactory",
        "VLASFTHDF5Spec",
        "VLASFTRLDSDatasetBundle",
        "VLASFTRLDSDatasetFactory",
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
        "dreamervla/dataset/libero_dataset.py",
        "dreamervla/dataset/transition_dataset.py",
    }

    for relative_path in removed_files:
        assert not (project_root / relative_path).exists()


def test_configs_use_importable_dreamervla_dataset_targets() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config_names = sorted(
        str(path.relative_to(config_dir).with_suffix(""))
        for path in config_dir.rglob("*.yaml")
        if "experiment" not in path.relative_to(config_dir).parts
    )

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfgs = [compose(config_name=config_name) for config_name in config_names]
        experiment_names = sorted(path.stem for path in (config_dir / "experiment").glob("*.yaml"))
        cfgs.extend(
            compose(config_name="train", overrides=[f"experiment={experiment_name}"])
            for experiment_name in experiment_names
        )
        for cfg in cfgs:
            for section in ("dataset", "dataset_val_ind", "dataset_val_ood"):
                target = (
                    cfg.get(section, {}).get("_target_") if cfg.get(section) is not None else None
                )
                if target is not None:
                    assert str(target).startswith("dreamervla.dataset.")
                    get_class(str(target))


def test_task_classifier_dataset_targets_are_importable() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=classifier_official_upper_bound"],
        )

    get_class(str(cfg.task.classifier.dataset.train._target_))
    get_class(str(cfg.task.classifier.dataset.validation._target_))
