from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_dataloader_public_api_exports_only_retained_routes() -> None:
    import src.dataloader as dataloader

    expected = {
        "BaseDataset",
        "LIBEROPixelRynnHiddenSequenceDataset",
        "LIBEROPixelSequenceDataset",
        "LIBEROPixelSequenceSpec",
        "LIBEROTokenSequenceDataset",
        "LIBEROTokenSequenceSpec",
        "PretokenizeActionChunkDataset",
        "PretokenizeDataSpec",
        "PretokenizeDataset",
    }

    assert set(dataloader.__all__) == expected
    for name in expected:
        assert hasattr(dataloader, name)

    removed_names = {
        "LIBERODataSpec",
        "LIBEROTransitionDataset",
        "PretokenizeFlatDataset",
        "TrainingDataSpec",
        "TransitionDataset",
    }
    for name in removed_names:
        assert name not in dataloader.__all__
        assert not hasattr(dataloader, name)


def test_removed_dataloader_files_are_absent() -> None:
    project_root = Path(__file__).resolve().parents[1]
    removed_files = {
        "src/dataloader/libero_dataset.py",
        "src/dataloader/transition_dataset.py",
    }

    for relative_path in removed_files:
        assert not (project_root / relative_path).exists()


def test_configs_use_only_retained_dataloader_targets() -> None:
    retained_targets = {
        "src.dataloader.libero_pixel_rynn_hidden_sequence_dataset.LIBEROPixelRynnHiddenSequenceDataset",
        "src.dataloader.libero_pixel_sequence_dataset.LIBEROPixelSequenceDataset",
        "src.dataloader.libero_token_sequence_dataset.LIBEROTokenSequenceDataset",
        "src.dataloader.pretokenize_dataset.PretokenizeActionChunkDataset",
        "src.dataloader.pretokenize_dataset.PretokenizeDataset",
    }

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    config_names = sorted(str(path.relative_to(config_dir).with_suffix("")) for path in config_dir.rglob("*.yaml"))

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for config_name in config_names:
            cfg = compose(config_name=config_name)
            for section in ("dataset", "dataset_val_ind", "dataset_val_ood"):
                target = cfg.get(section, {}).get("_target_") if cfg.get(section) is not None else None
                if target is not None:
                    assert str(target) in retained_targets
