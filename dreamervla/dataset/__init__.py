"""Public datasets, loaded on demand without importing optional model stacks."""

from importlib import import_module
from typing import Any

_MODULES = {
    "BaseDataset": "base.base_dataloader",
    "DatasetLoaderBundle": "base.base_dataloader",
    "HDF5Dataset": "base.hdf5_dataloader",
    "HDF5ActionChunkDataset": "base.hdf5_dataloader",
    "HDF5ActionChunkSpec": "base.hdf5_dataloader",
    "PixelSequenceDataset": "base.hdf5_dataloader",
    "PixelSequenceSpec": "base.hdf5_dataloader",
    "DinoTokenTrajectoryDataset": "base.latent_token_dataloader",
    "PixelHiddenSequenceDataset": "base.latent_token_dataloader",
    "LeRobotV3DataLoader": "base.lerobot_v3_dataloader",
    "MultiDataset": "base.multi_dataloader",
    "DistributedMixtureSampler": "base.multi_dataloader",
    "LiberoDataset": "libero",
    "LeRobotV3LIBERODataLoaderFactory": "libero",
    "VLASFTHDF5Dataset": "libero",
    "VLASFTHDF5DatasetFactory": "libero",
    "BalancedTerminalDataset": "classifier_dataset",
    "BalancedTerminalSampler": "classifier_dataset",
    "CollectedRolloutClassifierDataset": "classifier_dataset",
    "LumosAlignedLatentTrainDataset": "classifier_dataset",
    "LumosAlignedLatentValDataset": "classifier_dataset",
}

__all__ = list(_MODULES)


def __getattr__(name: str) -> Any:
    """Resolve retained dataset exports without eagerly loading every reader."""
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
