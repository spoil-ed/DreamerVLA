from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RLDSBatchTransform:
    action_tokenizer: Any
    tokenizer: Any
    image_transform: Callable
    prompt_builder_fn: Callable
    use_wrist_image: bool = True
    use_proprio: bool = True

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "RLDS transforms are not included in the lightweight OpenVLA-OFT layer."
        )


class RLDSDataset:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "RLDSDataset requires the full OpenVLA-OFT data pipeline; use OpenVLAOFTHDF5DatasetFactory instead."
        )


class EpisodicRLDSDataset(RLDSDataset):
    pass


class DummyDataset(RLDSDataset):
    pass


__all__ = ["DummyDataset", "EpisodicRLDSDataset", "RLDSBatchTransform", "RLDSDataset"]
