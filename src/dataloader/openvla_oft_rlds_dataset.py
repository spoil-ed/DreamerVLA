from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.utils.openvla_oft_imports import ensure_openvla_oft_on_path


@dataclass
class OpenVLAOFTRLDSDatasetBundle:
    dataset: Any
    dataloader: DataLoader
    dataset_statistics: dict[str, Any]


class OpenVLAOFTRLDSDatasetFactory:
    """Factory for OpenVLA-OFT RLDS datasets.

    The factory is intentionally separate from Hydra instantiation because the
    OpenVLA processor and image size come from the loaded policy.
    """

    def __init__(
        self,
        data_root_dir: str | Path,
        dataset_name: str = "libero_goal_no_noops",
        shuffle_buffer_size: int = 100_000,
        image_aug: bool = True,
        use_wrist_image: bool = True,
        use_proprio: bool = True,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        self.data_root_dir = str(Path(data_root_dir).expanduser().resolve())
        self.dataset_name = str(dataset_name)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.image_aug = bool(image_aug)
        self.use_wrist_image = bool(use_wrist_image)
        self.use_proprio = bool(use_proprio)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def build(self, policy: Any, *, train: bool = True) -> OpenVLAOFTRLDSDatasetBundle:
        ensure_openvla_oft_on_path()
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

        processor = policy.processor
        if processor is None:
            raise ValueError(
                "OpenVLAOFTRLDSDatasetFactory requires a policy loaded from a real processor."
            )
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=self.use_wrist_image,
            use_proprio=self.use_proprio,
        )
        dataset = RLDSDataset(
            Path(self.data_root_dir),
            self.dataset_name,
            batch_transform,
            resize_resolution=tuple(policy.vla.config.image_sizes),
            shuffle_buffer_size=self.shuffle_buffer_size,
            image_aug=self.image_aug,
            train=train,
        )
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collator,
            num_workers=self.num_workers,
        )
        return OpenVLAOFTRLDSDatasetBundle(
            dataset=dataset,
            dataloader=dataloader,
            dataset_statistics=dict(dataset.dataset_statistics),
        )


__all__ = ["OpenVLAOFTRLDSDatasetBundle", "OpenVLAOFTRLDSDatasetFactory"]
