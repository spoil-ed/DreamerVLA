from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class RandomMultimodalDataset(Dataset):
    def __init__(self, data_config: DictConfig, model_config: DictConfig) -> None:
        self.data_config = data_config
        self.model_config = model_config

        if self.data_config.min_language_tokens > self.model_config.max_language_length:
            raise ValueError("min_language_tokens must be <= max_language_length.")
        if self.model_config.vocab_size <= self.model_config.pad_token_id + 1:
            raise ValueError("vocab_size must be larger than pad_token_id + 1.")

    def __len__(self) -> int:
        return int(self.data_config.num_samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        del index
        data_cfg = self.data_config
        model_cfg = self.model_config

        image = torch.randn(
            model_cfg.image_channels,
            model_cfg.image_size,
            model_cfg.image_size,
        )
        proprio = torch.randn(model_cfg.proprio_dim)

        seq_len = torch.randint(
            low=data_cfg.min_language_tokens,
            high=model_cfg.max_language_length + 1,
            size=(1,),
        ).item()
        language = torch.full(
            (model_cfg.max_language_length,),
            fill_value=model_cfg.pad_token_id,
            dtype=torch.long,
        )
        language[:seq_len] = torch.randint(
            low=model_cfg.pad_token_id + 1,
            high=model_cfg.vocab_size,
            size=(seq_len,),
            dtype=torch.long,
        )
        language_attention_mask = language.ne(model_cfg.pad_token_id)

        return {
            "image": image,
            "language": language,
            "language_attention_mask": language_attention_mask,
            "proprio": proprio,
        }


class RandomRynnVLADataset(Dataset):
    def __init__(self, data_config: DictConfig, model_config: DictConfig) -> None:
        self.data_config = data_config
        self.model_config = model_config

    def __len__(self) -> int:
        return int(self.data_config.num_samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_size = int(self.data_config.get("image_size", self.model_config.get("image_size", 224)))
        state_dim = int(self.model_config.get("state_dim", self.model_config.get("proprio_dim", 16)))

        rgb_static = np.random.randint(
            low=0,
            high=256,
            size=(image_size, image_size, 3),
            dtype=np.uint8,
        )
        wrist_static = np.random.randint(
            low=0,
            high=256,
            size=(image_size, image_size, 3),
            dtype=np.uint8,
        )
        state = np.random.randn(state_dim).astype(np.float32)
        text = f"synthetic task instruction {index}"

        return {
            "obs": {
                "rgb_obs": {
                    "rgb_static": rgb_static,
                    "wrist_static": wrist_static,
                },
                "state": state,
            },
            "text": text,
        }


def _collate_rynn_vla_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "obs": [sample["obs"] for sample in batch],
        "text": [sample["text"] for sample in batch],
    }


def create_random_dataloader(data_config: DictConfig, model_config: DictConfig) -> DataLoader:
    encoder_type = str(model_config.get("encoder_type", "multimodal"))

    if encoder_type == "rynn_vla":
        dataset: Dataset = RandomRynnVLADataset(data_config=data_config, model_config=model_config)
        collate_fn = _collate_rynn_vla_batch
    else:
        dataset = RandomMultimodalDataset(data_config=data_config, model_config=model_config)
        collate_fn = None

    return DataLoader(
        dataset=dataset,
        batch_size=int(data_config.batch_size),
        shuffle=bool(data_config.shuffle),
        num_workers=int(data_config.num_workers),
        pin_memory=bool(data_config.pin_memory),
        drop_last=bool(data_config.drop_last),
        collate_fn=collate_fn,
    )
