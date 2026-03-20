from __future__ import annotations

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


def create_random_dataloader(data_config: DictConfig, model_config: DictConfig) -> DataLoader:
    dataset = RandomMultimodalDataset(data_config=data_config, model_config=model_config)
    return DataLoader(
        dataset=dataset,
        batch_size=int(data_config.batch_size),
        shuffle=bool(data_config.shuffle),
        num_workers=int(data_config.num_workers),
        pin_memory=bool(data_config.pin_memory),
        drop_last=bool(data_config.drop_last),
    )
