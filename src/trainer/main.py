from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "debug.yaml"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.model import DreamerVLA
from utils.dataset.random_dataloader import create_random_dataloader


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> DictConfig:
    return OmegaConf.load(Path(config_path))


def build_optimizer(model: nn.Module, optim_cfg: DictConfig) -> torch.optim.Optimizer:
    if optim_cfg.name.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(optim_cfg.learning_rate),
            betas=tuple(float(beta) for beta in optim_cfg.betas),
            weight_decay=float(optim_cfg.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer: {optim_cfg.name}")


def build_loss(loss_cfg: DictConfig) -> nn.Module:
    if loss_cfg.name.lower() == "mse":
        return nn.MSELoss(reduction=loss_cfg.reduction)
    raise ValueError(f"Unsupported loss: {loss_cfg.name}")


class SimpleTrainer:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = torch.device(_resolve_device(self.config.trainer.device))

        if "seed" in self.config.trainer:
            torch.manual_seed(int(self.config.trainer.seed))

        model_cfg = self.config.model
        data_cfg = self.config.data
        optim_cfg = self.config.optim
        loss_cfg = self.config.loss
        self.target_value = float(loss_cfg.target_value)

        self.model = DreamerVLA(
            image_size=model_cfg.image_size,
            patch_size=model_cfg.patch_size,
            image_channels=model_cfg.image_channels,
            vocab_size=model_cfg.vocab_size,
            max_language_length=model_cfg.max_language_length,
            proprio_dim=model_cfg.proprio_dim,
            embed_dim=model_cfg.embed_dim,
            fused_dim=model_cfg.fused_dim,
            image_depth=model_cfg.image_depth,
            language_depth=model_cfg.language_depth,
            num_heads=model_cfg.num_heads,
            mlp_ratio=model_cfg.mlp_ratio,
            proprio_hidden_dim=model_cfg.proprio_hidden_dim,
            dropout=model_cfg.dropout,
            pad_token_id=model_cfg.pad_token_id,
        ).to(self.device)

        self.optimizer = build_optimizer(self.model, optim_cfg=optim_cfg)
        self.criterion = build_loss(loss_cfg=loss_cfg)
        self.dataloader = create_random_dataloader(data_config=data_cfg, model_config=model_cfg)

    def train(self) -> None:
        self.model.train()
        for epoch in range(self.config.trainer.num_epochs):
            for step, batch in enumerate(self.dataloader):
                batch = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                }

                output = self.model(
                    image=batch["image"],
                    language=batch["language"],
                    proprio=batch["proprio"],
                    language_attention_mask=batch["language_attention_mask"],
                )

                target = torch.full_like(output["latent"], fill_value=self.target_value)
                loss = self.criterion(output["latent"], target)

                self.optimizer.zero_grad(set_to_none=bool(self.config.optim.zero_grad_set_to_none))
                loss.backward()
                self.optimizer.step()

                if step % int(self.config.trainer.log_every) == 0:
                    print(
                        f"epoch={epoch} step={step} "
                        f"latent_shape={tuple(output['latent'].shape)} "
                        f"tokens_shape={tuple(output['multimodal_tokens'].shape)} "
                        f"loss={loss.item():.6f}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the OmegaConf YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if bool(config.trainer.print_config):
        print(OmegaConf.to_yaml(config, resolve=True))
    trainer = SimpleTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
