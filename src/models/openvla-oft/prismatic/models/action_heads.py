import math

import torch
import torch.nn as nn

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim % 2 != 0:
            raise ValueError(f"dim must be even, got {self.dim}")
        half_dim = self.dim // 2
        exponent = (
            torch.arange(half_dim, device=x.device) * -math.log(10000) / (half_dim - 1)
        )
        emb = torch.exp(exponent)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x) + x


class MLPResNet(nn.Module):
    def __init__(
        self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            MLPResNetBlock(hidden_dim) for _ in range(num_blocks)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=int(input_dim) * ACTION_DIM,
            hidden_dim=int(hidden_dim),
            output_dim=self.action_dim,
        )

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged)

    def predict_action_with_intermediates(
        self, actions_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = actions_hidden_states.shape[0]
        x = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        model = self.model
        x = model.layer_norm1(x)
        x = model.fc1(x)
        x = model.relu(x)
        hidden_c = x
        for block in model.mlp_resnet_blocks:
            x = block(x)
        hidden_d = x
        x = model.layer_norm2(x)
        action = model.fc2(x)
        return action, hidden_c, hidden_d


class DiffusionActionHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "The lightweight OpenVLA-OFT compatibility layer only supports L1 regression."
        )


__all__ = [
    "DiffusionActionHead",
    "L1RegressionActionHead",
    "MLPResNet",
    "MLPResNetBlock",
    "SinusoidalPositionalEncoding",
]
