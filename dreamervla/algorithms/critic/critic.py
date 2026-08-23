from __future__ import annotations

from torch import Tensor, nn


class Critic(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        critic_hidden_dim = int(critic_hidden_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim!r}")
        if critic_hidden_dim <= 0:
            raise ValueError(f"critic_hidden_dim must be > 0, got {critic_hidden_dim!r}")
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, critic_hidden_dim),
            nn.GELU(),
            nn.Linear(critic_hidden_dim, 1),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        # Value score
        return self.value_head(hidden).squeeze(-1)
