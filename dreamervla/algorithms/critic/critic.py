from __future__ import annotations

from torch import Tensor, nn


class Critic(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(critic_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(critic_hidden_dim), 1),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        # Value score
        return self.value_head(hidden).squeeze(-1)
