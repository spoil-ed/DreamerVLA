"""Protocol for the LUMOS success verifier — DreamerVLA's value source."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class SuccessVerifier(Protocol):
    """Scores an imagined latent video and returns per-rollout success.

    This is DreamerVLA's ``V(e_t)=P(future success)``. ``LatentSuccessClassifier``
    satisfies it today; an MLP / transformer / two-hot / ensemble / calibrated
    critic can be swapped in via the ``classifier`` component's Hydra ``_target_``
    as long as it implements this method with this return contract.
    """

    def predict_success(
        self,
        latent_video: torch.Tensor,
        *,
        threshold: float,
        stride: int = 1,
        min_steps: int = 1,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Return sparse outcome plus optional continuous score tensors."""
        ...
