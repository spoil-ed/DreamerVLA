"""Explicit weights-only migration of retired WM-owned visual readouts."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

logger = logging.getLogger(__name__)


def discard_legacy_wm_readout(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop only the retired decoder namespace for evaluation/weights-only loading.

    Callers must still strictly validate the remaining transition state. This
    does not migrate optimizer state or authorize resuming image supervision.
    """
    removed = [
        key for key in state if key.removeprefix("module.").startswith("decoded_visual_loss.")
    ]
    if removed:
        logger.warning(
            "Weights-only migration: discard retired WM decoder (%d tensors, %d values); keys=%s",
            len(removed),
            sum(state[key].numel() for key in removed),
            sorted(removed),
        )
    return {key: value for key, value in state.items() if key not in set(removed)}
