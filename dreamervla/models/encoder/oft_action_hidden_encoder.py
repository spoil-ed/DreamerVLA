from __future__ import annotations

from pathlib import Path

import torch

from .base_encoder import BaseEncoder


class OFTActionHiddenEncoder(BaseEncoder):
    """Placeholder encoder for precomputed OpenVLA-OFT action-hidden WM runs.

    The current WM pretraining path reads OFT hidden states from sidecar HDF5
    files, so this class only preserves the config surface that online eval will
    later fill in with live OFT inference.
    """

    def __init__(
        self,
        oft_ckpt_path: str | Path,
        resolution: int = 224,
        action_dim: int = 7,
        time_horizon: int = 8,
        action_head_type: str = "oft_l1_regression",
        pool: str = "none",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.oft_ckpt_path = str(Path(oft_ckpt_path).expanduser())
        self.resolution = int(resolution)
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        self.action_head_type = str(action_head_type)
        self.pool = str(pool)
        self.freeze_backbone = bool(freeze_backbone)

    def encode(self, obs: dict[str, object]) -> torch.Tensor:
        raise NotImplementedError(
            "OFTActionHiddenEncoder live inference is not implemented yet; "
            "WM pretraining should use precomputed obs_embedding sidecars."
        )
