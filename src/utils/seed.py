from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    # Python seed
    random.seed(seed)
    # Torch seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # CUDA seed
        torch.cuda.manual_seed_all(seed)
