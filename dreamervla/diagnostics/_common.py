from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    return torch.device(name if torch.cuda.is_available() else "cpu")
