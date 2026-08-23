"""Minimal Stateful checkpoint wrapper for FSDP strategies.

A lightweight stand-in for RLinf's DCP ``Checkpoint(Stateful)``: bundles a model
and its optimizers behind the ``Stateful`` protocol so a single ``state_dict`` /
``load_state_dict`` round-trips both. Single-node verifiable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful


class Checkpoint(Stateful):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer] = (),
    ) -> None:
        self.model = model
        if isinstance(optimizers, torch.optim.Optimizer):
            self.optimizers = [optimizers]
        else:
            self.optimizers = list(optimizers)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        for opt, opt_state in zip(
            self.optimizers,
            state_dict.get("optimizers", []),
            strict=False,
        ):
            opt.load_state_dict(opt_state)
