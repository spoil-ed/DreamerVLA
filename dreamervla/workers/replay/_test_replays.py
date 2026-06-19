"""Tiny importable replay stand-ins for Ray learner e2e tests.

Lives in the package (like ``workers/actor/_test_models.py`` and
``workers/env/_test_envs.py``) so Ray actors can import it when a learner
worker is launched with one of these as its replay.
"""

from __future__ import annotations

import torch


class FixedBatchReplay:
    """Return one fixed batch on every ``sample`` (no RNG).

    Used to feed an in-process learner and a Ray-actor learner byte-identical
    input so their update math can be compared. Picklable: it only holds CPU
    tensors and is importable in the actor process.
    """

    def __init__(self, batch: dict) -> None:
        self._batch = {
            key: (value.detach().cpu().clone() if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }

    def sample(self, batch_size: int) -> dict:
        del batch_size
        return {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in self._batch.items()
        }
