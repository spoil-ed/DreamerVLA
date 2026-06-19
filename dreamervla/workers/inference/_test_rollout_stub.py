"""CPU stub implementing the rollout-bundle contract for tests."""

from __future__ import annotations

from typing import Any

import numpy as np

BACKBONE_DIM = 8  # Small synthetic backbone-latent dim for fast plumbing tests.
HIDDEN_DIM = BACKBONE_DIM


class StubExtractor:
    """Minimal per-env extractor with observable reset state."""

    def __init__(self) -> None:
        self.n = 0

    def reset(self) -> None:
        self.n = 0

    def prepare(self, obs: dict[str, Any], task_description: str) -> dict[str, Any]:
        self.n += 1
        return {"seed": int(obs.get("seed", 0))}


class StubRolloutBundle:
    """Deterministic rollout bundle for CPU-only worker tests."""

    def __init__(self, action_dim: int = 7, hidden_dim: int = HIDDEN_DIM) -> None:
        self._action_dim = int(action_dim)
        self._hidden_dim = int(hidden_dim)

    def make_extractor(self) -> StubExtractor:
        return StubExtractor()

    def predict_batch(self, preps: list[dict[str, Any]]) -> list[tuple[list[np.ndarray], np.ndarray]]:
        out = []
        for prep in preps:
            seed = int(prep["seed"])
            action_chunk = [
                np.full((self._action_dim,), float(seed) + j, dtype=np.float32)
                for j in range(8)
            ]
            flat_hidden = np.full((self._hidden_dim,), float(seed), dtype=np.float16)
            out.append((action_chunk, flat_hidden))
        return out
