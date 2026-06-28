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
        return {
            "seed": int(obs.get("seed", 0)),
            "task_description": str(task_description),
            "reset_count": int(self.n),
        }


class StubDecodeOutput:
    """Tuple-compatible decode output carrying optional language sidecar."""

    def __init__(
        self,
        action_chunk: list[np.ndarray],
        hidden: np.ndarray,
        lang_emb: np.ndarray | None,
    ) -> None:
        self.action_chunk = action_chunk
        self.hidden_state = hidden
        self.lang_emb = lang_emb

    def __iter__(self):
        yield self.action_chunk
        yield self.hidden_state

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        return (self.action_chunk, self.hidden_state)[index]


class StubRolloutBundle:
    """Deterministic rollout bundle for CPU-only worker tests."""

    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = HIDDEN_DIM,
        emit_lang: bool = False,
    ) -> None:
        self._action_dim = int(action_dim)
        self._hidden_dim = int(hidden_dim)
        self._emit_lang = bool(emit_lang)

    def make_extractor(self) -> StubExtractor:
        return StubExtractor()

    def predict_batch(self, preps: list[dict[str, Any]]) -> list[Any]:
        out = []
        for prep in preps:
            seed = int(prep["seed"])
            action_chunk = [
                np.full((self._action_dim,), float(seed) + j, dtype=np.float32)
                for j in range(8)
            ]
            flat_hidden = np.full((self._hidden_dim,), float(seed), dtype=np.float16)
            lang_emb = (
                np.full((2,), float(seed) + 0.5, dtype=np.float16)
                if self._emit_lang
                else None
            )
            out.append(StubDecodeOutput(action_chunk, flat_hidden, lang_emb))
        return out
