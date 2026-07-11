"""Small shared utilities used by the mainline online runners."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import torch


def load_world_model_state_from_dict(
    world_model: torch.nn.Module,
    state: dict[str, Any],
    *,
    reset_reward_head: bool = False,
) -> tuple[list[str], list[str]]:
    """Load an exact mainline world-model state dict."""

    dtype = next(world_model.parameters()).dtype
    cleaned: dict[str, torch.Tensor] = {}
    skipped_reward = 0
    for raw_key, value in state.items():
        key = str(raw_key).removeprefix("_fsdp_wrapped_module.").removeprefix("module.")
        if reset_reward_head and key.startswith("reward_head."):
            skipped_reward += 1
            continue
        cleaned[key] = (
            value.to(dtype=dtype) if torch.is_floating_point(value) else value
        )
    if reset_reward_head:
        missing, unexpected = world_model.load_state_dict(cleaned, strict=False)
        non_reward_missing = [key for key in missing if not key.startswith("reward_head.")]
        if non_reward_missing or unexpected:
            raise RuntimeError(
                "world-model checkpoint mismatch while resetting reward head: "
                f"missing={non_reward_missing}, unexpected={list(unexpected)}"
            )
    else:
        world_model.load_state_dict(cleaned, strict=True)
        missing, unexpected = [], []
    print(
        f"[init] world_model loaded: tensors={len(cleaned)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if skipped_reward:
        print(f"[init] skipped reward head tensors: {skipped_reward}", flush=True)
    return missing, unexpected


def load_world_model_state(
    world_model: torch.nn.Module,
    ckpt_path: str,
    reset_reward_head: bool = False,
) -> None:
    """Load a world model from a component or runner checkpoint."""

    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"world model ckpt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dicts", {}).get("world_model") or payload.get("model")
    if state is None:
        raise RuntimeError(f"{path} has no state_dicts.world_model or model")
    load_world_model_state_from_dict(
        world_model,
        state,
        reset_reward_head=reset_reward_head,
    )


class SuccessTracker:
    """Windowed episode success rate with best-so-far and print delta."""

    def __init__(self, window: int) -> None:
        self._buf: deque[float] = deque(maxlen=max(1, int(window)))
        self._best: float = 0.0
        self._last_printed: float | None = None

    def update(self, success: bool) -> None:
        self._buf.append(1.0 if success else 0.0)
        if len(self._buf) == self._buf.maxlen:
            rate = self.rate()
            if rate > self._best:
                self._best = rate

    def rate(self) -> float:
        return (sum(self._buf) / len(self._buf)) if self._buf else 0.0

    @property
    def best(self) -> float:
        return self._best

    def delta(self) -> float:
        if self._last_printed is None:
            return 0.0
        return self.rate() - self._last_printed

    def mark_printed(self) -> None:
        self._last_printed = self.rate()

    def __len__(self) -> int:
        return len(self._buf)


__all__ = [
    "SuccessTracker",
    "load_world_model_state",
    "load_world_model_state_from_dict",
]
