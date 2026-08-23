from __future__ import annotations

import random
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG that :func:`set_seed` controls, for bit-exact resume.

    Covers Python ``random``, NumPy, torch CPU, and all CUDA devices.
    """

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(
    state: Mapping[str, Any] | None,
    *,
    strict: bool = False,
) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`.

    A ``None`` or partial payload is a no-op for the missing pieces, so old
    checkpoints without an ``"rng"`` entry resume unchanged.
    """

    if strict and not isinstance(state, Mapping):
        raise RuntimeError("strict RNG state must be a mapping")
    if not state:
        if strict:
            raise RuntimeError("strict RNG restore requires a non-empty state")
        return
    if strict:
        missing = sorted({"python", "numpy", "torch", "cuda"}.difference(state))
        if missing:
            raise RuntimeError(f"strict RNG state is missing keys: {missing}")

    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_state = state.get("torch")
    cuda_state = state.get("cuda")

    if strict:
        if not isinstance(python_state, tuple):
            raise RuntimeError("strict RNG python state must be a tuple")
        if not isinstance(numpy_state, tuple):
            raise RuntimeError("strict RNG numpy state must be a tuple")
        if not isinstance(torch_state, torch.Tensor):
            raise RuntimeError("strict RNG torch state must be a tensor")
        if not isinstance(cuda_state, list):
            raise RuntimeError("strict RNG cuda state must be a list")
        if not all(isinstance(item, torch.Tensor) for item in cuda_state):
            raise RuntimeError("strict RNG cuda state contains a non-tensor entry")
        cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if len(cuda_state) != cuda_device_count:
            raise RuntimeError(
                "strict RNG cuda topology mismatch: "
                f"checkpoint={len(cuda_state)} current={cuda_device_count}"
            )

    if isinstance(python_state, tuple):
        try:
            random.setstate(python_state)
        except Exception as exc:
            if strict:
                raise RuntimeError("could not strictly restore python RNG") from exc
    if isinstance(numpy_state, tuple):
        try:
            np.random.set_state(numpy_state)
        except Exception as exc:
            if strict:
                raise RuntimeError("could not strictly restore numpy RNG") from exc
    if isinstance(torch_state, torch.Tensor):
        try:
            torch.set_rng_state(torch_state)
        except Exception as exc:
            if strict:
                raise RuntimeError("could not strictly restore torch RNG") from exc
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        if not cuda_state:
            return
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as exc:  # GPU topology may differ on resume
            if strict:
                raise RuntimeError("could not strictly restore cuda RNG") from exc
            warnings.warn(f"could not restore cuda RNG: {exc}", stacklevel=2)
    elif strict and isinstance(cuda_state, list) and cuda_state:
        raise RuntimeError("strict RNG restore cannot apply cuda state without cuda")


def select_rank_rng_state(states: Any, rank: int) -> Mapping[str, Any] | None:
    """Select a rank's RNG mapping from a gathered list or legacy mapping."""

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        return None
    if isinstance(states, Mapping):
        return states
    if not isinstance(states, (list, tuple)) or rank >= len(states):
        return None
    selected = states[rank]
    return selected if isinstance(selected, Mapping) else None
