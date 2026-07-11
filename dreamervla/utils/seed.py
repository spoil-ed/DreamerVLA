from __future__ import annotations

import random
import warnings
from collections.abc import Mapping
from typing import Any

import torch


def set_seed(seed: int) -> None:
    # Python seed
    random.seed(seed)
    # Torch seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # CUDA seed
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG that :func:`set_seed` controls, for bit-exact resume.

    Covers Python ``random``, torch CPU, and all CUDA devices (RLinf-style).
    NumPy is intentionally omitted: ``set_seed`` does not seed it, so it is
    outside this project's determinism contract.
    """

    return {
        "python": random.getstate(),
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
        missing = sorted({"python", "torch", "cuda"}.difference(state))
        if missing:
            raise RuntimeError(f"strict RNG state is missing keys: {missing}")
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    elif strict:
        raise RuntimeError("strict RNG state has no Python state")
    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state)
    elif strict:
        raise RuntimeError("strict RNG state has no Torch tensor state")
    cuda_state = state.get("cuda")
    if strict and not isinstance(cuda_state, list):
        raise RuntimeError("strict RNG CUDA state must be a list")
    if strict and isinstance(cuda_state, list) and not all(
        isinstance(item, torch.Tensor) for item in cuda_state
    ):
        raise RuntimeError("strict RNG CUDA state contains a non-tensor entry")
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        if strict and len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "strict RNG CUDA topology mismatch: "
                f"checkpoint={len(cuda_state)} current={torch.cuda.device_count()}"
            )
        if not cuda_state:
            return
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as exc:  # GPU topology may differ on resume
            if strict:
                raise RuntimeError("could not strictly restore CUDA RNG") from exc
            warnings.warn(f"could not restore CUDA RNG: {exc}", stacklevel=2)
    elif strict and isinstance(cuda_state, list) and cuda_state:
        raise RuntimeError("strict RNG restore cannot apply CUDA state without CUDA")
