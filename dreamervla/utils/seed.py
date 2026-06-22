from __future__ import annotations

import random
import warnings
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


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`.

    A ``None`` or partial payload is a no-op for the missing pieces, so old
    checkpoints without an ``"rng"`` entry resume unchanged.
    """

    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list) and cuda_state:
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as exc:  # GPU topology may differ on resume
            warnings.warn(f"could not restore CUDA RNG: {exc}", stacklevel=2)
