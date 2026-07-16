"""Pure helpers extracted from ``LIBEROVLAEvaluationRunner`` (P3 god-file split).

Stateless functions (no runner ``self`` coupling); ``LIBEROVLAEvaluationRunner`` exposes them
as static-method delegators so call sites are unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image


def normalize_vla_encoder_state_for_single_process_eval(payload: dict[str, Any]) -> None:
    """Make DDP-saved VLA encoder checkpoints load into single-process eval.

    VLA SFT checkpoints saved under DDP can contain backbone keys like
    ``backbone.module.model...``. Eval constructs the unwrapped encoder, so those keys
    need to become ``backbone.model...`` before strict loading.
    """
    encoder_state = payload.get("state_dicts", {}).get("encoder")
    if not isinstance(encoder_state, dict):
        return
    if not any(str(key).startswith("backbone.module.") for key in encoder_state):
        return
    payload["state_dicts"]["encoder"] = {
        (
            str(key).replace("backbone.module.", "backbone.", 1)
            if str(key).startswith("backbone.module.")
            else key
        ): value
        for key, value in encoder_state.items()
    }


def checkpoint_cfg_from_payload(payload: dict[str, Any]) -> DictConfig:
    cfg = payload.get("cfg")
    if cfg is None:
        raise RuntimeError("Dreamer checkpoint has no saved cfg; cannot rebuild Dreamer modules.")
    if isinstance(cfg, DictConfig):
        return copy.deepcopy(cfg)
    if isinstance(cfg, dict):
        return OmegaConf.create(copy.deepcopy(cfg))
    raise TypeError(f"Dreamer checkpoint cfg must be DictConfig or dict, got {type(cfg).__name__}")


def real_relabel_sparse_rewards(success: bool, finish_step: int, max_steps: int) -> list[float]:
    length = max(1, min(int(finish_step), int(max_steps)))
    rewards = [0.0] * length
    if success:
        rewards[length - 1] = 1.0
    return rewards


def to_numpy_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def array_summary(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "l2": float(np.linalg.norm(arr.reshape(-1))),
    }


def action_clip_bounds() -> tuple[np.ndarray, np.ndarray]:
    min_values = np.array([-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0])
    max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
    return min_values, max_values


def action_stats(prefix: str, left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    diff = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    return {
        f"{prefix}_mse": float(np.mean(np.square(diff))),
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_max_abs": float(np.max(np.abs(diff))),
    }


def strip_wrapping_prefix(key: str) -> str:
    for prefix in ("_fsdp_wrapped_module.", "module."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def resize_hwc_uint8(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return np.ascontiguousarray(image)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    return np.asarray(
        Image.fromarray(image).resize((size, size), resample=resample),
        dtype=np.uint8,
    )
