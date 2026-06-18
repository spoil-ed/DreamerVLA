"""Shared helpers for OpenVLA-OFT rollout collection paths."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def process_action(action: Any) -> np.ndarray:
    """Gripper post-process for OpenVLA-OFT LIBERO actions (shared by eval + collectors).

    The OFT model gripper output is in ``[0, 1]``; map to ``[-1, 1]`` (``2g-1``),
    binarize with ``sign``, then invert (``*-1``) for LIBERO (-1=open, +1=close).
    This MUST be applied to every action before ``env.step`` — without it the
    gripper is wrong and grasping (hence task success) fails. Matches the canonical
    OpenVLA-OFT / RLinf eval (``normalize_gripper_action(binarize=True)`` +
    ``invert_gripper_action``).
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    a[-1] = np.sign(2.0 * a[-1] - 1.0) * -1.0
    return a


def resolve_model_path(model_path: str) -> str:
    """Absolute path for a checkpoint dir; relative paths resolve against cwd."""
    p = Path(model_path)
    return str(p.expanduser().resolve() if p.is_absolute() else Path.cwd() / model_path)


def load_policy(cfg: dict[str, Any], gpu_id: int) -> Any:
    """Load OpenVLAOFTPolicy from checkpoint on the specified GPU.

    Under torchrun, gpu_id = LOCAL_RANK; CUDA_VISIBLE_DEVICES is NOT
    overridden (torchrun sets it per-process already via LOCAL_RANK env).
    """
    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()

    from dreamervla.models.encoder.openvla_oft_policy import OpenVLAOFTPolicy

    model_path = resolve_model_path(cfg["model_path"])

    device = torch.device(f"cuda:{gpu_id}")

    # Auto-detect head mode (l1 vs discrete) from the checkpoint, mirroring the offline
    # preprocess (resolve_oft_policy_mode).  The one-trajectory cold-start ckpt is DISCRETE
    # (no action_head -> actions decoded from LM logits), and discrete implies no proprio.
    from dreamervla.preprocess.preprocess_oft_action_hidden import resolve_oft_policy_mode

    mode = resolve_oft_policy_mode(model_path, str(cfg["policy_mode"]))
    use_l1 = mode == "l1"
    use_proprio = use_l1
    cfg["_policy_mode"] = mode
    cfg["_use_proprio"] = use_proprio

    print(
        f"[collector rank={cfg['_rank']}] Loading OFT policy ({mode}) from {model_path} on {device} ...",
        flush=True,
    )
    t0 = time.time()
    policy = OpenVLAOFTPolicy(
        model_path=model_path,
        component_ckpt_dir=model_path,
        torch_dtype="bf16",
        num_images_in_input=int(cfg["num_images_in_input"]),
        use_lora=False,
        use_l1_regression=use_l1,
        use_diffusion=False,
        use_proprio=use_proprio,
        use_film=False,
        freeze_vla_backbone=True,
    )
    policy.eval()
    policy.to(device)

    # Load LIBERO-specific norm_stats from dataset_statistics.json
    stats_path = Path(model_path) / "dataset_statistics.json"
    with stats_path.open() as fh:
        policy.vla.norm_stats = json.load(fh)

    unnorm_key = cfg["unnorm_key"]
    assert unnorm_key in policy.vla.norm_stats, (
        f"{unnorm_key!r} not in norm_stats; found: {list(policy.vla.norm_stats)}"
    )

    # Cast proprio_projector to bfloat16 (matches sidecar-generation runtime)
    if policy.proprio_projector is not None:
        policy.proprio_projector.to(dtype=torch.bfloat16)

    print(f"[collector rank={cfg['_rank']}] Policy loaded in {time.time() - t0:.1f}s", flush=True)
    return policy


def make_preprocess_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build preprocess_config.json from cfg (no hardcoded extraction defaults).

    Matches BalancedTerminalDataset._validate_hidden_sidecar.  action_head_type
    and include_state reflect the DETECTED policy mode (cfg["_policy_mode"] /
    cfg["_use_proprio"], set by load_policy and asserted == task expectation by
    assert_policy_mode_matches); the remaining fields come straight from cfg.
    """
    mode = cfg["_policy_mode"]
    use_proprio = cfg["_use_proprio"]
    return {
        "action_head_type": "oft_l1_regression" if mode == "l1" else "oft_discrete_token",
        "obs_hidden_source": cfg["expected_obs_hidden_source"],
        "prompt_style": cfg["expected_prompt_style"],
        "history": int(cfg["expected_history"]),
        "include_state": bool(use_proprio),
        "rotate_images_180": bool(cfg["expected_rotate_images_180"]),
        "time_horizon": int(cfg["time_horizon"]),
        "token_dim": int(cfg["token_dim"]),
        "action_dim": int(cfg["action_dim"]),
        "num_images_in_input": int(cfg["num_images_in_input"]),
        "chunk_size": int(cfg["chunk_size"]),
        "hidden_key": "obs_embedding",
        "resolution": int(cfg["resolution"]),
        "model_path": resolve_model_path(cfg["model_path"]),
        "unnorm_key": cfg["unnorm_key"],
        "center_crop": True,
        "task_suite_name": cfg["task_suite_name"],
    }


def assert_policy_mode_matches(cfg: dict[str, Any]) -> None:
    """Early validation: ckpt-detected mode == task expected_* (RLinf-style)."""
    detected_head = (
        "oft_l1_regression" if cfg["_policy_mode"] == "l1" else "oft_discrete_token"
    )
    if detected_head != cfg["expected_action_head_type"]:
        raise ValueError(
            f"Detected OFT head {detected_head!r} from ckpt {cfg['model_path']!r} "
            f"!= task expected_action_head_type {cfg['expected_action_head_type']!r}. "
            "Point the cold-start task at a checkpoint matching the WM's expected head."
        )
    if bool(cfg["_use_proprio"]) != bool(cfg["expected_include_state"]):
        raise ValueError(
            f"Detected proprio={cfg['_use_proprio']!r} != task expected_include_state "
            f"{cfg['expected_include_state']!r} for ckpt {cfg['model_path']!r}."
        )


def resolve_num_images_in_input(collect_cfg: Any) -> int:
    """Resolve OFT deployment image count from the central collect config."""
    val = OmegaConf.select(collect_cfg, "num_images_in_input", default=None)
    return int(val) if val is not None else 1


def select_vla_image_keys(
    image_keys: list[str],
    *,
    history: int,
    num_images_in_input: int,
) -> list[str]:
    """Select camera keys that produce the requested OFT image count.

    ``task.image_keys`` names the camera views stored in rollout dumps. The OFT
    policy input count is a separate deployment knob: the one-trajectory
    discrete checkpoints use one current-frame camera even though dumps still
    store both LIBERO views. The extractor stacks ``history * len(image_keys)``
    images, so select the prefix of camera keys that matches the policy input
    count when possible.
    """
    keys = list(image_keys)
    if not keys:
        raise ValueError("task.image_keys must contain at least one camera key")
    hist = max(1, int(history))
    n_images = max(1, int(num_images_in_input))
    if n_images % hist != 0:
        raise ValueError(
            "num_images_in_input must be divisible by expected_history: "
            f"{n_images} % {hist} != 0"
        )
    n_views = max(1, n_images // hist)
    if n_views > len(keys):
        raise ValueError(
            "task.image_keys does not contain enough camera views for "
            f"num_images_in_input={n_images} and expected_history={hist}: "
            f"need {n_views}, got {len(keys)}"
        )
    return keys[: min(len(keys), n_views)]
