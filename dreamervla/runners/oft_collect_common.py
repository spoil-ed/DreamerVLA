"""Shared helpers for OpenVLA-OFT rollout collection paths."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from dreamervla.preprocess.sidecar_schema import (
    SIDECAR_SCHEMA_VERSION,
    required_demo_datasets,
    validate_hidden_token_preprocess_config,
)


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


def process_action_batch(actions: Any) -> np.ndarray:
    """Vectorized OpenVLA-OFT LIBERO gripper post-process for action batches."""

    a = np.asarray(actions, dtype=np.float32).copy()
    if a.shape[-1] <= 0:
        raise ValueError("actions must have a non-empty action dimension")
    a[..., -1] = np.sign(2.0 * a[..., -1] - 1.0) * -1.0
    return a


def pop_open_loop_action(
    action_chunk: Any,
    action_queue: list,
    action_steps: int | None = None,
) -> np.ndarray:
    """The open-loop action CORE shared by ALL three OFT rollouts (single-env
    collector, batched vectorized collector, online cotrain): refill
    ``action_queue`` from ``action_chunk`` once drained, pop one action, and
    gripper-post-process it (``process_action`` — mandatory before ``env.step`` or
    grasping/success fails). ``action_steps`` (default = full chunk) caps how many
    of the predicted chunk run open-loop before re-querying. Mutates
    ``action_queue`` in place; returns the post-processed action."""
    if not action_queue:
        chunk = list(action_chunk)
        n = len(chunk) if action_steps is None else int(action_steps)
        if len(chunk) < n:
            raise ValueError(
                f"policy returned {len(chunk)} actions, need action_steps={n}"
            )
        action_queue.extend(chunk[:n])
    return process_action(action_queue.pop(0))


@dataclass(frozen=True)
class OFTOpenLoopStep:
    """Tuple-compatible rollout step output with optional sidecars."""

    action: np.ndarray
    hidden_state: Any
    lang_emb: Any | None = None

    def __iter__(self):
        yield self.action
        yield self.hidden_state

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        return (self.action, self.hidden_state)[index]


def sidecar_to_numpy(value: Any, dtype: Any | None = None) -> np.ndarray | None:
    """Convert a torch/numpy/list sidecar to a CPU numpy array."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        arr = value.numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def oft_open_loop_action(
    extractor: Any,
    extractor_obs: Any,
    task_description: str,
    action_queue: list,
    action_steps: int | None = None,
) -> OFTOpenLoopStep:
    """One SINGLE-ENV OFT open-loop rollout step — the shared implementation used
    by the collector (``collect_parallel_rollouts``) and the online cotrain rollout
    (``OnlineCotrainRunner._rollout_action``) so the two can never drift. Runs the
    OFT forward (``extractor.step``) for this frame's hidden state (= the
    ``obs_embedding`` the WM/classifier consume), then takes the open-loop action
    via ``pop_open_loop_action`` (the same core the batched vectorized collector
    uses). Returns a tuple-compatible ``(action, hidden_state)`` with optional
    ``lang_emb``."""
    decoded = extractor.step(extractor_obs, task_description)
    action_chunk, hidden_state = decoded
    action = pop_open_loop_action(action_chunk, action_queue, action_steps)
    return OFTOpenLoopStep(action, hidden_state, getattr(decoded, "lang_emb", None))


def resolve_model_path(model_path: str) -> str:
    """Absolute path for a checkpoint dir; relative paths resolve against cwd."""
    p = Path(model_path)
    return str(p.expanduser().resolve() if p.is_absolute() else Path.cwd() / model_path)


def _resolve_token_dim(vla: Any) -> int:
    for path in (
        ("token_dim",),
        ("hidden_size",),
        ("config", "hidden_size"),
        ("language_model", "config", "hidden_size"),
        ("llm_backbone", "llm", "config", "hidden_size"),
    ):
        cur = vla
        for attr in path:
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if cur is not None:
            return int(cur)
    raise ValueError("Could not derive token_dim from loaded VLA")


def vla_hidden_token_spec(vla: Any, image_keys: list[str]) -> dict[str, int]:
    """Derive hidden-token geometry from the loaded VLA input backbone."""

    token_dim = _resolve_token_dim(vla)
    patches_per_image = int(vla.vision_backbone.get_num_patches())
    num_images_in_input = int(vla.vision_backbone.get_num_images_in_input())
    keys = list(image_keys)
    if keys != ["agentview_rgb"]:
        raise ValueError(
            "OpenVLA-OFT hidden-token mainline requires image_keys=['agentview_rgb'], "
            f"got {keys!r}"
        )
    if patches_per_image <= 0 or token_dim <= 0 or num_images_in_input <= 0:
        raise ValueError(
            "loaded VLA exposes non-positive hidden-token geometry: "
            f"patches={patches_per_image}, token_dim={token_dim}, "
            f"num_images_in_input={num_images_in_input}"
        )
    if num_images_in_input != len(keys):
        raise ValueError(
            "loaded VLA image count does not match collection image_keys: "
            f"num_images_in_input={num_images_in_input}, image_keys={keys!r}"
        )
    token_count = patches_per_image * num_images_in_input
    return {
        "per_image": patches_per_image,
        "patches_per_image": patches_per_image,
        "views": num_images_in_input,
        "num_images_in_input": num_images_in_input,
        "token_dim": token_dim,
        "token_count": token_count,
        "flat_dim": token_count * token_dim,
    }


def _policy_device_from_id(device_ref: int | str | torch.device) -> torch.device:
    """Normalize a collector device reference into a torch device."""

    if isinstance(device_ref, torch.device):
        return device_ref
    value = str(device_ref).strip().lower()
    if value in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda"):
        return torch.device(value if ":" in value else "cuda:0")
    gpu_id = int(value)
    return torch.device("cpu" if gpu_id < 0 else f"cuda:{gpu_id}")


def load_policy(cfg: dict[str, Any], gpu_id: int | str | torch.device) -> Any:
    """Load OpenVLAOFTPolicy from checkpoint on the specified device.

    Under torchrun, gpu_id = LOCAL_RANK; CUDA_VISIBLE_DEVICES is NOT
    overridden (torchrun sets it per-process already via LOCAL_RANK env).
    """
    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()

    from dreamervla.models.embodiment.openvla_oft_policy import OpenVLAOFTPolicy

    model_path = resolve_model_path(cfg["model_path"])

    device = _policy_device_from_id(gpu_id)

    # Validate the only supported checkpoint mode before constructing the policy.
    from dreamervla.preprocess.preprocess_oft_hidden_token import resolve_oft_policy_mode

    mode = resolve_oft_policy_mode(model_path, str(cfg["policy_mode"]))
    use_proprio = False
    cfg["_policy_mode"] = mode
    cfg["_use_proprio"] = use_proprio

    print(
        f"[collector rank={cfg['_rank']}] Loading OFT policy ({mode}) from {model_path} on {device} ...",
        flush=True,
    )
    t0 = time.time()
    policy = OpenVLAOFTPolicy(
        model_path=model_path,
        torch_dtype="bf16",
        num_images_in_input=int(cfg["num_images_in_input"]),
        use_lora=False,
        use_l1_regression=False,
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
    obs_hidden_source = str(cfg["expected_obs_hidden_source"])
    if obs_hidden_source != "hidden_token":
        raise ValueError(
            "OpenVLA-OFT collection only supports "
            "expected_obs_hidden_source='hidden_token'; "
            f"got {obs_hidden_source!r}"
        )

    mode = cfg["_policy_mode"]
    if mode != "discrete":
        raise ValueError("L1/action-query checkpoints are closed")
    use_proprio = cfg["_use_proprio"]
    config = {
        "action_head_type": "oft_discrete_token",
        "obs_hidden_source": obs_hidden_source,
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
    token_count = int(cfg["token_count"])
    hidden_dim = int(cfg["hidden_dim"])
    patches_per_image = int(cfg["patches_per_image"])
    config["token_count"] = token_count
    config["obs_embedding_shape"] = [token_count, int(cfg["token_dim"])]
    config["hidden_dim"] = hidden_dim
    config["patches_per_image"] = patches_per_image
    config["hidden_storage_format"] = "tokenized"
    config["sidecar_schema_version"] = SIDECAR_SCHEMA_VERSION
    config["required_demo_datasets"] = required_demo_datasets()
    token_dim = int(cfg["token_dim"])
    num_images = int(cfg["num_images_in_input"])
    if token_count <= 0 or token_dim <= 0 or patches_per_image <= 0 or num_images <= 0:
        raise ValueError(
            "OpenVLA-OFT collection token geometry must be positive: "
            f"token_count={token_count}, token_dim={token_dim}, "
            f"patches_per_image={patches_per_image}, num_images_in_input={num_images}"
        )
    if token_count != patches_per_image * num_images:
        raise ValueError(
            "token_count must equal patches_per_image * num_images_in_input: "
            f"{token_count} != {patches_per_image} * {num_images}"
        )
    if hidden_dim != token_count * token_dim:
        raise ValueError(
            f"hidden_dim must be token_count * token_dim ({token_count * token_dim}), "
            f"got {hidden_dim}"
        )
    validate_hidden_token_preprocess_config(
        config,
        context="OpenVLA-OFT collection preprocess config",
    )
    return config


def assert_policy_mode_matches(cfg: dict[str, Any]) -> None:
    """Early validation: ckpt-detected mode == task expected_* (RLinf-style)."""
    if cfg["_policy_mode"] != "discrete":
        raise ValueError("L1/action-query checkpoints are closed")
    detected_head = "oft_discrete_token"
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
    """Resolve and validate the fixed one-image mainline deployment."""
    val = OmegaConf.select(collect_cfg, "num_images_in_input", default=None)
    count = int(val) if val is not None else 1
    if count != 1:
        raise ValueError(
            "OpenVLA-OFT hidden-token mainline requires num_images_in_input=1, "
            f"got {count}"
        )
    return count


def select_vla_image_keys(
    image_keys: list[str],
    *,
    history: int,
    num_images_in_input: int,
) -> list[str]:
    """Return the single primary camera admitted by the mainline."""
    keys = list(image_keys)
    if int(history) != 1:
        raise ValueError(
            "OpenVLA-OFT hidden-token mainline requires expected_history=1, "
            f"got {int(history)}"
        )
    if int(num_images_in_input) != 1:
        raise ValueError(
            "OpenVLA-OFT hidden-token mainline requires num_images_in_input=1, "
            f"got {int(num_images_in_input)}"
        )
    if keys != ["agentview_rgb"]:
        raise ValueError(
            "OpenVLA-OFT hidden-token mainline requires exactly one primary camera "
            f"'agentview_rgb', got {keys!r}"
        )
    return ["agentview_rgb"]
