#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from dreamervla.utils.paths import checkpoints_path, processed_data_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _demo_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        try:
            return int(name.split("_")[-1]), name
        except ValueError:
            pass
    return 10**9, name


def _list_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted((str(key) for key in data_group.keys()), key=_demo_sort_key)


def _task_prompt_from_path(path: str | Path) -> str:
    stem = Path(path).name
    if stem.endswith("_demo.hdf5"):
        stem = stem[: -len("_demo.hdf5")]
    else:
        stem = Path(stem).stem
    return stem.replace("_", " ")


def _init_distributed() -> tuple[int, int, torch.device]:
    manual_world_size = int(os.environ.get("MANUAL_SHARD_WORLD_SIZE", "1"))
    if manual_world_size > 1:
        rank = int(os.environ.get("MANUAL_SHARD_RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return rank, manual_world_size, device

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, device


def _is_complete_hdf5(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return bool(handle.attrs.get("complete", False))
    except OSError:
        return False


def _state_from_obs_group(obs_group: h5py.Group, index: int) -> np.ndarray:
    required = ("ee_pos", "ee_ori", "gripper_states")
    missing = [key for key in required if key not in obs_group]
    if missing:
        raise KeyError(f"missing state keys under obs: {missing}")
    return np.concatenate(
        [
            np.asarray(obs_group["ee_pos"][index], dtype=np.float32).reshape(-1),
            np.asarray(obs_group["ee_ori"][index], dtype=np.float32).reshape(-1),
            np.asarray(obs_group["gripper_states"][index], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )


def _history_indices(index: int, history: int) -> list[int]:
    history = max(1, int(history))
    start = max(0, int(index) - history + 1)
    indices = list(range(start, int(index) + 1))
    if len(indices) < history:
        indices = [indices[0]] * (history - len(indices)) + indices
    return indices


def _image_from_hdf5(
    obs_group: h5py.Group, key: str, index: int, *, rotate_images_180: bool
) -> np.ndarray:
    image = np.asarray(obs_group[key][index], dtype=np.uint8)
    if bool(rotate_images_180):
        image = image[::-1, ::-1].copy()
    return image


def _normalize_proprio(proprio: np.ndarray, norm_stats: dict[str, Any]) -> np.ndarray:
    from openvla_oft.constants import (
        ACTION_PROPRIO_NORMALIZATION_TYPE,
        NormalizationType,
    )

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = (
            np.array(norm_stats["max"]),
            np.array(norm_stats["min"]),
        )
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = (
            np.array(norm_stats["q99"]),
            np.array(norm_stats["q01"]),
        )
    else:
        raise ValueError("Unsupported action/proprio normalization type detected")
    return np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )


def _center_crop_image(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    width, height = image.size
    ratio = float(crop_scale) ** 0.5
    crop_width = max(1, int(width * ratio))
    crop_height = max(1, int(height * ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    image = image.crop((left, top, left + crop_width, top + crop_height))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def _prepare_images_for_vla(images: list[np.ndarray], cfg: Any) -> list[Image.Image]:
    processed_images: list[Image.Image] = []
    for image in images:
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[-1] != 3
            or image.dtype != np.uint8
        ):
            raise ValueError("OpenVLA-OFT images must be uint8 arrays with shape (H, W, 3)")
        pil_image = Image.fromarray(image).convert("RGB")
        if pil_image.size != (224, 224):
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        if bool(cfg.center_crop):
            pil_image = _center_crop_image(pil_image)
        processed_images.append(pil_image)
    return processed_images


def resolve_oft_policy_mode(checkpoint: str | Path, policy_mode: str = "auto") -> str:
    """Resolve the OFT action-head mode for a checkpoint directory.

    ``l1`` is the component-wise OFT format and requires
    ``action_head--*_checkpoint.pt``; ``discrete`` is the merged LM-head
    action-token format with no component files. ``auto`` probes for the
    action-head component, mirroring the eval-side detection.
    """
    mode = str(policy_mode).lower()
    if mode not in {"auto", "l1", "discrete"}:
        raise ValueError(f"Unsupported policy mode: {policy_mode!r}")
    checkpoint = Path(checkpoint).expanduser()
    has_action_head = bool(sorted(checkpoint.glob("action_head--*_checkpoint.pt")))
    if mode == "auto":
        return "l1" if has_action_head else "discrete"
    if mode == "l1" and not has_action_head:
        raise FileNotFoundError(
            f"Missing action_head--*_checkpoint.pt under {checkpoint}; "
            "use --policy-mode discrete for merged LM-head checkpoints."
        )
    return mode


def _resolve_num_images_in_input(args: argparse.Namespace) -> int:
    if args.num_images_in_input is not None:
        return int(args.num_images_in_input)
    return int(args.history) * len(args.image_keys)


def _load_oft_components(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if bool(args.load_in_8bit) or bool(args.load_in_4bit):
        raise NotImplementedError(
            "The lightweight OpenVLA-OFT loader does not support 8-bit or 4-bit loading."
        )

    from dreamervla.models.encoder.openvla_oft_policy import OpenVLAOFTPolicy
    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()
    mode = resolve_oft_policy_mode(
        _project_path(args.oft_ckpt), getattr(args, "policy_mode", "auto")
    )
    use_l1_regression = mode == "l1"
    use_proprio = bool(args.include_state) and use_l1_regression
    cfg = SimpleNamespace(
        pretrained_checkpoint=str(_project_path(args.oft_ckpt)),
        use_l1_regression=use_l1_regression,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=_resolve_num_images_in_input(args),
        use_proprio=use_proprio,
        center_crop=bool(args.center_crop),
        unnorm_key=str(args.unnorm_key),
    )
    policy = OpenVLAOFTPolicy(
        model_path=cfg.pretrained_checkpoint,
        component_ckpt_dir=cfg.pretrained_checkpoint,
        torch_dtype="bf16",
        num_images_in_input=cfg.num_images_in_input,
        use_lora=False,
        use_l1_regression=use_l1_regression,
        use_diffusion=False,
        use_proprio=cfg.use_proprio,
        use_film=False,
        freeze_vla_backbone=True,
    )
    policy.eval()
    policy.to(device)
    vla = policy.vla
    if getattr(vla, "norm_stats", None) is None:
        stats_path = Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
        if stats_path.is_file():
            with stats_path.open("r", encoding="utf-8") as handle:
                vla.norm_stats = json.load(handle)
    return {
        "cfg": cfg,
        "mode": mode,
        "vla": vla,
        "processor": policy.processor,
        "action_head": policy.action_head,
        "proprio_projector": policy.proprio_projector,
        "normalize_proprio": _normalize_proprio,
        "prepare_images_for_vla": _prepare_images_for_vla,
        "device": device,
    }


def _predict_intermediates_chunk(
    *,
    components: dict[str, Any],
    args: argparse.Namespace,
    obs_group: h5py.Group,
    image_keys: tuple[str, ...],
    prompt: str,
    start: int,
    end: int,
    want_action: bool = True,
    want_input_tokens: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    cfg = components["cfg"]
    vla = components["vla"]
    processor = components["processor"]
    action_head = components["action_head"]
    proprio_projector = components["proprio_projector"]
    normalize_proprio = components["normalize_proprio"]
    prepare_images_for_vla = components["prepare_images_for_vla"]
    device = components["device"]

    model_prompt = f"In: What action should the robot take to {prompt.lower()}?\nOut:"
    input_ids: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    primary_pixels: list[torch.Tensor] = []
    extra_pixels_by_view: list[list[torch.Tensor]] = []
    proprio_values: list[np.ndarray] = []

    for index in range(start, end):
        images = [
            _image_from_hdf5(obs_group, key, hidx, rotate_images_180=bool(args.rotate_images_180))
            for hidx in _history_indices(index, int(args.history))
            for key in image_keys
        ]
        images = prepare_images_for_vla(images, cfg)
        primary_inputs = processor(model_prompt, images[0]).to(device, dtype=torch.bfloat16)
        input_ids.append(primary_inputs["input_ids"])
        attention_masks.append(primary_inputs["attention_mask"])
        primary_pixels.append(primary_inputs["pixel_values"])
        for view_idx, image in enumerate(images[1:]):
            while len(extra_pixels_by_view) <= view_idx:
                extra_pixels_by_view.append([])
            extra_inputs = processor(model_prompt, image).to(device, dtype=torch.bfloat16)
            extra_pixels_by_view[view_idx].append(extra_inputs["pixel_values"])
        if bool(cfg.use_proprio):
            proprio = _state_from_obs_group(obs_group, index)
            proprio_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            proprio_values.append(normalize_proprio(proprio, proprio_stats))

    pixel_values = torch.cat(primary_pixels, dim=0)
    if extra_pixels_by_view:
        extra_batches = [torch.cat(view_pixels, dim=0) for view_pixels in extra_pixels_by_view]
        pixel_values = torch.cat([pixel_values] + extra_batches, dim=1)
    inputs = {
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_masks, dim=0),
        "pixel_values": pixel_values,
    }

    proprio_batch = None
    if bool(cfg.use_proprio):
        proprio_batch = np.stack(proprio_values, axis=0).astype(np.float32, copy=False)

    with torch.inference_mode():
        from openvla_oft.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK

        input_ids_tensor = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if not torch.all(input_ids_tensor[:, -1] == 29871):
            empty_token = torch.full(
                (input_ids_tensor.shape[0], 1),
                29871,
                dtype=input_ids_tensor.dtype,
                device=input_ids_tensor.device,
            )
            input_ids_tensor = torch.cat((input_ids_tensor, empty_token), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones_like(empty_token, dtype=attention_mask.dtype),
                ),
                dim=1,
            )

        labels = input_ids_tensor.clone()
        labels[:] = IGNORE_INDEX
        num_prompt_tokens = input_ids_tensor.shape[-1] - 1
        input_ids_tensor, attention_mask = vla._prepare_input_for_action_prediction(
            input_ids_tensor, attention_mask
        )
        labels = vla._prepare_labels_for_action_prediction(labels, input_ids_tensor)
        input_embeddings = vla.get_input_embeddings()(input_ids_tensor)
        all_actions_mask = vla._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0],
            -1,
            input_embeddings.shape[2],
        )
        projected_patch_embeddings = vla._process_vision_features(
            inputs["pixel_values"], language_embeddings, False
        )
        input_token_emb = None
        if want_input_tokens:
            # Scheme-B frame latent: projected vision patch tokens of the CURRENT
            # frame's views (input order is [history ... current] x views).
            per_image = int(vla.vision_backbone.get_num_patches())
            current_tokens = projected_patch_embeddings[:, -per_image * len(image_keys) :, :]
            input_token_emb = (
                current_tokens.reshape(current_tokens.shape[0], -1).float().cpu().numpy()
            )
        if not want_action:
            return None, None, None, input_token_emb
        if proprio_projector is not None and proprio_batch is not None:
            proprio_tensor = torch.as_tensor(
                proprio_batch,
                device=projected_patch_embeddings.device,
                dtype=projected_patch_embeddings.dtype,
            )
            projected_patch_embeddings = vla._process_proprio_features(
                projected_patch_embeddings,
                proprio_tensor,
                proprio_projector,
            )
        num_patches = (
            vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
        )
        if proprio_projector is not None and proprio_batch is not None:
            num_patches += 1
        if action_head is not None:
            _normalized_actions, actions_hidden_states, hidden_c, hidden_d = (
                vla._regression_or_discrete_prediction(
                    input_embeddings,
                    all_actions_mask,
                    projected_patch_embeddings,
                    attention_mask,
                    labels,
                    num_patches,
                    num_prompt_tokens,
                    action_head,
                    return_intermediates=True,
                )
            )
        else:
            # Discrete LM-head checkpoint: the action hidden states come from the
            # same backbone layer, but there is no L1 MLP head, hence no C/D
            # intermediates.
            _normalized_actions, actions_hidden_states = vla._regression_or_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                num_patches,
                num_prompt_tokens,
                action_head=None,
            )
            hidden_c = None
            hidden_d = None
        expected = int(NUM_ACTIONS_CHUNK * ACTION_DIM)
        if int(actions_hidden_states.shape[1]) != expected:
            raise RuntimeError(
                f"Unexpected OFT action hidden token count: {actions_hidden_states.shape[1]} != {expected}"
            )
    return (
        None if hidden_c is None else hidden_c.reshape(hidden_c.shape[0], -1).float().cpu().numpy(),
        None if hidden_d is None else hidden_d.reshape(hidden_d.shape[0], -1).float().cpu().numpy(),
        actions_hidden_states.float().cpu().numpy(),
        input_token_emb,
    )


def _action_head_type_for_mode(mode: str) -> str:
    return "oft_l1_regression" if str(mode) == "l1" else "oft_discrete_token"


def _input_token_sidecar_dims(
    vla: Any, *, image_keys: Sequence[str], token_dim: int
) -> tuple[int, int]:
    per_image_patches = int(vla.vision_backbone.get_num_patches())
    token_count = per_image_patches * len(list(image_keys))
    return token_count, token_count * int(token_dim)


def _write_attrs(
    handle: h5py.File,
    args: argparse.Namespace,
    *,
    source_path: Path,
    obs_hidden_source: str,
    hidden_dim: int,
    action_hidden_sequence_dim: int = 0,
    action_hidden_dim: int = 0,
) -> None:
    handle.attrs["complete"] = False
    handle.attrs["source_hdf5"] = str(source_path)
    handle.attrs["source_hdf5_dir"] = str(source_path.parent)
    handle.attrs["hidden_key"] = str(args.hidden_key)
    handle.attrs["hidden_dim"] = int(hidden_dim)
    handle.attrs["obs_hidden_source"] = str(obs_hidden_source)
    handle.attrs["output_dtype"] = str(np.dtype(args.output_dtype))
    handle.attrs["image_keys"] = json.dumps(list(args.image_keys))
    handle.attrs["prompt_style"] = str(args.prompt_style)
    handle.attrs["history"] = int(args.history)
    handle.attrs["include_state"] = bool(args.include_state)
    handle.attrs["rotate_images_180"] = bool(args.rotate_images_180)
    handle.attrs["resolution"] = int(args.resolution)
    handle.attrs["model_path"] = str(_project_path(args.oft_ckpt))
    handle.attrs["encoder_state_ckpt"] = ""
    handle.attrs["pool"] = "none"
    handle.attrs["action_head_type"] = _action_head_type_for_mode(
        getattr(args, "resolved_policy_mode", "l1")
    )
    handle.attrs["save_actor_sequence"] = False
    handle.attrs["save_action_hidden"] = bool(args.save_action_hidden)
    handle.attrs["action_trigger_token_id"] = -1
    handle.attrs["actor_sequence_dim"] = 0
    handle.attrs["actor_hidden_dim"] = 0
    handle.attrs["action_hidden_sequence_dim"] = int(action_hidden_sequence_dim)
    handle.attrs["action_hidden_dim"] = int(action_hidden_dim)
    handle.attrs["time_horizon"] = int(args.time_horizon)
    handle.attrs["token_dim"] = int(args.token_dim)
    handle.attrs["chunk_size"] = int(args.chunk_size)


def _write_source_sidecars(
    *,
    source_path: Path,
    out_c_path: Path | None,
    out_d_path: Path | None,
    out_action_path: Path | None,
    out_input_path: Path | None = None,
    components: dict[str, Any],
    args: argparse.Namespace,
    rank: int,
) -> dict[str, int]:
    tmp_c = (
        None if out_c_path is None else out_c_path.with_name(f"{out_c_path.name}.rank{rank}.tmp")
    )
    tmp_d = (
        None if out_d_path is None else out_d_path.with_name(f"{out_d_path.name}.rank{rank}.tmp")
    )
    tmp_action = (
        None
        if out_action_path is None
        else out_action_path.with_name(f"{out_action_path.name}.rank{rank}.tmp")
    )
    tmp_input = (
        None
        if out_input_path is None
        else out_input_path.with_name(f"{out_input_path.name}.rank{rank}.tmp")
    )
    for tmp in (tmp_c, tmp_d, tmp_action, tmp_input):
        if tmp is None:
            continue
        if tmp.exists():
            tmp.unlink()

    image_keys = tuple(args.image_keys)
    cd_hidden_dim = int(args.time_horizon * args.token_dim)
    action_hidden_seq_len = int(args.time_horizon * args.action_dim)
    action_hidden_dim = int(args.token_dim)
    action_flat_dim = int(action_hidden_seq_len * action_hidden_dim)
    dtype = np.dtype(args.output_dtype)
    prompt = _task_prompt_from_path(source_path)
    demos_written = 0
    frames_written = 0

    with ExitStack() as stack:
        source = stack.enter_context(h5py.File(source_path, "r", swmr=True, libver="latest"))
        out_c = (
            None if tmp_c is None else stack.enter_context(h5py.File(tmp_c, "w", libver="latest"))
        )
        out_d = (
            None if tmp_d is None else stack.enter_context(h5py.File(tmp_d, "w", libver="latest"))
        )
        out_action = (
            None
            if tmp_action is None
            else stack.enter_context(h5py.File(tmp_action, "w", libver="latest"))
        )
        out_input = (
            None
            if tmp_input is None
            else stack.enter_context(h5py.File(tmp_input, "w", libver="latest"))
        )
        input_token_count, input_flat_dim = _input_token_sidecar_dims(
            components["vla"], image_keys=image_keys, token_dim=int(args.token_dim)
        )
        if out_c is not None:
            _write_attrs(
                out_c,
                args,
                source_path=source_path,
                obs_hidden_source="oft_mlpresnet_fc1_relu",
                hidden_dim=cd_hidden_dim,
            )
        if out_d is not None:
            _write_attrs(
                out_d,
                args,
                source_path=source_path,
                obs_hidden_source="oft_mlpresnet_post_resblocks",
                hidden_dim=cd_hidden_dim,
            )
        if out_action is not None:
            _write_attrs(
                out_action,
                args,
                source_path=source_path,
                obs_hidden_source="action_query",
                hidden_dim=action_flat_dim,
                action_hidden_sequence_dim=action_hidden_seq_len,
                action_hidden_dim=action_hidden_dim,
            )
        if out_input is not None:
            _write_attrs(
                out_input,
                args,
                source_path=source_path,
                obs_hidden_source="input_token_embedding",
                hidden_dim=input_flat_dim,
            )
            out_input.attrs["save_action_hidden"] = False
            out_input.attrs["token_count"] = int(input_token_count)
        data_group = source["data"]
        out_c_data = None if out_c is None else out_c.create_group("data")
        out_d_data = None if out_d is None else out_d.create_group("data")
        out_action_data = None if out_action is None else out_action.create_group("data")
        out_input_data = None if out_input is None else out_input.create_group("data")
        demo_keys = _list_demo_keys(data_group)
        if args.max_demos_per_file is not None:
            demo_keys = demo_keys[: int(args.max_demos_per_file)]

        for demo_key in tqdm(demo_keys, desc=f"rank{rank} {source_path.name}", leave=False):
            demo = data_group[demo_key]
            obs_group = demo["obs"]
            for key in image_keys:
                if key not in obs_group:
                    raise KeyError(f"{source_path}:{demo_key} missing obs/{key}")
            length = int(demo["actions"].shape[0])
            demo_c = None if out_c_data is None else out_c_data.create_group(demo_key)
            demo_d = None if out_d_data is None else out_d_data.create_group(demo_key)
            demo_action = (
                None if out_action_data is None else out_action_data.create_group(demo_key)
            )
            for demo_out in (demo_c, demo_d):
                if demo_out is not None:
                    demo_out.attrs["length"] = length
                    demo_out.attrs["task_prompt"] = prompt
            if demo_action is not None:
                demo_action.attrs["length"] = length
                demo_action.attrs["task_prompt"] = prompt
            demo_input = None if out_input_data is None else out_input_data.create_group(demo_key)
            if demo_input is not None:
                demo_input.attrs["length"] = length
                demo_input.attrs["task_prompt"] = prompt
            dset_c = (
                None
                if demo_c is None
                else demo_c.create_dataset(
                    args.hidden_key,
                    shape=(length, cd_hidden_dim),
                    dtype=dtype,
                    chunks=(min(max(1, int(args.chunk_size)), length), cd_hidden_dim),
                    compression=None,
                )
            )
            dset_d = (
                None
                if demo_d is None
                else demo_d.create_dataset(
                    args.hidden_key,
                    shape=(length, cd_hidden_dim),
                    dtype=dtype,
                    chunks=(min(max(1, int(args.chunk_size)), length), cd_hidden_dim),
                    compression=None,
                )
            )
            for dset in (dset_c, dset_d):
                if dset is not None:
                    dset.attrs["hidden_dim"] = cd_hidden_dim
                    dset.attrs["source_dtype"] = "float32"
                    dset.attrs["sequence_dim"] = int(args.time_horizon)
                    dset.attrs["token_dim"] = int(args.token_dim)
            action_embedding_dset = None
            action_hidden_dset = None
            if demo_action is not None:
                action_embedding_dset = demo_action.create_dataset(
                    args.hidden_key,
                    shape=(length, action_flat_dim),
                    dtype=dtype,
                    chunks=(min(max(1, int(args.chunk_size)), length), action_flat_dim),
                    compression=None,
                )
                action_hidden_dset = demo_action.create_dataset(
                    "action_hidden_states",
                    shape=(length, action_hidden_seq_len, action_hidden_dim),
                    dtype=dtype,
                    chunks=(1, action_hidden_seq_len, action_hidden_dim),
                    compression=None,
                )
                action_embedding_dset.attrs["hidden_dim"] = action_flat_dim
                action_embedding_dset.attrs["source_dtype"] = "float32"
                action_hidden_dset.attrs["hidden_dim"] = action_hidden_dim
                action_hidden_dset.attrs["source_dtype"] = "float32"
                action_hidden_dset.attrs["sequence_dim"] = action_hidden_seq_len
            input_dset = None
            if demo_input is not None:
                input_dset = demo_input.create_dataset(
                    args.hidden_key,
                    shape=(length, input_flat_dim),
                    dtype=dtype,
                    chunks=(min(max(1, int(args.chunk_size)), length), input_flat_dim),
                    compression=None,
                )
                input_dset.attrs["hidden_dim"] = input_flat_dim
                input_dset.attrs["source_dtype"] = "float32"
                input_dset.attrs["token_count"] = int(input_token_count)
                input_dset.attrs["token_dim"] = int(args.token_dim)

            for start in range(0, length, int(args.chunk_size)):
                end = min(start + int(args.chunk_size), length)
                hidden_c, hidden_d, action_hidden, input_tokens = _predict_intermediates_chunk(
                    components=components,
                    args=args,
                    obs_group=obs_group,
                    image_keys=image_keys,
                    prompt=prompt,
                    start=start,
                    end=end,
                    want_action=(
                        dset_c is not None
                        or dset_d is not None
                        or action_embedding_dset is not None
                    ),
                    want_input_tokens=input_dset is not None,
                )
                if dset_c is not None:
                    dset_c[start:end] = hidden_c.astype(dtype, copy=False)
                if dset_d is not None:
                    dset_d[start:end] = hidden_d.astype(dtype, copy=False)
                if action_embedding_dset is not None and action_hidden_dset is not None:
                    action_embedding_dset[start:end] = action_hidden.reshape(
                        action_hidden.shape[0], -1
                    ).astype(dtype, copy=False)
                    action_hidden_dset[start:end] = action_hidden.astype(dtype, copy=False)
                if input_dset is not None and input_tokens is not None:
                    input_dset[start:end] = input_tokens.astype(dtype, copy=False)
                frames_written += int(end - start)
            demos_written += 1

        if out_c is not None:
            out_c.attrs["complete"] = True
        if out_d is not None:
            out_d.attrs["complete"] = True
        if out_action is not None:
            out_action.attrs["complete"] = True
        if out_input is not None:
            out_input.attrs["complete"] = True

    if tmp_c is not None and out_c_path is not None:
        tmp_c.replace(out_c_path)
    if tmp_d is not None and out_d_path is not None:
        tmp_d.replace(out_d_path)
    if tmp_action is not None and out_action_path is not None:
        tmp_action.replace(out_action_path)
    if tmp_input is not None and out_input_path is not None:
        tmp_input.replace(out_input_path)
    return {"demos": demos_written, "frames": frames_written}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute OpenVLA-OFT action hidden C/D sidecars for LIBERO HDF5."
    )
    parser.add_argument(
        "--openvla-oft-dir",
        default=str(
            PROJECT_ROOT / "dreamervla" / "models" / "embodiment" / "openvla_oft"
        ),
    )
    parser.add_argument(
        "--hdf5-dir",
        default=str(processed_data_path("libero_goal/no_noops_t_256")),
    )
    parser.add_argument(
        "--out-c-dir",
        default=str(processed_data_path("libero_goal/no_noops_t_256_oft_action_hidden_c_h8")),
    )
    parser.add_argument(
        "--out-d-dir",
        default=str(processed_data_path("libero_goal/no_noops_t_256_oft_action_hidden_d_h8")),
    )
    parser.add_argument("--out-action-dir", default=None)
    parser.add_argument(
        "--out-input-token-dir",
        default=None,
        help="Optional Scheme-B sidecar: current-frame projected vision patch "
        "tokens (input embeddings) instead of action-slot hidden states.",
    )
    parser.add_argument("--skip-cd-sidecars", action="store_true")
    parser.add_argument("--oft-ckpt", default=str(checkpoints_path("OpenVLA-OFT", "libero_goal")))
    parser.add_argument("--unnorm-key", default="libero_goal_no_noops")
    parser.add_argument(
        "--policy-mode",
        default="auto",
        choices=["auto", "l1", "discrete"],
        help="OFT action-head format: component-wise L1 head or merged discrete "
        "LM-head; auto probes for action_head--*_checkpoint.pt.",
    )
    parser.add_argument("--image-keys", nargs="+", default=["agentview_rgb", "eye_in_hand_rgb"])
    parser.add_argument(
        "--num-images-in-input",
        type=int,
        default=None,
        help="Defaults to history * len(image_keys).",
    )
    parser.add_argument("--include-state", action="store_true", default=True)
    parser.add_argument("--center-crop", action="store_true", default=True)
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--rotate-images-180", action="store_true", default=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--prompt-style", default="vla_policy")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--hidden-key", default="obs_embedding")
    parser.add_argument("--time-horizon", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--token-dim", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--save-action-hidden", action="store_true", default=True)
    parser.add_argument("--output-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-demos-per-file", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.resolved_policy_mode = resolve_oft_policy_mode(
        _project_path(args.oft_ckpt), args.policy_mode
    )
    if args.resolved_policy_mode == "discrete":
        if not args.skip_cd_sidecars:
            raise SystemExit(
                "C/D sidecars are L1 action-head MLP intermediates and do not "
                "exist for discrete LM-head checkpoints; pass --skip-cd-sidecars."
            )
        # Merged discrete checkpoints carry no proprio_projector component.
        args.include_state = False
    hdf5_dir = _project_path(args.hdf5_dir)
    out_c_dir = None if args.skip_cd_sidecars else _project_path(args.out_c_dir)
    out_d_dir = None if args.skip_cd_sidecars else _project_path(args.out_d_dir)
    out_action_dir = None if args.out_action_dir is None else _project_path(args.out_action_dir)
    out_input_dir = (
        None if args.out_input_token_dir is None else _project_path(args.out_input_token_dir)
    )
    if out_c_dir is None and out_d_dir is None and out_action_dir is None and out_input_dir is None:
        raise SystemExit(
            "No output requested: set --out-action-dir and/or --out-input-token-dir "
            "(or drop --skip-cd-sidecars)."
        )
    if out_c_dir is not None:
        out_c_dir.mkdir(parents=True, exist_ok=True)
    if out_d_dir is not None:
        out_d_dir.mkdir(parents=True, exist_ok=True)
    if out_action_dir is not None:
        out_action_dir.mkdir(parents=True, exist_ok=True)
    if out_input_dir is not None:
        out_input_dir.mkdir(parents=True, exist_ok=True)
    rank, world_size, device = _init_distributed()

    files = sorted(hdf5_dir.glob("*.hdf5"))
    if args.max_files is not None:
        files = files[: int(args.max_files)]
    if not files:
        raise RuntimeError(f"No HDF5 files found under {hdf5_dir}")
    assigned = files[rank::world_size]
    if rank == 0:
        base_config = vars(args).copy()
        base_config["hdf5_dir"] = str(hdf5_dir)
        base_config["out_c_dir"] = None if out_c_dir is None else str(out_c_dir)
        base_config["out_d_dir"] = None if out_d_dir is None else str(out_d_dir)
        base_config["out_action_dir"] = None if out_action_dir is None else str(out_action_dir)
        base_config["out_input_token_dir"] = None if out_input_dir is None else str(out_input_dir)
        base_config["world_size"] = world_size
        base_config["start_time"] = time.time()
        base_config["model_path"] = str(_project_path(args.oft_ckpt))
        base_config["encoder_state_ckpt"] = ""
        base_config["action_head_type"] = _action_head_type_for_mode(args.resolved_policy_mode)
        base_config["prompt_style"] = str(args.prompt_style)
        base_config["history"] = int(args.history)
        base_config["include_state"] = bool(args.include_state)
        base_config["rotate_images_180"] = bool(args.rotate_images_180)
        config_c = dict(base_config, obs_hidden_source="oft_mlpresnet_fc1_relu")
        config_d = dict(base_config, obs_hidden_source="oft_mlpresnet_post_resblocks")
        if out_c_dir is not None:
            (out_c_dir / "preprocess_config.json").write_text(
                json.dumps(config_c, indent=2, sort_keys=True) + "\n"
            )
        if out_d_dir is not None:
            (out_d_dir / "preprocess_config.json").write_text(
                json.dumps(config_d, indent=2, sort_keys=True) + "\n"
            )
        if out_action_dir is not None:
            config_action = dict(base_config, obs_hidden_source="action_query")
            (out_action_dir / "preprocess_config.json").write_text(
                json.dumps(config_action, indent=2, sort_keys=True) + "\n"
            )
        if out_input_dir is not None:
            config_input = dict(
                base_config,
                obs_hidden_source="input_token_embedding",
                save_action_hidden=False,
            )
            (out_input_dir / "preprocess_config.json").write_text(
                json.dumps(config_input, indent=2, sort_keys=True) + "\n"
            )
        print(
            f"[oft-hidden] source={hdf5_dir} files={len(files)} assigned/rank={len(assigned)} "
            f"world_size={world_size} policy_mode={args.resolved_policy_mode}"
        )

    using_torch_dist = bool(world_size > 1 and dist.is_available() and dist.is_initialized())
    if using_torch_dist and rank != 0:
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
    components = _load_oft_components(args, device)
    if using_torch_dist:
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
    total_demos = 0
    total_frames = 0
    for source_path in assigned:
        out_c_path = None if out_c_dir is None else out_c_dir / source_path.name
        out_d_path = None if out_d_dir is None else out_d_dir / source_path.name
        out_action_path = None if out_action_dir is None else out_action_dir / source_path.name
        out_input_path = None if out_input_dir is None else out_input_dir / source_path.name
        existing_paths = [
            path
            for path in (out_c_path, out_d_path, out_action_path, out_input_path)
            if path is not None
        ]
        if any(path.exists() for path in existing_paths):
            if args.overwrite:
                for path in existing_paths:
                    if path.exists():
                        path.unlink()
            elif all(_is_complete_hdf5(path) for path in existing_paths):
                print(f"[rank{rank}] skip complete {source_path.name}")
                continue
            else:
                raise RuntimeError(
                    f"Refusing to use incomplete sidecar without --overwrite: {existing_paths}"
                )
        stats = _write_source_sidecars(
            source_path=source_path,
            out_c_path=out_c_path,
            out_d_path=out_d_path,
            out_action_path=out_action_path,
            out_input_path=out_input_path,
            components=components,
            args=args,
            rank=rank,
        )
        total_demos += stats["demos"]
        total_frames += stats["frames"]
        print(
            f"[rank{rank}] wrote {source_path.name}: demos={stats['demos']} frames={stats['frames']}"
        )
    print(f"[rank{rank}] done demos={total_demos} frames={total_frames}")


if __name__ == "__main__":
    main()
