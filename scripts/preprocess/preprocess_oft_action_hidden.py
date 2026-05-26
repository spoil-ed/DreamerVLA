#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _image_from_hdf5(obs_group: h5py.Group, key: str, index: int, *, rotate_images_180: bool) -> np.ndarray:
    image = np.asarray(obs_group[key][index], dtype=np.uint8)
    if bool(rotate_images_180):
        image = image[::-1, ::-1].copy()
    return image


def _load_oft_components(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    openvla_dir = _project_path(args.openvla_oft_dir)
    if str(openvla_dir) not in sys.path:
        sys.path.insert(0, str(openvla_dir))

    previous_cwd = Path.cwd()
    os.chdir(openvla_dir)
    try:
        import experiments.robot.openvla_utils as openvla_utils

        openvla_utils.DEVICE = device

        cfg = SimpleNamespace(
            pretrained_checkpoint=str(_project_path(args.oft_ckpt)),
            use_l1_regression=True,
            use_diffusion=False,
            num_diffusion_steps_train=50,
            num_diffusion_steps_inference=50,
            use_film=False,
            num_images_in_input=int(args.num_images_in_input),
            use_proprio=bool(args.include_state),
            center_crop=bool(args.center_crop),
            lora_rank=int(args.lora_rank),
            unnorm_key=str(args.unnorm_key),
            load_in_8bit=bool(args.load_in_8bit),
            load_in_4bit=bool(args.load_in_4bit),
        )
        vla = openvla_utils.get_vla(cfg)
        processor = openvla_utils.get_processor(cfg)
        action_head = openvla_utils.get_action_head(cfg, vla.llm_dim)
        proprio_projector = None
        if bool(args.include_state):
            from prismatic.vla.constants import PROPRIO_DIM

            proprio_projector = openvla_utils.get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM)
        return {
            "cfg": cfg,
            "vla": vla,
            "processor": processor,
            "action_head": action_head,
            "proprio_projector": proprio_projector,
            "normalize_proprio": openvla_utils.normalize_proprio,
            "prepare_images_for_vla": openvla_utils.prepare_images_for_vla,
            "device": device,
        }
    finally:
        os.chdir(previous_cwd)


def _predict_intermediates_chunk(
    *,
    components: dict[str, Any],
    args: argparse.Namespace,
    obs_group: h5py.Group,
    image_keys: tuple[str, ...],
    prompt: str,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, IGNORE_INDEX

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
            attention_mask = torch.cat((attention_mask, torch.ones_like(empty_token, dtype=attention_mask.dtype)), dim=1)

        labels = input_ids_tensor.clone()
        labels[:] = IGNORE_INDEX
        num_prompt_tokens = input_ids_tensor.shape[-1] - 1
        input_ids_tensor, attention_mask = vla._prepare_input_for_action_prediction(input_ids_tensor, attention_mask)
        labels = vla._prepare_labels_for_action_prediction(labels, input_ids_tensor)
        input_embeddings = vla.get_input_embeddings()(input_ids_tensor)
        all_actions_mask = vla._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0],
            -1,
            input_embeddings.shape[2],
        )
        projected_patch_embeddings = vla._process_vision_features(inputs["pixel_values"], language_embeddings, False)
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
        num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
        if proprio_projector is not None and proprio_batch is not None:
            num_patches += 1
        _normalized_actions, actions_hidden_states, hidden_c, hidden_d = vla._regression_or_discrete_prediction(
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
        expected = int(NUM_ACTIONS_CHUNK * ACTION_DIM)
        if int(actions_hidden_states.shape[1]) != expected:
            raise RuntimeError(f"Unexpected OFT action hidden token count: {actions_hidden_states.shape[1]} != {expected}")
    return (
        hidden_c.reshape(hidden_c.shape[0], -1).float().cpu().numpy(),
        hidden_d.reshape(hidden_d.shape[0], -1).float().cpu().numpy(),
        actions_hidden_states.float().cpu().numpy(),
    )


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
    handle.attrs["action_head_type"] = "oft_l1_regression"
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
    components: dict[str, Any],
    args: argparse.Namespace,
    rank: int,
) -> dict[str, int]:
    tmp_c = None if out_c_path is None else out_c_path.with_name(f"{out_c_path.name}.rank{rank}.tmp")
    tmp_d = None if out_d_path is None else out_d_path.with_name(f"{out_d_path.name}.rank{rank}.tmp")
    tmp_action = None if out_action_path is None else out_action_path.with_name(f"{out_action_path.name}.rank{rank}.tmp")
    for tmp in (tmp_c, tmp_d, tmp_action):
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
        out_c = None if tmp_c is None else stack.enter_context(h5py.File(tmp_c, "w", libver="latest"))
        out_d = None if tmp_d is None else stack.enter_context(h5py.File(tmp_d, "w", libver="latest"))
        out_action = None if tmp_action is None else stack.enter_context(h5py.File(tmp_action, "w", libver="latest"))
        if out_c is not None:
            _write_attrs(out_c, args, source_path=source_path, obs_hidden_source="oft_mlpresnet_fc1_relu", hidden_dim=cd_hidden_dim)
        if out_d is not None:
            _write_attrs(out_d, args, source_path=source_path, obs_hidden_source="oft_mlpresnet_post_resblocks", hidden_dim=cd_hidden_dim)
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
        data_group = source["data"]
        out_c_data = None if out_c is None else out_c.create_group("data")
        out_d_data = None if out_d is None else out_d.create_group("data")
        out_action_data = None if out_action is None else out_action.create_group("data")
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
            demo_action = None if out_action_data is None else out_action_data.create_group(demo_key)
            for demo_out in (demo_c, demo_d):
                if demo_out is not None:
                    demo_out.attrs["length"] = length
                    demo_out.attrs["task_prompt"] = prompt
            if demo_action is not None:
                demo_action.attrs["length"] = length
                demo_action.attrs["task_prompt"] = prompt
            dset_c = None if demo_c is None else demo_c.create_dataset(
                    args.hidden_key,
                    shape=(length, cd_hidden_dim),
                    dtype=dtype,
                    chunks=(min(max(1, int(args.chunk_size)), length), cd_hidden_dim),
                    compression=None,
                )
            dset_d = None if demo_d is None else demo_d.create_dataset(
                    args.hidden_key,
                    shape=(length, cd_hidden_dim),
                    dtype=dtype,
                    chunks=(min(max(1, int(args.chunk_size)), length), cd_hidden_dim),
                    compression=None,
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

            for start in range(0, length, int(args.chunk_size)):
                end = min(start + int(args.chunk_size), length)
                hidden_c, hidden_d, action_hidden = _predict_intermediates_chunk(
                    components=components,
                    args=args,
                    obs_group=obs_group,
                    image_keys=image_keys,
                    prompt=prompt,
                    start=start,
                    end=end,
                )
                if dset_c is not None:
                    dset_c[start:end] = hidden_c.astype(dtype, copy=False)
                if dset_d is not None:
                    dset_d[start:end] = hidden_d.astype(dtype, copy=False)
                if action_embedding_dset is not None and action_hidden_dset is not None:
                    action_embedding_dset[start:end] = action_hidden.reshape(action_hidden.shape[0], -1).astype(dtype, copy=False)
                    action_hidden_dset[start:end] = action_hidden.astype(dtype, copy=False)
                frames_written += int(end - start)
            demos_written += 1

        if out_c is not None:
            out_c.attrs["complete"] = True
        if out_d is not None:
            out_d.attrs["complete"] = True
        if out_action is not None:
            out_action.attrs["complete"] = True

    if tmp_c is not None and out_c_path is not None:
        tmp_c.replace(out_c_path)
    if tmp_d is not None and out_d_path is not None:
        tmp_d.replace(out_d_path)
    if tmp_action is not None and out_action_path is not None:
        tmp_action.replace(out_action_path)
    return {"demos": demos_written, "frames": frames_written}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute OpenVLA-OFT action hidden C/D sidecars for LIBERO HDF5.")
    parser.add_argument("--openvla-oft-dir", default=str(PROJECT_ROOT.parent / "openvla-oft"))
    parser.add_argument("--hdf5-dir", default=str(PROJECT_ROOT / "data/processed_data/libero_goal_no_noops_t_256"))
    parser.add_argument("--out-c-dir", default=str(PROJECT_ROOT / "data/processed_data/libero_goal_no_noops_t_256_oft_action_hidden_c_h8"))
    parser.add_argument("--out-d-dir", default=str(PROJECT_ROOT / "data/processed_data/libero_goal_no_noops_t_256_oft_action_hidden_d_h8"))
    parser.add_argument("--out-action-dir", default=None)
    parser.add_argument("--skip-cd-sidecars", action="store_true")
    parser.add_argument("--oft-ckpt", default=str(PROJECT_ROOT / "data/ckpts/OpenVLA-OFT/libero_goal"))
    parser.add_argument("--unnorm-key", default="libero_goal_no_noops")
    parser.add_argument("--image-keys", nargs="+", default=["agentview_rgb", "eye_in_hand_rgb"])
    parser.add_argument("--num-images-in-input", type=int, default=4)
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
    hdf5_dir = _project_path(args.hdf5_dir)
    out_c_dir = None if args.skip_cd_sidecars else _project_path(args.out_c_dir)
    out_d_dir = None if args.skip_cd_sidecars else _project_path(args.out_d_dir)
    out_action_dir = None if args.out_action_dir is None else _project_path(args.out_action_dir)
    if out_c_dir is not None:
        out_c_dir.mkdir(parents=True, exist_ok=True)
    if out_d_dir is not None:
        out_d_dir.mkdir(parents=True, exist_ok=True)
    if out_action_dir is not None:
        out_action_dir.mkdir(parents=True, exist_ok=True)
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
        base_config["world_size"] = world_size
        base_config["start_time"] = time.time()
        base_config["model_path"] = str(_project_path(args.oft_ckpt))
        base_config["encoder_state_ckpt"] = ""
        base_config["action_head_type"] = "oft_l1_regression"
        base_config["prompt_style"] = str(args.prompt_style)
        base_config["history"] = int(args.history)
        base_config["include_state"] = bool(args.include_state)
        base_config["rotate_images_180"] = bool(args.rotate_images_180)
        config_c = dict(base_config, obs_hidden_source="oft_mlpresnet_fc1_relu")
        config_d = dict(base_config, obs_hidden_source="oft_mlpresnet_post_resblocks")
        if out_c_dir is not None:
            (out_c_dir / "preprocess_config.json").write_text(json.dumps(config_c, indent=2, sort_keys=True) + "\n")
        if out_d_dir is not None:
            (out_d_dir / "preprocess_config.json").write_text(json.dumps(config_d, indent=2, sort_keys=True) + "\n")
        if out_action_dir is not None:
            config_action = dict(base_config, obs_hidden_source="action_query")
            (out_action_dir / "preprocess_config.json").write_text(
                json.dumps(config_action, indent=2, sort_keys=True) + "\n"
            )
        print(f"[oft-hidden] source={hdf5_dir} files={len(files)} assigned/rank={len(assigned)} world_size={world_size}")

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
        existing_paths = [path for path in (out_c_path, out_d_path, out_action_path) if path is not None]
        if any(path.exists() for path in existing_paths):
            if args.overwrite:
                for path in existing_paths:
                    if path.exists():
                        path.unlink()
            elif all(_is_complete_hdf5(path) for path in existing_paths):
                print(f"[rank{rank}] skip complete {source_path.name}")
                continue
            else:
                raise RuntimeError(f"Refusing to use incomplete sidecar without --overwrite: {existing_paths}")
        stats = _write_source_sidecars(
            source_path=source_path,
            out_c_path=out_c_path,
            out_d_path=out_d_path,
            out_action_path=out_action_path,
            components=components,
            args=args,
            rank=rank,
        )
        total_demos += stats["demos"]
        total_frames += stats["frames"]
        print(f"[rank{rank}] wrote {source_path.name}: demos={stats['demos']} frames={stats['frames']}")
    print(f"[rank{rank}] done demos={total_demos} frames={total_frames}")


if __name__ == "__main__":
    main()
