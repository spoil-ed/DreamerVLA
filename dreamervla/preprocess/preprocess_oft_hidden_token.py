#!/usr/bin/env python
"""Build OpenVLA-OFT projected hidden-token sidecars for LIBERO HDF5 demos."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from dreamervla.preprocess.artifact_utils import plan_hdf5_preprocess_tasks
from dreamervla.preprocess.sidecar_schema import (
    SIDECAR_SCHEMA_VERSION,
    required_demo_datasets,
    validate_hidden_token_preprocess_config,
)
from dreamervla.utils.hydra_config import script_namespace
from dreamervla.utils.progress import ProgressReporter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBS_HIDDEN_SOURCE = "hidden_token"


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
            return rank, manual_world_size, torch.device(f"cuda:{local_rank}")
        return rank, manual_world_size, torch.device("cpu")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device(f"cuda:{local_rank}")
    return rank, world_size, torch.device("cpu")


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
    obs_group: h5py.Group,
    key: str,
    index: int,
    *,
    rotate_images_180: bool,
) -> np.ndarray:
    image = np.asarray(obs_group[key][index], dtype=np.uint8)
    if rotate_images_180:
        image = image[::-1, ::-1].copy()
    return image


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


def resolve_oft_policy_mode(checkpoint: str | Path, policy_mode: str = "discrete") -> str:
    """Validate the discrete one-trajectory checkpoint contract."""

    mode = str(policy_mode).lower()
    if mode != "discrete":
        if mode == "l1":
            raise ValueError("L1/action-query checkpoints are closed")
        if mode == "auto":
            checkpoint = Path(checkpoint).expanduser()
            if any(checkpoint.glob("action_head--*.pt")):
                raise ValueError("L1/action-query checkpoints are closed")
        raise ValueError(
            f"OpenVLA-OFT mainline requires policy_mode='discrete', got {policy_mode!r}"
        )
    checkpoint = Path(checkpoint).expanduser()
    has_action_head = bool(sorted(checkpoint.glob("action_head--*.pt")))
    if has_action_head:
        raise ValueError("L1/action-query checkpoints are closed")
    has_proprio_projector = bool(sorted(checkpoint.glob("proprio_projector--*.pt")))
    if has_proprio_projector:
        raise ValueError("OpenVLA-OFT hidden-token mainline does not include proprio")
    return "discrete"


def _action_head_type_for_mode(mode: str) -> str:
    if str(mode) != "discrete":
        raise ValueError("only the discrete OpenVLA-OFT action head is supported")
    return "oft_discrete_token"


def _resolve_num_images_in_input(args: SimpleNamespace) -> int:
    if int(args.history) != 1:
        raise ValueError(
            f"OpenVLA-OFT hidden-token mainline requires history=1, got {int(args.history)}"
        )
    if list(args.image_keys) != ["agentview_rgb"]:
        raise ValueError(
            "OpenVLA-OFT hidden-token mainline requires image_keys=['agentview_rgb'], "
            f"got {list(args.image_keys)!r}"
        )
    count = 1 if args.num_images_in_input is None else int(args.num_images_in_input)
    if count != 1:
        raise ValueError(
            f"OpenVLA-OFT hidden-token mainline requires num_images_in_input=1, got {count}"
        )
    return count


class _FakeVisionBackbone:
    def __init__(self, *, num_patches: int, num_images_in_input: int) -> None:
        self._num_patches = int(num_patches)
        self._num_images_in_input = int(num_images_in_input)

    def get_num_patches(self) -> int:
        return self._num_patches

    def get_num_images_in_input(self) -> int:
        return self._num_images_in_input


def _load_oft_components(args: SimpleNamespace, device: torch.device) -> dict[str, Any]:
    """Load only the OFT components required to project image and language tokens."""

    if bool(args.fake_oft_components):
        vla = SimpleNamespace(
            token_dim=int(args.token_dim),
            vision_backbone=_FakeVisionBackbone(
                num_patches=int(args.fake_num_patches),
                num_images_in_input=_resolve_num_images_in_input(args),
            ),
        )
        cfg = SimpleNamespace(
            pretrained_checkpoint=str(_project_path(args.oft_ckpt)),
            num_images_in_input=_resolve_num_images_in_input(args),
            center_crop=bool(args.center_crop),
        )
        return {
            "cfg": cfg,
            "mode": "fake",
            "vla": vla,
            "processor": None,
            "prepare_images_for_vla": _prepare_images_for_vla,
            "device": device,
            "fake": True,
        }

    if bool(args.load_in_8bit) or bool(args.load_in_4bit):
        raise NotImplementedError(
            "The lightweight OpenVLA-OFT loader does not support 8-bit or 4-bit loading."
        )

    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()
    from dreamervla.models.embodiment.openvla_oft_policy import OpenVLAOFTPolicy

    mode = resolve_oft_policy_mode(
        _project_path(args.oft_ckpt), getattr(args, "policy_mode", "discrete")
    )
    cfg = SimpleNamespace(
        pretrained_checkpoint=str(_project_path(args.oft_ckpt)),
        num_images_in_input=_resolve_num_images_in_input(args),
        center_crop=bool(args.center_crop),
    )
    policy = OpenVLAOFTPolicy(
        model_path=cfg.pretrained_checkpoint,
        torch_dtype="bf16",
        num_images_in_input=cfg.num_images_in_input,
        use_lora=False,
        use_l1_regression=False,
        use_diffusion=False,
        use_proprio=False,
        use_film=False,
        freeze_vla_backbone=True,
        unnorm_key=str(args.unnorm_key),
    )
    policy.eval()
    policy.to(device)
    return {
        "cfg": cfg,
        "mode": mode,
        "vla": policy.vla,
        "processor": policy.processor,
        "prepare_images_for_vla": _prepare_images_for_vla,
        "device": device,
        "fake": False,
    }


def _hidden_token_sidecar_dims(
    vla: Any,
    *,
    image_keys: Sequence[str],
    token_dim: int,
) -> tuple[int, int]:
    per_image_patches = int(vla.vision_backbone.get_num_patches())
    keys = tuple(image_keys)
    if keys != ("agentview_rgb",):
        raise ValueError(
            f"OpenVLA-OFT hidden-token mainline requires one agentview image, got {keys!r}"
        )
    resolved_token_dim = _loaded_token_dim(vla)
    if per_image_patches <= 0 or int(token_dim) <= 0:
        raise ValueError(
            "OpenVLA-OFT hidden-token geometry must be positive, "
            f"got patches={per_image_patches}, token_dim={int(token_dim)}"
        )
    if int(token_dim) != resolved_token_dim:
        raise ValueError(
            "OpenVLA-OFT token_dim metadata does not match the loaded backbone: "
            f"config={int(token_dim)}, loaded={resolved_token_dim}"
        )
    return per_image_patches, per_image_patches * resolved_token_dim


def _loaded_token_dim(vla: Any) -> int:
    for path in (
        ("token_dim",),
        ("hidden_size",),
        ("config", "hidden_size"),
        ("language_model", "config", "hidden_size"),
        ("llm_backbone", "llm", "config", "hidden_size"),
    ):
        value = vla
        for attribute in path:
            value = getattr(value, attribute, None)
            if value is None:
                break
        if value is not None:
            return int(value)
    raise ValueError("could not derive token_dim from loaded OpenVLA-OFT policy")


def _predict_hidden_token_chunk(
    *,
    components: dict[str, Any],
    args: SimpleNamespace,
    obs_group: h5py.Group,
    image_keys: tuple[str, ...],
    prompt: str,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a frame chunk to current-frame hidden tokens and language embeddings."""

    cfg = components["cfg"]
    vla = components["vla"]
    batch = int(end - start)
    token_count, _ = _hidden_token_sidecar_dims(
        vla,
        image_keys=image_keys,
        token_dim=int(args.token_dim),
    )
    if bool(components.get("fake", False)):
        for index in range(start, end):
            for hidx in _history_indices(index, int(args.history)):
                for key in image_keys:
                    _image_from_hdf5(
                        obs_group,
                        key,
                        hidx,
                        rotate_images_180=bool(args.rotate_images_180),
                    )
            if bool(args.include_state):
                _state_from_obs_group(obs_group, index)
        return (
            np.zeros((batch, token_count, int(args.token_dim)), dtype=np.float32),
            np.zeros((batch, int(args.token_dim)), dtype=np.float32),
        )

    processor = components["processor"]
    prepare_images_for_vla = components["prepare_images_for_vla"]
    device = components["device"]
    model_prompt = f"In: What action should the robot take to {prompt.lower()}?\nOut:"
    input_ids: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    primary_pixels: list[torch.Tensor] = []
    extra_pixels_by_view: list[list[torch.Tensor]] = []

    for index in range(start, end):
        images = [
            _image_from_hdf5(
                obs_group,
                key,
                hidx,
                rotate_images_180=bool(args.rotate_images_180),
            )
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

    pixel_values = torch.cat(primary_pixels, dim=0)
    if extra_pixels_by_view:
        extra_batches = [torch.cat(view_pixels, dim=0) for view_pixels in extra_pixels_by_view]
        pixel_values = torch.cat([pixel_values, *extra_batches], dim=1)

    with torch.inference_mode():
        from prismatic.vla.constants import IGNORE_INDEX

        input_ids_tensor = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0)
        if not torch.all(input_ids_tensor[:, -1] == 29871):
            empty_token = torch.full(
                (input_ids_tensor.shape[0], 1),
                29871,
                dtype=input_ids_tensor.dtype,
                device=input_ids_tensor.device,
            )
            input_ids_tensor = torch.cat((input_ids_tensor, empty_token), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(empty_token, dtype=attention_mask.dtype)),
                dim=1,
            )
        labels = input_ids_tensor.clone()
        labels[:] = IGNORE_INDEX
        input_ids_tensor, attention_mask = vla._prepare_input_for_action_prediction(
            input_ids_tensor, attention_mask
        )
        labels = vla._prepare_labels_for_action_prediction(labels, input_ids_tensor)
        input_embeddings = vla.get_input_embeddings()(input_ids_tensor)
        all_actions_mask = vla._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        lang_emb = language_embeddings.mean(dim=1).float()
        projected = vla._process_vision_features(pixel_values, language_embeddings, False)
        current_tokens = projected[:, -token_count:, :]

    if current_tokens.ndim != 3 or tuple(current_tokens.shape[1:]) != (
        token_count,
        int(args.token_dim),
    ):
        raise RuntimeError(
            "OpenVLA-OFT projected hidden-token shape mismatch: "
            f"got {tuple(current_tokens.shape)}, expected "
            f"[B, {token_count}, {int(args.token_dim)}]"
        )
    return (
        current_tokens.float().cpu().numpy(),
        lang_emb.float().cpu().numpy(),
    )


def _write_attrs(
    handle: h5py.File,
    args: SimpleNamespace,
    *,
    source_path: Path,
    token_count: int,
) -> None:
    token_dim = int(args.token_dim)
    handle.attrs["complete"] = False
    handle.attrs["source_hdf5"] = str(source_path)
    handle.attrs["source_hdf5_dir"] = str(source_path.parent)
    handle.attrs["hidden_key"] = str(args.hidden_key)
    handle.attrs["hidden_dim"] = int(token_count * token_dim)
    handle.attrs["obs_hidden_source"] = OBS_HIDDEN_SOURCE
    handle.attrs["obs_embedding_shape"] = np.asarray([token_count, token_dim], dtype=np.int64)
    handle.attrs["hidden_storage_format"] = "tokenized"
    handle.attrs["token_count"] = int(token_count)
    handle.attrs["token_dim"] = token_dim
    handle.attrs["output_dtype"] = str(np.dtype(args.output_dtype))
    handle.attrs["image_keys"] = json.dumps(list(args.image_keys))
    handle.attrs["prompt_style"] = str(args.prompt_style)
    handle.attrs["history"] = int(args.history)
    handle.attrs["include_state"] = bool(args.include_state)
    handle.attrs["rotate_images_180"] = bool(args.rotate_images_180)
    handle.attrs["resolution"] = int(args.resolution)
    handle.attrs["model_path"] = str(_project_path(args.oft_ckpt))
    handle.attrs["action_head_type"] = _action_head_type_for_mode(
        getattr(args, "resolved_policy_mode", "discrete")
    )
    handle.attrs["time_horizon"] = int(args.time_horizon)
    handle.attrs["chunk_size"] = int(args.chunk_size)


def _write_source_hidden_token(
    *,
    source_path: Path,
    out_hidden_token_path: Path,
    components: dict[str, Any],
    args: SimpleNamespace,
    rank: int,
) -> dict[str, int]:
    """Atomically write one hidden-token HDF5 sidecar for one reward shard."""

    tmp_path = out_hidden_token_path.with_name(f"{out_hidden_token_path.name}.rank{rank}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    image_keys = tuple(args.image_keys)
    token_count, flat_dim = _hidden_token_sidecar_dims(
        components["vla"],
        image_keys=image_keys,
        token_dim=int(args.token_dim),
    )
    dtype = np.dtype(args.output_dtype)
    prompt = _task_prompt_from_path(source_path)
    demos_written = 0
    frames_written = 0

    with (
        h5py.File(source_path, "r", swmr=True, libver="latest") as source,
        h5py.File(tmp_path, "w", libver="latest") as output,
    ):
        _write_attrs(
            output,
            args,
            source_path=source_path,
            token_count=token_count,
        )
        source_data = source["data"]
        output_data = output.create_group("data")
        demo_keys = _list_demo_keys(source_data)
        if args.max_demos_per_file is not None:
            demo_keys = demo_keys[: int(args.max_demos_per_file)]
        pbar = ProgressReporter(len(demo_keys), f"rank{rank} {source_path.name}", unit="demo")
        for demo_key in demo_keys:
            demo = source_data[demo_key]
            obs_group = demo["obs"]
            for key in image_keys:
                if key not in obs_group:
                    raise KeyError(f"{source_path}:{demo_key} missing obs/{key}")
            length = int(demo["actions"].shape[0])
            demo_out = output_data.create_group(demo_key)
            demo_out.attrs["length"] = length
            demo_out.attrs["task_prompt"] = prompt
            hidden_dset = demo_out.create_dataset(
                str(args.hidden_key),
                shape=(length, token_count, int(args.token_dim)),
                dtype=dtype,
                chunks=(1, token_count, int(args.token_dim)),
                compression=None,
            )
            hidden_dset.attrs["hidden_dim"] = flat_dim
            hidden_dset.attrs["source_dtype"] = "float32"
            hidden_dset.attrs["token_count"] = token_count
            hidden_dset.attrs["token_dim"] = int(args.token_dim)
            hidden_dset.attrs["hidden_storage_format"] = "tokenized"
            lang_emb_dset = demo_out.create_dataset(
                "lang_emb",
                shape=(int(args.token_dim),),
                dtype=dtype,
                compression=None,
            )
            lang_emb_dset.attrs["hidden_dim"] = int(args.token_dim)
            lang_emb_dset.attrs["source_dtype"] = "float32"

            for start in range(0, length, int(args.chunk_size)):
                end = min(start + int(args.chunk_size), length)
                hidden_token, lang_emb = _predict_hidden_token_chunk(
                    components=components,
                    args=args,
                    obs_group=obs_group,
                    image_keys=image_keys,
                    prompt=prompt,
                    start=start,
                    end=end,
                )
                hidden_dset[start:end] = hidden_token.astype(dtype, copy=False)
                if start == 0:
                    lang_emb_dset[...] = lang_emb[0].astype(dtype, copy=False)
                frames_written += int(end - start)
            demo_out.attrs["complete"] = True
            demos_written += 1
            pbar.update()
        pbar.close()
        output.attrs["complete"] = True

    tmp_path.replace(out_hidden_token_path)
    return {"demos": demos_written, "frames": frames_written}


def build_hidden_token_preprocess_config(
    args: SimpleNamespace,
    *,
    hdf5_dir: Path,
    out_hidden_token_dir: Path,
    world_size: int,
    token_count: int,
) -> dict[str, Any]:
    """Build and validate metadata exactly as it will be persisted.

    Keeping this in one function makes the data producer and the training-side
    validator share a testable boundary instead of duplicating schema fields in
    the command-line entry point.
    """

    config = vars(args).copy()
    config.update(
        hdf5_dir=str(hdf5_dir),
        out_hidden_token_dir=str(out_hidden_token_dir),
        world_size=int(world_size),
        start_time=time.time(),
        model_path=str(_project_path(args.oft_ckpt)),
        encoder_state_ckpt="",
        action_head_type=_action_head_type_for_mode(args.resolved_policy_mode),
        obs_hidden_source=OBS_HIDDEN_SOURCE,
        token_count=int(token_count),
        hidden_dim=int(token_count) * int(args.token_dim),
        obs_embedding_shape=[int(token_count), int(args.token_dim)],
        hidden_storage_format="tokenized",
        sidecar_schema_version=SIDECAR_SCHEMA_VERSION,
        required_demo_datasets=required_demo_datasets(),
    )
    validate_hidden_token_preprocess_config(
        config,
        context="OpenVLA-OFT offline preprocess config",
    )
    return config


def parse_args() -> SimpleNamespace:
    return script_namespace("preprocess_oft_hidden_token")


def main() -> None:
    args = parse_args()
    if str(args.obs_hidden_source) != OBS_HIDDEN_SOURCE:
        raise SystemExit("OpenVLA-OFT preprocessing only supports obs_hidden_source=hidden_token")
    if bool(args.fake_oft_components):
        args.resolved_policy_mode = "discrete"
    else:
        args.resolved_policy_mode = resolve_oft_policy_mode(
            _project_path(args.oft_ckpt), args.policy_mode
        )
    if args.resolved_policy_mode != "discrete":
        raise SystemExit(
            "OpenVLA-OFT preprocessing supports only the discrete one-trajectory "
            "mainline; L1/action-query checkpoints are closed"
        )
    args.include_state = False
    if int(args.history) != 1 or int(_resolve_num_images_in_input(args)) != 1:
        raise SystemExit("OpenVLA-OFT preprocessing requires history=1 and num_images_in_input=1")
    if int(args.patches_per_image) <= 0 or int(args.token_dim) <= 0:
        raise SystemExit("OpenVLA-OFT patches_per_image and token_dim must be positive")

    hdf5_dir = _project_path(args.hdf5_dir)
    out_hidden_token_dir = _project_path(args.out_hidden_token_dir)
    out_hidden_token_dir.mkdir(parents=True, exist_ok=True)
    rank, world_size, device = _init_distributed()
    files = sorted(hdf5_dir.glob("*.hdf5"))
    if args.max_files is not None:
        files = files[: int(args.max_files)]
    if not files:
        raise RuntimeError(f"No HDF5 files found under {hdf5_dir}")

    def _output_paths(source_path: Path) -> list[Path]:
        return [out_hidden_token_dir / source_path.name]

    required = {
        out_hidden_token_dir / source_path.name: required_demo_datasets() for source_path in files
    }
    task_plan = plan_hdf5_preprocess_tasks(
        files,
        rank=rank,
        world_size=world_size,
        output_paths=_output_paths,
        required_demo_datasets=required,
        overwrite=bool(args.overwrite),
    )
    assigned = [task.source_path for task in task_plan.assigned]

    if rank == 0:
        token_count = int(args.patches_per_image) * len(tuple(args.image_keys))
        config = build_hidden_token_preprocess_config(
            args,
            hdf5_dir=hdf5_dir,
            out_hidden_token_dir=out_hidden_token_dir,
            world_size=world_size,
            token_count=token_count,
        )
        (out_hidden_token_dir / "preprocess_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[oft-hidden-tokens] source={hdf5_dir} files={len(files)} "
            f"pending={len(task_plan.pending)} skipped={len(task_plan.skipped)} "
            f"repaired={len(task_plan.repaired)} assigned/rank={len(assigned)} "
            f"loads_by_rank={task_plan.loads_by_rank} world_size={world_size} "
            f"policy_mode={args.resolved_policy_mode}"
        )

    if not assigned:
        print(f"[rank{rank}] done demos=0 frames=0")
        return

    components = _load_oft_components(args, device)
    actual_patches = int(components["vla"].vision_backbone.get_num_patches())
    if actual_patches != int(args.patches_per_image):
        raise ValueError(
            "OpenVLA-OFT patches_per_image mismatch: "
            f"model={actual_patches}, config={int(args.patches_per_image)}"
        )
    actual_token_dim = _loaded_token_dim(components["vla"])
    if actual_token_dim != int(args.token_dim):
        raise ValueError(
            "OpenVLA-OFT token_dim mismatch: "
            f"model={actual_token_dim}, config={int(args.token_dim)}"
        )
    total_demos = 0
    total_frames = 0
    for source_path in assigned:
        out_path = out_hidden_token_dir / source_path.name
        if bool(args.overwrite) and out_path.exists():
            out_path.unlink()
        stats = _write_source_hidden_token(
            source_path=source_path,
            out_hidden_token_path=out_path,
            components=components,
            args=args,
            rank=rank,
        )
        total_demos += stats["demos"]
        total_frames += stats["frames"]
        print(
            f"[rank{rank}] wrote {source_path.name}: "
            f"demos={stats['demos']} frames={stats['frames']}"
        )
    print(f"[rank{rank}] done demos={total_demos} frames={total_frames}")


if __name__ == "__main__":
    main()
