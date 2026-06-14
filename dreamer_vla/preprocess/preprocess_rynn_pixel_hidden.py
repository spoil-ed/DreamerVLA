#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from dreamer_vla.models.encoder.rynnvla_encoder import RynnVLAEncoder
from dreamer_vla.utils.hf_checkpoint import is_hf_checkpoint, load_runner_payload
from dreamer_vla.utils.paths import checkpoints_path, processed_data_path


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _default_ckpt_path(*parts: str) -> str:
    return str(checkpoints_path(*parts).resolve())


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


def _init_distributed() -> tuple[int, int, int, torch.device]:
    """Read torchrun rank env vars without creating a NCCL process group."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def _is_complete_hdf5(path: Path, *, require_actor_sequence: bool = False) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                return False
            if not require_actor_sequence:
                return True
            if not bool(handle.attrs.get("save_actor_sequence", False)):
                return False
            data_group = handle.get("data")
            if data_group is None:
                return False
            for demo_key in data_group.keys():
                demo = data_group[demo_key]
                for key in (
                    "actor_hidden_states",
                    "actor_input_ids",
                    "actor_attention_mask",
                    "actor_seq_lens",
                ):
                    if key not in demo:
                        return False
            return True
    except OSError:
        return False


def _hidden_dtype(name: str) -> np.dtype:
    normalized = str(name).lower()
    if normalized in {"fp16", "float16", "half"}:
        return np.dtype("float16")
    if normalized in {"fp32", "float32"}:
        return np.dtype("float32")
    raise ValueError(f"Unsupported output dtype: {name}")


def _compression(name: str) -> str | None:
    normalized = str(name).lower()
    if normalized in {"none", "null", "false", "0", ""}:
        return None
    if normalized not in {"lzf", "gzip"}:
        raise ValueError(f"Unsupported HDF5 compression: {name}")
    return normalized


def _prepare_actor_sequence_arrays(
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: list[list[int]],
    target_token_id: int,
) -> dict[str, np.ndarray]:
    """Prepare full VLA token-hidden sequence arrays for HDF5 storage.

    ``VLAActionHeadActor`` sequence mode expects the hidden states before the
    action trigger, plus ``input_ids`` that contain the trigger token.  The
    backbone returns hidden states for the current prompt only, so we append the
    trigger to the stored ids and mark it valid in the actor attention mask.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be [B,L,D], got {tuple(hidden_states.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [B,L], got {tuple(attention_mask.shape)}")
    batch_size, seq_len, _hidden_dim = hidden_states.shape
    if attention_mask.shape != (batch_size, seq_len):
        raise ValueError(
            "attention_mask shape mismatch: "
            f"got {tuple(attention_mask.shape)}, expected {(batch_size, seq_len)}"
        )
    if len(input_ids) != batch_size:
        raise ValueError(f"input_ids batch mismatch: got {len(input_ids)}, expected {batch_size}")

    input_rows = np.zeros((batch_size, seq_len + 1), dtype=np.int32)
    mask_rows = np.zeros((batch_size, seq_len + 1), dtype=np.bool_)
    seq_lens = np.zeros((batch_size,), dtype=np.int32)
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1).detach().cpu().tolist()
    for idx, valid_len_raw in enumerate(lengths):
        valid_len = int(valid_len_raw)
        if valid_len < 1:
            raise ValueError(f"sample {idx} has no valid hidden tokens")
        if valid_len > seq_len:
            raise ValueError(f"sample {idx} valid_len={valid_len} exceeds seq_len={seq_len}")
        if len(input_ids[idx]) < valid_len:
            raise ValueError(
                f"sample {idx} input_ids shorter than attention_mask: "
                f"len(input_ids)={len(input_ids[idx])}, valid_len={valid_len}"
            )
        row_ids = [int(tok) for tok in input_ids[idx][:valid_len]]
        input_rows[idx, :valid_len] = np.asarray(row_ids, dtype=np.int32)
        input_rows[idx, valid_len] = int(target_token_id)
        mask_rows[idx, : valid_len + 1] = True
        seq_lens[idx] = valid_len

    return {
        "actor_hidden_states": hidden_states.detach().cpu().numpy(),
        "actor_input_ids": input_rows,
        "actor_attention_mask": mask_rows,
        "actor_seq_lens": seq_lens,
    }


def _image_content_token_spans(
    tokens: list[int], *, start_id: int, end_id: int, new_line_id: int
) -> list[list[int]]:
    """Per-image VQ content token ids; grid-size, newline, and marker tokens stripped."""
    spans: list[list[int]] = []
    inside = False
    current: list[int] = []
    skip = 0
    for tok in tokens:
        if tok == start_id:
            inside, current, skip = True, [], 2
            continue
        if not inside:
            continue
        if tok == end_id:
            spans.append(current)
            inside = False
            continue
        if skip > 0:
            skip -= 1
            continue
        if tok == new_line_id:
            continue
        current.append(tok)
    return spans


def _input_token_embedding_obs(
    *,
    encoder: RynnVLAEncoder,
    processor: Any,
    input_ids_list: list[list[int]],
    num_views: int,
) -> torch.Tensor:
    """Scheme-B frame latent: current-frame VQ image tokens through the backbone
    input-embedding table (no transformer forward). Returns [T, N*token_dim]."""
    start_id = int(processor.token2id(processor.image_start_token))
    end_id = int(processor.token2id(processor.image_end_token))
    new_line_id = int(processor.token2id(processor.new_line_token))
    backbone = encoder.backbone
    embed = (
        backbone.get_input_embeddings()
        if hasattr(backbone, "get_input_embeddings")
        else backbone.model.embed_tokens
    )
    frames: list[torch.Tensor] = []
    expected: int | None = None
    for tokens in input_ids_list:
        spans = _image_content_token_spans(
            tokens, start_id=start_id, end_id=end_id, new_line_id=new_line_id
        )
        if len(spans) < num_views:
            raise RuntimeError(
                f"expected at least {num_views} image spans per frame, got {len(spans)}"
            )
        # Record layout is [history ... current] x views; current frame is last.
        current = [tok for span in spans[-num_views:] for tok in span]
        if expected is None:
            expected = len(current)
        elif len(current) != expected:
            raise RuntimeError(
                f"inconsistent image token count across frames: {len(current)} != {expected}"
            )
        ids = torch.as_tensor(current, dtype=torch.long, device=encoder.device)
        with torch.no_grad():
            emb = embed(ids)
        frames.append(emb.reshape(-1).float())
    return torch.stack(frames, dim=0)


def _select_obs_hidden(
    *,
    pooled_hidden: torch.Tensor,
    action_hidden: torch.Tensor | None,
    obs_hidden_source: str,
) -> torch.Tensor:
    source = str(obs_hidden_source).lower()
    if source == "pooled":
        return pooled_hidden
    if source != "action_query":
        raise ValueError("obs_hidden_source must be one of: pooled, action_query")
    if action_hidden is None:
        raise ValueError("obs_hidden_source='action_query' requires action_hidden")
    if action_hidden.ndim != 3:
        raise ValueError(f"action_hidden must be [B,H,D], got {tuple(action_hidden.shape)}")
    return _legacy_flat_obs_embedding_from_action_hidden(action_hidden)


def _legacy_flat_obs_embedding_from_action_hidden(
    action_hidden: torch.Tensor,
) -> torch.Tensor:
    """Flatten action-hidden tokens only at the legacy HDF5 sidecar boundary."""
    return action_hidden.flatten(start_dim=1)


def _source_stats(source_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    demos = 0
    frames = 0
    with h5py.File(source_path, "r", swmr=True, libver="latest") as handle:
        data_group = handle["data"]
        demo_keys = _list_demo_keys(data_group)
        if args.max_demos_per_file is not None:
            demo_keys = demo_keys[: int(args.max_demos_per_file)]
        for demo_key in demo_keys:
            demos += 1
            frames += int(data_group[demo_key]["actions"].shape[0])
    return {
        "source": str(source_path),
        "file": source_path.name,
        "demos": demos,
        "frames": frames,
    }


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
) -> Image.Image:
    image = np.asarray(obs_group[key][index], dtype=np.uint8)
    if bool(rotate_images_180):
        image = image[::-1, ::-1].copy()
    return Image.fromarray(image)


def _build_vla_policy_record(
    *,
    prompt: str,
    obs_group: h5py.Group,
    image_keys: tuple[str, ...],
    index: int,
    history: int,
    include_state: bool,
    rotate_images_180: bool,
) -> dict[str, Any]:
    if len(image_keys) != 2:
        raise ValueError(
            "prompt_style='vla_policy' expects exactly two image keys: third-view and wrist-view"
        )
    images: list[Image.Image] = []
    for hidx in _history_indices(index, history):
        for key in image_keys:
            images.append(
                _image_from_hdf5(obs_group, key, hidx, rotate_images_180=rotate_images_180)
            )
    human_val = f"Finish the task: {prompt}."
    if include_state:
        human_val += "<|state|>"
    human_val += "<|image|>" * len(images)
    record: dict[str, Any] = {
        "conversations": [{"from": "human", "value": human_val}],
        "image": images,
        "action": [],
    }
    if include_state:
        record["state"] = [_state_from_obs_group(obs_group, index)]
    return record


def _write_rank_progress(
    progress_dir: Path,
    rank: int,
    *,
    run_id: str,
    completed_demos: int,
    completed_frames: int,
    current_file: str | None = None,
    current_demo: str | None = None,
    done: bool = False,
) -> None:
    progress_dir.mkdir(parents=True, exist_ok=True)
    path = progress_dir / f"rank{rank:03d}.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "run_id": run_id,
        "rank": rank,
        "completed_demos": int(completed_demos),
        "completed_frames": int(completed_frames),
        "current_file": current_file,
        "current_demo": current_demo,
        "done": bool(done),
        "time": time.time(),
    }
    tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _read_progress(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_rank_progress(progress_dir: Path, rank: int, run_id: str) -> dict[str, Any] | None:
    progress = _read_progress(progress_dir / f"rank{rank:03d}.json")
    if not progress or progress.get("run_id") != run_id:
        return None
    return progress


def _all_ranks_done(progress_dir: Path, run_id: str, world_size: int) -> bool:
    for rank in range(world_size):
        progress = _read_rank_progress(progress_dir, rank, run_id)
        if not progress or not bool(progress.get("done", False)):
            return False
    return True


def _monitor_global_progress(
    progress_dir: Path,
    *,
    run_id: str,
    total_demos: int,
    total_frames: int,
    world_size: int,
    stop_event: threading.Event,
) -> None:
    if total_demos <= 0:
        return
    last_demos = 0
    with tqdm(
        total=total_demos,
        desc="total demos",
        position=world_size,
        leave=True,
        mininterval=1.0,
    ) as pbar:
        while not stop_event.is_set():
            completed_demos = 0
            completed_frames = 0
            for rank in range(world_size):
                progress = _read_rank_progress(progress_dir, rank, run_id)
                if not progress:
                    continue
                completed_demos += int(progress.get("completed_demos", 0))
                completed_frames += int(progress.get("completed_frames", 0))
            if completed_demos > last_demos:
                pbar.update(completed_demos - last_demos)
                last_demos = completed_demos
            pbar.set_postfix(
                frames=f"{completed_frames}/{total_frames}",
                refresh=False,
            )
            if completed_demos >= total_demos:
                break
            time.sleep(2.0)

        completed_demos = 0
        completed_frames = 0
        for rank in range(world_size):
            progress = _read_rank_progress(progress_dir, rank, run_id)
            if not progress:
                continue
            completed_demos += int(progress.get("completed_demos", 0))
            completed_frames += int(progress.get("completed_frames", 0))
        if completed_demos > last_demos:
            pbar.update(completed_demos - last_demos)
        pbar.set_postfix(frames=f"{completed_frames}/{total_frames}", refresh=True)


def _make_encoder(args: argparse.Namespace, device: torch.device) -> RynnVLAEncoder:
    model_path = args.model_path
    encoder_state_is_hf = bool(
        args.encoder_state_ckpt and is_hf_checkpoint(args.encoder_state_ckpt)
    )
    if encoder_state_is_hf:
        model_path = args.encoder_state_ckpt
    encoder = RynnVLAEncoder(
        model_path=model_path,
        tokenizer_path=args.tokenizer_path,
        text_tokenizer_path=args.text_tokenizer_path,
        chameleon_vqgan_config=args.chameleon_vqgan_config,
        chameleon_vqgan_ckpt=args.chameleon_vqgan_ckpt,
        resolution=args.resolution,
        action_dim=args.action_dim,
        time_horizon=args.time_horizon,
        action_head_type=args.action_head_type,
        pool=args.pool,
        freeze_backbone=True,
    ).to(device)
    if args.encoder_state_ckpt and not encoder_state_is_hf:
        payload = load_runner_payload(args.encoder_state_ckpt)
        encoder_state = payload.get("state_dicts", {}).get("encoder")
        if encoder_state is None:
            raise RuntimeError(f"{args.encoder_state_ckpt} has no state_dicts.encoder")
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(
                f"[rynn-hidden] warning: missing encoder keys while loading "
                f"{args.encoder_state_ckpt}: {len(missing)}"
            )
        if unexpected:
            print(
                f"[rynn-hidden] warning: unexpected encoder keys while loading "
                f"{args.encoder_state_ckpt}: {len(unexpected)}"
            )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def _encode_chunk(
    encoder: RynnVLAEncoder,
    prompt: str,
    obs_group: h5py.Group,
    image_keys: tuple[str, ...],
    start: int,
    end: int,
    *,
    save_actor_sequence: bool,
    save_action_hidden: bool,
    obs_hidden_source: str,
    target_token_id: int,
    prompt_style: str,
    history: int,
    include_state: bool,
    rotate_images_180: bool,
) -> dict[str, np.ndarray]:
    if (
        str(prompt_style).lower() != "vla_policy"
        or not bool(include_state)
        or not bool(rotate_images_180)
        or int(history) != 2
    ):
        raise ValueError(
            "RynnVLA action-hidden preprocessing must match the existing sidecar: "
            "vla_policy + history=2 + state + rotate180"
        )

    processor = encoder._build_processor(encoder.device)
    input_ids_list: list[list[int]] = []
    labels_list: list[list[int]] = []
    for tidx in range(start, end):
        record = _build_vla_policy_record(
            prompt=prompt,
            obs_group=obs_group,
            image_keys=image_keys,
            index=tidx,
            history=2,
            include_state=True,
            rotate_images_180=True,
        )
        tokens = processor.process_item(record, training_mode=False)
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        tokens = [int(tok) for tok in tokens]
        input_ids_list.append(tokens)
        labels_list.append([-100] * len(tokens))
    lengths = [len(seq) for seq in input_ids_list]

    if str(obs_hidden_source).lower() == "input_token_embedding":
        if bool(save_actor_sequence) or bool(save_action_hidden):
            raise ValueError(
                "obs_hidden_source='input_token_embedding' writes a pure input-token "
                "sidecar; disable --save-actor-sequence and --save-action-hidden."
            )
        hidden = _input_token_embedding_obs(
            encoder=encoder,
            processor=processor,
            input_ids_list=input_ids_list,
            num_views=len(image_keys),
        )
        return {"hidden": hidden.detach().cpu().numpy()}

    with torch.no_grad():
        _, _, _, hidden_states, labels_tensor, _, _ = encoder.backbone(
            input_ids=input_ids_list,
            labels=labels_list,
            training=True,
            output_hidden_states=True,
            att_mask=False,
        )
    attention_mask = torch.zeros(
        hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
    )
    for idx, length in enumerate(lengths):
        attention_mask[idx, :length] = True
    weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    encoded = type(
        "_EncodedChunk",
        (),
        {
            "hidden": pooled.float(),
            "hidden_states": hidden_states.float(),
            "attention_mask": attention_mask,
            "labels": labels_tensor,
            "input_ids": input_ids_list,
            "lengths": lengths,
        },
    )()

    need_actor_arrays = (
        bool(save_actor_sequence)
        or bool(save_action_hidden)
        or str(obs_hidden_source).lower() == "action_query"
    )
    actor_arrays = None
    if need_actor_arrays:
        actor_arrays = _prepare_actor_sequence_arrays(
            hidden_states=encoded.hidden_states,
            attention_mask=encoded.attention_mask,
            input_ids=encoded.input_ids,
            target_token_id=int(target_token_id),
        )
    action_hidden = None
    if bool(save_action_hidden) or str(obs_hidden_source).lower() == "action_query":
        if actor_arrays is None:
            raise RuntimeError("internal error: action hidden requested without actor input arrays")
        device = encoded.hidden_states.device
        input_ids = torch.as_tensor(
            actor_arrays["actor_input_ids"], device=device, dtype=torch.long
        )
        attention_mask = torch.as_tensor(
            actor_arrays["actor_attention_mask"], device=device, dtype=torch.bool
        )
        with torch.no_grad():
            action_hidden = encoder.extract_action_hidden(
                hidden_states=encoded.hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_token_id=int(target_token_id),
                eval=True,
            )

    hidden = _select_obs_hidden(
        pooled_hidden=encoded.hidden,
        action_hidden=action_hidden,
        obs_hidden_source=str(obs_hidden_source),
    )
    result = {"hidden": hidden.detach().cpu().numpy()}
    if bool(save_actor_sequence) and actor_arrays is not None:
        result.update(actor_arrays)
    if bool(save_action_hidden) and action_hidden is not None:
        result["action_hidden_states"] = action_hidden.detach().cpu().numpy()
    return result


def _write_source_sidecar(
    source_path: Path,
    output_path: Path,
    encoder: RynnVLAEncoder,
    args: argparse.Namespace,
    rank: int,
    run_id: str,
    progress_dir: Path,
    completed_demos_base: int,
    completed_frames_base: int,
) -> dict[str, Any]:
    tmp_path = output_path.with_name(f"{output_path.name}.rank{rank}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    image_keys = tuple(args.image_keys)
    hidden_dtype = _hidden_dtype(args.output_dtype)
    compression = _compression(args.compression)
    prompt = _task_prompt_from_path(source_path)
    demos_written = 0
    frames_written = 0
    hidden_dim: int | None = None
    token_count: int | None = None
    actor_seq_len: int | None = None
    actor_hidden_dim: int | None = None
    action_hidden_seq_len: int | None = None
    action_hidden_dim: int | None = None

    with (
        h5py.File(source_path, "r", swmr=True, libver="latest") as source,
        h5py.File(tmp_path, "w", libver="latest") as out,
    ):
        data_group = source["data"]
        out_data = out.create_group("data")
        demo_keys = _list_demo_keys(data_group)
        if args.max_demos_per_file is not None:
            demo_keys = demo_keys[: int(args.max_demos_per_file)]

        iterator = tqdm(
            demo_keys,
            desc=f"rank{rank} {source_path.name}",
            position=rank,
            leave=False,
            mininterval=1.0,
        )
        for demo_key in iterator:
            demo = data_group[demo_key]
            obs_group = demo["obs"]
            for key in image_keys:
                if key not in obs_group:
                    raise KeyError(f"{source_path}:{demo_key} missing obs/{key}")
            length = int(demo["actions"].shape[0])
            demo_out = out_data.create_group(demo_key)
            demo_out.attrs["length"] = length
            demo_out.attrs["task_prompt"] = prompt
            dset = None
            actor_hidden_dset = None
            actor_input_ids_dset = None
            actor_attention_mask_dset = None
            actor_seq_lens_dset = None
            action_hidden_dset = None
            for start in range(0, length, int(args.chunk_size)):
                end = min(start + int(args.chunk_size), length)
                encoded = _encode_chunk(
                    encoder=encoder,
                    prompt=prompt,
                    obs_group=obs_group,
                    image_keys=image_keys,
                    start=start,
                    end=end,
                    save_actor_sequence=bool(args.save_actor_sequence),
                    save_action_hidden=bool(args.save_action_hidden),
                    obs_hidden_source=str(args.obs_hidden_source),
                    target_token_id=int(args.action_trigger_token_id),
                    prompt_style=str(args.prompt_style),
                    history=int(args.history),
                    include_state=bool(args.include_state),
                    rotate_images_180=bool(args.rotate_images_180),
                )
                hidden = encoded["hidden"]
                if dset is None:
                    hidden_dim = int(hidden.shape[-1])
                    if str(args.obs_hidden_source).lower() == "input_token_embedding":
                        token_dim = int(args.token_dim)
                        if hidden_dim % token_dim != 0:
                            raise RuntimeError(
                                f"input-token hidden_dim={hidden_dim} is not divisible by token_dim={token_dim}"
                            )
                        demo_token_count = hidden_dim // token_dim
                        if token_count is None:
                            token_count = demo_token_count
                        elif token_count != demo_token_count:
                            raise RuntimeError(
                                "inconsistent input-token count across demos: "
                                f"{demo_token_count} != {token_count}"
                            )
                    dset = demo_out.create_dataset(
                        args.hidden_key,
                        shape=(length, hidden_dim),
                        dtype=hidden_dtype,
                        chunks=(min(max(1, int(args.chunk_size)), length), hidden_dim),
                        compression=compression,
                    )
                    dset.attrs["hidden_dim"] = hidden_dim
                    dset.attrs["source_dtype"] = "float32"
                    if token_count is not None:
                        dset.attrs["token_count"] = int(token_count)
                        dset.attrs["token_dim"] = int(args.token_dim)
                dset[start:end] = hidden.astype(hidden_dtype, copy=False)
                if bool(args.save_actor_sequence):
                    actor_hidden = encoded["actor_hidden_states"]
                    actor_input_ids = encoded["actor_input_ids"]
                    actor_attention_mask = encoded["actor_attention_mask"]
                    actor_seq_lens = encoded["actor_seq_lens"]
                    chunk_seq_len = int(actor_hidden.shape[1])
                    if actor_hidden_dset is None:
                        actor_seq_len = chunk_seq_len
                        actor_hidden_dim = int(actor_hidden.shape[-1])
                        actor_hidden_dset = demo_out.create_dataset(
                            "actor_hidden_states",
                            shape=(length, actor_seq_len, actor_hidden_dim),
                            maxshape=(length, None, actor_hidden_dim),
                            dtype=hidden_dtype,
                            chunks=(1, actor_seq_len, actor_hidden_dim),
                            compression=compression,
                        )
                        actor_input_ids_dset = demo_out.create_dataset(
                            "actor_input_ids",
                            shape=(length, actor_seq_len + 1),
                            maxshape=(length, None),
                            dtype=np.int32,
                            chunks=(
                                min(max(1, int(args.chunk_size)), length),
                                actor_seq_len + 1,
                            ),
                            compression=compression,
                        )
                        actor_attention_mask_dset = demo_out.create_dataset(
                            "actor_attention_mask",
                            shape=(length, actor_seq_len + 1),
                            maxshape=(length, None),
                            dtype=np.bool_,
                            chunks=(
                                min(max(1, int(args.chunk_size)), length),
                                actor_seq_len + 1,
                            ),
                            compression=compression,
                        )
                        actor_seq_lens_dset = demo_out.create_dataset(
                            "actor_seq_lens",
                            shape=(length,),
                            dtype=np.int32,
                            chunks=(min(max(1, int(args.chunk_size)), length),),
                            compression=compression,
                        )
                        actor_hidden_dset.attrs["hidden_dim"] = actor_hidden_dim
                        actor_hidden_dset.attrs["source_dtype"] = "float32"
                        actor_hidden_dset.attrs["sequence_dim"] = actor_seq_len
                        actor_input_ids_dset.attrs["target_token_id"] = int(
                            args.action_trigger_token_id
                        )
                    elif chunk_seq_len > int(actor_hidden_dset.shape[1]):
                        actor_seq_len = chunk_seq_len
                        actor_hidden_dset.resize(
                            (
                                length,
                                actor_seq_len,
                                int(actor_hidden_dim or actor_hidden.shape[-1]),
                            )
                        )
                        actor_input_ids_dset.resize((length, actor_seq_len + 1))
                        actor_attention_mask_dset.resize((length, actor_seq_len + 1))
                        actor_hidden_dset.attrs["sequence_dim"] = actor_seq_len

                    target_seq_len = int(actor_hidden_dset.shape[1])
                    if chunk_seq_len < target_seq_len:
                        pad = target_seq_len - chunk_seq_len
                        actor_hidden = np.pad(actor_hidden, ((0, 0), (0, pad), (0, 0)))
                        actor_input_ids = np.pad(actor_input_ids, ((0, 0), (0, pad)))
                        actor_attention_mask = np.pad(actor_attention_mask, ((0, 0), (0, pad)))

                    actor_hidden_dset[start:end] = actor_hidden.astype(hidden_dtype, copy=False)
                    actor_input_ids_dset[start:end] = actor_input_ids.astype(np.int32, copy=False)
                    actor_attention_mask_dset[start:end] = actor_attention_mask.astype(
                        np.bool_, copy=False
                    )
                    actor_seq_lens_dset[start:end] = actor_seq_lens.astype(np.int32, copy=False)
                if bool(args.save_action_hidden):
                    action_hidden = encoded["action_hidden_states"]
                    chunk_action_horizon = int(action_hidden.shape[1])
                    if action_hidden_dset is None:
                        action_hidden_seq_len = chunk_action_horizon
                        action_hidden_dim = int(action_hidden.shape[-1])
                        action_hidden_dset = demo_out.create_dataset(
                            "action_hidden_states",
                            shape=(length, action_hidden_seq_len, action_hidden_dim),
                            dtype=hidden_dtype,
                            chunks=(1, action_hidden_seq_len, action_hidden_dim),
                            compression=compression,
                        )
                        action_hidden_dset.attrs["hidden_dim"] = action_hidden_dim
                        action_hidden_dset.attrs["source_dtype"] = "float32"
                        action_hidden_dset.attrs["sequence_dim"] = action_hidden_seq_len
                    action_hidden_dset[start:end] = action_hidden.astype(hidden_dtype, copy=False)
                frames_written += end - start
            demos_written += 1
            _write_rank_progress(
                progress_dir,
                rank,
                run_id=run_id,
                completed_demos=completed_demos_base + demos_written,
                completed_frames=completed_frames_base + frames_written,
                current_file=source_path.name,
                current_demo=demo_key,
            )

        out.attrs["complete"] = False
        out.attrs["source_hdf5"] = str(source_path)
        out.attrs["source_hdf5_dir"] = str(source_path.parent)
        out.attrs["hidden_key"] = str(args.hidden_key)
        out.attrs["hidden_dim"] = int(hidden_dim or 0)
        if token_count is not None:
            out.attrs["token_count"] = int(token_count)
            out.attrs["token_dim"] = int(args.token_dim)
        out.attrs["obs_hidden_source"] = str(args.obs_hidden_source)
        out.attrs["output_dtype"] = str(hidden_dtype)
        out.attrs["image_keys"] = json.dumps(list(image_keys))
        out.attrs["prompt_style"] = str(args.prompt_style)
        out.attrs["history"] = int(args.history)
        out.attrs["include_state"] = bool(args.include_state)
        out.attrs["rotate_images_180"] = bool(args.rotate_images_180)
        out.attrs["resolution"] = int(args.resolution)
        out.attrs["model_path"] = str(args.model_path)
        out.attrs["encoder_state_ckpt"] = str(args.encoder_state_ckpt or "")
        out.attrs["pool"] = str(args.pool)
        out.attrs["save_actor_sequence"] = bool(args.save_actor_sequence)
        out.attrs["save_action_hidden"] = bool(args.save_action_hidden)
        out.attrs["action_trigger_token_id"] = int(args.action_trigger_token_id)
        out.attrs["actor_sequence_dim"] = int(actor_seq_len or 0)
        out.attrs["actor_hidden_dim"] = int(actor_hidden_dim or 0)
        out.attrs["action_hidden_sequence_dim"] = int(action_hidden_seq_len or 0)
        out.attrs["action_hidden_dim"] = int(action_hidden_dim or 0)
        out.attrs["complete"] = True

    tmp_path.replace(output_path)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "demos": demos_written,
        "frames": frames_written,
        "hidden_dim": int(hidden_dim or 0),
        "actor_sequence_dim": int(actor_seq_len or 0),
        "actor_hidden_dim": int(actor_hidden_dim or 0),
        "action_hidden_sequence_dim": int(action_hidden_seq_len or 0),
        "action_hidden_dim": int(action_hidden_dim or 0),
        "save_actor_sequence": bool(args.save_actor_sequence),
        "save_action_hidden": bool(args.save_action_hidden),
        "obs_hidden_source": str(args.obs_hidden_source),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute frozen RynnVLA hidden vectors for LIBERO pixel HDF5 files. "
            "The original image dataset is not modified; matching sidecar HDF5 "
            "files are written under --out-dir."
        )
    )
    parser.add_argument(
        "--hdf5-dir",
        default=str(processed_data_path("libero_goal_no_noops_t_256")),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Sidecar output directory. If omitted, the default is derived from "
            "the legacy RynnVLA action-hidden sidecar layout."
        ),
    )
    parser.add_argument("--image-keys", nargs="+", default=["agentview_rgb", "eye_in_hand_rgb"])
    parser.add_argument(
        "--prompt-style",
        default="vla_policy",
        choices=["vla_policy"],
        help="Construct inputs exactly like the existing RynnVLA action-hidden sidecar.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=2,
        help="Number of timesteps of two-view image history; the existing sidecar uses 2.",
    )
    parser.add_argument(
        "--include-state",
        action="store_true",
        default=True,
        help="Include ee_pos + ee_ori + gripper_states in the VLA-policy prompt.",
    )
    parser.add_argument(
        "--rotate-images-180",
        action="store_true",
        default=True,
        help="Rotate HDF5 images by 180 degrees before tokenization, matching get_libero_image().",
    )
    parser.add_argument("--hidden-key", default="obs_embedding")
    parser.add_argument(
        "--obs-hidden-source",
        default="action_query",
        choices=["pooled", "action_query", "input_token_embedding"],
        help=(
            "Which VLA representation is written to --hidden-key. "
            "'action_query' writes the flattened RynnVLA action hidden "
            "after the action-head transformer; 'pooled' keeps the old "
            "4096-d pooled backbone hidden for explicit ablations."
        ),
    )
    parser.add_argument(
        "--save-actor-sequence",
        action="store_true",
        help=(
            "Also store full token hidden sequences for native VLA action-head "
            "training/eval: actor_hidden_states, actor_input_ids, "
            "actor_attention_mask, actor_seq_lens."
        ),
    )
    parser.add_argument(
        "--save-action-hidden",
        action="store_true",
        help="Also store unflattened RynnVLA action hidden states as action_hidden_states.",
    )
    parser.add_argument("--action-trigger-token-id", type=int, default=10004)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--output-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--compression", default="none", choices=["none", "lzf", "gzip"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-demos-per-file", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-global-progress", action="store_true")
    parser.add_argument("--model-path", default=_default_ckpt_path("VLA_model_256", "libero_goal"))
    parser.add_argument("--encoder-state-ckpt", default=None)
    parser.add_argument(
        "--tokenizer-path",
        default=_default_ckpt_path("models--Alpha-VLLM--Lumina-mGPT-7B-768"),
    )
    parser.add_argument(
        "--text-tokenizer-path",
        default=_default_ckpt_path("chameleon", "tokenizer", "text_tokenizer.json"),
    )
    parser.add_argument(
        "--chameleon-vqgan-config",
        default=_default_ckpt_path("chameleon", "tokenizer", "vqgan.yaml"),
    )
    parser.add_argument(
        "--chameleon-vqgan-ckpt",
        default=_default_ckpt_path("chameleon", "tokenizer", "vqgan.ckpt"),
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--time-horizon", type=int, default=5)
    parser.add_argument("--token-dim", type=int, default=4096)
    parser.add_argument("--action-head-type", default="legacy", choices=["legacy"])
    parser.add_argument("--pool", default="mean", choices=["mean", "last"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_dir = _project_path(args.hdf5_dir)
    if args.out_dir is None:
        args.out_dir = str(
            PROJECT_ROOT
            / "data"
            / "processed_data"
            / "libero_goal_no_noops_t_256_legacy_action_hidden_vla_policy_h2"
        )
    out_dir = _project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank, world_size, _local_rank, device = _init_distributed()
    run_id = (
        os.environ.get("RYNN_HIDDEN_RUN_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or f"manual_{int(time.time())}"
    )

    files = sorted(hdf5_dir.glob("*.hdf5"))
    if args.max_files is not None:
        files = files[: int(args.max_files)]
    if not files:
        raise RuntimeError(f"No HDF5 files found under {hdf5_dir}")
    assigned = files[rank::world_size]
    assigned_stats = {
        stat["file"]: stat for stat in (_source_stats(path, args) for path in assigned)
    }
    progress_dir = out_dir / ".progress"
    total_demos = 0
    total_frames = 0
    stop_event: threading.Event | None = None
    monitor_thread: threading.Thread | None = None

    if rank == 0:
        all_stats = [_source_stats(path, args) for path in files]
        total_demos = sum(int(stat["demos"]) for stat in all_stats)
        total_frames = sum(int(stat["frames"]) for stat in all_stats)
        config_path = out_dir / "preprocess_config.json"
        config = vars(args).copy()
        config["hdf5_dir"] = str(hdf5_dir)
        config["out_dir"] = str(out_dir)
        config["num_source_files"] = len(files)
        config["world_size"] = world_size
        config["run_id"] = run_id
        config["total_demos"] = total_demos
        config["total_frames"] = total_frames
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        (out_dir / "preprocess_plan.json").write_text(
            json.dumps(
                {
                    "total_demos": total_demos,
                    "total_frames": total_frames,
                    "run_id": run_id,
                    "files": all_stats,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        progress_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[rynn-hidden] source={hdf5_dir} out={out_dir} "
            f"files={len(files)} demos={total_demos} frames={total_frames} "
            f"world_size={world_size} run_id={run_id}"
        )

    _write_rank_progress(
        progress_dir,
        rank,
        run_id=run_id,
        completed_demos=0,
        completed_frames=0,
    )
    if rank == 0 and not bool(args.no_global_progress):
        if total_demos <= 0:
            plan_path = out_dir / "preprocess_plan.json"
            if plan_path.is_file():
                plan = json.loads(plan_path.read_text())
                total_demos = int(plan.get("total_demos", 0))
                total_frames = int(plan.get("total_frames", 0))
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_global_progress,
            kwargs={
                "progress_dir": progress_dir,
                "run_id": run_id,
                "total_demos": total_demos,
                "total_frames": total_frames,
                "world_size": world_size,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        monitor_thread.start()

    encoder = _make_encoder(args, device=device)
    rank_records: list[dict[str, Any]] = []
    completed_demos = 0
    completed_frames = 0
    for source_path in assigned:
        output_path = out_dir / source_path.name
        source_stat = assigned_stats[source_path.name]
        if output_path.exists():
            if args.overwrite:
                output_path.unlink()
            elif _is_complete_hdf5(
                output_path,
                require_actor_sequence=bool(args.save_actor_sequence),
            ):
                print(f"[rank{rank}] skip complete {output_path}")
                completed_demos += int(source_stat["demos"])
                completed_frames += int(source_stat["frames"])
                _write_rank_progress(
                    progress_dir,
                    rank,
                    run_id=run_id,
                    completed_demos=completed_demos,
                    completed_frames=completed_frames,
                    current_file=source_path.name,
                    done=False,
                )
                continue
            else:
                raise RuntimeError(
                    f"Refusing to use incomplete sidecar without --overwrite: {output_path}"
                )
        record = _write_source_sidecar(
            source_path=source_path,
            output_path=output_path,
            encoder=encoder,
            args=args,
            rank=rank,
            run_id=run_id,
            progress_dir=progress_dir,
            completed_demos_base=completed_demos,
            completed_frames_base=completed_frames,
        )
        completed_demos += int(record["demos"])
        completed_frames += int(record["frames"])
        _write_rank_progress(
            progress_dir,
            rank,
            run_id=run_id,
            completed_demos=completed_demos,
            completed_frames=completed_frames,
            current_file=source_path.name,
        )
        rank_records.append(record)
        print(
            f"[rank{rank}] wrote {output_path} "
            f"demos={record['demos']} frames={record['frames']} hidden_dim={record['hidden_dim']}"
        )

    _write_rank_progress(
        progress_dir,
        rank,
        run_id=run_id,
        completed_demos=completed_demos,
        completed_frames=completed_frames,
        done=True,
    )

    (out_dir / f"manifest_rank{rank:03d}.json").write_text(
        json.dumps(rank_records, indent=2, sort_keys=True) + "\n"
    )
    if rank == 0:
        if monitor_thread is not None:
            while not _all_ranks_done(progress_dir, run_id, world_size):
                time.sleep(2.0)
        if stop_event is not None:
            stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
        print("[rynn-hidden] done")


if __name__ == "__main__":
    main()
