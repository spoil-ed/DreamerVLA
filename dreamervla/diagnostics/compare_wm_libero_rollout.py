"""Render aligned collected/LIBERO/world-model trajectory comparisons.

The diagnostic starts every branch from one collected rollout.  Stored raw
LIBERO actions are replayed in a freshly constructed physics environment while
the same actions drive a closed-loop ChunkAwareWorldModel.  A separately
trained latent pixel decoder turns both the true encoder tokens and imagined
tokens into base/wrist RGB, isolating decoder-only error from world-model drift.

Example::

    python -m dreamervla.diagnostics.compare_wm_libero_rollout \
      --data-dir /path/to/collected_rollouts/pi05_libero_object/reward \
      --wm-ckpt /path/to/wm-run/checkpoints/latest.ckpt \
      --decoder-ckpt /path/to/decoder-run/checkpoints/latest.ckpt \
      --output-dir /path/to/comparison-output \
      --task-id 0 --outcomes success failure --num-chunks 10 --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hydra
import imageio
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from dreamervla.models.embodiment.pi05.policy import _libero_eval_image
from dreamervla.models.embodiment.world_model.latent_pixel_decoder import (
    latent_pixel_reconstruction_loss,
)
from dreamervla.models.embodiment.world_model.wm_chunk import ChunkAwareWorldModel
from dreamervla.utils.run_config import load_run_config

VIEW_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
VIEW_LABELS = ("base", "wrist")


@dataclass(frozen=True)
class CollectedTrajectory:
    """The time-aligned fields needed by both rollout branches."""

    path: Path
    demo_key: str
    task_id: int
    episode_id: int
    init_state_index: int
    success: bool
    task_description: str
    actions: np.ndarray
    proprio: np.ndarray
    states: np.ndarray
    base_images: np.ndarray
    wrist_images: np.ndarray
    init_state: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


def _policy_hydra_config(value: Any) -> DictConfig:
    """Normalize the two supported component-config spellings without runner imports.

    Importing a training runner pulls the complete dataset stack (including
    TensorFlow) into this process.  Mesa must be initialized before that stack
    on the local workstation, so this diagnostic keeps component normalization
    deliberately lightweight.
    """

    if isinstance(value, DictConfig) and OmegaConf.select(value, "_target_", default=None):
        return value
    raw = OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value
    if not isinstance(raw, Mapping):
        raise TypeError("policy config must be a mapping")
    target = raw.get("target")
    kwargs = raw.get("kwargs", {})
    if not target or not isinstance(kwargs, Mapping):
        raise ValueError("policy config requires target and kwargs")
    return OmegaConf.create({"_target_": str(target), **dict(kwargs)})


def _world_model_hydra_config(cfg: DictConfig) -> DictConfig:
    direct = cfg.get("world_model")
    if isinstance(direct, DictConfig) and OmegaConf.select(direct, "_target_", default=None):
        return direct
    component = OmegaConf.select(cfg, "ray_components.world_model", default=None)
    if not isinstance(component, DictConfig):
        raise ValueError("run config has no world_model or ray_components.world_model")
    target = component.get("target")
    kwargs = component.get("kwargs", {})
    if not target or not isinstance(kwargs, (DictConfig, Mapping)):
        raise ValueError("ray_components.world_model requires target and kwargs")
    raw_kwargs = (
        OmegaConf.to_container(kwargs, resolve=True) if isinstance(kwargs, DictConfig) else kwargs
    )
    return OmegaConf.create({"_target_": str(target), **dict(raw_kwargs)})


def _load_chunk_wm(checkpoint_path: Path, device: torch.device) -> ChunkAwareWorldModel:
    cfg = load_run_config(checkpoint_path)
    model = hydra.utils.instantiate(_world_model_hydra_config(cfg))
    if not isinstance(model, ChunkAwareWorldModel):
        raise TypeError(f"world-model checkpoint instantiated {type(model).__name__}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dicts = payload.get("state_dicts")
    if isinstance(state_dicts, dict) and isinstance(state_dicts.get("world_model"), dict):
        state = state_dicts["world_model"]
    elif isinstance(payload.get("model"), dict):
        state = payload["model"]
    elif isinstance(payload.get("world_model"), dict):
        state = payload["world_model"]
    else:
        raise ValueError(f"checkpoint has no world-model state: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    print(
        f"[load] WM warmup_epoch={payload.get('warmup_epoch')} "
        f"warmup_step={payload.get('warmup_step')}",
        flush=True,
    )
    return model.eval().to(device)


@torch.inference_mode()
def _rollout_closed_loop(
    wm: ChunkAwareWorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    num_chunks: int,
    *,
    proprio: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact training-aligned autoregressive chunk protocol."""

    history_frames = int(wm.num_hist)
    chunk_size = int(wm.chunk_size)
    time_steps = int(obs.shape[0])
    chunks = min(int(num_chunks), (time_steps - history_frames) // chunk_size)
    if chunks < 1:
        raise ValueError(
            f"trajectory has T={time_steps}, need at least {history_frames + chunk_size}"
        )
    vision_tokens = wm._normalize_raw_vision_tokens(obs.unsqueeze(0))
    if int(wm.proprio_condition_dim) > 0:
        if proprio is None:
            raise ValueError("conditioned WM rollout requires proprio")
        proprio_batch = proprio.unsqueeze(0)
        obs_tokens = wm._observation_tokens(vision_tokens, proprio_batch)[0]
    else:
        proprio_batch = None
        obs_tokens = vision_tokens[0]
    history = obs_tokens[:history_frames].unsqueeze(0)
    action_history = torch.zeros(
        1,
        history_frames,
        int(wm.action_dim),
        device=obs.device,
        dtype=obs.dtype,
    )
    if history_frames > 1:
        action_history[:, : history_frames - 1] = actions[: history_frames - 1].unsqueeze(0)
    current: dict[str, torch.Tensor | None] = {
        "hidden": history[:, -1],
        "history": history,
        "actions": action_history,
        "lang": None,
    }
    if proprio_batch is not None:
        current["proprio"] = proprio_batch[:, history_frames - 1]

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for chunk_index in range(chunks):
        begin = history_frames - 1 + chunk_index * chunk_size
        action_chunk = actions[begin : begin + chunk_size].unsqueeze(0)
        output = wm.predict_next_chunk(current, action_chunk)
        predictions.append(output["hidden_seq"][0])
        target_begin = history_frames + chunk_index * chunk_size
        targets.append(obs_tokens[target_begin : target_begin + chunk_size])
        current = {
            "hidden": output["hidden"],
            "history": output["history"],
            "actions": output["actions"],
            "lang": output.get("lang"),
        }
        if isinstance(output.get("proprio"), torch.Tensor):
            current["proprio"] = output["proprio"]
    return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)


def _select_trajectory_rows(
    rows: list[dict[str, Any]],
    *,
    task_id: int,
    outcomes: tuple[str, ...],
    min_length: int,
) -> list[dict[str, Any]]:
    """Select the first deterministic episode for every requested outcome."""

    selected: list[dict[str, Any]] = []
    for outcome in outcomes:
        normalized = str(outcome).strip().lower()
        if normalized not in {"success", "failure"}:
            raise ValueError("outcomes must contain only 'success' and/or 'failure'")
        wanted_success = normalized == "success"
        candidates = [
            row
            for row in rows
            if int(row.get("task_id", -1)) == int(task_id)
            and bool(row.get("success", False)) is wanted_success
            and int(row.get("horizon", 0)) >= int(min_length)
        ]
        if not candidates:
            raise RuntimeError(
                f"no task {task_id} {normalized} trajectory has at least {min_length} frames"
            )
        selected.append(
            min(
                candidates,
                key=lambda row: (
                    int(row.get("episode_id", 0)),
                    str(row.get("file", "")),
                ),
            )
        )
    return selected


def _load_trajectory(data_dir: Path, row: dict[str, Any], length: int) -> CollectedTrajectory:
    file_name = row.get("file")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("episode_index.jsonl row is missing a non-empty 'file'")
    path = data_dir / file_name
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        if data is None or not data.keys():
            raise ValueError(f"collected shard has no demo under data/: {path}")
        demo_key = str(row.get("demo_key") or next(iter(data.keys())))
        demo = data[demo_key]
        obs = demo["obs"]
        available = min(
            int(demo["actions"].shape[0]),
            int(demo["states"].shape[0]),
            *(int(obs[key].shape[0]) for key in (*VIEW_KEYS, "ee_pos", "ee_ori", "gripper_states")),
        )
        if available < int(length):
            raise ValueError(f"{path.name}/{demo_key} has {available} frames, need {length}")
        proprio = np.concatenate(
            [
                np.asarray(obs[key][:length], dtype=np.float32)
                for key in ("ee_pos", "ee_ori", "gripper_states")
            ],
            axis=-1,
        )
        return CollectedTrajectory(
            path=path,
            demo_key=demo_key,
            task_id=int(demo.attrs.get("task_id", row["task_id"])),
            episode_id=int(demo.attrs.get("episode_id", row["episode_id"])),
            init_state_index=int(demo.attrs.get("init_state_index", row["init_state_index"])),
            success=bool(demo.attrs.get("success", row["success"])),
            task_description=str(demo.attrs.get("task_description", "")),
            actions=np.asarray(demo["actions"][:length], dtype=np.float32),
            proprio=proprio,
            states=np.asarray(demo["states"][:length], dtype=np.float64),
            base_images=np.asarray(obs[VIEW_KEYS[0]][:length], dtype=np.uint8),
            wrist_images=np.asarray(obs[VIEW_KEYS[1]][:length], dtype=np.uint8),
            init_state=np.asarray(demo.attrs["init_state"], dtype=np.float64),
        )


def _load_pixel_decoder(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    cfg = load_run_config(checkpoint_path)
    decoder_cfg = cfg.get("pixel_decoder")
    if not isinstance(decoder_cfg, DictConfig):
        raise ValueError(f"decoder run config has no pixel_decoder target: {checkpoint_path}")
    decoder = hydra.utils.instantiate(decoder_cfg)
    if not isinstance(decoder, torch.nn.Module):
        raise TypeError("pixel_decoder config did not instantiate torch.nn.Module")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dicts = payload.get("state_dicts")
    if not isinstance(state_dicts, dict) or not isinstance(state_dicts.get("pixel_decoder"), dict):
        raise ValueError(f"checkpoint has no state_dicts.pixel_decoder: {checkpoint_path}")
    state = state_dicts["pixel_decoder"]
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[len("module.") :]: value for key, value in state.items()}
    decoder.load_state_dict(state, strict=True)
    return decoder.eval().to(device)


@torch.inference_mode()
def _encode_trajectories(
    trajectories: list[CollectedTrajectory],
    *,
    policy_cfg: DictConfig,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    policy = hydra.utils.instantiate(_policy_hydra_config(policy_cfg))
    if not isinstance(policy, torch.nn.Module):
        raise TypeError("online_latent.policy did not instantiate torch.nn.Module")
    encode = getattr(policy, "encode_raw_observation_prefix_batch", None)
    if not callable(encode):
        raise TypeError("online_latent.policy must implement encode_raw_observation_prefix_batch")
    policy.eval().to(device)
    encoded_trajectories: list[torch.Tensor] = []
    for trajectory in trajectories:
        chunks: list[torch.Tensor] = []
        for begin in range(0, trajectory.length, int(batch_size)):
            end = min(trajectory.length, begin + int(batch_size))
            raw = [
                {
                    "observation/image": _libero_eval_image(
                        trajectory.base_images[index], rotate_180=True
                    ),
                    "observation/wrist_image": _libero_eval_image(
                        trajectory.wrist_images[index], rotate_180=True
                    ),
                    "observation/state": trajectory.proprio[index],
                    "prompt": trajectory.task_description,
                }
                for index in range(begin, end)
            ]
            prefix = encode(raw)
            if not isinstance(prefix, torch.Tensor):
                raise TypeError("π0.5 prefix encoder must return torch.Tensor")
            chunks.append(prefix.detach().to(device="cpu", dtype=torch.float16))
            print(
                f"[encode] task={trajectory.task_id} episode={trajectory.episode_id} "
                f"frames={end}/{trajectory.length}",
                flush=True,
            )
        encoded_trajectories.append(torch.cat(chunks, dim=0))
    del policy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return encoded_trajectories


@torch.inference_mode()
def _closed_loop_tokens(
    wm: torch.nn.Module,
    trajectory: CollectedTrajectory,
    encoded: torch.Tensor,
    *,
    num_chunks: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    obs = encoded.to(device=device, dtype=torch.float32)
    actions = torch.from_numpy(trajectory.actions).to(device=device, dtype=torch.float32)
    proprio = torch.from_numpy(trajectory.proprio).to(device=device, dtype=torch.float32)
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )
    with autocast:
        predicted, target = _rollout_closed_loop(
            wm,
            obs,
            actions,
            int(num_chunks),
            proprio=proprio,
        )
    visual_width = int(wm.token_dim)
    pred_visual = predicted[..., :visual_width].float()
    target_visual = target[..., :visual_width].float()
    pred_flat = pred_visual.flatten(1)
    target_flat = target_visual.flatten(1)
    mse = torch.mean((pred_flat - target_flat).square(), dim=1)
    cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1)
    metrics = {
        "latent_mse_mean": float(mse.mean().item()),
        "latent_mse_final": float(mse[-1].item()),
        "latent_cosine_mean": float(cosine.mean().item()),
        "latent_cosine_final": float(cosine[-1].item()),
    }
    return predicted.detach().to(device="cpu", dtype=torch.float16), metrics


@torch.inference_mode()
def _decode_tokens(
    decoder: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    images: list[np.ndarray] = []
    for begin in range(0, int(tokens.shape[0]), int(batch_size)):
        batch = tokens[begin : begin + int(batch_size)].to(device=device, dtype=torch.float32)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = decoder(batch)
        output = (
            output.detach()
            .float()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 1, 3, 4, 2)
            .cpu()
            .numpy()
        )
        images.append(output)
    return np.concatenate(images, axis=0)


def _replay_libero(
    trajectory: CollectedTrajectory,
    *,
    suite_name: str,
    resolution: int,
    render_backend: str,
    render_gpu: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay in an isolated process so Mesa and OpenPI LLVM never collide."""

    worker_env = os.environ.copy()
    worker_env["MUJOCO_GL"] = str(render_backend)
    worker_env["PYOPENGL_PLATFORM"] = str(render_backend)
    with tempfile.TemporaryDirectory(prefix="dvla-libero-replay-") as temp_dir:
        output_path = Path(temp_dir) / "replay.npz"
        command = [
            sys.executable,
            "-u",
            "-m",
            "dreamervla.diagnostics.replay_collected_libero",
            "--trajectory-file",
            str(trajectory.path),
            "--demo-key",
            trajectory.demo_key,
            "--length",
            str(trajectory.length),
            "--suite-name",
            str(suite_name),
            "--task-id",
            str(trajectory.task_id),
            "--resolution",
            str(int(resolution)),
            "--render-backend",
            str(render_backend),
            "--render-gpu",
            str(int(render_gpu)),
            "--seed",
            str(int(seed)),
            "--output",
            str(output_path),
        ]
        subprocess.run(command, check=True, env=worker_env)
        with np.load(output_path, allow_pickle=False) as replay:
            frames = np.asarray(replay["frames"], dtype=np.uint8).copy()
            states = np.asarray(replay["states"], dtype=np.float64).copy()
    return frames, states


def _model_space_reference(trajectory: CollectedTrajectory, image_size: int) -> np.ndarray:
    source = np.stack([trajectory.base_images, trajectory.wrist_images], axis=1)
    output = np.empty((trajectory.length, 2, int(image_size), int(image_size), 3), dtype=np.uint8)
    for time_index in range(trajectory.length):
        for view_index in range(2):
            image = np.ascontiguousarray(source[time_index, view_index][::-1, ::-1])
            output[time_index, view_index] = np.asarray(
                Image.fromarray(image).resize(
                    (int(image_size), int(image_size)), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
    return output


def _resize_views(frames: np.ndarray, image_size: int) -> np.ndarray:
    output = np.empty(
        (int(frames.shape[0]), 2, int(image_size), int(image_size), 3), dtype=np.uint8
    )
    for time_index in range(int(frames.shape[0])):
        for view_index in range(2):
            output[time_index, view_index] = np.asarray(
                Image.fromarray(frames[time_index, view_index]).resize(
                    (int(image_size), int(image_size)), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
    return output


def _source_panel(
    views: np.ndarray,
    *,
    source_label: str,
    step: int,
    phase: str,
) -> np.ndarray:
    """Stack base/wrist views with an even-height textual header."""

    image_size = int(views.shape[1])
    header = 28
    canvas = Image.new("RGB", (image_size, header + 2 * image_size), color=(24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), f"{source_label}  t={step:03d}  {phase}", fill=(245, 245, 245))
    canvas.paste(Image.fromarray(views[0]), (0, header))
    canvas.paste(Image.fromarray(views[1]), (0, header + image_size))
    draw.rectangle((0, header, 43, header + 15), fill=(0, 0, 0))
    draw.text((4, header + 1), VIEW_LABELS[0], fill=(255, 255, 255))
    draw.rectangle((0, header + image_size, 43, header + image_size + 15), fill=(0, 0, 0))
    draw.text((4, header + image_size + 1), VIEW_LABELS[1], fill=(255, 255, 255))
    return np.asarray(canvas, dtype=np.uint8)


def _comparison_frame(
    reference: np.ndarray,
    libero: np.ndarray,
    oracle: np.ndarray,
    imagined: np.ndarray,
    *,
    step: int,
    warmup_frames: int,
) -> np.ndarray:
    phase = "warm-start" if int(step) < int(warmup_frames) else "closed-loop"
    panels = [
        _source_panel(reference, source_label="collected", step=step, phase="reference"),
        _source_panel(libero, source_label="LIBERO", step=step, phase="physics"),
        _source_panel(oracle, source_label="true-latent decoder", step=step, phase="decoder-only"),
        _source_panel(imagined, source_label="WM decoder", step=step, phase=phase),
    ]
    return np.concatenate(panels, axis=1)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=int(fps),
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def _pixel_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = torch.from_numpy(prediction).permute(0, 1, 4, 2, 3).float() / 255.0
    true = torch.from_numpy(target).permute(0, 1, 4, 2, 3).float() / 255.0
    metrics = latent_pixel_reconstruction_loss(pred, true, l1_weight=1.0, ssim_weight=0.0)
    return {
        "mae": float(metrics["l1"].item()),
        "ssim": float(metrics["ssim"].item()),
        "psnr_db": float(metrics["psnr"].item()),
    }


def _pixel_metrics_by_view(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Report joint and per-camera reconstruction metrics."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "pixel arrays must have equal [T,V,H,W,C] shapes, got "
            f"{prediction.shape} and {target.shape}"
        )
    if int(prediction.shape[1]) != len(VIEW_LABELS):
        raise ValueError(f"expected {len(VIEW_LABELS)} views, got {prediction.shape[1]}")
    return {
        "all_views": _pixel_metrics(prediction, target),
        "by_view": {
            label: _pixel_metrics(prediction[:, index : index + 1], target[:, index : index + 1])
            for index, label in enumerate(VIEW_LABELS)
        },
    }


def _per_frame_pixel_metrics(
    oracle: np.ndarray,
    imagined: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, Any]]:
    """Return inexpensive per-frame/per-view MAE and PSNR attribution metrics."""

    if oracle.shape != imagined.shape or oracle.shape != target.shape or oracle.ndim != 5:
        raise ValueError("oracle, imagined, and target must share [T,V,H,W,C] shape")

    def series(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        error = prediction.astype(np.float32) / 255.0 - target.astype(np.float32) / 255.0
        mae = np.mean(np.abs(error), axis=(2, 3, 4))
        mse = np.mean(np.square(error), axis=(2, 3, 4))
        psnr = -10.0 * np.log10(np.maximum(mse, 1.0e-10))
        return mae, psnr

    oracle_mae, oracle_psnr = series(oracle)
    wm_mae, wm_psnr = series(imagined)
    rows: list[dict[str, Any]] = []
    for step in range(int(target.shape[0])):
        rows.append(
            {
                "step": step,
                "decoder_only": {
                    label: {
                        "mae": float(oracle_mae[step, index]),
                        "psnr_db": float(oracle_psnr[step, index]),
                    }
                    for index, label in enumerate(VIEW_LABELS)
                },
                "wm_decoder": {
                    label: {
                        "mae": float(wm_mae[step, index]),
                        "psnr_db": float(wm_psnr[step, index]),
                    }
                    for index, label in enumerate(VIEW_LABELS)
                },
            }
        )
    return rows


def _run_one(
    trajectory: CollectedTrajectory,
    encoded: torch.Tensor,
    live_raw: np.ndarray,
    live_states: np.ndarray,
    *,
    wm: ChunkAwareWorldModel,
    decoder: torch.nn.Module,
    output_dir: Path,
    num_chunks: int,
    decode_batch_size: int,
    device: torch.device,
    fps: int,
) -> dict[str, Any]:
    predicted, latent_metrics = _closed_loop_tokens(
        wm,
        trajectory,
        encoded,
        num_chunks=num_chunks,
        device=device,
    )
    warmup_frames = int(wm.num_hist)
    oracle_reconstruction = _decode_tokens(
        decoder,
        encoded,
        device=device,
        batch_size=decode_batch_size,
    )
    imagined_reconstruction = _decode_tokens(
        decoder,
        predicted,
        device=device,
        batch_size=decode_batch_size,
    )
    imagined = np.concatenate(
        [oracle_reconstruction[:warmup_frames], imagined_reconstruction], axis=0
    )
    reference = _model_space_reference(trajectory, imagined.shape[2])
    live = _resize_views(live_raw, imagined.shape[2])
    if not (len(reference) == len(live) == len(oracle_reconstruction) == len(imagined)):
        raise RuntimeError(
            f"aligned video length mismatch: reference={len(reference)}, "
            f"libero={len(live)}, decoder={len(oracle_reconstruction)}, wm={len(imagined)}"
        )

    outcome = "success" if trajectory.success else "failure"
    name = f"task={trajectory.task_id:02d}_episode={trajectory.episode_id:06d}_{outcome}"
    trajectory_dir = output_dir / name
    comparison_frames = [
        _comparison_frame(
            reference[index],
            live[index],
            oracle_reconstruction[index],
            imagined[index],
            step=index,
            warmup_frames=warmup_frames,
        )
        for index in range(len(reference))
    ]
    reference_frames = [
        _source_panel(frame, source_label="collected", step=index, phase="reference")
        for index, frame in enumerate(reference)
    ]
    libero_frames = [
        _source_panel(frame, source_label="LIBERO", step=index, phase="physics")
        for index, frame in enumerate(live)
    ]
    oracle_frames = [
        _source_panel(
            frame,
            source_label="true-latent decoder",
            step=index,
            phase="decoder-only",
        )
        for index, frame in enumerate(oracle_reconstruction)
    ]
    imagined_frames = [
        _source_panel(
            frame,
            source_label="WM decoder",
            step=index,
            phase="warm-start" if index < warmup_frames else "closed-loop",
        )
        for index, frame in enumerate(imagined)
    ]
    videos = {
        "comparison": trajectory_dir / "comparison.mp4",
        "collected_reference": trajectory_dir / "collected_reference.mp4",
        "libero_replay": trajectory_dir / "libero_replay.mp4",
        "decoder_only": trajectory_dir / "decoder_only.mp4",
        "wm_imagined": trajectory_dir / "wm_imagined.mp4",
    }
    _write_video(videos["comparison"], comparison_frames, fps)
    _write_video(videos["collected_reference"], reference_frames, fps)
    _write_video(videos["libero_replay"], libero_frames, fps)
    _write_video(videos["decoder_only"], oracle_frames, fps)
    _write_video(videos["wm_imagined"], imagined_frames, fps)

    horizon_slice = slice(warmup_frames, None)
    decoder_only_metrics = _pixel_metrics_by_view(
        oracle_reconstruction[horizon_slice], reference[horizon_slice]
    )
    wm_decoder_metrics = _pixel_metrics_by_view(imagined[horizon_slice], reference[horizon_slice])
    decoder_joint = decoder_only_metrics["all_views"]
    wm_joint = wm_decoder_metrics["all_views"]
    state_delta = live_states - trajectory.states
    state_rmse = np.sqrt(np.mean(np.square(state_delta), axis=1))
    metrics: dict[str, Any] = {
        "task_id": trajectory.task_id,
        "episode_id": trajectory.episode_id,
        "init_state_index": trajectory.init_state_index,
        "collected_success": trajectory.success,
        "task_description": trajectory.task_description,
        "frames": trajectory.length,
        "warmup_frames": warmup_frames,
        "imagined_frames": int(predicted.shape[0]),
        "world_model": latent_metrics,
        "decoder_only_pixels_vs_collected": decoder_only_metrics,
        "wm_decoder_pixels_vs_collected": wm_decoder_metrics,
        "wm_penalty_over_decoder": {
            "mae_increase": float(wm_joint["mae"] - decoder_joint["mae"]),
            "ssim_decrease": float(decoder_joint["ssim"] - wm_joint["ssim"]),
            "psnr_drop_db": float(decoder_joint["psnr_db"] - wm_joint["psnr_db"]),
        },
        "libero_pixels_vs_collected": _pixel_metrics(live, reference),
        "libero_state_vs_collected": {
            "rmse_mean": float(state_rmse.mean()),
            "rmse_final": float(state_rmse[-1]),
            "max_abs": float(np.max(np.abs(state_delta))),
        },
        "videos": {key: str(value) for key, value in videos.items()},
    }
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    (trajectory_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    per_frame = _per_frame_pixel_metrics(
        oracle_reconstruction,
        imagined,
        reference,
    )
    (trajectory_dir / "per_frame_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_frame),
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--wm-ckpt", type=Path, required=True)
    parser.add_argument("--decoder-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite-name", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--outcomes", nargs="+", default=("success", "failure"))
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-backend", choices=("osmesa", "egl"), default="osmesa")
    parser.add_argument("--render-gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.num_chunks) <= 0:
        raise ValueError("num_chunks must be positive")
    if int(args.encode_batch_size) <= 0 or int(args.decode_batch_size) <= 0:
        raise ValueError("encode/decode batch sizes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    data_dir = args.data_dir.expanduser().resolve()
    index_path = data_dir / "episode_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]

    wm_cfg = load_run_config(args.wm_ckpt)
    chunk_size = int(wm_cfg.ray_components.world_model.kwargs.chunk_size)
    num_hist = int(wm_cfg.ray_components.world_model.kwargs.num_hist)
    length = num_hist + int(args.num_chunks) * chunk_size
    max_seq_len = int(wm_cfg.ray_components.world_model.kwargs.max_seq_len)
    if length > max_seq_len:
        raise ValueError(
            f"requested {length} frames exceeds WM max_seq_len={max_seq_len}; reduce --num-chunks"
        )
    selected_rows = _select_trajectory_rows(
        rows,
        task_id=int(args.task_id),
        outcomes=tuple(args.outcomes),
        min_length=length,
    )
    trajectories = [_load_trajectory(data_dir, row, length) for row in selected_rows]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[selection] "
        + ", ".join(
            f"task={item.task_id} episode={item.episode_id} "
            f"outcome={'success' if item.success else 'failure'} frames={item.length}"
            for item in trajectories
        ),
        flush=True,
    )

    online_latent = wm_cfg.offline_warmup.online_latent
    policy_cfg = online_latent.policy
    # OSMesa must own the process before OpenPI potentially imports the wider
    # training/data stack. Replaying first avoids a Mesa/TensorFlow shared-
    # library conflict observed on the local Ubuntu 24.04 workstation.
    live_replays = [
        _replay_libero(
            trajectory,
            suite_name=str(args.suite_name),
            resolution=int(args.resolution),
            render_backend=str(args.render_backend),
            render_gpu=int(args.render_gpu),
            seed=int(args.seed),
        )
        for trajectory in trajectories
    ]
    encoded = _encode_trajectories(
        trajectories,
        policy_cfg=policy_cfg,
        device=device,
        batch_size=int(args.encode_batch_size),
    )
    wm = _load_chunk_wm(args.wm_ckpt, device)
    decoder = _load_pixel_decoder(args.decoder_ckpt, device)
    results: list[dict[str, Any]] = []
    for trajectory, trajectory_tokens, live_replay in zip(
        trajectories, encoded, live_replays, strict=True
    ):
        live_raw, live_states = live_replay
        results.append(
            _run_one(
                trajectory,
                trajectory_tokens,
                live_raw,
                live_states,
                wm=wm,
                decoder=decoder,
                output_dir=output_dir,
                num_chunks=int(args.num_chunks),
                decode_batch_size=int(args.decode_batch_size),
                device=device,
                fps=int(args.fps),
            )
        )
    manifest = {
        "protocol": "same stored init_state and raw action sequence",
        "decoder_ablation": (
            "collected pixels vs true encoder latent decoded pixels vs "
            "closed-loop world-model latent decoded pixels"
        ),
        "suite_name": str(args.suite_name),
        "task_id": int(args.task_id),
        "num_hist": num_hist,
        "chunk_size": chunk_size,
        "num_chunks": int(args.num_chunks),
        "wm_checkpoint": str(args.wm_ckpt.expanduser().resolve()),
        "decoder_checkpoint": str(args.decoder_ckpt.expanduser().resolve()),
        "data_dir": str(data_dir),
        "results": results,
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] comparison videos: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
