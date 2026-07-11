#!/usr/bin/env python3
"""Real VLA-to-Chunk-WM overfit on one raw LIBERO demo, without sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from hydra.utils import instantiate

from dreamervla.diagnostics.wm_single_trajectory_overfit import (
    EpisodeArrays,
    RunSettings,
    _compose_config,
    run_overfit,
)
from dreamervla.diagnostics.wm_single_trajectory_raw_overfit import load_raw_episode
from dreamervla.preprocess.preprocess_oft_hidden_token import (
    _load_oft_components,
    _predict_hidden_token_chunk,
    _task_prompt_from_path,
)
from dreamervla.utils.paths import data_path


def _vla_args(checkpoint: Path, image_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        fake_oft_components=False,
        load_in_8bit=False,
        load_in_4bit=False,
        oft_ckpt=str(checkpoint),
        policy_mode="discrete",
        include_state=False,
        num_images_in_input=1,
        history=1,
        image_keys=(image_key,),
        center_crop=False,
        unnorm_key="libero_goal_no_noops",
        rotate_images_180=True,
        token_dim=4096,
        action_dim=7,
        time_horizon=8,
    )


def encode_raw_demo_with_vla(
    *,
    raw_hdf5: Path,
    demo_key: str,
    checkpoint: Path,
    device: torch.device,
    image_key: str,
    batch_size: int,
) -> tuple[EpisodeArrays, dict[str, int]]:
    """Run frozen VLA inference in memory and return WM-ready arrays."""

    raw_episode = load_raw_episode(raw_hdf5, demo_key)
    args = _vla_args(checkpoint, image_key)
    components = _load_oft_components(args, device)
    prompt = _task_prompt_from_path(raw_hdf5)
    hidden_chunks: list[np.ndarray] = []
    lang_embedding: np.ndarray | None = None
    with h5py.File(raw_hdf5, "r") as hdf5:
        obs_group = hdf5["data"][demo_key]["obs"]
        for start in range(0, raw_episode.states.shape[0], batch_size):
            end = min(start + batch_size, raw_episode.states.shape[0])
            hidden_token, lang_emb = _predict_hidden_token_chunk(
                components=components,
                args=args,
                obs_group=obs_group,
                image_keys=(image_key,),
                prompt=prompt,
                start=start,
                end=end,
            )
            hidden_chunks.append(hidden_token.astype(np.float32, copy=False))
            if lang_embedding is None:
                lang_embedding = lang_emb[0].astype(np.float32, copy=False)
    if lang_embedding is None:
        raise RuntimeError("VLA produced no language embedding")
    hidden = np.concatenate(hidden_chunks, axis=0)
    episode = EpisodeArrays(
        hidden=hidden,
        lang=lang_embedding,
        actions=raw_episode.actions,
        rewards=np.zeros((raw_episode.states.shape[0],), dtype=np.float32),
        proprio=raw_episode.states,
    )
    return episode, {
        "frames": int(hidden.shape[0]),
        "token_count": int(hidden.shape[1]),
        "token_dim": int(hidden.shape[2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--raw-hdf5",
        type=Path,
        default=data_path(
            "processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/"
            "no_noops_t_256_remaining_reward/"
            "open_the_middle_drawer_of_the_cabinet_demo.hdf5"
        ),
    )
    parser.add_argument(
        "--vla-ckpt",
        type=Path,
        default=data_path("checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"),
    )
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument("--image-key", default="agentview_rgb")
    parser.add_argument("--vla-batch-size", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=data_path("outputs/world_model_probe/single_trajectory_vla_overfit"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--mse-threshold", type=float, default=0.03)
    parser.add_argument("--cosine-threshold", type=float, default=0.95)
    parser.add_argument("--required-passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    plan = {
        "initialization": "random WM + frozen runtime VLA",
        "raw_hdf5": str(args.raw_hdf5),
        "vla_ckpt": str(args.vla_ckpt),
        "demo_key": args.demo_key,
        "image_key": args.image_key,
        "device": args.device,
        "precomputed_hidden_sidecar": False,
    }
    if not args.run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not args.vla_ckpt.is_dir():
        raise FileNotFoundError(f"VLA checkpoint directory not found: {args.vla_ckpt}")
    if args.vla_batch_size <= 0:
        raise ValueError("vla-batch-size must be positive")

    cfg = _compose_config("openvla_onetraj_libero")
    wm = instantiate(cfg.world_model)
    episode, hidden_meta = encode_raw_demo_with_vla(
        raw_hdf5=args.raw_hdf5,
        demo_key=args.demo_key,
        checkpoint=args.vla_ckpt,
        device=device,
        image_key=args.image_key,
        batch_size=args.vla_batch_size,
    )
    settings = RunSettings(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=1.0,
        eval_every=args.eval_every,
        mse_threshold=args.mse_threshold,
        cosine_threshold=args.cosine_threshold,
        required_passes=args.required_passes,
        seed=args.seed,
    )
    summary = run_overfit(
        model=wm,
        episode=episode,
        settings=settings,
        out_dir=args.out_dir,
        device=device,
    )
    summary.update(plan)
    summary["vla_hidden_meta"] = hidden_meta
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "converged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
