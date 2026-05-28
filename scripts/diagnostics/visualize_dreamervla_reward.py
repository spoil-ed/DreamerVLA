#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.world_model.dreamerv3_torch import _reward_pred


def _strip_prefix(key: str) -> str:
    for prefix in ("_fsdp_wrapped_module.", "module."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _load_world_model(
    ckpt_path: Path, device: torch.device
) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    wm_cfg = OmegaConf.select(cfg, "world_model")
    if wm_cfg is None:
        raise RuntimeError(f"{ckpt_path} has no world_model config")
    world_model = instantiate(wm_cfg).to(device)
    fsdp_precision = str(
        OmegaConf.select(cfg, "training.fsdp_mixed_precision", default="bf16")
    )
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    world_model = world_model.to(dtype=dtype_map.get(fsdp_precision, torch.bfloat16))
    state = payload.get("state_dicts", {}).get("world_model")
    if state is None:
        raise RuntimeError(f"{ckpt_path} has no state_dicts.world_model")
    target_sd = world_model.state_dict()
    remapped = {}
    for key, value in state.items():
        key = _strip_prefix(key)
        if key.startswith("reward_head.net.") and not key.startswith(
            "reward_head.net.net."
        ):
            candidate = key.replace("reward_head.net.", "reward_head.net.net.", 1)
            if candidate in target_sd:
                key = candidate
        remapped[key] = value
    missing, unexpected = world_model.load_state_dict(remapped, strict=False)
    world_model.eval()
    return world_model, {
        "missing": missing,
        "unexpected": unexpected,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }


def _iter_demo_keys(hdf5_dir: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for h5_path in sorted(hdf5_dir.glob("*.hdf5")):
        with h5py.File(h5_path, "r") as handle:
            for demo_key in sorted(handle["data"].keys()):
                out.append((h5_path, demo_key))
    return out


def _load_demo_arrays(
    h5_path: Path, hidden_dir: Path, demo_key: str, hidden_key: str
) -> dict[str, np.ndarray]:
    hidden_path = hidden_dir / h5_path.name
    with h5py.File(h5_path, "r") as src, h5py.File(hidden_path, "r") as hid:
        demo = src["data"][demo_key]
        actions_raw = np.asarray(demo["actions"], dtype=np.float32)
        prev_actions = np.zeros_like(actions_raw, dtype=np.float32)
        if len(actions_raw) > 1:
            prev_actions[1:] = actions_raw[:-1]
        rewards = np.asarray(demo["rewards"], dtype=np.float32)
        dones = np.asarray(demo["dones"], dtype=np.float32)
        hidden = np.asarray(hid["data"][demo_key][hidden_key], dtype=np.float32)
    n = min(len(prev_actions), len(rewards), len(dones), len(hidden))
    return {
        "actions": prev_actions[:n],
        "rewards": rewards[:n],
        "dones": dones[:n],
        "hidden": hidden[:n],
    }


@torch.no_grad()
def _predict_reward(
    world_model: Any, arrays: dict[str, np.ndarray], device: torch.device
) -> np.ndarray:
    obs = torch.from_numpy(arrays["hidden"]).unsqueeze(0).to(device)
    actions = torch.from_numpy(arrays["actions"]).unsqueeze(0).to(device)
    is_first = torch.zeros((1, obs.shape[1]), dtype=torch.bool, device=device)
    is_first[:, 0] = True
    batch = {"obs_embedding": obs, "actions": actions, "is_first": is_first}
    seq = world_model.observe_sequence(batch)
    logits = world_model.reward_head(seq["feat"])
    pred = _reward_pred(world_model.reward_head, logits).squeeze(0).squeeze(-1)
    return pred.float().detach().cpu().numpy()


def _plot_demo(
    out_png: Path, pred: np.ndarray, true_reward: np.ndarray, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(pred))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, pred, label="WM predicted reward", color="#2563eb", linewidth=2.0)
    positive = np.where(true_reward > 0)[0]
    if len(positive):
        ax.vlines(
            positive,
            ymin=0.0,
            ymax=max(float(pred.max()), float(true_reward.max()), 1e-3),
            colors="#111827",
            linestyles="--",
            linewidth=1.2,
            label="true reward > 0",
        )
    ax.plot(
        x,
        true_reward,
        label="true sparse reward",
        color="#dc2626",
        alpha=0.75,
        linewidth=1.0,
    )
    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--hdf5-dir", required=True, type=Path)
    parser.add_argument("--hidden-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hidden-key", default="obs_embedding")
    parser.add_argument("--num-demos", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefer-positive", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device
        if torch.cuda.is_available() or not args.device.startswith("cuda")
        else "cpu"
    )
    world_model, load_info = _load_world_model(args.ckpt, device)

    candidates = _iter_demo_keys(args.hdf5_dir)
    selected: list[tuple[Path, str, dict[str, np.ndarray]]] = []
    fallback: list[tuple[Path, str, dict[str, np.ndarray]]] = []
    for h5_path, demo_key in candidates:
        arrays = _load_demo_arrays(h5_path, args.hidden_dir, demo_key, args.hidden_key)
        item = (h5_path, demo_key, arrays)
        if arrays["rewards"].max(initial=0.0) > 0:
            selected.append(item)
        elif not args.prefer_positive:
            selected.append(item)
        else:
            fallback.append(item)
        if len(selected) >= args.num_demos:
            break
    if not selected:
        selected = fallback[: args.num_demos]

    summary_rows = []
    for idx, (h5_path, demo_key, arrays) in enumerate(selected):
        pred = _predict_reward(world_model, arrays, device)
        rewards = arrays["rewards"]
        stem = f"{idx:02d}_{h5_path.stem}_{demo_key}"
        csv_path = args.out_dir / f"{stem}.csv"
        png_path = args.out_dir / f"{stem}.png"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestep", "pred_reward", "true_reward", "done"])
            for t, (p, r, d) in enumerate(zip(pred, rewards, arrays["dones"])):
                writer.writerow([t, float(p), float(r), float(d)])
        _plot_demo(png_path, pred, rewards, f"{h5_path.name}:{demo_key}")
        positive = np.where(rewards > 0)[0]
        tail = slice(max(0, len(pred) - 10), len(pred))
        pre = slice(0, max(1, len(pred) - 10))
        summary_rows.append(
            {
                "file": h5_path.name,
                "demo": demo_key,
                "timesteps": int(len(pred)),
                "true_positive_steps": positive.tolist(),
                "pred_min": float(pred.min()),
                "pred_max": float(pred.max()),
                "pred_mean": float(pred.mean()),
                "pred_tail10_mean": float(pred[tail].mean()),
                "pred_before_tail_mean": float(pred[pre].mean()),
                "csv": str(csv_path),
                "plot": str(png_path),
            }
        )

    with (args.out_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "ckpt": str(args.ckpt),
                "hdf5_dir": str(args.hdf5_dir),
                "hidden_dir": str(args.hidden_dir),
                "device": str(device),
                "load_missing": list(load_info["missing"]),
                "load_unexpected": list(load_info["unexpected"]),
                "demos": summary_rows,
            },
            handle,
            indent=2,
        )
    print(f"wrote reward visualizations -> {args.out_dir}")
    for row in summary_rows:
        print(
            f"{row['file']}:{row['demo']} max={row['pred_max']:.6f} "
            f"tail10={row['pred_tail10_mean']:.6f} before_tail={row['pred_before_tail_mean']:.6f} "
            f"true_pos={row['true_positive_steps']}"
        )


if __name__ == "__main__":
    main()
