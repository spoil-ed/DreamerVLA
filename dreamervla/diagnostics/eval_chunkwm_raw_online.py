"""Evaluate a Chunk-WM on raw rollout frames with transient OpenVLA encoding.

No hidden-token sidecars are read or written. Each selected trajectory is
encoded in memory, evaluated, and released before the next trajectory.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from dreamervla.diagnostics.eval_chunkwm_closeloop import (
    load_chunk_wm,
    per_step_metrics,
    rollout,
)
from dreamervla.runtime.online_vla_hidden_encoder import OnlineVLAHiddenEncoder


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _select_task_balanced(
    raw_dir: Path,
    *,
    per_task: int,
) -> list[tuple[Path, str, int, bool]]:
    candidates: dict[int, list[tuple[Path, str, int, bool]]] = defaultdict(list)
    for path in sorted(raw_dir.glob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            for demo_name, demo in handle["data"].items():
                task_id = int(demo.attrs["task_id"])
                success = bool(
                    demo.attrs.get("success", demo.attrs.get("episode_success", False))
                )
                candidates[task_id].append(
                    (path, f"data/{demo_name}", task_id, success)
                )

    selected: list[tuple[Path, str, int, bool]] = []
    for task_id in sorted(candidates):
        rows = candidates[task_id]
        successes = [row for row in rows if row[3]]
        failures = [row for row in rows if not row[3]]
        task_rows: list[tuple[Path, str, int, bool]] = []
        while len(task_rows) < int(per_task) and (successes or failures):
            pool = successes if len(task_rows) % 2 == 0 and successes else failures
            if not pool:
                pool = successes
            task_rows.append(pool.pop(0))
        selected.extend(task_rows)
    return selected


def _load_raw_demo(
    path: Path,
    demo_key: str,
    *,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with h5py.File(path, "r") as handle:
        demo = handle[demo_key]
        steps = min(int(demo["actions"].shape[0]), int(max_steps))
        images = np.asarray(
            demo["obs"]["agentview_rgb"][:steps], dtype=np.uint8
        )
        actions = np.asarray(demo["actions"][:steps], dtype=np.float32)
        proprio = np.concatenate(
            [
                np.asarray(demo["obs"][key][:steps], dtype=np.float32).reshape(
                    steps, -1
                )
                for key in ("ee_pos", "ee_ori", "gripper_states")
            ],
            axis=-1,
        )
        prompt = _text(demo.attrs["task_description"]).replace("_", " ")
    return images, actions, proprio, prompt


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n_demos": len(records),
        "n_steps": int(sum(len(row["open"]["cos"]) for row in records)),
        "open_cos": float(np.mean([np.mean(row["open"]["cos"]) for row in records])),
        "close_cos": float(np.mean([np.mean(row["close"]["cos"]) for row in records])),
        "open_mse": float(np.mean([np.mean(row["open"]["mse"]) for row in records])),
        "close_mse": float(np.mean([np.mean(row["close"]["mse"]) for row in records])),
        "open_rel_l2": float(
            np.mean([np.mean(row["open"]["rel_l2"]) for row in records])
        ),
        "close_rel_l2": float(
            np.mean([np.mean(row["close"]["rel_l2"]) for row in records])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--unnorm-key", default="libero_goal_no_noops")
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--num-chunks", type=int, default=20)
    parser.add_argument("--encoder-micro-batch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    wm = load_chunk_wm(args.ckpt, device, config_path=args.config)
    encoder = OnlineVLAHiddenEncoder(
        model_path=args.model_path,
        unnorm_key=args.unnorm_key,
        device=device,
        micro_batch_size=args.encoder_micro_batch,
    )
    selected = _select_task_balanced(
        Path(args.raw_dir).expanduser().resolve(),
        per_task=args.per_task,
    )
    print(
        f"[data] selected={len(selected)} tasks={len(set(row[2] for row in selected))} "
        f"per_task<={args.per_task}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    try:
        for index, (path, demo_key, task_id, success) in enumerate(selected):
            images, actions, proprio, prompt = _load_raw_demo(
                path,
                demo_key,
                max_steps=int(wm.max_seq_len),
            )
            image_tensor = torch.from_numpy(images).unsqueeze(0)
            obs, language = encoder.encode(image_tensor, [prompt])
            obs = obs[0].to(device=device, dtype=torch.float32)
            action_tensor = torch.from_numpy(actions).to(device=device)
            proprio_tensor = torch.from_numpy(proprio).to(device=device)
            language_tensor = language[0].to(device=device, dtype=torch.float32)
            pred_open, target_open = rollout(
                wm,
                obs,
                action_tensor,
                args.num_chunks,
                "open",
                proprio=proprio_tensor,
                lang_emb=language_tensor,
            )
            pred_close, target_close = rollout(
                wm,
                obs,
                action_tensor,
                args.num_chunks,
                "close",
                proprio=proprio_tensor,
                lang_emb=language_tensor,
            )
            open_metrics = per_step_metrics(pred_open, target_open)
            close_metrics = per_step_metrics(pred_close, target_close)
            records.append(
                {
                    "task_id": task_id,
                    "success": success,
                    "path": str(path),
                    "demo_key": demo_key,
                    "open": open_metrics,
                    "close": close_metrics,
                }
            )
            print(
                f"[{index + 1}/{len(selected)}] task={task_id} success={int(success)} "
                f"open_cos={np.mean(open_metrics['cos']):.4f} "
                f"close_cos={np.mean(close_metrics['cos']):.4f}",
                flush=True,
            )
            del obs, language, image_tensor
    finally:
        encoder.close()

    by_task = {
        str(task_id): _aggregate([row for row in records if row["task_id"] == task_id])
        for task_id in sorted({int(row["task_id"]) for row in records})
    }
    summary = _aggregate(records)
    output = {
        "checkpoint": args.ckpt,
        "raw_dir": str(Path(args.raw_dir).resolve()),
        "summary": summary,
        "by_task": by_task,
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[summary] {json.dumps(summary, sort_keys=True)}", flush=True)
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
