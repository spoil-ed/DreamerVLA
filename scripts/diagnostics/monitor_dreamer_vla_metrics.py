#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


METRICS = [
    ("train_wm_loss", "WM loss"),
    ("train_wm_image_recon_ce_loss", "Token CE"),
    ("train_wm_image_recon_accuracy", "Token acc"),
    ("train_wm_kl_loss", "KL"),
    ("train_actor_loss", "Actor loss"),
    ("train_critic_loss", "Critic loss"),
    ("train_returns_mean", "Returns mean"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render DreamerVLA training curves from dreamer_vla_logs.json.txt."
    )
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--step-interval", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--save-step-copy", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "global_step" in row:
                rows.append(row)
    rows.sort(key=lambda x: int(x.get("global_step", -1)))
    return rows


def finite_metric(
    rows: list[dict[str, Any]], key: str
) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for row in rows:
        if key not in row:
            continue
        try:
            value = float(row[key])
            step = int(row["global_step"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            steps.append(step)
            values.append(value)
    return steps, values


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    out: list[float] = []
    total = 0.0
    q: list[float] = []
    for value in values:
        q.append(value)
        total += value
        if len(q) > window:
            total -= q.pop(0)
        out.append(total / len(q))
    return out


def summarize(rows: list[dict[str, Any]], window: int) -> dict[str, Any]:
    first = rows[:window]
    last = rows[-window:]
    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "first_step": int(rows[0]["global_step"]),
        "last_step": int(rows[-1]["global_step"]),
        "epoch": rows[-1].get("epoch"),
        "window": window,
        "metrics": {},
    }
    for key, _label in METRICS:
        first_vals = [
            float(x[key]) for x in first if key in x and math.isfinite(float(x[key]))
        ]
        last_vals = [
            float(x[key]) for x in last if key in x and math.isfinite(float(x[key]))
        ]
        if first_vals and last_vals:
            summary["metrics"][key] = {
                "first_mean": sum(first_vals) / len(first_vals),
                "last_mean": sum(last_vals) / len(last_vals),
                "last": last_vals[-1],
            }
    return summary


def render(
    rows: list[dict[str, Any]], out_dir: Path, window: int, save_step_copy: bool
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    last_step = int(rows[-1]["global_step"])
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True)
    axes_flat = axes.reshape(-1)
    for ax, (key, label) in zip(axes_flat, METRICS):
        steps, values = finite_metric(rows, key)
        if not steps:
            ax.set_title(f"{label} (missing)")
            ax.axis("off")
            continue
        ax.plot(steps, values, color="#9aa4b2", alpha=0.35, linewidth=0.8, label="raw")
        ax.plot(
            steps,
            moving_average(values, window),
            color="#2563eb",
            linewidth=1.6,
            label=f"ma{window}",
        )
        ax.set_title(label)
        ax.set_xlabel("global_step")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes_flat[-1].axis("off")
    fig.suptitle(f"DreamerVLA metrics through step {last_step}", fontsize=14)
    latest_path = out_dir / "loss_curve_latest.png"
    fig.savefig(latest_path, dpi=160)
    if save_step_copy:
        fig.savefig(out_dir / f"loss_curve_step_{last_step:07d}.png", dpi=160)
    plt.close(fig)

    summary_path = out_dir / "metrics_summary_latest.json"
    summary_path.write_text(
        json.dumps(summarize(rows, window), indent=2), encoding="utf-8"
    )
    return latest_path


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.log_path.parent / "plots")
    last_rendered_step: int | None = None
    while True:
        rows = read_rows(args.log_path)
        if rows:
            latest_step = int(rows[-1]["global_step"])
            should_render = (
                last_rendered_step is None
                or latest_step // args.step_interval
                > last_rendered_step // args.step_interval
            )
            if should_render:
                path = render(rows, out_dir, args.window, args.save_step_copy)
                print(f"[monitor] rendered {path} at step {latest_step}", flush=True)
                last_rendered_step = latest_step
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
