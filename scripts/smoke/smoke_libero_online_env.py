#!/usr/bin/env python3
# ruff: noqa: E402
"""Smoke-test the online LIBERO env wrapper.

Example:
  MUJOCO_GL=osmesa python scripts/smoke_libero_online_env.py --task-id 0 --steps 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env import ACTION_LOW, LIBEROOnlineEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test LIBEROOnlineEnv")
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = LIBEROOnlineEnv(
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        resolution=args.resolution,
        image_size=args.image_size,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
    try:
        obs, info = env.reset()
        print(
            "reset:",
            f"task={info['task_id']}",
            f"init={info['init_state_index']}",
            f"image={obs['image'].shape}/{obs['image'].dtype}",
            f"state={obs['state'].shape}/{obs['state'].dtype}",
            f"text={obs['task_description']!r}",
        )
        action = ACTION_LOW.copy()
        for idx in range(max(args.steps, 0)):
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"step {idx + 1}:",
                f"reward={reward:.1f}",
                f"terminated={terminated}",
                f"truncated={truncated}",
                f"success={info['success']}",
                f"image={obs['image'].shape}",
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
