"""Lightweight isolated LIBERO action replay used by comparison diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-file", type=Path, required=True)
    parser.add_argument("--demo-key", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-backend", choices=("osmesa", "egl"), default="osmesa")
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.length) <= 0:
        raise ValueError("length must be positive")
    from dreamervla.utils.egl_device import apply_libero_render_regime

    apply_libero_render_regime(str(args.render_backend), 0, [int(args.render_gpu)])
    from libero.libero import benchmark as libero_benchmark

    from dreamervla.envs.libero.utils import get_libero_env

    trajectory_path = args.trajectory_file.expanduser().resolve()
    with h5py.File(trajectory_path, "r") as handle:
        demo = handle["data"][str(args.demo_key)]
        actions = np.asarray(demo["actions"][: int(args.length)], dtype=np.float32)
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
    if int(actions.shape[0]) != int(args.length):
        raise ValueError(
            f"trajectory has {int(actions.shape[0])} actions, requested {int(args.length)}"
        )

    suite = libero_benchmark.get_benchmark_dict()[str(args.suite_name)]()
    task = suite.get_task(int(args.task_id))
    env, _description = get_libero_env(
        task,
        resolution=int(args.resolution),
        seed=int(args.seed),
    )
    frames: list[np.ndarray] = []
    states: list[np.ndarray] = []

    def capture(raw_obs: dict[str, Any]) -> None:
        frames.append(
            np.stack(
                [
                    np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1]),
                    np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                ],
                axis=0,
            )
        )
        states.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64))

    try:
        env.reset()
        raw_obs = env.set_init_state(init_state)
        capture(raw_obs)
        for index, action in enumerate(actions[:-1]):
            raw_obs, _reward, _done, _info = env.step(action.tolist())
            capture(raw_obs)
            if (index + 1) % 20 == 0:
                print(
                    f"[libero] task={int(args.task_id)} frames={index + 2}/{int(args.length)}",
                    flush=True,
                )
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frames=np.stack(frames, axis=0),
        states=np.stack(states, axis=0),
    )


if __name__ == "__main__":
    main()
