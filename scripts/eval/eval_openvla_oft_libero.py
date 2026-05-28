#!/usr/bin/env python3
"""Run LIBERO eval with an OpenVLA-OFT obs-to-action policy adapter.

The LIBERO eval internals stay in OpenVLA-OFT's official
`experiments.robot.libero.run_libero_eval`. This script only replaces the
action-returning function with a policy object whose interface is:

    actions = policy(obs, task_description)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.openvla_oft_obs_action_policy import (
    OpenVLAOFTObsActionPolicy,
    ensure_openvla_oft_importable,
    set_runtime_env,
)


TASK_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
SUITE_ALIASES = {
    "libero10": "libero_10",
    "libero_long": "libero_10",
}
CAMERA_INPUTS = ("primary", "primary+wrist")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize(text: str, max_len: int = 72) -> str:
    text = text.lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len] or "run"


def _parse_task_ids(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    task_ids: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise argparse.ArgumentTypeError(f"Bad task range: {part}")
            task_ids.extend(range(start, end + 1))
        else:
            task_ids.append(int(part))
    return list(dict.fromkeys(task_ids))


def parse_suite_name(value: str) -> str:
    suite = SUITE_ALIASES.get(value, value)
    if suite not in TASK_SUITES:
        raise argparse.ArgumentTypeError(f"Invalid task suite: {value}; expected one of {TASK_SUITES}")
    return suite


def resolve_num_images_for_camera_inputs(camera_inputs: str | None, num_images: int | None) -> int | None:
    if camera_inputs is None:
        return num_images
    if camera_inputs not in CAMERA_INPUTS:
        raise ValueError(f"Unsupported camera_inputs={camera_inputs!r}; expected one of {CAMERA_INPUTS}")
    expected_num_images = 1 if camera_inputs == "primary" else 2
    if num_images is not None and int(num_images) != expected_num_images:
        raise ValueError(
            f"--camera-inputs {camera_inputs!r} conflicts with --num-images {num_images}; "
            f"expected --num-images {expected_num_images}"
        )
    return expected_num_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=parse_suite_name)
    parser.add_argument("--task-ids", type=_parse_task_ids, default=None)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--openvla-oft-root", type=Path, default=None)
    parser.add_argument("--policy-mode", choices=("auto", "discrete", "l1"), default="auto")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument(
        "--camera-inputs",
        choices=CAMERA_INPUTS,
        default=None,
        help="Select visual inputs for LIBERO eval. primary uses only agentview; primary+wrist also includes wrist.",
    )
    parser.add_argument("--use-proprio", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-open-loop-steps", type=int, default=8)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--initial-states-path", default="DEFAULT")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--run-note", default=None)
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Accepted for launcher compatibility; official OpenVLA-OFT eval controls video saving.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    set_runtime_env(args.gpu_id)
    openvla_oft_root = ensure_openvla_oft_importable(args.openvla_oft_root)

    from experiments.robot.libero import run_libero_eval as eval_mod
    from libero.libero import benchmark

    num_images_in_input = resolve_num_images_for_camera_inputs(args.camera_inputs, args.num_images)

    policy = OpenVLAOFTObsActionPolicy.from_checkpoint(
        args.ckpt,
        task_suite_name=args.suite,
        openvla_oft_root=openvla_oft_root,
        policy_mode=args.policy_mode,
        num_images_in_input=num_images_in_input,
        use_proprio=args.use_proprio,
        center_crop=args.center_crop,
        num_open_loop_steps=args.num_open_loop_steps,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    cfg = policy.cfg
    cfg.num_trials_per_task = int(args.num_trials)
    cfg.initial_states_path = args.initial_states_path
    cfg.env_img_res = int(args.env_img_res)
    cfg.seed = int(args.seed)
    cfg.use_wandb = False
    cfg.camera_inputs = args.camera_inputs or ("primary+wrist" if cfg.num_images_in_input > 1 else "primary")

    run_tag = _timestamp()
    task_note = args.task_ids if args.task_ids is not None else "all"
    cfg.run_id_note = args.run_note or _sanitize(f"{Path(args.ckpt).name} tasks {task_note}")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            _project_root()
            / "data"
            / "outputs"
            / "eval"
            / "openvla_oft_libero"
            / f"{args.suite}_{Path(args.ckpt).name}_{run_tag}"
        )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.local_log_dir = str(output_dir / "logs")

    eval_mod.get_action = policy.as_get_action()
    eval_mod.validate_config(cfg)
    eval_mod.set_seed_everywhere(cfg.seed)
    resize_size = eval_mod.get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = eval_mod.setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    valid_task_ids = list(range(task_suite.n_tasks))
    task_ids = args.task_ids if args.task_ids is not None else valid_task_ids
    invalid = [task_id for task_id in task_ids if task_id not in valid_task_ids]
    if invalid:
        raise ValueError(f"Invalid task ids for {cfg.task_suite_name}: {invalid}; valid={valid_task_ids}")

    eval_mod.log_message(
        f"Using OpenVLA-OFT obs-action policy; task_ids={task_ids}; unnorm_key={cfg.unnorm_key}",
        log_file,
    )

    total_episodes = 0
    total_successes = 0
    per_task: list[dict[str, Any]] = []
    try:
        for task_id in task_ids:
            before_episodes = total_episodes
            before_successes = total_successes
            total_episodes, total_successes = eval_mod.run_task(
                cfg,
                task_suite,
                task_id,
                policy.model,
                resize_size,
                policy.processor,
                policy.action_head,
                policy.proprio_projector,
                policy.noisy_action_projector,
                total_episodes,
                total_successes,
                log_file,
            )
            episodes = total_episodes - before_episodes
            successes = total_successes - before_successes
            task = task_suite.get_task(task_id)
            per_task.append(
                {
                    "task_id": task_id,
                    "task_description": task.language,
                    "episodes": episodes,
                    "successes": successes,
                    "success_rate": successes / episodes if episodes else 0.0,
                }
            )
    finally:
        log_file.close()

    summary = {
        "run_id": run_id,
        "checkpoint": str(Path(args.ckpt).expanduser().resolve()),
        "suite": cfg.task_suite_name,
        "task_ids": task_ids,
        "num_trials_per_task": cfg.num_trials_per_task,
        "unnorm_key": cfg.unnorm_key,
        "use_l1_regression": cfg.use_l1_regression,
        "use_proprio": cfg.use_proprio,
        "num_images_in_input": cfg.num_images_in_input,
        "camera_inputs": cfg.camera_inputs,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "per_task": per_task,
        "log_file": local_log_filepath,
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
