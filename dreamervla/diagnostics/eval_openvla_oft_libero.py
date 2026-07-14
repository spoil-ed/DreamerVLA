#!/usr/bin/env python3
"""Run LIBERO eval with an OpenVLA-OFT obs-to-action policy adapter.

The LIBERO eval internals stay in OpenVLA-OFT's official
`experiments.robot.libero.run_libero_eval`. This script only replaces the
action-returning function with a policy object whose interface is:

    actions = policy(obs, task_description)
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.diagnostics.openvla_oft_obs_action_policy import (
    OpenVLAOFTObsActionPolicy,
    ensure_openvla_oft_importable,
    set_runtime_env,
)

TASK_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
SUITE_ALIASES = {
    "libero10": "libero_10",
    "libero_long": "libero_10",
}
CAMERA_INPUTS = ("primary",)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


CONFIG_DIR = _project_root() / "configs" / "experiment" / "openvla_oft_official_eval"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize(text: str, max_len: int = 72) -> str:
    text = text.lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len] or "run"


@dataclass(frozen=True)
class EvalOpenVLAOFTConfig:
    ckpt: Path
    suite: str
    task_ids: list[int] | None
    num_trials: int
    gpu_id: str | None
    seed: int
    output_dir: Path | None
    openvla_oft_root: Path | None
    policy_mode: str
    num_images: int | None
    camera_inputs: str | None
    use_proprio: bool | None
    center_crop: bool
    num_open_loop_steps: int
    env_img_res: int
    initial_states_path: str
    load_in_8bit: bool
    load_in_4bit: bool
    run_note: str | None
    save_videos: bool


def _parse_task_ids(value: str) -> list[int] | None:
    if value.strip() == "":
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
                raise ValueError(f"Bad task range: {part}")
            task_ids.extend(range(start, end + 1))
        else:
            task_ids.append(int(part))
    return list(dict.fromkeys(task_ids))


def parse_suite_name(value: str) -> str:
    suite = SUITE_ALIASES.get(value, value)
    if suite not in TASK_SUITES:
        raise ValueError(f"Invalid task suite: {value}; expected one of {TASK_SUITES}")
    return suite


def resolve_num_images_for_camera_inputs(
    camera_inputs: str | None,
    num_images: int | None,
) -> int:
    if camera_inputs not in {None, "primary"}:
        raise ValueError("OpenVLA-OFT mainline requires camera_inputs='primary'")
    if num_images not in {None, 1}:
        raise ValueError("OpenVLA-OFT mainline requires num_images=1")
    return 1


def _plain(value: Any) -> Any:
    return (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (DictConfig, ListConfig))
        else value
    )


def _empty_to_none(value: Any) -> Any | None:
    value = _plain(value)
    return None if value in (None, "") else value


def _optional_path(value: Any) -> Path | None:
    value = _empty_to_none(value)
    return None if value is None else Path(str(value))


def _optional_int(value: Any) -> int | None:
    value = _empty_to_none(value)
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    value = _empty_to_none(value)
    return None if value is None else str(value)


def _optional_bool(value: Any) -> bool | None:
    value = _empty_to_none(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean-like value, got {value!r}")


def _bool(value: Any, *, default: bool = False) -> bool:
    parsed = _optional_bool(value)
    return default if parsed is None else parsed


def _task_ids_from_config(value: Any) -> list[int] | None:
    value = _empty_to_none(value)
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_task_ids(value)
    if isinstance(value, int):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(dict.fromkeys(int(item) for item in value))
    raise TypeError(f"Unsupported task_ids value: {value!r}")


def _config_from_mapping(cfg: Mapping[str, Any]) -> EvalOpenVLAOFTConfig:
    suite = parse_suite_name(str(cfg["suite"]))
    policy_mode = str(cfg.get("policy_mode", "discrete"))
    if policy_mode != "discrete":
        raise ValueError("policy_mode must be discrete")
    camera_inputs = _optional_str(cfg.get("camera_inputs"))
    if camera_inputs not in {None, "primary"}:
        raise ValueError("OpenVLA-OFT mainline requires camera_inputs=primary")
    num_images = _optional_int(cfg.get("num_images"))
    if num_images not in {None, 1}:
        raise ValueError("OpenVLA-OFT mainline requires num_images=1")
    use_proprio = _optional_bool(cfg.get("use_proprio"))
    if bool(use_proprio):
        raise ValueError("OpenVLA-OFT mainline does not include proprio")
    return EvalOpenVLAOFTConfig(
        ckpt=Path(str(cfg["ckpt"])),
        suite=suite,
        task_ids=_task_ids_from_config(cfg.get("task_ids")),
        num_trials=int(cfg.get("num_trials", 10)),
        gpu_id=_optional_str(cfg.get("gpu_id")),
        seed=int(cfg.get("seed", 7)),
        output_dir=_optional_path(cfg.get("output_dir")),
        openvla_oft_root=_optional_path(cfg.get("openvla_oft_root")),
        policy_mode=policy_mode,
        num_images=num_images,
        camera_inputs=camera_inputs,
        use_proprio=use_proprio,
        center_crop=_bool(cfg.get("center_crop", True), default=True),
        num_open_loop_steps=int(cfg.get("num_open_loop_steps", 8)),
        env_img_res=int(cfg.get("env_img_res", 256)),
        initial_states_path=str(cfg.get("initial_states_path", "DEFAULT")),
        load_in_8bit=_bool(cfg.get("load_in_8bit", False)),
        load_in_4bit=_bool(cfg.get("load_in_4bit", False)),
        run_note=_optional_str(cfg.get("run_note")),
        save_videos=_bool(cfg.get("save_videos", False)),
    )


def _parse_hydra_like_argv(argv: Sequence[str]) -> tuple[str, list[str]]:
    config_name = "eval"
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == "--config-name":
            if i + 1 >= len(argv):
                raise SystemExit("--config-name requires a value")
            config_name = argv[i + 1]
            i += 2
            continue
        if item.startswith("--config-name="):
            config_name = item.split("=", 1)[1]
            i += 1
            continue
        if item.startswith("--"):
            raise SystemExit(
                f"Unsupported eval flag {item!r}. Use Hydra override syntax like "
                "ckpt=/path/to/ckpt, suite=libero_goal, task_ids=0-2."
            )
        overrides.append(item)
        i += 1
    return config_name, overrides


def run_eval(cfg: Mapping[str, Any]) -> int:
    args = _config_from_mapping(cfg)
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
    cfg.camera_inputs = "primary"

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


def main(argv: Sequence[str] | None = None) -> int:
    config_name, overrides = _parse_hydra_like_argv(list(sys.argv[1:] if argv is None else argv))
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="openvla_oft_official_eval",
        version_base=None,
    ):
        cfg_obj = compose(config_name=config_name, overrides=overrides)
    cfg: dict[str, Any] = OmegaConf.to_container(cfg_obj, resolve=True)  # type: ignore[assignment]
    return run_eval(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
