"""Lightweight checks for experiment-stage shell launchers.

These commands validate contracts and summarize artifacts around the real Hydra
training/eval entrypoints. They intentionally do not implement training loops.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.dataset.collection_manifest import summarize_collection, write_manifest
from dreamervla.utils.paths import PROJECT_ROOT, data_path

DEFAULT_COLLECT_EXPERIMENT = "collect_rollouts_ray"
DEFAULT_COLLECT_TASK = "openvla_onetraj_coldstart_libero"
DEFAULT_CLASSIFIER_EXPERIMENT = "wmpo_token_classifier_openvla_onetraj_libero_goal_h1"


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _count_hdf5(path: Path) -> int:
    _require_path(path, "directory")
    if not path.is_dir():
        raise NotADirectoryError(path)
    return sum(1 for item in path.glob("*.hdf5") if item.is_file())


def _compose_train_config(experiment: str, overrides: list[str]) -> DictConfig:
    register_dreamervla_resolvers()
    config_dir = PROJECT_ROOT / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}", *overrides],
        )
    OmegaConf.resolve(cfg)
    return cfg


def _compose_collect_config(args: argparse.Namespace) -> DictConfig:
    overrides = [f"task={args.task}", *list(args.overrides)]
    return _compose_train_config(args.experiment, overrides)


def _latest_child(root: Path) -> Path:
    _require_path(root, "run family directory")
    children = [item for item in root.iterdir() if item.is_dir()]
    if not children:
        raise FileNotFoundError(f"no run directories under {root}")
    return max(children, key=lambda item: item.stat().st_mtime)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _collect_num_tasks(cfg: DictConfig) -> int:
    configured = OmegaConf.select(cfg, "collect.num_tasks", default=None)
    if configured is not None:
        return int(configured)
    task_ids = OmegaConf.select(cfg, "collect.task_ids", default="all")
    if OmegaConf.is_config(task_ids):
        task_ids = OmegaConf.to_container(task_ids, resolve=True)
    if task_ids == "all":
        return 10
    if isinstance(task_ids, int):
        return 1
    if isinstance(task_ids, str):
        return len([item for item in task_ids.split(",") if item.strip()])
    if isinstance(task_ids, (list, tuple)):
        return len(task_ids)
    return 10


def _collect_target(args: argparse.Namespace, cfg: DictConfig) -> tuple[int, int]:
    num_tasks = int(args.num_tasks) if args.num_tasks is not None else _collect_num_tasks(cfg)
    target = (
        int(args.target_episodes)
        if args.target_episodes is not None
        else int(cfg.collect.episodes_per_task) * num_tasks
    )
    return target, num_tasks


def collect_check(args: argparse.Namespace) -> int:
    cfg = _compose_collect_config(args)
    oft = cfg.task.openvla_oft
    ckpt = Path(str(oft.ckpt_path)).expanduser()
    stats = Path(str(oft.dataset_statistics_path)).expanduser()
    _require_path(ckpt, "OpenVLA-OFT checkpoint")
    _require_path(stats, "OpenVLA-OFT dataset statistics")
    suite = str(cfg.task.suite)
    reward_dir = Path(args.reward_dir).expanduser() if args.reward_dir else data_path(
        "collected_rollouts",
        suite,
        "reward",
    )
    hidden_dir = Path(args.hidden_dir).expanduser() if args.hidden_dir else data_path(
        "collected_rollouts",
        suite,
        "hidden",
    )
    target_episodes, num_tasks = _collect_target(args, cfg)
    _print_json(
        {
            "status": "ok",
            "stage": "collect-check",
            "experiment": args.experiment,
            "task": args.task,
            "suite": suite,
            "checkpoint": str(ckpt),
            "dataset_statistics": str(stats),
            "reward_dir": str(reward_dir),
            "hidden_dir": str(hidden_dir),
            "episodes_per_task": int(cfg.collect.episodes_per_task),
            "target_episodes": target_episodes,
            "num_tasks": num_tasks,
            "task_ids": OmegaConf.to_container(cfg.collect.task_ids, resolve=True)
            if OmegaConf.is_config(cfg.collect.task_ids)
            else cfg.collect.task_ids,
        }
    )
    return 0


def _collection_output_payload(
    *,
    root: Path,
    reward_dir: Path,
    hidden_dir: Path,
    suite: str,
    task: str,
    target_episodes: int | None,
    num_tasks: int | None,
    resolved_config: Path | None,
    collect_cmd: list[str] | None,
) -> dict[str, Any]:
    summary = summarize_collection(
        reward_dir,
        hidden_dir,
        target_total=target_episodes,
        num_tasks=num_tasks,
    )
    manifest_data: dict[str, Any] = {
        "suite": suite,
        "task": task,
        "reward_dir": str(reward_dir),
        "hidden_dir": str(hidden_dir),
        "target_episodes": target_episodes,
        "num_tasks": num_tasks,
        "collected_counts": {
            "total": int(summary["total"]),
            "per_task": {str(k): int(v) for k, v in summary["per_task"].items()},
        },
        "collected_episodes": int(summary["total"]),
        "episodes_per_task": {str(k): int(v) for k, v in summary["per_task"].items()},
        "status": "complete" if summary["complete"] else "in_progress",
        "resume_status": {
            "complete": bool(summary["complete"]),
            "remaining": summary["remaining"],
            "target_total": summary["target_total"],
            "target_per_task": summary["target_per_task"],
            "num_tasks": int(summary["num_tasks"]),
        },
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if collect_cmd is not None:
        manifest_data["collect_cmd"] = list(collect_cmd)
    if resolved_config is not None and resolved_config.is_file():
        manifest_data["resolved_config_snapshot"] = resolved_config.read_text(
            encoding="utf-8"
        )
    root.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(root, manifest_data)
    copied_resolved = None
    if resolved_config is not None and resolved_config.is_file():
        copied_resolved = root / "resolved_config.yaml"
        if resolved_config.resolve() != copied_resolved.resolve():
            shutil.copy2(resolved_config, copied_resolved)
    printed_summary = dict(manifest_data)
    if "resolved_config_snapshot" in printed_summary:
        snapshot = str(printed_summary.pop("resolved_config_snapshot"))
        printed_summary["resolved_config_snapshot_bytes"] = len(snapshot.encode("utf-8"))
    return {
        "status": "ok",
        "stage": "collect-output",
        "root": str(root),
        "manifest": str(manifest),
        "resolved_config": str(copied_resolved) if copied_resolved else None,
        "summary": printed_summary,
    }


def collect_run(args: argparse.Namespace) -> int:
    cfg = _compose_collect_config(args)
    suite = str(cfg.task.suite)
    reward_dir = Path(args.reward_dir).expanduser() if args.reward_dir else data_path(
        "collected_rollouts",
        suite,
        "reward",
    )
    hidden_dir = Path(args.hidden_dir).expanduser() if args.hidden_dir else data_path(
        "collected_rollouts",
        suite,
        "hidden",
    )
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_path(
        "outputs",
        "collect_rollouts",
        suite,
        time.strftime("%Y%m%d_%H%M%S"),
    )
    python_bin = str(args.python or sys.executable)
    target_episodes, num_tasks = _collect_target(args, cfg)
    cmd = [
        python_bin,
        "-m",
        "dreamervla.train",
        f"experiment={args.experiment}",
        f"task={args.task}",
        "logger=tensorboard",
        f"task.openvla_oft.hdf5_reward_dir={reward_dir}",
        f"task.openvla_oft.input_token_hidden_dir={hidden_dir}",
        f"++collect.hdf5_reward_dir={reward_dir}",
        f"++collect.hidden_dir={hidden_dir}",
        f"training.out_dir={out_dir}",
        *list(args.overrides),
    ]
    plan = {
        "status": "dry_run" if args.dry_run else "running",
        "stage": "collect-run",
        "experiment": args.experiment,
        "task": args.task,
        "suite": suite,
        "reward_dir": str(reward_dir),
        "hidden_dir": str(hidden_dir),
        "out_dir": str(out_dir),
        "target_episodes": target_episodes,
        "num_tasks": num_tasks,
        "cmd": cmd,
    }
    _print_json(plan)
    if args.dry_run:
        return 0
    subprocess.run(cmd, check=True)
    payload = _collection_output_payload(
        root=reward_dir.parent,
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        suite=suite,
        task=args.task,
        target_episodes=target_episodes,
        num_tasks=num_tasks,
        resolved_config=out_dir / "resolved_config.yaml",
        collect_cmd=cmd,
    )
    _print_json(payload)
    return 0


def collect_output(args: argparse.Namespace) -> int:
    cfg = _compose_collect_config(args)
    suite = str(cfg.task.suite)
    reward_dir = Path(args.reward_dir).expanduser() if args.reward_dir else data_path(
        "collected_rollouts",
        suite,
        "reward",
    )
    hidden_dir = Path(args.hidden_dir).expanduser() if args.hidden_dir else data_path(
        "collected_rollouts",
        suite,
        "hidden",
    )
    _require_path(reward_dir, "collection reward directory")
    _require_path(hidden_dir, "collection hidden directory")
    root = Path(args.root).expanduser() if args.root else reward_dir.parent
    resolved_config = Path(args.resolved_config).expanduser() if args.resolved_config else None
    target_episodes, num_tasks = _collect_target(args, cfg)
    payload = _collection_output_payload(
        root=root,
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        suite=suite,
        task=args.task,
        target_episodes=target_episodes,
        num_tasks=num_tasks,
        resolved_config=resolved_config,
        collect_cmd=None,
    )
    _print_json(payload)
    return 0


def cls_check(args: argparse.Namespace) -> int:
    cfg = _compose_train_config(args.experiment, list(args.overrides))
    data_cfg = cfg.data
    directories = {
        "success_dir_raw": Path(str(data_cfg.success_dir_raw)).expanduser(),
        "success_dir_hidden": Path(str(data_cfg.success_dir_hidden)).expanduser(),
        "failure_dir_raw": Path(str(data_cfg.failure_dir_raw)).expanduser(),
        "failure_dir_hidden": Path(str(data_cfg.failure_dir_hidden)).expanduser(),
    }
    counts = {name: _count_hdf5(path) for name, path in directories.items()}
    missing = [name for name, count in counts.items() if count <= 0]
    if missing:
        raise FileNotFoundError(f"classifier data directories contain no HDF5 files: {missing}")

    _print_json(
        {
            "status": "ok",
            "stage": "cls-check",
            "experiment": args.experiment,
            "training_out_dir": str(cfg.training.out_dir),
            "directories": {name: str(path) for name, path in directories.items()},
            "hdf5_counts": counts,
            "window": int(data_cfg.window),
            "sampling_protocol": str(OmegaConf.select(data_cfg, "sampling_protocol")),
            "classifier_target": str(OmegaConf.select(cfg, "classifier._target_")),
        }
    )
    return 0


def cls_eval(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else _latest_child(
        data_path("outputs/classifier", args.family)
    )
    _require_path(run_dir, "classifier run directory")
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "log" / "train_log.jsonl"
    ckpt_dir = run_dir / "checkpoints"
    _require_path(summary_path, "classifier summary")
    _require_path(log_path, "classifier train log")
    _require_path(ckpt_dir, "classifier checkpoint directory")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = _read_jsonl(log_path)
    val_window = [record for record in records if record.get("event") == "val_window"]
    val_episode = [record for record in records if record.get("event") == "val_episode"]
    ckpts = sorted(str(path) for path in ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no classifier checkpoints under {ckpt_dir}")

    payload = {
        "status": "ok",
        "stage": "cls-eval",
        "run_dir": str(run_dir),
        "summary": summary,
        "num_val_window_records": len(val_window),
        "num_val_episode_records": len(val_episode),
        "checkpoints": ckpts,
    }
    out = Path(args.out).expanduser() if args.out else run_dir / "classifier_eval_summary.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["out"] = str(out)
    _print_json(payload)
    return 0


def wm_check(args: argparse.Namespace) -> int:
    reward_dir = Path(args.reward_dir).expanduser()
    hidden_dir = Path(args.hidden_dir).expanduser()
    reward_count = _count_hdf5(reward_dir)
    hidden_count = _count_hdf5(hidden_dir)
    if reward_count <= 0 or hidden_count <= 0:
        raise FileNotFoundError("WM warmup requires non-empty reward and hidden HDF5 dirs")
    resolved = Path(args.resolved_config).expanduser()
    manifest = Path(args.manifest).expanduser()
    if args.require_manifest:
        _require_path(manifest, "collection manifest")
    if args.require_resolved_config:
        _require_path(resolved, "collection resolved config")
    _print_json(
        {
            "status": "ok",
            "stage": "wm-check",
            "reward_dir": str(reward_dir),
            "hidden_dir": str(hidden_dir),
            "reward_hdf5_count": reward_count,
            "hidden_hdf5_count": hidden_count,
            "manifest": str(manifest),
            "resolved_config": str(resolved),
        }
    )
    return 0


def _extract_world_model(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    state_dicts = payload.get("state_dicts")
    if isinstance(state_dicts, dict) and isinstance(state_dicts.get("world_model"), dict):
        return state_dicts["world_model"]
    if isinstance(payload.get("world_model"), dict):
        return payload["world_model"]
    if isinstance(payload.get("model"), dict):
        return payload["model"]
    raise KeyError(f"{path} has no world_model/model state dict")


def _extract_classifier(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    state_dicts = payload.get("state_dicts")
    if isinstance(state_dicts, dict) and isinstance(state_dicts.get("classifier"), dict):
        return state_dicts["classifier"]
    if isinstance(payload.get("classifier"), dict):
        return payload["classifier"]
    if isinstance(payload.get("model"), dict):
        return payload["model"]
    raise KeyError(f"{path} has no classifier/model state dict")


def _extract_classifier_threshold(payload: dict[str, Any]) -> float:
    for key in ("classifier_threshold", "threshold"):
        if key in payload:
            return float(payload[key])
    state_dicts = payload.get("state_dicts")
    if isinstance(state_dicts, dict) and "classifier_threshold" in state_dicts:
        return float(state_dicts["classifier_threshold"])
    return 0.5


def pack_init(args: argparse.Namespace) -> int:
    wm_ckpt = Path(args.wm_ckpt).expanduser()
    cls_ckpt = Path(args.classifier_ckpt).expanduser()
    out = Path(args.out).expanduser()
    _require_path(wm_ckpt, "WM checkpoint")
    _require_path(cls_ckpt, "classifier checkpoint")
    wm_payload = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
    cls_payload = torch.load(cls_ckpt, map_location="cpu", weights_only=False)
    packed = {
        "state_dicts": {
            "world_model": _extract_world_model(wm_payload, wm_ckpt),
            "classifier": _extract_classifier(cls_payload, cls_ckpt),
        },
        "classifier_threshold": _extract_classifier_threshold(cls_payload),
        "metadata": {
            "source_world_model": str(wm_ckpt),
            "source_classifier": str(cls_ckpt),
            "schema": "state_dicts.world_model+classifier",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, out)
    _print_json(
        {
            "status": "ok",
            "stage": "pack-init",
            "out": str(out),
            "source_world_model": str(wm_ckpt),
            "source_classifier": str(cls_ckpt),
            "classifier_threshold": packed["classifier_threshold"],
            "components": sorted(packed["state_dicts"]),
        }
    )
    return 0


def cotrain_check(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).expanduser() if args.run_root else _latest_child(
        data_path("outputs/coldstart_warmup_cotrain")
    )
    cotrain_dir = run_root / "cotrain"
    ckpt_dir = cotrain_dir / "ckpt"
    wm = ckpt_dir / "wm_warmup.ckpt"
    cls = ckpt_dir / "classifier_warmup.ckpt"
    resolved = cotrain_dir / "resolved_config.yaml"
    _require_path(run_root, "run root")
    _require_path(resolved, "cotrain resolved config")
    if not wm.is_file() and not (ckpt_dir / "wm_warmup_hf").is_dir():
        raise FileNotFoundError(f"warmup world-model checkpoint not found: {wm}")
    if not cls.is_file() and not (ckpt_dir / "classifier_warmup_hf").is_dir():
        raise FileNotFoundError(f"warmup classifier checkpoint not found: {cls}")
    init_ckpt = Path(args.init_ckpt).expanduser() if args.init_ckpt else None
    if init_ckpt is not None:
        _require_path(init_ckpt, "consolidated init checkpoint")
    _print_json(
        {
            "status": "ok",
            "stage": "cotrain-check",
            "run_root": str(run_root),
            "cotrain_dir": str(cotrain_dir),
            "wm_warmup": str(wm),
            "classifier_warmup": str(cls),
            "resolved_config": str(resolved),
            "init_ckpt": str(init_ckpt) if init_ckpt else None,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect_check_p = sub.add_parser("collect-check")
    collect_check_p.add_argument("--experiment", default=DEFAULT_COLLECT_EXPERIMENT)
    collect_check_p.add_argument("--task", default=DEFAULT_COLLECT_TASK)
    collect_check_p.add_argument("--reward-dir", default=None)
    collect_check_p.add_argument("--hidden-dir", default=None)
    collect_check_p.add_argument("--target-episodes", type=int, default=None)
    collect_check_p.add_argument("--num-tasks", type=int, default=None)
    collect_check_p.add_argument("overrides", nargs="*")
    collect_check_p.set_defaults(func=collect_check)

    collect_run_p = sub.add_parser("collect-run")
    collect_run_p.add_argument("--experiment", default=DEFAULT_COLLECT_EXPERIMENT)
    collect_run_p.add_argument("--task", default=DEFAULT_COLLECT_TASK)
    collect_run_p.add_argument("--reward-dir", default=None)
    collect_run_p.add_argument("--hidden-dir", default=None)
    collect_run_p.add_argument("--out-dir", default=None)
    collect_run_p.add_argument("--python", default=None)
    collect_run_p.add_argument("--target-episodes", type=int, default=None)
    collect_run_p.add_argument("--num-tasks", type=int, default=None)
    collect_run_p.add_argument("--dry-run", action="store_true")
    collect_run_p.add_argument("overrides", nargs="*")
    collect_run_p.set_defaults(func=collect_run)

    collect_output_p = sub.add_parser("collect-output")
    collect_output_p.add_argument("--experiment", default=DEFAULT_COLLECT_EXPERIMENT)
    collect_output_p.add_argument("--task", default=DEFAULT_COLLECT_TASK)
    collect_output_p.add_argument("--reward-dir", default=None)
    collect_output_p.add_argument("--hidden-dir", default=None)
    collect_output_p.add_argument("--root", default=None)
    collect_output_p.add_argument("--resolved-config", default=None)
    collect_output_p.add_argument("--target-episodes", type=int, default=None)
    collect_output_p.add_argument("--num-tasks", type=int, default=None)
    collect_output_p.add_argument("overrides", nargs="*")
    collect_output_p.set_defaults(func=collect_output)

    cls_check_p = sub.add_parser("cls-check")
    cls_check_p.add_argument("--experiment", default=DEFAULT_CLASSIFIER_EXPERIMENT)
    cls_check_p.add_argument("overrides", nargs="*")
    cls_check_p.set_defaults(func=cls_check)

    cls_eval_p = sub.add_parser("cls-eval")
    cls_eval_p.add_argument("--run-dir", default=None)
    cls_eval_p.add_argument("--family", default=DEFAULT_CLASSIFIER_EXPERIMENT)
    cls_eval_p.add_argument("--out", default=None)
    cls_eval_p.set_defaults(func=cls_eval)

    wm_check_p = sub.add_parser("wm-check")
    wm_check_p.add_argument(
        "--reward-dir",
        default=str(data_path("collected_rollouts/libero_goal/reward")),
    )
    wm_check_p.add_argument(
        "--hidden-dir",
        default=str(data_path("collected_rollouts/libero_goal/hidden")),
    )
    wm_check_p.add_argument(
        "--manifest",
        default=str(data_path("collected_rollouts/libero_goal/collection_manifest.json")),
    )
    wm_check_p.add_argument(
        "--resolved-config",
        default=str(data_path("collected_rollouts/libero_goal/resolved_config.yaml")),
    )
    wm_check_p.add_argument(
        "--require-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    wm_check_p.add_argument(
        "--require-resolved-config",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    wm_check_p.set_defaults(func=wm_check)

    pack_p = sub.add_parser("pack-init")
    pack_p.add_argument("--wm-ckpt", required=True)
    pack_p.add_argument("--classifier-ckpt", required=True)
    pack_p.add_argument("--out", required=True)
    pack_p.set_defaults(func=pack_init)

    cotrain_p = sub.add_parser("cotrain-check")
    cotrain_p.add_argument("--run-root", default=None)
    cotrain_p.add_argument("--init-ckpt", default=None)
    cotrain_p.set_defaults(func=cotrain_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
