"""Hydra-owned public Docker reproduction workflow."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.launchers.task_cli import normalize_task_flag
from dreamervla.preprocess.check_artifacts import validate_hdf5_dir
from dreamervla.runtime.reproduction import (
    ReproductionError,
    atomic_write_json,
    decide_stage,
    select_metric_checkpoint,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "scripts"
IMAGE_METADATA_PATH = PROJECT_ROOT / ".dreamervla-image.json"


@dataclass(frozen=True)
class ReproductionWorkflow:
    """A resolved reproduction request."""

    config_name: str
    cfg: DictConfig
    dry_run: bool


def _parse_args(argv: Sequence[str]) -> tuple[str, list[str]]:
    config_name = "reproduce/prepare_assets"
    overrides: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--config-name":
            if index + 1 >= len(argv):
                raise SystemExit("--config-name requires a value")
            config_name = argv[index + 1]
            index += 2
            continue
        if item.startswith("--config-name="):
            config_name = item.split("=", 1)[1]
        else:
            overrides.append(item)
        index += 1
    return config_name, overrides


def build_workflow(argv: Sequence[str]) -> ReproductionWorkflow:
    """Compose one reproduction workflow and apply Hydra overrides."""

    register_dreamervla_resolvers()
    config_name, overrides = _parse_args(argv)
    overrides, task_override = normalize_task_flag(overrides, hydra_key="profile.task")
    if task_override is not None:
        overrides.append(task_override)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name="reproduce", version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    supported_tasks = {str(value) for value in cfg.get("supported_tasks", [cfg.profile.task])}
    if str(cfg.profile.task) not in supported_tasks:
        valid = ", ".join(sorted(supported_tasks))
        raise SystemExit(
            f"reproduction config {config_name!r} does not support task "
            f"{cfg.profile.task!r}; valid: {valid}"
        )
    return ReproductionWorkflow(
        config_name=config_name,
        cfg=cfg,
        dry_run=bool(cfg.get("dry_run", False)),
    )


def _stage_path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def build_stage_command(
    cfg: DictConfig,
    stage_name: str,
    *,
    action: str,
    selected_checkpoints: Mapping[str, Path],
) -> tuple[str, ...]:
    """Build one command using the repository's registered experiment launcher."""

    stage = cfg.stages[stage_name]
    run_root = _stage_path(stage.run_root)
    command = [
        "bash",
        str((PROJECT_ROOT / str(stage.launcher)).resolve()),
        "--config",
        str(stage.experiment),
    ]
    if action == "resume":
        command.extend(["--resume", str(run_root)])
    elif action == "fresh":
        command.append(f"training.out_dir={run_root}")
    else:
        raise ReproductionError(f"cannot build command for stage action {action!r}")
    if stage_name == "dreamer":
        try:
            world_model = selected_checkpoints["world_model"]
            classifier = selected_checkpoints["classifier"]
        except KeyError as exc:
            raise ReproductionError("Dreamer requires selected WM and CLS checkpoints") from exc
        command.extend(
            [
                "--wm_ckpt",
                str(world_model),
                "--cls_ckpt",
                str(classifier),
            ]
        )
    command.extend(
        [
            f"{stage.budget_key}={int(stage.budget)}",
            f"ngpu={int(cfg.profile.num_gpus)}",
            f"gpus={cfg.profile.gpus}",
        ]
    )
    command.extend(str(value) for value in stage.get("overrides", []))
    return tuple(command)


def _run(command: Sequence[str], *, env: Mapping[str, str], dry_run: bool) -> None:
    print(f"[reproduce] command: {shlex.join(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, env=dict(env), check=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReproductionError(f"invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReproductionError(f"JSON manifest must be an object: {path}")
    return value


def _image_metadata() -> dict[str, Any]:
    return _load_json(IMAGE_METADATA_PATH) if IMAGE_METADATA_PATH.is_file() else {}


def _git_revision(path: Path) -> str:
    if not (path / ".git").exists():
        raise ReproductionError(f"third-party source is not a Git checkout: {path}")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_third_party(cfg: DictConfig) -> dict[str, str]:
    directories = {
        "libero": "LIBERO",
        "robosuite": "robosuite",
        "robosuite_task_zoo": "robosuite-task-zoo",
        "robomimic": "robomimic",
        "mimicgen": "mimicgen",
        "openvla_oft": "openvla-oft",
        "egl_probe": "egl_probe",
    }
    source_root = _stage_path(
        OmegaConf.select(
            cfg,
            "third_party_root",
            default=PROJECT_ROOT / "third_party",
        )
    )
    revisions: dict[str, str] = {}
    for key, directory in directories.items():
        expected = str(cfg.third_party[key])
        actual = _git_revision(source_root / directory)
        if not actual.startswith(expected):
            raise ReproductionError(
                f"{source_root / directory} revision mismatch: expected={expected} actual={actual}"
            )
        revisions[key] = actual
    return revisions


def _verify_hardware(cfg: DictConfig, data_root: Path) -> None:
    import torch

    if torch.cuda.device_count() != int(cfg.profile.num_gpus):
        raise ReproductionError(
            f"GPU count mismatch: expected={cfg.profile.num_gpus} actual={torch.cuda.device_count()}"
        )
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        if "H100" not in properties.name:
            raise ReproductionError(f"GPU {index} is not H100: {properties.name}")
        memory_gib = properties.total_memory / 1024**3
        if memory_gib < float(cfg.profile.minimum_gpu_memory_gib):
            raise ReproductionError(
                f"GPU {index} memory is too small: expected>={cfg.profile.minimum_gpu_memory_gib} GiB "
                f"actual={memory_gib:.1f} GiB"
            )
    free_gib = shutil.disk_usage(data_root).free / 1024**3
    if free_gib < float(cfg.profile.minimum_free_disk_gib):
        raise ReproductionError(
            f"insufficient free space under {data_root}: "
            f"expected>={cfg.profile.minimum_free_disk_gib} GiB actual={free_gib:.1f} GiB"
        )


def _valid_openvla(cfg: DictConfig) -> bool:
    root = _stage_path(cfg.assets.openvla.target)
    required = [root / str(name) for name in cfg.assets.openvla.required_files]
    shards = sorted(root.glob("model-*.safetensors"))
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    if not shards or not (root / "model.safetensors.index.json").is_file():
        return False
    for path in [*required, *shards]:
        with path.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs.github.com/spec"):
                return False
    try:
        actual_revision = _git_revision(root)
    except (ReproductionError, subprocess.CalledProcessError):
        return False
    if actual_revision != str(cfg.assets.openvla.revision):
        return False
    return True


def _libero_hdf5_files(cfg: DictConfig) -> list[Path]:
    root = _stage_path(cfg.assets.libero.target)
    return sorted(path for path in root.rglob("*.hdf5") if path.is_file()) if root.is_dir() else []


def _valid_libero(cfg: DictConfig) -> bool:
    root = _stage_path(cfg.assets.libero.target)
    marker_path = root / ".dreamervla-source.json"
    marker = _load_json(marker_path)
    if marker != {
        "repo": str(cfg.assets.libero.repo),
        "revision": str(cfg.assets.libero.revision),
        "suite": str(cfg.profile.task),
    }:
        return False
    return len(_libero_hdf5_files(cfg)) >= int(cfg.assets.libero.minimum_hdf5_files)


def build_libero_download_command(cfg: DictConfig) -> tuple[str, ...]:
    """Build the immutable LIBERO dataset download command."""

    return (
        "python",
        "-m",
        "dreamervla.preprocess.download_libero",
        "--repo",
        str(cfg.assets.libero.repo),
        "--revision",
        str(cfg.assets.libero.revision),
        "--suite",
        str(cfg.profile.task),
        "--target",
        str(_stage_path(cfg.assets.libero.target)),
    )


def _asset_file_records(paths: Sequence[Path], root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve().relative_to(root.resolve())),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _prepare_assets(workflow: ReproductionWorkflow) -> None:
    cfg = workflow.cfg
    data_root = _stage_path(cfg.data_root)
    third_party_root = _stage_path(cfg.third_party_root)
    data_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "DVLA_ROOT": str(PROJECT_ROOT),
            "DVLA_DATA_ROOT": str(data_root),
            "DVLA_THIRD_PARTY_ROOT": str(third_party_root),
            "OPENVLA_OFT_ROOT": str(third_party_root / "openvla-oft"),
        }
    )
    if workflow.dry_run:
        third_party_revisions = {key: str(value) for key, value in cfg.third_party.items()}
    else:
        _verify_hardware(cfg, data_root)
        third_party_revisions = _verify_third_party(cfg)

    model_root = _stage_path(cfg.assets.openvla.target)
    if not _valid_openvla(cfg):
        if model_root.exists() and any(model_root.iterdir()):
            raise ReproductionError(
                f"OpenVLA target exists but is incomplete; move it aside and rerun: {model_root}"
            )
        download_model = (
            "bash",
            str(PROJECT_ROOT / "scripts/download_assets.sh"),
            "download.libero=false",
            "download.openvla_one_traj=true",
            "only=[10_openvla_oft_one_trajectory]",
            "env.OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD=git",
            "env.OPENVLA_ONE_TRAJ_REPOS="
            + json.dumps([f"{cfg.assets.openvla.repo}:{model_root.name}"]),
        )
        _run(download_model, env=env, dry_run=workflow.dry_run)
        _run(
            ("git", "-C", str(model_root), "checkout", str(cfg.assets.openvla.revision)),
            env=env,
            dry_run=workflow.dry_run,
        )
        _run(("git", "-C", str(model_root), "lfs", "pull"), env=env, dry_run=workflow.dry_run)

    if not _valid_libero(cfg):
        dataset_root = _stage_path(cfg.assets.libero.target)
        if dataset_root.exists() and any(dataset_root.iterdir()):
            raise ReproductionError(
                f"LIBERO target exists but is incomplete; move it aside and rerun: {dataset_root}"
            )
        _run(build_libero_download_command(cfg), env=env, dry_run=workflow.dry_run)

    preprocess = (
        "bash",
        str(PROJECT_ROOT / "scripts/preprocess/prepare_libero_data.sh"),
        f"task={cfg.profile.task}",
        f"artifact_name={cfg.preprocess.artifact_name}",
        f"ngpu={int(cfg.preprocess.ngpu)}",
        f"gpus={cfg.preprocess.gpus}",
    )
    _run(preprocess, env=env, dry_run=workflow.dry_run)
    if workflow.dry_run:
        return
    if not _valid_openvla(cfg):
        raise ReproductionError(f"OpenVLA validation failed after download: {model_root}")
    dataset_files = _libero_hdf5_files(cfg)
    if not _valid_libero(cfg):
        raise ReproductionError(
            "LIBERO validation failed for pinned source: "
            f"repo={cfg.assets.libero.repo} revision={cfg.assets.libero.revision} "
            f"expected at least {cfg.assets.libero.minimum_hdf5_files} HDF5 files, "
            f"got {len(dataset_files)}"
        )
    validate_hdf5_dir(
        cfg.preprocess.hidden_dir,
        reference_dir=cfg.preprocess.reward_dir,
        require_complete_attr=True,
        require_config=True,
        match_reference_demos=True,
        match_reference_lengths=True,
    )
    model_files = [model_root / str(name) for name in cfg.assets.openvla.required_files]
    model_files.extend(sorted(model_root.glob("model-*.safetensors")))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "profile": str(cfg.profile.id),
        "created_at": datetime.now(UTC).isoformat(),
        "image": _image_metadata(),
        "third_party": third_party_revisions,
        "openvla": {
            "repo": str(cfg.assets.openvla.repo),
            "revision": str(cfg.assets.openvla.revision),
            "files": _asset_file_records(model_files, data_root),
        },
        "libero": {
            "repo": str(cfg.assets.libero.repo),
            "revision": str(cfg.assets.libero.revision),
            "files": _asset_file_records(dataset_files, data_root),
        },
        "preprocess": {
            "reward_dir": str(_stage_path(cfg.preprocess.reward_dir)),
            "hidden_dir": str(_stage_path(cfg.preprocess.hidden_dir)),
            "token_count": int(cfg.preprocess.token_count),
            "token_dim": int(cfg.preprocess.token_dim),
            "history": int(cfg.preprocess.history),
            "chunk_size": int(cfg.preprocess.chunk_size),
        },
    }
    atomic_write_json(cfg.manifest_path, manifest)
    print(f"[reproduce] assets complete: {cfg.manifest_path}", flush=True)


def _validate_frozen_route(cfg: DictConfig, selected: Mapping[str, Path]) -> None:
    from dreamervla.launchers.train import build_launch

    launch = build_launch(
        [
            "--config",
            str(cfg.stages.dreamer.experiment),
            "--wm_ckpt",
            str(selected["world_model"]),
            "--cls_ckpt",
            str(selected["classifier"]),
            "dry_run=true",
        ]
    )
    expected = cfg.frozen_assertions
    if str(launch.cfg._target_) != str(expected._target_):
        raise ReproductionError(
            f"Dreamer target mismatch: expected={expected._target_} actual={launch.cfg._target_}"
        )
    for key, value in expected.manual_cotrain.items():
        actual = launch.cfg.manual_cotrain[key]
        if actual != value:
            raise ReproductionError(
                f"Dreamer frozen assertion failed: manual_cotrain.{key} "
                f"expected={value!r} actual={actual!r}"
            )


def _train_dreamer(workflow: ReproductionWorkflow) -> None:
    cfg = workflow.cfg
    data_root = _stage_path(cfg.data_root)
    env = os.environ.copy()
    env.update({"DVLA_ROOT": str(PROJECT_ROOT), "DVLA_DATA_ROOT": str(data_root)})
    if workflow.dry_run:
        selected = {
            "world_model": _stage_path(cfg.output_root) / "world_model/checkpoints/SELECTED.ckpt",
            "classifier": _stage_path(cfg.output_root) / "classifier/checkpoints/SELECTED.ckpt",
        }
        for name in ("world_model", "classifier", "dreamer"):
            command = build_stage_command(
                cfg,
                name,
                action="fresh",
                selected_checkpoints=selected,
            )
            _run(command, env=env, dry_run=True)
        return

    asset_manifest = _load_json(_stage_path(cfg.asset_manifest_path))
    if asset_manifest.get("status") != "complete" or asset_manifest.get("profile") != str(
        cfg.profile.id
    ):
        raise ReproductionError(
            f"asset manifest is missing, incomplete, or for another profile: "
            f"{cfg.asset_manifest_path}"
        )
    state_path = _stage_path(cfg.state_path)
    state = _load_json(state_path) or {
        "schema_version": 1,
        "profile": str(cfg.profile.id),
        "stages": {},
    }
    if state.get("profile") != str(cfg.profile.id):
        raise ReproductionError(
            f"training state profile mismatch: {state.get('profile')} != {cfg.profile.id}"
        )
    selected: dict[str, Path] = {}
    for name in ("world_model", "classifier", "dreamer"):
        stage = cfg.stages[name]
        decision = decide_stage(
            state,
            stage=name,
            run_root=stage.run_root,
            budget=int(stage.budget),
        )
        if decision.action == "skip":
            if decision.selected_checkpoint is not None:
                selected[name] = decision.selected_checkpoint
            print(f"[reproduce] skip completed stage={name}", flush=True)
            continue
        if name == "dreamer":
            _validate_frozen_route(cfg, selected)
        command = build_stage_command(
            cfg,
            name,
            action=decision.action,
            selected_checkpoints=selected,
        )
        record = {
            "status": "running",
            "budget": int(stage.budget),
            "run_root": str(_stage_path(stage.run_root)),
            "command": list(command),
            "started_at": datetime.now(UTC).isoformat(),
        }
        state["stages"][name] = record
        atomic_write_json(state_path, state)
        try:
            _run(command, env=env, dry_run=False)
        except subprocess.CalledProcessError:
            record["status"] = "interrupted"
            atomic_write_json(state_path, state)
            raise
        checkpoint_dir = _stage_path(stage.run_root) / "checkpoints"
        if "selection" in stage:
            chosen = select_metric_checkpoint(
                checkpoint_dir,
                metric_name=str(stage.selection.metric_name),
                mode=str(stage.selection.mode),
            )
            selected_path = chosen.path
            record["selected_metric"] = {
                "name": chosen.metric_name,
                "value": chosen.value,
                "epoch": chosen.epoch,
            }
        else:
            selected_path = checkpoint_dir / "latest.ckpt"
            if not selected_path.is_file():
                raise ReproductionError(f"Dreamer did not write latest checkpoint: {selected_path}")
        selected[name] = selected_path.resolve()
        record.update(
            {
                "status": "completed",
                "selected_checkpoint": str(selected[name]),
                "sha256": sha256_file(selected[name]),
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        atomic_write_json(state_path, state)
    print(f"[reproduce] training complete: {state_path}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one configured reproduction workflow."""

    workflow = build_workflow(list(sys.argv[1:] if argv is None else argv))
    print(
        f"[reproduce] mode={workflow.cfg.mode} profile={workflow.cfg.profile.id} "
        f"data_root={workflow.cfg.data_root} dry_run={str(workflow.dry_run).lower()}",
        flush=True,
    )
    try:
        if workflow.cfg.mode == "prepare_assets":
            _prepare_assets(workflow)
        elif workflow.cfg.mode == "train_dreamer":
            _train_dreamer(workflow)
        else:
            raise ReproductionError(f"unknown reproduction mode: {workflow.cfg.mode}")
    except ReproductionError as exc:
        print(f"[reproduce] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReproductionWorkflow",
    "build_libero_download_command",
    "build_stage_command",
    "build_workflow",
    "main",
]
