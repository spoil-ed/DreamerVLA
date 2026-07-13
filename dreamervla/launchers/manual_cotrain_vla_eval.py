"""Periodic real-LIBERO VLA evaluation for manual cotrain commands.

The training process is segmented at evaluation boundaries so Ray releases its
GPU actors before the single-GPU evaluator starts.  Evaluation is deliberately
outside replay and PPO: step zero runs the immutable base VLA, while later
steps run the same OFT observation/prompt/action-chunk path with the saved
ActorGroup policy state.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class PeriodicVLAEvalSpec:
    """Resolved Hydra contract for periodic real-LIBERO evaluation."""

    interval_global_steps: int
    include_initial: bool
    base_vla_ckpt: Path
    task_suite_name: str
    task_ids: tuple[int, ...]
    num_episodes_per_task: int
    num_envs: int
    action_steps: int
    history_length: int
    seed: int
    gpus: str
    render_backend: str
    significant_drop_threshold: float
    learner_updates_enabled: bool

    @property
    def enabled(self) -> bool:
        return bool(self.include_initial or self.interval_global_steps > 0)


def periodic_vla_eval_spec(cfg: DictConfig) -> PeriodicVLAEvalSpec:
    """Resolve and validate the manual-cotrain eval block from a train config."""

    protocol = OmegaConf.select(cfg, "manual_cotrain.eval_protocol", default={})
    interval = int(
        OmegaConf.select(
            cfg,
            "manual_cotrain.eval_interval_global_steps",
            default=0,
        )
        or 0
    )
    include_initial = bool(
        OmegaConf.select(
            cfg,
            "manual_cotrain.eval_initial_global_step",
            default=False,
        )
    )
    raw_base = OmegaConf.select(protocol, "base_vla_ckpt", default=None)
    if raw_base in (None, ""):
        raise ValueError(
            "manual_cotrain.eval_protocol.base_vla_ckpt is required"
        )
    base_vla_ckpt = Path(str(raw_base)).expanduser().resolve()
    task_ids = tuple(
        int(task_id)
        for task_id in (
            OmegaConf.select(protocol, "task_ids", default=[]) or []
        )
    )
    num_episodes = int(
        OmegaConf.select(protocol, "num_episodes_per_task", default=0) or 0
    )
    num_envs = int(OmegaConf.select(protocol, "num_envs", default=0) or 0)
    action_steps = int(
        OmegaConf.select(protocol, "action_steps", default=0) or 0
    )
    history_length = int(
        OmegaConf.select(protocol, "history_length", default=0) or 0
    )
    if interval < 0:
        raise ValueError("manual_cotrain.eval_interval_global_steps must be non-negative")
    if not task_ids:
        raise ValueError("manual_cotrain.eval_protocol.task_ids cannot be empty")
    if len(set(task_ids)) != len(task_ids) or min(task_ids) < 0:
        raise ValueError(
            "manual_cotrain.eval_protocol.task_ids must be distinct non-negative IDs"
        )
    for name, value in (
        ("num_episodes_per_task", num_episodes),
        ("num_envs", num_envs),
        ("action_steps", action_steps),
        ("history_length", history_length),
    ):
        if value <= 0:
            raise ValueError(
                f"manual_cotrain.eval_protocol.{name} must be positive"
            )
    render_backend = str(
        OmegaConf.select(protocol, "render_backend", default="osmesa")
    ).lower()
    if render_backend not in {"osmesa", "egl"}:
        raise ValueError(
            "manual_cotrain.eval_protocol.render_backend must be osmesa or egl"
        )
    return PeriodicVLAEvalSpec(
        interval_global_steps=interval,
        include_initial=include_initial,
        base_vla_ckpt=base_vla_ckpt,
        task_suite_name=str(
            OmegaConf.select(protocol, "task_suite_name", default="libero_goal")
        ),
        task_ids=task_ids,
        num_episodes_per_task=num_episodes,
        num_envs=num_envs,
        action_steps=action_steps,
        history_length=history_length,
        seed=int(OmegaConf.select(protocol, "seed", default=0)),
        gpus=str(OmegaConf.select(protocol, "gpus", default="0")),
        render_backend=render_backend,
        significant_drop_threshold=float(
            OmegaConf.select(
                protocol,
                "significant_drop_threshold",
                default=0.10,
            )
        ),
        learner_updates_enabled=bool(
            OmegaConf.select(
                cfg,
                "manual_cotrain.learner_updates_enabled",
                default=True,
            )
        ),
    )


def periodic_eval_steps(
    *,
    start_step: int,
    target_step: int,
    interval: int,
    include_initial: bool,
) -> list[int]:
    """Scheduled eval steps, with no implicit non-multiple final evaluation."""

    start = int(start_step)
    target = int(target_step)
    every = int(interval)
    if start < 0 or target < start:
        raise ValueError(
            f"invalid global-step range start={start} target={target}"
        )
    steps: list[int] = []
    if include_initial and start == 0:
        steps.append(0)
    if every <= 0:
        return steps
    if start > 0 and start % every == 0:
        steps.append(start)
    next_step = ((start // every) + 1) * every
    steps.extend(range(next_step, target + 1, every))
    return steps


def _override_key(item: str) -> str | None:
    if item.startswith("-"):
        return None
    return item.split("=", 1)[0].lstrip("+~")


def _last_int_override(command: Sequence[str], key: str) -> int | None:
    value = None
    for item in command:
        if _override_key(str(item)) == key:
            value = int(str(item).split("=", 1)[1])
    return value


def manual_cotrain_target_step(command: Sequence[str]) -> int:
    target = _last_int_override(command, "manual_cotrain.global_steps")
    if target is None or target <= 0:
        raise ValueError(
            "manual_cotrain.global_steps must be a positive explicit override"
        )
    return int(target)


def _replace_overrides(
    command: Sequence[str],
    values: Mapping[str, str],
) -> list[str]:
    keys = {_override_key(str(key)) for key in values}
    out = [item for item in command if _override_key(str(item)) not in keys]
    out.extend(f"{key}={value}" for key, value in values.items())
    return out


def _hydra_string(value: str | Path) -> str:
    return json.dumps(str(value))


def segment_train_command(
    command: Sequence[str],
    *,
    target_step: int,
    checkpoint_every: int,
    run_root: Path,
    resume_ckpt: Path | None,
    learner_updates_enabled: bool = False,
) -> list[str]:
    """Return one training segment ending at an absolute manual global step."""

    values = {
        "manual_cotrain.global_steps": str(int(target_step)),
        "manual_cotrain.checkpoint_every": str(int(checkpoint_every)),
    }
    if resume_ckpt is not None:
        values.update(
            {
                "training.resume": "true",
                "training.resume_dir": _hydra_string(run_root),
                "++manual_cotrain.resume_ckpt": _hydra_string(resume_ckpt),
            }
        )
        if learner_updates_enabled:
            values.update(
                {
                    "learner.init_ckpt.path": _hydra_string(resume_ckpt),
                    "learner.init_ckpt.components": (
                        "[world_model,classifier,world_model_optimizer,"
                        "classifier_optimizer]"
                    ),
                }
            )
    return _replace_overrides(command, values)


def manual_cotrain_checkpoint(run_root: Path, global_step: int) -> Path:
    return (
        Path(run_root)
        / "checkpoints"
        / f"manual_cotrain_step_{int(global_step)}"
        / "manual_cotrain.ckpt"
    )


def _format_list(values: Sequence[int]) -> str:
    return "[" + ",".join(str(int(value)) for value in values) + "]"


def vla_eval_command(
    python: str,
    spec: PeriodicVLAEvalSpec,
    *,
    global_step: int,
    policy_ckpt: Path | None,
    out_dir: Path,
) -> list[str]:
    """Build baseline-VLA or learned-VLA-policy standalone eval command."""

    is_baseline = int(global_step) == 0
    ckpt_path = spec.base_vla_ckpt if is_baseline else policy_ckpt
    if ckpt_path is None:
        raise ValueError("post-update VLA policy eval requires a policy checkpoint")
    command = [
        str(python),
        "-m",
        "dreamervla.launchers.train",
        "--config-name",
        "eval_libero_vla",
        "experiment=eval_libero_vla",
        f"out_dir={_hydra_string(out_dir)}",
        f"gpus={spec.gpus}",
        f"eval.ckpt_path={_hydra_string(ckpt_path)}",
        f"eval.ckpt_kind={'vla' if is_baseline else 'vla_policy'}",
        f"init.vla_ckpt_path={_hydra_string(spec.base_vla_ckpt)}",
        f"eval.task_suite_name={spec.task_suite_name}",
        f"eval.task_ids={_format_list(spec.task_ids)}",
        f"eval.num_episodes_per_task={spec.num_episodes_per_task}",
        f"eval.num_envs={spec.num_envs}",
        f"eval.action_steps={spec.action_steps}",
        f"eval.history_length={spec.history_length}",
        f"eval.seed={spec.seed}",
        "eval.scheme=rlinf_chunk",
        "eval.enumerate_all_init_states=false",
        "eval.reconfigure_per_episode=true",
        "eval.action_postprocess=openvla_oft",
        "eval.require_strict_component_load=true",
        f"eval.render_backend={spec.render_backend}",
    ]
    if not is_baseline and bool(spec.learner_updates_enabled):
        command.extend(
            [
                "eval.cotrain_diagnostics=true",
                "eval.cotrain_expected_trajectories="
                f"{len(spec.task_ids) * spec.num_episodes_per_task}",
                "eval.cotrain_encode_batch_size=4",
            ]
        )
    return command


def _eval_env(base_env: Mapping[str, str], spec: PeriodicVLAEvalSpec) -> dict[str, str]:
    env = {str(key): str(value) for key, value in base_env.items()}
    env["CUDA_VISIBLE_DEVICES"] = str(spec.gpus)
    env["MUJOCO_GL"] = spec.render_backend
    env["PYOPENGL_PLATFORM"] = spec.render_backend
    if spec.render_backend == "osmesa":
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    return env


def _checkpoint_global_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "global_step" not in payload:
        raise ValueError(f"resume checkpoint has no global_step: {path}")
    return int(payload["global_step"])


def _read_eval_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"eval metrics not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"eval metrics must be a JSON object: {path}")
    return {
        str(key): float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _append_eval_summary(
    path: Path,
    *,
    global_step: int,
    ckpt_path: Path,
    ckpt_kind: str,
    eval_out_dir: Path,
    metrics: Mapping[str, float],
    significant_drop_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = {"records": []}
    records = list(raw.get("records", [])) if isinstance(raw, dict) else []
    records = [
        record
        for record in records
        if int(record.get("global_step", -1)) != int(global_step)
    ]
    records.sort(key=lambda record: int(record.get("global_step", -1)))
    previous_rate = (
        float(records[-1]["eval_success_rate"])
        if records and "eval_success_rate" in records[-1]
        else None
    )
    success_rate = float(metrics.get("eval_success_rate", 0.0))
    previous_best = max(
        [float(record.get("eval_success_rate", 0.0)) for record in records],
        default=success_rate,
    )
    best_rate = max(previous_best, success_rate)
    drop = (
        max(0.0, float(previous_rate) - success_rate)
        if previous_rate is not None
        else 0.0
    )
    delta = (
        success_rate - float(previous_rate)
        if previous_rate is not None
        else 0.0
    )
    record: dict[str, Any] = {
        "global_step": int(global_step),
        "policy_source": "base_vla" if int(global_step) == 0 else "ppo_vla_policy",
        "ckpt_kind": str(ckpt_kind),
        "ckpt_path": str(ckpt_path),
        "eval_out_dir": str(eval_out_dir),
        "eval_success_rate": success_rate,
        "eval_success_rate_delta": float(delta),
        "eval_best_success_rate": float(best_rate),
        "eval_success_rate_drop": float(drop),
        "eval_significant_drop": float(
            drop > max(0.0, float(significant_drop_threshold))
        ),
    }
    for key, value in metrics.items():
        record.setdefault(str(key), float(value))
    records.append(record)
    records.sort(key=lambda item: int(item["global_step"]))
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "significant_drop_threshold": float(significant_drop_threshold),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def execute_periodic_vla_eval(
    *,
    training_command: Sequence[str],
    training_env: Mapping[str, str],
    run_root: Path,
    resume_ckpt: Path | None,
    spec: PeriodicVLAEvalSpec,
    cwd: Path,
) -> int:
    """Run segmented manual cotrain with isolated real-LIBERO eval stages."""

    if not spec.enabled:
        result = subprocess.run(
            list(training_command),
            cwd=cwd,
            env=dict(training_env),
            check=False,
        )
        return int(result.returncode)

    target_step = manual_cotrain_target_step(training_command)
    start_step = _checkpoint_global_step(resume_ckpt) if resume_ckpt else 0
    if target_step < start_step:
        raise ValueError(
            f"target global step {target_step} precedes resume step {start_step}"
        )
    scheduled = periodic_eval_steps(
        start_step=start_step,
        target_step=target_step,
        interval=spec.interval_global_steps,
        include_initial=spec.include_initial,
    )
    current_step = int(start_step)
    current_ckpt = resume_ckpt
    summary_path = Path(run_root) / "eval" / "eval_summary.json"

    def run_eval(step: int, checkpoint: Path | None) -> int:
        kind = "vla" if int(step) == 0 else "vla_policy"
        source_ckpt = spec.base_vla_ckpt if int(step) == 0 else checkpoint
        if source_ckpt is None:
            raise ValueError("learned VLA policy eval requires a saved checkpoint")
        out_dir = Path(run_root) / "eval" / f"global_step_{int(step):08d}"
        command = vla_eval_command(
            str(training_command[0]),
            spec,
            global_step=int(step),
            policy_ckpt=checkpoint,
            out_dir=out_dir,
        )
        print(
            "[periodic-vla-eval] "
            f"global_step={int(step)} kind={kind} "
            f"tasks={len(spec.task_ids)} episodes_per_task={spec.num_episodes_per_task} "
            f"checkpoint={source_ckpt}",
            flush=True,
        )
        result = subprocess.run(
            command,
            cwd=cwd,
            env=_eval_env(training_env, spec),
            check=False,
        )
        if int(result.returncode) != 0:
            return int(result.returncode)
        metrics = _read_eval_metrics(out_dir / "eval_libero_metrics.json")
        _append_eval_summary(
            summary_path,
            global_step=int(step),
            ckpt_path=source_ckpt,
            ckpt_kind=kind,
            eval_out_dir=out_dir,
            metrics=metrics,
            significant_drop_threshold=spec.significant_drop_threshold,
        )
        return 0

    for eval_step in scheduled:
        if int(eval_step) == 0:
            code = run_eval(0, None)
            if code != 0:
                return code
            continue
        if int(eval_step) == current_step:
            code = run_eval(current_step, current_ckpt)
            if code != 0:
                return code
            continue
        train_cmd = segment_train_command(
            training_command,
            target_step=int(eval_step),
            checkpoint_every=max(1, int(spec.interval_global_steps)),
            run_root=Path(run_root),
            resume_ckpt=current_ckpt,
            learner_updates_enabled=spec.learner_updates_enabled,
        )
        print(
            "[manual-cotrain-segment] "
            f"from_step={current_step} to_step={int(eval_step)} "
            f"resume={current_ckpt is not None}",
            flush=True,
        )
        result = subprocess.run(
            train_cmd,
            cwd=cwd,
            env=dict(training_env),
            check=False,
        )
        if int(result.returncode) != 0:
            return int(result.returncode)
        current_step = int(eval_step)
        current_ckpt = manual_cotrain_checkpoint(run_root, current_step)
        if not current_ckpt.is_file():
            raise FileNotFoundError(
                f"manual cotrain checkpoint for eval not found: {current_ckpt}"
            )
        code = run_eval(current_step, current_ckpt)
        if code != 0:
            return code

    if current_step < target_step:
        train_cmd = segment_train_command(
            training_command,
            target_step=target_step,
            checkpoint_every=max(1, int(spec.interval_global_steps)),
            run_root=Path(run_root),
            resume_ckpt=current_ckpt,
            learner_updates_enabled=spec.learner_updates_enabled,
        )
        print(
            "[manual-cotrain-segment] "
            f"from_step={current_step} to_step={target_step} "
            f"resume={current_ckpt is not None}",
            flush=True,
        )
        result = subprocess.run(
            train_cmd,
            cwd=cwd,
            env=dict(training_env),
            check=False,
        )
        return int(result.returncode)
    return 0


__all__ = [
    "PeriodicVLAEvalSpec",
    "execute_periodic_vla_eval",
    "manual_cotrain_checkpoint",
    "manual_cotrain_target_step",
    "periodic_eval_steps",
    "periodic_vla_eval_spec",
    "segment_train_command",
    "vla_eval_command",
]
