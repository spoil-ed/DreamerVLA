"""Run the official-data WM/CLS -> frozen imagined-RL feasibility chain."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dreamervla.utils.frozen_components import load_frozen_component
from dreamervla.utils.hydra_config import script_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STAGES = {"all", "wm", "classifier", "rl", "eval"}
_WM_LOSS_RE = re.compile(
    r"^wm_step=(?P<step>\d{8})-loss=(?P<loss>[-+0-9.eE]+)\.ckpt$"
)
_WM_PROGRESS_RE = re.compile(r"^wm_step_(?P<step>\d{8})\.ckpt$")
_CLASSIFIER_WINDOW_RE = re.compile(
    r"^best_window_f1(?P<f1>[-+0-9.eE]+)_th(?P<threshold>[-+0-9.eE]+)\.ckpt$"
)


@dataclass(frozen=True)
class FrozenPreMainlinePlan:
    """Fully resolved subprocess commands and artifact paths for one run."""

    stage: str
    run_root: Path
    summary_dir: Path
    selected_wm_ckpt: Path
    selected_classifier_ckpt: Path
    rl_final_ckpt: Path
    frozen_summary: Path
    feasibility_summary: Path
    wm_cmd: list[str]
    classifier_cmd: list[str]
    rl_cmd: list[str]
    eval_baseline_cmd: list[str]
    eval_rl_cmd: list[str]
    compare_cmd: list[str]

    @property
    def commands(self) -> tuple[list[str], ...]:
        return (
            self.wm_cmd,
            self.classifier_cmd,
            self.rl_cmd,
            self.eval_baseline_cmd,
            self.eval_rl_cmd,
            self.compare_cmd,
        )


def _as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return str(value)


def _override_key(value: Any) -> str:
    return str(value).split("=", 1)[0].lstrip("+~")


def _assert_no_reserved_overrides(
    values: Sequence[Any],
    *,
    reserved: set[str],
    label: str,
) -> None:
    for value in values:
        key = _override_key(value)
        if any(key == item or key.startswith(f"{item}.") for item in reserved):
            raise ValueError(f"{label} cannot override reserved causal boundary {key!r}")


def _train_command(
    *,
    python: str,
    experiment: str,
    task: str,
    out_dir: Path,
    overrides: Sequence[Any] = (),
    ngpu: int = 1,
    master_port: int = 29511,
    distributed: bool = False,
) -> list[str]:
    command = [python, "-m"]
    if distributed and int(ngpu) > 1:
        command.extend(
            [
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={int(ngpu)}",
                f"--master_port={int(master_port)}",
                "-m",
            ]
        )
    command.extend(
        [
            "dreamervla.train",
            "--config-name",
            "train",
            f"experiment={experiment}",
            f"task={task}",
            f"training.out_dir={out_dir}",
        ]
    )
    command.extend(str(item) for item in overrides)
    return command


def _task_spec(cfg: Mapping[str, Any]) -> dict[str, Any]:
    task = str(cfg.get("task", "goal")).lower().removeprefix("libero_")
    tasks = cfg.get("tasks")
    if not isinstance(tasks, Mapping) or task not in tasks:
        allowed = ", ".join(sorted(str(key) for key in (tasks or {})))
        raise ValueError(f"task must be one of: {allowed}")
    return dict(tasks[task])


def _validate_proof_contract(
    cfg: Mapping[str, Any],
    *,
    task_spec: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
) -> None:
    """Reject script-level overrides that weaken the fixed feasibility proof."""

    if str(cfg.get("task", "goal")).lower().removeprefix("libero_") != "goal":
        raise ValueError("frozen-model proof requires task=goal")
    if task_spec.get("hydra_task") != "openvla_onetraj_libero":
        raise ValueError("frozen-model proof requires the canonical goal Hydra task")
    if task_spec.get("suite") != "libero_goal":
        raise ValueError("frozen-model proof requires suite=libero_goal")
    expected_baseline = (
        Path(str(cfg["data_root"]))
        / "checkpoints"
        / "Openvla-oft-SFT-traj1"
        / "Openvla-oft-SFT-libero-goal-traj1"
    ).expanduser().resolve()
    actual_baseline = Path(str(task_spec.get("baseline_ckpt", ""))).expanduser().resolve()
    if actual_baseline != expected_baseline:
        raise ValueError(
            "frozen-model proof requires the canonical one-trajectory baseline checkpoint"
        )

    expected_eval: dict[str, Any] = {
        "task_ids": list(range(10)),
        "max_tasks": 10,
        "num_episodes_per_task": 10,
        "num_envs": 64,
        "seed": 7,
        "num_steps_wait": 10,
        "action_steps": 8,
        "max_steps": 300,
        "enumerate_all_init_states": False,
        "scheme": "rlinf_chunk",
        "reconfigure_per_episode": True,
        "history_length": 1,
        "action_postprocess": "none",
        "render_backend": "osmesa",
        "dreamer_policy_source": "ckpt",
        "dreamer_deterministic": True,
        "dreamer_action_repeat": 1,
        "dreamer_clip_actions": True,
        "dreamer_unnorm_actions": "auto",
        "dreamer_latent_action_source": "env",
        "dreamer_rollout_mode": "stateless",
        "dreamer_actor_input_source": "latent",
        "dreamer_wm_history_length": 2,
        "dreamer_wm_rotate_images": True,
        "dreamer_wm_prompt_style": "vla_policy",
        "dreamer_wm_include_state": False,
    }
    for key, expected in expected_eval.items():
        actual = eval_cfg.get(key)
        if actual != expected:
            raise ValueError(
                f"frozen-model proof requires eval.{key}={expected!r}, got {actual!r}"
            )


def build_pipeline_plan(cfg: Mapping[str, Any]) -> FrozenPreMainlinePlan:
    """Build commands without touching checkpoints or starting subprocesses."""

    stage = str(cfg.get("stage", "all")).lower()
    if stage not in _STAGES:
        raise ValueError(f"stage must be one of: {', '.join(sorted(_STAGES))}")
    run_root = Path(str(cfg["run_root"])).expanduser().resolve()
    summary_dir = run_root / "summary"
    selected_wm = summary_dir / "selected_wm.ckpt"
    selected_classifier = summary_dir / "selected_classifier.ckpt"
    task_spec = _task_spec(cfg)
    hydra_task = str(task_spec["hydra_task"])
    suite = str(task_spec["suite"])
    python = str(cfg.get("python") or sys.executable)
    ngpu = int(cfg.get("ngpu", 1) or 1)
    master_port = int(cfg.get("master_port", 29511) or 29511)
    wm_overrides = _as_list(cfg.get("wm_overrides"))
    classifier_overrides = _as_list(cfg.get("classifier_overrides"))
    rl_overrides = _as_list(cfg.get("rl_overrides"))
    eval_overrides = _as_list(cfg.get("eval_overrides"))
    common_reserved = {
        "experiment",
        "task",
        "pre_mainline",
        "training.out_dir",
        "_target_",
    }
    _assert_no_reserved_overrides(
        wm_overrides,
        reserved=common_reserved
        | {
            "pre_mainline.stage",
            "offline_warmup.data_dir",
            "offline_warmup.hidden_dir",
            "training.classifier_warmup_steps",
            "online_rollout.total_env_steps",
            "training.debug",
        },
        label="wm_overrides",
    )
    _assert_no_reserved_overrides(
        classifier_overrides,
        reserved=common_reserved
        | {
            "pre_mainline.stage",
            "data.success_dir_raw",
            "data.success_dir_hidden",
            "data.failure_dir_raw",
            "data.failure_dir_hidden",
        },
        label="classifier_overrides",
    )
    _assert_no_reserved_overrides(
        rl_overrides,
        reserved=common_reserved
        | {
            "pre_mainline.stage",
            "init.world_model_state_ckpt",
            "init.classifier_state_ckpt",
            "official_replay.data_dir",
            "official_replay.hidden_dir",
            "algorithm.update_type",
        },
        label="rl_overrides",
    )

    wm_cmd = _train_command(
        python=python,
        experiment="wm_official_upper_bound",
        task=hydra_task,
        out_dir=run_root / "wm",
        overrides=wm_overrides,
        ngpu=ngpu,
        master_port=master_port,
        distributed=True,
    )
    classifier_cmd = _train_command(
        python=python,
        experiment="classifier_official_upper_bound",
        task=hydra_task,
        out_dir=run_root / "classifier",
        overrides=classifier_overrides,
        ngpu=ngpu,
        master_port=master_port + 1,
        distributed=True,
    )
    rl_cmd = _train_command(
        python=python,
        experiment="dreamervla_frozen_models_rl",
        task=hydra_task,
        out_dir=run_root / "rl",
        overrides=[
            f"init.world_model_state_ckpt={selected_wm}",
            f"init.classifier_state_ckpt={selected_classifier}",
            *rl_overrides,
        ],
    )

    eval_cfg = cfg.get("eval", {})
    if not isinstance(eval_cfg, Mapping):
        raise TypeError("eval config must be a mapping")
    _validate_proof_contract(cfg, task_spec=task_spec, eval_cfg=eval_cfg)
    protocol_keys = {
        "eval.ckpt_path",
        "eval.ckpt_kind",
        "eval.task_suite_name",
        "eval.task_ids",
        "eval.task_start",
        "eval.max_tasks",
        "eval.num_episodes_per_task",
        "eval.num_envs",
        "eval.seed",
        "eval.num_steps_wait",
        "eval.action_steps",
        "eval.max_steps",
        "eval.enumerate_all_init_states",
        "eval.scheme",
        "eval.reconfigure_per_episode",
        "eval.history_length",
        "eval.action_postprocess",
        "eval.render_backend",
        "eval.dreamer_policy_source",
        "eval.dreamer_deterministic",
        "eval.dreamer_action_repeat",
        "eval.dreamer_clip_actions",
        "eval.dreamer_unnorm_actions",
        "eval.dreamer_latent_action_source",
        "eval.dreamer_rollout_mode",
        "eval.dreamer_actor_input_source",
        "eval.dreamer_wm_history_length",
        "eval.dreamer_wm_rotate_images",
        "eval.dreamer_wm_prompt_style",
        "eval.dreamer_wm_include_state",
        "eval.require_strict_component_load",
        "eval.libero_env",
    }
    _assert_no_reserved_overrides(
        eval_overrides,
        reserved=common_reserved
        | protocol_keys
        | {
            "init",
            "encoder",
            "world_model",
            "classifier",
            "policy",
            "algorithm",
            "optim",
            "dataloader",
        },
        label="eval_overrides",
    )
    task_ids = eval_cfg.get("task_ids")
    max_tasks = eval_cfg.get("max_tasks")
    protocol = [
        f"eval.task_suite_name={suite}",
        f"eval.task_ids={_hydra_value(task_ids)}",
        "eval.task_start=0",
        f"eval.max_tasks={_hydra_value(max_tasks)}",
        f"eval.num_episodes_per_task={int(eval_cfg.get('num_episodes_per_task', 10))}",
        f"eval.num_envs={int(eval_cfg.get('num_envs', 64))}",
        f"eval.seed={int(eval_cfg.get('seed', 7))}",
        f"eval.num_steps_wait={int(eval_cfg.get('num_steps_wait', 10))}",
        f"eval.action_steps={int(eval_cfg.get('action_steps', 8))}",
        f"eval.max_steps={_hydra_value(eval_cfg.get('max_steps'))}",
        "eval.enumerate_all_init_states="
        f"{_hydra_value(eval_cfg.get('enumerate_all_init_states', False))}",
        f"eval.scheme={eval_cfg.get('scheme', 'rlinf_chunk')}",
        "eval.reconfigure_per_episode="
        f"{_hydra_value(eval_cfg.get('reconfigure_per_episode', True))}",
        f"eval.history_length={int(eval_cfg.get('history_length', 1))}",
        f"eval.action_postprocess={eval_cfg.get('action_postprocess', 'none')}",
        f"eval.render_backend={eval_cfg.get('render_backend', 'osmesa')}",
        f"eval.dreamer_policy_source={eval_cfg.get('dreamer_policy_source', 'ckpt')}",
        f"eval.dreamer_deterministic={_hydra_value(eval_cfg.get('dreamer_deterministic', True))}",
        f"eval.dreamer_action_repeat={int(eval_cfg.get('dreamer_action_repeat', 1))}",
        f"eval.dreamer_clip_actions={_hydra_value(eval_cfg.get('dreamer_clip_actions', True))}",
        f"eval.dreamer_unnorm_actions={eval_cfg.get('dreamer_unnorm_actions', 'auto')}",
        f"eval.dreamer_latent_action_source={eval_cfg.get('dreamer_latent_action_source', 'env')}",
        f"eval.dreamer_rollout_mode={eval_cfg.get('dreamer_rollout_mode', 'stateless')}",
        f"eval.dreamer_actor_input_source={eval_cfg.get('dreamer_actor_input_source', 'latent')}",
        f"eval.dreamer_wm_history_length={int(eval_cfg.get('dreamer_wm_history_length', 2))}",
        "eval.dreamer_wm_rotate_images="
        f"{_hydra_value(eval_cfg.get('dreamer_wm_rotate_images', True))}",
        f"eval.dreamer_wm_prompt_style={eval_cfg.get('dreamer_wm_prompt_style', 'vla_policy')}",
        "eval.dreamer_wm_include_state="
        f"{_hydra_value(eval_cfg.get('dreamer_wm_include_state', False))}",
        "eval.require_strict_component_load=true",
        *eval_overrides,
    ]
    baseline_cmd = _train_command(
        python=python,
        experiment="eval_libero_vla",
        task=hydra_task,
        out_dir=run_root / "eval_baseline",
        overrides=[
            f"eval.ckpt_path={task_spec['baseline_ckpt']}",
            "eval.ckpt_kind=vla",
            *protocol,
        ],
    )
    rl_final = run_root / "rl" / "checkpoints" / "final.ckpt"
    eval_rl_cmd = _train_command(
        python=python,
        experiment="eval_libero_vla",
        task=hydra_task,
        out_dir=run_root / "eval_rl",
        overrides=[
            f"eval.ckpt_path={rl_final}",
            "eval.ckpt_kind=dreamer",
            *protocol,
        ],
    )
    frozen_summary = run_root / "rl" / "frozen_rl_summary.json"
    feasibility_summary = summary_dir / "feasibility_summary.json"
    compare_cmd = [
        python,
        "-m",
        "dreamervla.diagnostics.compare_frozen_rl_eval",
        "--baseline",
        str(run_root / "eval_baseline" / "eval_libero_metrics.json"),
        "--rl",
        str(run_root / "eval_rl" / "eval_libero_metrics.json"),
        "--frozen-summary",
        str(frozen_summary),
        "--output",
        str(feasibility_summary),
    ]
    return FrozenPreMainlinePlan(
        stage=stage,
        run_root=run_root,
        summary_dir=summary_dir,
        selected_wm_ckpt=selected_wm,
        selected_classifier_ckpt=selected_classifier,
        rl_final_ckpt=rl_final,
        frozen_summary=frozen_summary,
        feasibility_summary=feasibility_summary,
        wm_cmd=wm_cmd,
        classifier_cmd=classifier_cmd,
        rl_cmd=rl_cmd,
        eval_baseline_cmd=baseline_cmd,
        eval_rl_cmd=eval_rl_cmd,
        compare_cmd=compare_cmd,
    )


def select_world_model_checkpoint(wm_root: str | Path) -> Path:
    """Choose a validated ranked WM only after its run completed successfully."""

    root = Path(wm_root).expanduser().resolve()
    complete_path = root / "ckpt" / "wm_warmup.ckpt"
    if not complete_path.is_file():
        raise FileNotFoundError(
            f"complete WM checkpoint does not exist: {complete_path}"
        )
    complete = load_frozen_component(complete_path, "world_model")
    complete_config = complete.metadata.get("config")
    if complete.metadata.get("complete") is not True:
        raise ValueError(f"complete WM checkpoint is not marked complete: {complete_path}")
    complete_step = int(complete.metadata.get("warmup_step", 0) or 0)
    complete_total = int(complete.metadata.get("warmup_total_steps", 0) or 0)
    if (
        complete.metadata.get("warmup_component") != "wm"
        or complete_step <= 0
        or complete_step != complete_total
    ):
        raise ValueError(
            f"complete WM checkpoint has invalid completion metadata: {complete_path}"
        )
    if (
        not isinstance(complete_config, Mapping)
        or not isinstance(complete_config.get("world_model"), Mapping)
    ):
        raise ValueError(
            f"complete WM checkpoint has no config.world_model metadata: {complete_path}"
        )

    ranked: list[tuple[float, Path]] = []
    ranked_dir = root / "ckpt" / "warmup_topk" / "wm"
    for path in ranked_dir.glob("*.ckpt"):
        match = _WM_LOSS_RE.match(path.name)
        if match is None:
            raise ValueError(f"unexpected ranked WM checkpoint name: {path}")
        loaded = load_frozen_component(path, "world_model")
        metadata = loaded.metadata
        step = int(match.group("step"))
        filename_loss = float(match.group("loss"))
        metrics = metadata.get("metrics")
        config = metadata.get("config")
        if metadata.get("warmup_component") != "wm":
            raise ValueError(f"ranked checkpoint is not a WM warmup checkpoint: {path}")
        if metadata.get("complete") is not False:
            raise ValueError(f"ranked WM checkpoint must be marked complete=false: {path}")
        warmup_step = int(metadata.get("warmup_step", 0) or 0)
        total_steps = int(metadata.get("warmup_total_steps", 0) or 0)
        if warmup_step != step or warmup_step <= 0 or total_steps <= 0 or step > total_steps:
            raise ValueError(f"ranked WM checkpoint step metadata is invalid: {path}")
        if not isinstance(metrics, Mapping) or "loss" not in metrics:
            raise ValueError(f"ranked WM checkpoint has no metrics.loss: {path}")
        loss = float(metrics["loss"])
        if not math.isfinite(loss) or not math.isfinite(filename_loss):
            raise ValueError(f"ranked WM checkpoint loss is not finite: {path}")
        expected_name = f"wm_step={step:08d}-loss={loss:.6f}.ckpt"
        if path.name != expected_name:
            raise ValueError(
                f"ranked WM filename does not match its rounded loss metadata: {path}"
            )
        if config != complete_config:
            raise ValueError(
                f"ranked WM checkpoint config differs from the completed run: {path}"
            )
        if total_steps != complete_total:
            raise ValueError(
                f"ranked WM checkpoint budget differs from the completed run: {path}"
            )
        ranked.append((loss, path.resolve()))

    if ranked:
        return min(ranked, key=lambda item: (item[0], str(item[1])))[1]
    return complete_path.resolve()


def _validate_available_world_model_checkpoint(path: Path) -> Mapping[str, Any]:
    loaded = load_frozen_component(path, "world_model")
    metadata = loaded.metadata
    component = metadata.get("warmup_component")
    if component not in (None, "wm"):
        raise ValueError(f"checkpoint is not a WM warmup checkpoint: {path}")
    config = metadata.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("world_model"), Mapping):
        raise ValueError(f"WM checkpoint has no config.world_model metadata: {path}")
    return metadata


def select_available_world_model_checkpoint(wm_root: str | Path) -> Path:
    """Select the best currently available WM checkpoint without a completion gate."""

    root = Path(wm_root).expanduser().resolve()
    ranked: list[tuple[float, Path]] = []
    ranked_dir = root / "ckpt" / "warmup_topk" / "wm"
    for path in sorted(ranked_dir.glob("*.ckpt")):
        match = _WM_LOSS_RE.match(path.name)
        if match is None:
            continue
        metadata = _validate_available_world_model_checkpoint(path)
        metrics = metadata.get("metrics")
        if not isinstance(metrics, Mapping) or "loss" not in metrics:
            raise ValueError(f"ranked WM checkpoint has no metrics.loss: {path}")
        loss = float(metrics["loss"])
        filename_loss = float(match.group("loss"))
        if not math.isfinite(loss) or not math.isfinite(filename_loss):
            raise ValueError(f"ranked WM checkpoint loss is not finite: {path}")
        expected_name = f"wm_step={int(match.group('step')):08d}-loss={loss:.6f}.ckpt"
        if path.name != expected_name:
            raise ValueError(f"ranked WM filename does not match its rounded loss metadata: {path}")
        ranked.append((loss, path.resolve()))
    if ranked:
        return min(ranked, key=lambda item: (item[0], str(item[1])))[1]

    final_path = root / "ckpt" / "wm_warmup.ckpt"
    if final_path.is_file():
        _validate_available_world_model_checkpoint(final_path)
        return final_path.resolve()

    progress: list[tuple[int, Path]] = []
    progress_dir = root / "ckpt" / "warmup_progress"
    for path in sorted(progress_dir.glob("wm_step_*.ckpt")):
        match = _WM_PROGRESS_RE.match(path.name)
        if match is not None:
            progress.append((int(match.group("step")), path.resolve()))
    if progress:
        step, selected = max(progress, key=lambda item: (item[0], str(item[1])))
        metadata = _validate_available_world_model_checkpoint(selected)
        recorded_step = int(metadata.get("warmup_step", 0) or 0)
        if recorded_step != step:
            raise ValueError(
                f"WM progress filename does not match warmup_step metadata: {selected}"
            )
        return selected

    raise FileNotFoundError(
        "no world-model checkpoint found under run directory; expected an existing "
        f"warmup_topk, wm_warmup.ckpt, or warmup_progress checkpoint under {root}"
    )


def select_classifier_checkpoint(classifier_root: str | Path) -> Path:
    """Resolve and validate the held-out-window-F1 classifier checkpoint."""

    root = Path(classifier_root).expanduser().resolve()
    summary_path = root / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"classifier summary does not exist: {summary_path}") from exc
    if not isinstance(summary, dict) or not summary.get("best_window_ckpt_path"):
        raise ValueError(f"classifier summary has no best_window_ckpt_path: {summary_path}")
    selected = Path(str(summary["best_window_ckpt_path"])).expanduser()
    if not selected.is_absolute():
        selected = root / selected
    selected = selected.resolve()
    checkpoint_root = (root / "checkpoints").resolve()
    try:
        selected.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError(f"selected checkpoint is outside classifier stage: {selected}") from exc
    loaded = load_frozen_component(selected, "classifier")
    required_metadata = {"threshold", "f1", "step", "config", "extra"}
    missing = sorted(required_metadata.difference(loaded.metadata))
    if missing:
        raise ValueError(f"classifier checkpoint lacks required metadata {missing}: {selected}")
    threshold = float(loaded.metadata["threshold"])
    f1 = float(loaded.metadata["f1"])
    step = int(loaded.metadata["step"])
    config = loaded.metadata["config"]
    extra = loaded.metadata["extra"]
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"classifier threshold is invalid: {threshold}")
    if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
        raise ValueError(f"classifier F1 is invalid: {f1}")
    if (
        step <= 0
        or not isinstance(config, Mapping)
        or not isinstance(config.get("classifier"), Mapping)
    ):
        raise ValueError("classifier checkpoint step/config metadata is invalid")
    val_window = extra.get("val_window") if isinstance(extra, Mapping) else None
    if not isinstance(val_window, Mapping):
        raise ValueError("classifier checkpoint has no held-out val_window metadata")
    if not math.isclose(float(val_window.get("best_f1", math.nan)), f1) or not math.isclose(
        float(val_window.get("best_thresh", math.nan)), threshold
    ):
        raise ValueError("classifier checkpoint validation metrics are inconsistent")
    summary_f1 = float(summary.get("best_window_f1", math.nan))
    total_steps = int(summary.get("total_steps", 0) or 0)
    if not math.isclose(summary_f1, f1) or total_steps < step:
        raise ValueError("classifier summary does not bind the selected checkpoint")
    return selected


def _available_classifier_window_checkpoints(
    checkpoint_dir: Path,
) -> list[tuple[float, int, float, Path]]:
    ranked: list[tuple[float, int, float, Path]] = []
    for path in sorted(checkpoint_dir.glob("best_window_f1*_th*.ckpt")):
        match = _CLASSIFIER_WINDOW_RE.match(path.name)
        if match is None:
            continue
        loaded = load_frozen_component(path, "classifier")
        metadata = loaded.metadata
        config = metadata.get("config")
        if not isinstance(config, Mapping) or not isinstance(
            config.get("classifier"), Mapping
        ):
            raise ValueError(
                f"classifier checkpoint has no config.classifier metadata: {path}"
            )
        f1 = float(metadata.get("f1", math.nan))
        threshold = float(metadata.get("threshold", math.nan))
        step = int(metadata.get("step", 0) or 0)
        if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
            raise ValueError(f"classifier checkpoint F1 is invalid: {path}")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"classifier checkpoint threshold is invalid: {path}")
        if step <= 0:
            raise ValueError(f"classifier checkpoint step is invalid: {path}")
        expected_name = f"best_window_f1{f1:.4f}_th{threshold:.2f}.ckpt"
        if path.name != expected_name:
            raise ValueError(
                "classifier checkpoint filename does not match its validation "
                f"metadata: {path}"
            )
        ranked.append((f1, step, threshold, path.resolve()))
    return ranked


def _validate_available_classifier_checkpoint(path: Path) -> Mapping[str, Any]:
    loaded = load_frozen_component(path, "classifier")
    config = loaded.metadata.get("config")
    if not isinstance(config, Mapping) or not isinstance(
        config.get("classifier"), Mapping
    ):
        raise ValueError(
            f"classifier checkpoint has no config.classifier metadata: {path}"
        )
    return loaded.metadata


def select_available_classifier_checkpoint(classifier_root: str | Path) -> Path:
    """Select the best available compatible classifier without a summary gate."""

    root = Path(classifier_root).expanduser().resolve()
    checkpoint_dir = root if root.name == "checkpoints" else root / "checkpoints"
    ranked = _available_classifier_window_checkpoints(checkpoint_dir)
    if ranked:
        return max(ranked, key=lambda item: (item[0], item[1], str(item[3])))[3]

    for name in ("final.ckpt", "latest.ckpt"):
        path = checkpoint_dir / name
        if path.is_file():
            _validate_available_classifier_checkpoint(path)
            return path.resolve()

    raise FileNotFoundError(
        "no classifier checkpoint found under run directory; expected an existing "
        f"best_window, final.ckpt, or latest.ckpt under {checkpoint_dir}"
    )


def resolve_available_classifier_threshold(
    checkpoint: str | Path,
    *,
    default: float | None = None,
) -> float:
    """Resolve calibration for a manual classifier checkpoint handoff."""

    path = Path(checkpoint).expanduser().resolve()
    metadata = _validate_available_classifier_checkpoint(path)
    checkpoint_value = metadata.get("threshold")
    if checkpoint_value is not None:
        threshold = float(checkpoint_value)
    else:
        ranked = _available_classifier_window_checkpoints(path.parent)
        if ranked:
            threshold = max(
                ranked,
                key=lambda item: (item[0], item[1], str(item[3])),
            )[2]
        elif default is not None:
            threshold = float(default)
        else:
            raise ValueError(
                "classifier checkpoint has no validation threshold and no "
                f"best_window companion exists: {path}"
            )
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"classifier threshold must be finite and within [0,1], got {threshold}"
        )
    return threshold


def _materialize_selection(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source.resolve())
    os.replace(temporary, destination)


def refresh_upstream_selections(plan: FrozenPreMainlinePlan) -> None:
    """Revalidate stage artifacts and atomically replace both selection links."""

    _materialize_selection(
        select_world_model_checkpoint(plan.run_root / "wm"),
        plan.selected_wm_ckpt,
    )
    _materialize_selection(
        select_classifier_checkpoint(plan.run_root / "classifier"),
        plan.selected_classifier_ckpt,
    )


def _require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _environment(cfg: Mapping[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["DVLA_ROOT"] = str(PROJECT_ROOT)
    env["DVLA_DATA_ROOT"] = str(cfg["data_root"])
    env.setdefault("NCCL_NVLS_ENABLE", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    pythonpath = env.get("PYTHONPATH", "")
    if str(PROJECT_ROOT) not in pythonpath.split(":"):
        env["PYTHONPATH"] = f"{PROJECT_ROOT}{':' + pythonpath if pythonpath else ''}"
    if cfg.get("gpus") not in (None, ""):
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"])
    return env


def _run(command: Sequence[str], *, env: Mapping[str, str]) -> None:
    print(f"[frozen-pre-mainline] run: {shlex.join(command)}", flush=True)
    subprocess.run(list(command), cwd=PROJECT_ROOT, env=dict(env), check=True)


def main(argv: Sequence[str] | None = None) -> int:
    cfg = script_config(
        "frozen_model_pre_mainline",
        sys.argv[1:] if argv is None else argv,
    )
    plan = build_pipeline_plan(cfg)
    print(
        f"[frozen-pre-mainline] stage={plan.stage} root={plan.run_root} "
        f"task={cfg.get('task')} dry_run={bool(cfg.get('dry_run', False))}"
    )
    selected_commands: list[list[str]] = []
    if plan.stage in {"all", "wm"}:
        selected_commands.append(plan.wm_cmd)
    if plan.stage in {"all", "classifier"}:
        selected_commands.append(plan.classifier_cmd)
    if plan.stage in {"all", "rl"}:
        selected_commands.append(plan.rl_cmd)
    if plan.stage in {"all", "eval"}:
        selected_commands.extend([plan.eval_baseline_cmd, plan.eval_rl_cmd, plan.compare_cmd])
    if bool(cfg.get("dry_run", False)):
        for command in selected_commands:
            print(f"[frozen-pre-mainline] dry-run: {shlex.join(command)}")
        return 0

    plan.run_root.mkdir(parents=True, exist_ok=True)
    env = _environment(cfg)
    if plan.stage in {"all", "wm"}:
        _run(plan.wm_cmd, env=env)
        _materialize_selection(
            select_world_model_checkpoint(plan.run_root / "wm"),
            plan.selected_wm_ckpt,
        )
    if plan.stage in {"all", "classifier"}:
        _run(plan.classifier_cmd, env=env)
        _materialize_selection(
            select_classifier_checkpoint(plan.run_root / "classifier"),
            plan.selected_classifier_ckpt,
        )
    if plan.stage in {"all", "rl"}:
        refresh_upstream_selections(plan)
        _run(plan.rl_cmd, env=env)
        _require_file(plan.rl_final_ckpt, label="final frozen-RL checkpoint")
        _require_file(plan.frozen_summary, label="frozen-RL summary")
    if plan.stage in {"all", "eval"}:
        _require_file(plan.rl_final_ckpt, label="final frozen-RL checkpoint")
        _require_file(plan.frozen_summary, label="frozen-RL summary")
        _run(plan.eval_baseline_cmd, env=env)
        _run(plan.eval_rl_cmd, env=env)
        _run(plan.compare_cmd, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
