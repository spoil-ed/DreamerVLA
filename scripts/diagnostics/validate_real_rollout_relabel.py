#!/usr/bin/env python3
# ruff: noqa: E402
"""Closed-loop real-rollout relabel diagnostic for DreamerVLA.

This script is deliberately offline/diagnostic. It does not modify datasets,
checkpoints, or the active training loop. It runs real LIBERO closed-loop
rollouts, records WMPO-style outcome fields (`complete`, `finish_step`, `acc`),
and exports sparse real-outcome labels that can later be used as hard positives
or hard negatives for reward correction.
"""

from __future__ import annotations

import argparse
import os
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero.libero import benchmark as libero_benchmark

from scripts.diagnostics.diagnose_ppo_imagine_vs_real import (
    _array_stats,
    _build_workspace,
    _dreamer_action_from_latent,
    _observe,
    _sft_action,
)
from src.algorithms.dreamer_vla import _world_model_state_reward
from src.env import TASK_MAX_STEPS, get_libero_dummy_action, get_libero_env

try:
    import faulthandler

    faulthandler.enable()
except Exception:
    pass


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _safe_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else math.nan


def _finish_sparse_rewards(
    success: bool, finish_step: int, max_steps: int
) -> list[float]:
    length = max(1, min(int(finish_step), int(max_steps)))
    rewards = [0.0] * length
    if success:
        rewards[length - 1] = 1.0
    return rewards


def _wmpo_group_filter(
    records: list[dict[str, Any]], lower: float, upper: float
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["prompt_key"])].append(record)

    kept_prompt_keys: set[str] = set()
    group_rows = []
    for prompt_key, rows in sorted(groups.items()):
        acc = float(np.mean([float(row["acc"]) for row in rows])) if rows else 0.0
        keep = bool(float(lower) <= acc <= float(upper))
        if keep:
            kept_prompt_keys.add(prompt_key)
        group_rows.append(
            {
                "prompt_key": prompt_key,
                "num_samples": len(rows),
                "successes": int(sum(int(row["complete"]) for row in rows)),
                "acc_mean": acc,
                "keep_by_accuracy_band": keep,
            }
        )

    kept_records = [
        row for row in records if str(row["prompt_key"]) in kept_prompt_keys
    ]
    return {
        "accuracy_lower_bound": float(lower),
        "accuracy_upper_bound": float(upper),
        "num_prompt_groups": len(group_rows),
        "num_kept_prompt_groups": int(
            sum(int(row["keep_by_accuracy_band"]) for row in group_rows)
        ),
        "num_records": len(records),
        "num_kept_records": len(kept_records),
        "groups": group_rows,
        "kept_record_ids": [row["trajectory_id"] for row in kept_records],
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _aggregate_records(
    args: argparse.Namespace, records: list[dict[str, Any]], elapsed_sec: float
) -> dict[str, Any]:
    policy_modes = sorted({str(row["policy_mode"]) for row in records})
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in policy_modes:
        rows = [row for row in records if row["policy_mode"] == mode]
        by_mode[mode] = {
            "num_records": len(rows),
            "successes": int(sum(int(row["complete"]) for row in rows)),
            "success_rate": float(np.mean([row["acc"] for row in rows]))
            if rows
            else 0.0,
            "finish_step_mean": _safe_mean([float(row["finish_step"]) for row in rows]),
            "wm_reward_mean": _safe_mean(
                [float(row["wm_reward_pred"]["mean"]) for row in rows]
            ),
            "action_mse_to_sft_mean": _safe_mean(
                [
                    float(
                        row["action_compare"].get(
                            "mean_dreamer_vs_sft_env_action_mse", math.nan
                        )
                    )
                    for row in rows
                ]
            ),
        }

    return {
        "ckpt": str(args.ckpt),
        "task_suite": args.task_suite,
        "task_ids": _parse_ints(args.task_ids),
        "episode_indices": _parse_ints(args.episode_indices),
        "samples_per_init": int(args.samples_per_init),
        "policy_modes": policy_modes,
        "rollout_mode": args.rollout_mode,
        "num_records": len(records),
        "successes": int(sum(int(row["complete"]) for row in records)),
        "success_rate": float(np.mean([row["acc"] for row in records]))
        if records
        else 0.0,
        "by_policy_mode": by_mode,
        "wmpo_style_filter": _wmpo_group_filter(
            records,
            lower=float(args.accuracy_lower_bound),
            upper=float(args.accuracy_upper_bound),
        ),
        "elapsed_sec": float(elapsed_sec),
        "mismatch_high_reward_failures": [
            {
                "trajectory_id": row["trajectory_id"],
                "policy_mode": row["policy_mode"],
                "task_id": row["task_id"],
                "episode_idx": row["episode_idx"],
                "complete": row["complete"],
                "finish_step": row["finish_step"],
                "wm_reward_mean": row["wm_reward_pred"]["mean"],
                "wm_reward_max": row["wm_reward_pred"]["max"],
                "first_ge_0p8_step": row["wm_reward_pred"]["first_ge_0p8_step"],
                "action_mse_to_sft": row["action_compare"].get(
                    "mean_dreamer_vs_sft_env_action_mse", math.nan
                ),
            }
            for row in records
            if (not row["complete"])
            and float(row["wm_reward_pred"]["mean"])
            >= float(args.high_reward_fail_threshold)
        ],
    }


def _run_isolated(args: argparse.Namespace) -> int:
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    worker_root = args.out_dir / "isolated_workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "real_rollout_relabel_records.jsonl"
    summary_path = args.out_dir / "real_rollout_relabel_summary.json"

    task_ids = _parse_ints(args.task_ids)
    episode_indices = _parse_ints(args.episode_indices)
    policy_modes = [
        item.strip() for item in str(args.policy_modes).split(",") if item.strip()
    ]
    all_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for task_id in task_ids:
        for episode_idx in episode_indices:
            for policy_mode in policy_modes:
                n_samples = (
                    1 if policy_mode == "deterministic" else int(args.samples_per_init)
                )
                for sample_idx in range(n_samples):
                    sample_seed = (
                        int(args.seed)
                        + int(task_id) * 10000
                        + int(episode_idx) * 100
                        + sample_idx
                    )
                    worker_dir = (
                        worker_root
                        / f"task{task_id:02d}_ep{episode_idx:03d}_{policy_mode}_sample{sample_idx:03d}"
                    )
                    if worker_dir.exists():
                        shutil.rmtree(worker_dir)
                    worker_dir.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--ckpt",
                        str(args.ckpt),
                        "--out-dir",
                        str(worker_dir),
                        "--task-suite",
                        str(args.task_suite),
                        "--task-ids",
                        str(task_id),
                        "--episode-indices",
                        str(episode_idx),
                        "--samples-per-init",
                        "1",
                        "--policy-modes",
                        str(policy_mode),
                        "--max-steps",
                        str(args.max_steps),
                        "--action-steps",
                        str(args.action_steps),
                        "--history-length",
                        str(args.history_length),
                        "--rollout-mode",
                        str(args.rollout_mode),
                        "--trace-steps",
                        str(args.trace_steps),
                        "--sft-compare-steps",
                        str(args.sft_compare_steps),
                        "--accuracy-lower-bound",
                        str(args.accuracy_lower_bound),
                        "--accuracy-upper-bound",
                        str(args.accuracy_upper_bound),
                        "--high-reward-fail-threshold",
                        str(args.high_reward_fail_threshold),
                        "--progress-every",
                        str(args.progress_every),
                        "--seed",
                        str(sample_seed),
                        "--single-sample-idx",
                        str(sample_idx),
                    ]
                    log_path = worker_dir / "worker.log"
                    print(
                        f"[isolate] launch task={task_id} ep={episode_idx} mode={policy_mode} "
                        f"sample={sample_idx} seed={sample_seed}",
                        flush=True,
                    )
                    env = os.environ.copy()
                    with log_path.open("w") as log_file:
                        proc = subprocess.run(
                            cmd,
                            cwd=str(PROJECT_ROOT),
                            env=env,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                    worker_records = _load_jsonl(
                        worker_dir / "real_rollout_relabel_records.jsonl"
                    )
                    if proc.returncode != 0 or not worker_records:
                        failures.append(
                            {
                                "task_id": task_id,
                                "episode_idx": episode_idx,
                                "policy_mode": policy_mode,
                                "sample_idx": sample_idx,
                                "returncode": proc.returncode,
                                "log_path": str(log_path),
                            }
                        )
                        print(
                            f"[isolate] failed returncode={proc.returncode} log={log_path}",
                            flush=True,
                        )
                    else:
                        all_records.extend(worker_records)
                        with records_path.open("a") as out:
                            for row in worker_records:
                                out.write(json.dumps(row, default=_json_default) + "\n")
                        print(
                            f"[isolate] done id={worker_records[0]['trajectory_id']} "
                            f"success={worker_records[0]['complete']} finish={worker_records[0]['finish_step']}",
                            flush=True,
                        )
                    summary = _aggregate_records(args, all_records, time.time() - t0)
                    summary["records_jsonl"] = str(records_path)
                    summary["worker_failures"] = failures
                    summary_path.write_text(
                        json.dumps(summary, indent=2, default=_json_default)
                    )

    summary = _aggregate_records(args, all_records, time.time() - t0)
    summary["records_jsonl"] = str(records_path)
    summary["worker_failures"] = failures
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"[save] records={records_path}")
    print(f"[save] summary={summary_path}")
    return 1 if failures and not all_records else 0


def _rollout_one(
    *,
    ws: Any,
    env: Any,
    initial_state: Any,
    task_description: str,
    item_processor: Any,
    task_id: int,
    episode_idx: int,
    sample_idx: int,
    policy_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    resolution = int(OmegaConf.select(ws.cfg, "encoder.resolution", default=256))
    history_length = int(args.history_length)
    max_steps = int(args.max_steps)
    compare_steps = int(args.sft_compare_steps)
    trace_steps = int(args.trace_steps)

    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    for _ in range(10):
        obs, _, done, _ = env.step(get_libero_dummy_action())
        if done:
            break

    ws._dreamer_online_reset()
    frame_history: list[tuple[Image.Image, Image.Image]] = []
    action_compare_rows: list[dict[str, Any]] = []
    reward_preds: list[float] = []
    action_norms: list[float] = []
    steps = 0
    success = False

    deterministic = policy_mode == "deterministic"
    for step_idx in range(max_steps):
        obs_embedding, input_ids, _state, frame_history = _observe(
            ws,
            item_processor,
            obs,
            frame_history,
            task_description,
            resolution,
            history_length,
        )
        with torch.no_grad():
            if args.rollout_mode == "online_rssm":
                latent = ws._dreamer_online_update_latent(obs_embedding)
            else:
                latent = ws.world_model(
                    {"mode": "encode_latent", "hidden": obs_embedding}
                )
            pred_reward = (
                _world_model_state_reward(ws.world_model, latent)
                .detach()
                .float()
                .reshape(-1)[0]
            )

        action_tensor, raw_action, env_action, wm_action = _dreamer_action_from_latent(
            ws,
            latent,
            deterministic=deterministic,
        )
        reward_preds.append(float(pred_reward.cpu()))
        action_norms.append(float(np.linalg.norm(env_action)))

        if compare_steps < 0 or step_idx < compare_steps:
            _sft_raw, sft_env_action = _sft_action(ws, input_ids)
            compare_row = {
                "step": int(step_idx),
                "dreamer_env_action": env_action.tolist()
                if step_idx < trace_steps
                else None,
                "sft_env_action": sft_env_action.tolist()
                if step_idx < trace_steps
                else None,
            }
            compare_row.update(
                _array_stats("dreamer_vs_sft_env_action", env_action, sft_env_action)
            )
            action_compare_rows.append(compare_row)

        obs, reward, done, _info = env.step(env_action.tolist())
        if args.rollout_mode == "online_rssm":
            ws._dreamer_online_prev_action = wm_action.reshape(1, -1).to(ws.device)
        steps = step_idx + 1
        progress_every = int(getattr(args, "progress_every", 0))
        if progress_every > 0 and steps % progress_every == 0:
            print(
                f"[rollout-progress] task={task_id} ep={episode_idx} mode={policy_mode} "
                f"sample={sample_idx} step={steps}/{max_steps} reward={float(reward):.3f}",
                flush=True,
            )
        if reward > 0.0 or done:
            success = bool(reward > 0.0 or done)
            break

    compare_summary: dict[str, float] = {}
    for key in (
        "dreamer_vs_sft_env_action_mse",
        "dreamer_vs_sft_env_action_mae",
        "dreamer_vs_sft_env_action_max_abs",
        "dreamer_vs_sft_env_action_cos",
    ):
        compare_summary[f"mean_{key}"] = _safe_mean(
            [float(row[key]) for row in action_compare_rows]
        )

    finish_step = int(steps)
    sparse_rewards = _finish_sparse_rewards(success, finish_step, max_steps)
    trajectory_id = (
        f"task{task_id:02d}_ep{episode_idx:03d}_{policy_mode}_sample{sample_idx:03d}"
    )
    prompt_key = f"task{task_id:02d}_ep{episode_idx:03d}_{policy_mode}"
    return {
        "trajectory_id": trajectory_id,
        "prompt_key": prompt_key,
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "sample_idx": int(sample_idx),
        "policy_mode": policy_mode,
        "complete": bool(success),
        "acc": float(success),
        "finish_step": finish_step,
        "max_steps": int(max_steps),
        "valid_action_tokens": int(finish_step * 7),
        "real_sparse_rewards": sparse_rewards,
        "reward_relabel": {
            "type": "terminal_outcome",
            "positive_step": finish_step - 1 if success else -1,
            "target_return": float(success),
        },
        "wm_reward_pred": {
            "mean": _safe_mean(reward_preds),
            "max": float(np.max(reward_preds)) if reward_preds else math.nan,
            "last": float(reward_preds[-1]) if reward_preds else math.nan,
            "first_ge_0p8_step": next(
                (idx for idx, val in enumerate(reward_preds) if val >= 0.8), -1
            ),
            "trace": reward_preds[:trace_steps],
        },
        "action_norm_mean": _safe_mean(action_norms),
        "action_compare": compare_summary,
        "action_compare_rows": action_compare_rows[:trace_steps],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-ids", default="7")
    parser.add_argument("--episode-indices", default="0")
    parser.add_argument("--samples-per-init", type=int, default=4)
    parser.add_argument("--policy-modes", default="deterministic,sample")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--action-steps", type=int, default=5)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument(
        "--rollout-mode", choices=["stateless", "online_rssm"], default="stateless"
    )
    parser.add_argument("--trace-steps", type=int, default=20)
    parser.add_argument("--sft-compare-steps", type=int, default=80)
    parser.add_argument("--accuracy-lower-bound", type=float, default=0.01)
    parser.add_argument("--accuracy-upper-bound", type=float, default=0.99)
    parser.add_argument("--high-reward-fail-threshold", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--isolate-rollouts", action="store_true")
    parser.add_argument("--single-sample-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if bool(args.isolate_rollouts):
        raise SystemExit(_run_isolated(args))

    task_ids = _parse_ints(args.task_ids)
    episode_indices = _parse_ints(args.episode_indices)
    policy_modes = [
        item.strip() for item in str(args.policy_modes).split(",") if item.strip()
    ]
    invalid_modes = sorted(set(policy_modes) - {"deterministic", "sample"})
    if invalid_modes:
        raise ValueError(f"--policy-modes contains invalid values: {invalid_modes}")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ws = _build_workspace(args)
    item_processor = ws.encoder._build_processor(ws.device)
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    records: list[dict[str, Any]] = []
    records_path = args.out_dir / "real_rollout_relabel_records.jsonl"
    partial_summary_path = args.out_dir / "real_rollout_relabel_summary.partial.json"
    t0 = time.time()

    with records_path.open("w") as records_file:
        for task_id in task_ids:
            task = task_suite.get_task(int(task_id))
            initial_states = task_suite.get_task_init_states(int(task_id))
            max_steps = int(args.max_steps)
            if max_steps <= 0:
                max_steps = int(TASK_MAX_STEPS[args.task_suite])
                args.max_steps = max_steps
            for episode_idx in episode_indices:
                initial_state = initial_states[int(episode_idx)]
                for policy_mode in policy_modes:
                    n_samples = (
                        1
                        if policy_mode == "deterministic"
                        else int(args.samples_per_init)
                    )
                    for sample_idx in range(n_samples):
                        record_sample_idx = (
                            int(args.single_sample_idx)
                            if args.single_sample_idx is not None
                            else sample_idx
                        )
                        if args.single_sample_idx is not None:
                            sample_seed = int(args.seed)
                        else:
                            sample_seed = (
                                int(args.seed)
                                + int(task_id) * 10000
                                + int(episode_idx) * 100
                                + record_sample_idx
                            )
                        torch.manual_seed(sample_seed)
                        np.random.seed(sample_seed)
                        print(
                            f"[rollout] task={task_id} ep={episode_idx} mode={policy_mode} "
                            f"sample={sample_idx}/{n_samples} seed={sample_seed}",
                            flush=True,
                        )
                        env, task_description = get_libero_env(
                            task,
                            resolution=int(
                                OmegaConf.select(
                                    ws.cfg, "encoder.resolution", default=256
                                )
                            ),
                        )
                        try:
                            record = _rollout_one(
                                ws=ws,
                                env=env,
                                initial_state=initial_state,
                                task_description=task_description,
                                item_processor=item_processor,
                                task_id=int(task_id),
                                episode_idx=int(episode_idx),
                                sample_idx=int(record_sample_idx),
                                policy_mode=policy_mode,
                                args=args,
                            )
                        finally:
                            env.env.close()
                        records.append(record)
                        records_file.write(
                            json.dumps(record, default=_json_default) + "\n"
                        )
                        records_file.flush()
                        print(
                            f"[rollout] done id={record['trajectory_id']} success={record['complete']} "
                            f"finish_step={record['finish_step']} wm_reward_mean={record['wm_reward_pred']['mean']:.4f} "
                            f"sft_mse={record['action_compare'].get('mean_dreamer_vs_sft_env_action_mse')}",
                            flush=True,
                        )
                        partial = {
                            "num_records": len(records),
                            "success_rate": float(
                                np.mean([row["acc"] for row in records])
                            )
                            if records
                            else 0.0,
                            "filter": _wmpo_group_filter(
                                records,
                                lower=float(args.accuracy_lower_bound),
                                upper=float(args.accuracy_upper_bound),
                            ),
                        }
                        partial_summary_path.write_text(
                            json.dumps(partial, indent=2, default=_json_default)
                        )

    filter_summary = _wmpo_group_filter(
        records,
        lower=float(args.accuracy_lower_bound),
        upper=float(args.accuracy_upper_bound),
    )
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in policy_modes:
        rows = [row for row in records if row["policy_mode"] == mode]
        by_mode[mode] = {
            "num_records": len(rows),
            "successes": int(sum(int(row["complete"]) for row in rows)),
            "success_rate": float(np.mean([row["acc"] for row in rows]))
            if rows
            else 0.0,
            "finish_step_mean": _safe_mean([float(row["finish_step"]) for row in rows]),
            "wm_reward_mean": _safe_mean(
                [float(row["wm_reward_pred"]["mean"]) for row in rows]
            ),
            "action_mse_to_sft_mean": _safe_mean(
                [
                    float(
                        row["action_compare"].get(
                            "mean_dreamer_vs_sft_env_action_mse", math.nan
                        )
                    )
                    for row in rows
                ]
            ),
        }

    mismatch_rows = [
        {
            "trajectory_id": row["trajectory_id"],
            "policy_mode": row["policy_mode"],
            "task_id": row["task_id"],
            "episode_idx": row["episode_idx"],
            "complete": row["complete"],
            "finish_step": row["finish_step"],
            "wm_reward_mean": row["wm_reward_pred"]["mean"],
            "wm_reward_max": row["wm_reward_pred"]["max"],
            "first_ge_0p8_step": row["wm_reward_pred"]["first_ge_0p8_step"],
            "action_mse_to_sft": row["action_compare"].get(
                "mean_dreamer_vs_sft_env_action_mse", math.nan
            ),
        }
        for row in records
        if (not row["complete"])
        and float(row["wm_reward_pred"]["mean"])
        >= float(args.high_reward_fail_threshold)
    ]

    summary = {
        "ckpt": str(args.ckpt),
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "episode_indices": episode_indices,
        "samples_per_init": int(args.samples_per_init),
        "policy_modes": policy_modes,
        "rollout_mode": args.rollout_mode,
        "num_records": len(records),
        "successes": int(sum(int(row["complete"]) for row in records)),
        "success_rate": float(np.mean([row["acc"] for row in records]))
        if records
        else 0.0,
        "by_policy_mode": by_mode,
        "wmpo_style_filter": filter_summary,
        "records_jsonl": str(records_path),
        "elapsed_sec": float(time.time() - t0),
        "mismatch_high_reward_failures": mismatch_rows,
    }
    summary_path = args.out_dir / "real_rollout_relabel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"[save] records={records_path}")
    print(f"[save] summary={summary_path}")


if __name__ == "__main__":
    main()
