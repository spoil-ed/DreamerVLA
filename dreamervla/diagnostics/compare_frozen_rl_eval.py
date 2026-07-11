"""Fail-closed feasibility verdict for frozen-model policy RL."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dreamervla.utils.frozen_components import state_dict_sha256
from dreamervla.utils.hf_checkpoint import load_runner_payload

_PROTOCOL_FIELDS = (
    "task_suite",
    "num_episodes_per_task",
    "num_envs",
    "seed",
    "num_steps_wait",
    "action_steps",
    "task_ids",
    "task_start",
    "max_tasks",
    "max_steps",
    "enumerate_all_init_states",
    "scheme",
    "reconfigure_per_episode",
    "history_length",
    "action_postprocess",
    "render_backend",
    "eval_tasks",
    "eval_total_episodes",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FULL_SUITE_PROTOCOL = {
    "task_suite": "libero_goal",
    "num_episodes_per_task": 10,
    "num_envs": 64,
    "seed": 7,
    "num_steps_wait": 10,
    "action_steps": 8,
    "task_ids": list(range(10)),
    "task_start": 0,
    "max_tasks": 10,
    "max_steps": 300,
    "enumerate_all_init_states": False,
    "scheme": "rlinf_chunk",
    "reconfigure_per_episode": True,
    "history_length": 1,
    "action_postprocess": "none",
    "render_backend": "osmesa",
    "eval_tasks": 10,
    "eval_total_episodes": 100,
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} JSON does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON is invalid: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} JSON must contain an object: {source}")
    return payload


def _resolved_path(value: Any) -> str | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    return str(Path(value).expanduser().resolve())


def _checkpoint_state_hashes(path: Any) -> tuple[dict[str, str], str | None]:
    resolved = _resolved_path(path)
    if resolved is None:
        return {}, "checkpoint path is missing"
    try:
        payload = load_runner_payload(resolved)
        state_dicts = payload.get("state_dicts", {})
        required = ("world_model", "classifier", "policy")
        hashes = {
            name: state_dict_sha256(state_dicts[name])
            for name in required
        }
        return hashes, None
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _protocol_mismatches(
    baseline: dict[str, Any],
    rl: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    mismatches: dict[str, dict[str, Any]] = {}
    missing = object()
    for field in _PROTOCOL_FIELDS:
        base_value = baseline.get(field, missing)
        rl_value = rl.get(field, missing)
        if base_value is missing or rl_value is missing or base_value != rl_value:
            mismatches[field] = {
                "baseline": None if base_value is missing else base_value,
                "rl": None if rl_value is missing else rl_value,
            }

    base_task_keys = sorted(
        key for key in baseline if key.startswith("eval_task_") and key.endswith("_success_rate")
    )
    rl_task_keys = sorted(
        key for key in rl if key.startswith("eval_task_") and key.endswith("_success_rate")
    )
    if base_task_keys != rl_task_keys:
        mismatches["evaluated_task_keys"] = {
            "baseline": base_task_keys,
            "rl": rl_task_keys,
        }
    return mismatches


def compare_feasibility(
    baseline_metrics_path: str | Path,
    rl_metrics_path: str | Path,
    frozen_summary_path: str | Path,
) -> dict[str, Any]:
    """Return the strict real-evaluation and immutability verdict."""

    baseline = _read_json(baseline_metrics_path, label="baseline metrics")
    rl = _read_json(rl_metrics_path, label="RL metrics")
    frozen = _read_json(frozen_summary_path, label="frozen RL summary")

    if "eval_success_rate" not in baseline or "eval_success_rate" not in rl:
        raise ValueError("both evaluations must contain eval_success_rate")
    baseline_rate = float(baseline["eval_success_rate"])
    rl_rate = float(rl["eval_success_rate"])
    delta = round(rl_rate - baseline_rate, 12)
    success_rates_valid = (
        math.isfinite(baseline_rate)
        and math.isfinite(rl_rate)
        and 0.0 <= baseline_rate <= 1.0
        and 0.0 <= rl_rate <= 1.0
    )

    mismatches = _protocol_mismatches(baseline, rl)
    full_suite_proof_contract = all(
        baseline.get(field) == expected and rl.get(field) == expected
        for field, expected in _FULL_SUITE_PROTOCOL.items()
    )
    expected_task_metric_keys = {
        f"eval_task_{task_id}_success_rate" for task_id in range(10)
    }
    full_suite_proof_contract = full_suite_proof_contract and all(
        expected_task_metric_keys.issubset(metrics)
        for metrics in (baseline, rl)
    )
    frozen_before = frozen.get("frozen_hashes_before")
    frozen_after = frozen.get("frozen_hashes_after")
    required_frozen_keys = {"world_model", "classifier"}
    frozen_unchanged = (
        isinstance(frozen_before, dict)
        and isinstance(frozen_after, dict)
        and set(frozen_before) == required_frozen_keys
        and set(frozen_after) == required_frozen_keys
        and frozen_before == frozen_after
        and all(_is_sha256(value) for value in frozen_before.values())
        and all(_is_sha256(value) for value in frozen_after.values())
    )
    policy_before = frozen.get("policy_hash_before")
    policy_after = frozen.get("policy_hash_after")
    policy_changed = (
        _is_sha256(policy_before)
        and _is_sha256(policy_after)
        and policy_before != policy_after
        and frozen.get("policy_changed") is True
    )
    applied_steps = int(frozen.get("applied_policy_steps", 0) or 0)
    expected_state_hashes = (
        {
            "world_model": frozen_after.get("world_model"),
            "classifier": frozen_after.get("classifier"),
            "policy": policy_after,
        }
        if isinstance(frozen_after, dict)
        else {}
    )
    evaluated_state_hashes = rl.get("checkpoint_state_hashes")
    recomputed_state_hashes, checkpoint_hash_error = _checkpoint_state_hashes(
        rl.get("ckpt_path")
    )
    state_hashes_match = (
        set(expected_state_hashes) == {"world_model", "classifier", "policy"}
        and all(_is_sha256(value) for value in expected_state_hashes.values())
        and evaluated_state_hashes == expected_state_hashes
        and recomputed_state_hashes == expected_state_hashes
        and checkpoint_hash_error is None
    )
    checkpoint_kinds_match = (
        baseline.get("ckpt_kind") == "vla" and rl.get("ckpt_kind") == "dreamer"
    )
    rl_checkpoint_matches_summary = (
        _resolved_path(rl.get("ckpt_path")) is not None
        and _resolved_path(rl.get("ckpt_path"))
        == _resolved_path(frozen.get("final_checkpoint"))
    )

    checks = {
        "evaluation_protocol_matches": not mismatches,
        "full_suite_proof_contract": full_suite_proof_contract,
        "success_rates_valid": success_rates_valid,
        "strict_success_improvement": success_rates_valid and rl_rate > baseline_rate,
        "frozen_models_unchanged": frozen_unchanged,
        "policy_changed": policy_changed,
        "policy_optimizer_step_applied": applied_steps > 0,
        "checkpoint_kinds_match": checkpoint_kinds_match,
        "rl_checkpoint_matches_summary": rl_checkpoint_matches_summary,
        "evaluated_states_match_summary": state_hashes_match,
    }
    return {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_success_rate": baseline_rate,
        "rl_success_rate": rl_rate,
        "success_rate_delta": delta,
        "protocol_mismatches": mismatches,
        "applied_policy_steps": applied_steps,
        "frozen_hashes_before": frozen_before,
        "frozen_hashes_after": frozen_after,
        "policy_hash_before": policy_before,
        "policy_hash_after": policy_after,
        "evaluated_state_hashes": evaluated_state_hashes,
        "recomputed_state_hashes": recomputed_state_hashes,
        "checkpoint_hash_error": checkpoint_hash_error,
        "inputs": {
            "baseline_metrics": str(Path(baseline_metrics_path).expanduser().resolve()),
            "rl_metrics": str(Path(rl_metrics_path).expanduser().resolve()),
            "frozen_summary": str(Path(frozen_summary_path).expanduser().resolve()),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--rl", required=True, type=Path)
    parser.add_argument("--frozen-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    verdict = compare_feasibility(args.baseline, args.rl, args.frozen_summary)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[frozen-pre-mainline] "
        f"passed={verdict['passed']} "
        f"baseline={verdict['baseline_success_rate']:.4f} "
        f"rl={verdict['rl_success_rate']:.4f} "
        f"delta={verdict['success_rate_delta']:+.4f} "
        f"output={output}"
    )
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
