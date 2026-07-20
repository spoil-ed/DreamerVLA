"""Machine-checkable training-signal contracts for diagnostic experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingSignalResult:
    """Pass/fail result plus compact evidence for one signal probe."""

    passed: bool
    failures: tuple[str, ...]
    evidence: dict[str, float | int | str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "passed": bool(self.passed),
            "failures": list(self.failures),
            "evidence": dict(self.evidence),
        }


def evaluate_imagined_success_sft_signal(
    metrics: Mapping[str, Any],
    *,
    training_mode: str,
    policy_initial_hash: str,
    policy_final_hash: str,
    applied_policy_steps: int,
) -> TrainingSignalResult:
    """Decide whether the imagined-success SFT probe produced a real update."""

    successes = _metric(metrics, "actor/success_sft_trajectories")
    valid_samples = _metric(metrics, "actor/success_sft_valid_samples")
    optimizer_steps = _metric(metrics, "actor/success_sft_optimizer_steps")
    grad_norm = _metric(metrics, "actor/success_sft_grad_norm")
    committed = _metric(metrics, "actor/success_sft_update_committed")
    skipped = _metric(metrics, "actor/success_sft_skipped_no_success")
    initial_hash = str(policy_initial_hash or "")
    final_hash = str(policy_final_hash or "")

    failures: list[str] = []
    if str(training_mode) != "imagined_success_sft":
        failures.append("checkpoint is not an imagined_success_sft run")
    if successes < 1.0:
        failures.append("classifier selected no successful imagined trajectory")
    if valid_samples < 1.0:
        failures.append("successful imagined trajectories contained no valid SFT sample")
    if optimizer_steps < 1.0 or int(applied_policy_steps) < 1:
        failures.append("actor optimizer applied no committed step")
    if not math.isfinite(grad_norm) or grad_norm <= 0.0:
        failures.append("actor gradient norm was not finite and positive")
    if committed < 0.5:
        failures.append("policy update was not committed")
    if skipped >= 0.5:
        failures.append("success SFT reported a no-success skip")
    if not initial_hash or not final_hash:
        failures.append("checkpoint omitted the initial or final policy hash")
    elif initial_hash == final_hash:
        failures.append("policy hash did not change")

    return TrainingSignalResult(
        passed=not failures,
        failures=tuple(failures),
        evidence={
            "successful_imagined_trajectories": successes,
            "valid_sft_samples": valid_samples,
            "optimizer_steps": optimizer_steps,
            "applied_policy_steps": int(applied_policy_steps),
            "grad_norm": grad_norm,
            "training_mode": str(training_mode),
            "policy_initial_hash": initial_hash,
            "policy_final_hash": final_hash,
        },
    )


def _metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["TrainingSignalResult", "evaluate_imagined_success_sft_signal"]
