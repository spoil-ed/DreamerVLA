from __future__ import annotations

import json
from pathlib import Path

import torch

from dreamervla.utils.frozen_components import state_dict_sha256


def _metrics(
    success_rate: float,
    *,
    ckpt_kind: str,
    ckpt_path: str,
    checkpoint_state_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "ckpt_kind": ckpt_kind,
        "ckpt_path": ckpt_path,
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
        "eval_success_rate": success_rate,
        **{f"eval_task_{task_id}_success_rate": success_rate for task_id in range(10)},
    }
    if checkpoint_state_hashes is not None:
        payload["checkpoint_state_hashes"] = checkpoint_state_hashes
    return payload


def _frozen_summary(final_checkpoint: str) -> dict[str, object]:
    return {
        "frozen_hashes_before": {
            "world_model": "a" * 64,
            "classifier": "b" * 64,
        },
        "frozen_hashes_after": {
            "world_model": "a" * 64,
            "classifier": "b" * 64,
        },
        "policy_hash_before": "c" * 64,
        "policy_hash_after": "d" * 64,
        "policy_changed": True,
        "applied_policy_steps": 12,
        "final_checkpoint": final_checkpoint,
    }


def _inputs(tmp_path: Path, baseline_rate: float = 0.50, rl_rate: float = 0.61):
    final_path = tmp_path / "rl" / "checkpoints" / "final.ckpt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    states = {
        "world_model": {"weight": torch.tensor([1.0])},
        "classifier": {"weight": torch.tensor([2.0])},
        "policy": {"weight": torch.tensor([3.0])},
    }
    torch.save({"state_dicts": states}, final_path)
    final = str(final_path.resolve())
    frozen = _frozen_summary(final)
    frozen["frozen_hashes_before"] = {
        name: state_dict_sha256(states[name])
        for name in ("world_model", "classifier")
    }
    frozen["frozen_hashes_after"] = dict(frozen["frozen_hashes_before"])
    frozen["policy_hash_after"] = state_dict_sha256(states["policy"])
    hashes = {
        **frozen["frozen_hashes_after"],
        "policy": frozen["policy_hash_after"],
    }
    baseline = _metrics(
        baseline_rate,
        ckpt_kind="vla",
        ckpt_path=str((tmp_path / "baseline").resolve()),
    )
    rl = _metrics(
        rl_rate,
        ckpt_kind="dreamer",
        ckpt_path=final,
        checkpoint_state_hashes=hashes,
    )
    return baseline, rl, frozen


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_comparator_passes_only_strict_real_success_improvement(tmp_path: Path) -> None:
    from dreamervla.diagnostics.compare_frozen_rl_eval import compare_feasibility

    baseline, rl, frozen = _inputs(tmp_path)
    verdict = compare_feasibility(
        _write(tmp_path / "baseline.json", baseline),
        _write(tmp_path / "rl.json", rl),
        _write(tmp_path / "frozen.json", frozen),
    )

    assert verdict["passed"] is True
    assert verdict["success_rate_delta"] == 0.11
    assert verdict["checks"]["evaluation_protocol_matches"] is True
    assert verdict["checks"]["frozen_models_unchanged"] is True


def test_comparator_rejects_protocol_mismatch(tmp_path: Path) -> None:
    from dreamervla.diagnostics.compare_frozen_rl_eval import compare_feasibility

    baseline, rl, frozen = _inputs(tmp_path)
    rl["seed"] = 8
    verdict = compare_feasibility(
        _write(tmp_path / "baseline.json", baseline),
        _write(tmp_path / "rl.json", rl),
        _write(tmp_path / "frozen.json", frozen),
    )

    assert verdict["passed"] is False
    assert verdict["checks"]["evaluation_protocol_matches"] is False
    assert verdict["protocol_mismatches"]["seed"] == {"baseline": 7, "rl": 8}


def test_comparator_rejects_matching_but_incomplete_suite(tmp_path: Path) -> None:
    from dreamervla.diagnostics.compare_frozen_rl_eval import compare_feasibility

    baseline, rl, frozen = _inputs(tmp_path)
    for metrics in (baseline, rl):
        metrics["task_ids"] = [0]
        metrics["max_tasks"] = 1
        metrics["eval_tasks"] = 1
        metrics["eval_total_episodes"] = 10
        for task_id in range(1, 10):
            metrics.pop(f"eval_task_{task_id}_success_rate")
    verdict = compare_feasibility(
        _write(tmp_path / "baseline.json", baseline),
        _write(tmp_path / "rl.json", rl),
        _write(tmp_path / "frozen.json", frozen),
    )

    assert verdict["passed"] is False
    assert verdict["checks"]["evaluation_protocol_matches"] is True
    assert verdict["checks"]["full_suite_proof_contract"] is False


def test_comparator_rejects_unchanged_policy(tmp_path: Path) -> None:
    from dreamervla.diagnostics.compare_frozen_rl_eval import compare_feasibility

    baseline, rl, frozen = _inputs(tmp_path)
    frozen["policy_hash_after"] = frozen["policy_hash_before"]
    frozen["policy_changed"] = False
    verdict = compare_feasibility(
        _write(tmp_path / "baseline.json", baseline),
        _write(tmp_path / "rl.json", rl),
        _write(tmp_path / "frozen.json", frozen),
    )

    assert verdict["passed"] is False
    assert verdict["checks"]["policy_changed"] is False


def test_comparator_rejects_equal_real_success_rate(tmp_path: Path) -> None:
    from dreamervla.diagnostics.compare_frozen_rl_eval import compare_feasibility

    baseline, rl, frozen = _inputs(tmp_path, baseline_rate=0.50, rl_rate=0.50)
    verdict = compare_feasibility(
        _write(tmp_path / "baseline.json", baseline),
        _write(tmp_path / "rl.json", rl),
        _write(tmp_path / "frozen.json", frozen),
    )

    assert verdict["passed"] is False
    assert verdict["checks"]["strict_success_improvement"] is False


def test_comparator_rejects_nonfinite_success_rate(tmp_path: Path) -> None:
    from dreamervla.diagnostics.compare_frozen_rl_eval import compare_feasibility

    baseline, rl, frozen = _inputs(tmp_path)
    rl["eval_success_rate"] = float("inf")
    verdict = compare_feasibility(
        _write(tmp_path / "baseline.json", baseline),
        _write(tmp_path / "rl.json", rl),
        _write(tmp_path / "frozen.json", frozen),
    )

    assert verdict["passed"] is False
    assert verdict["checks"]["success_rates_valid"] is False


def test_comparator_rejects_malformed_hash_manifest(tmp_path: Path) -> None:
    from dreamervla.diagnostics.compare_frozen_rl_eval import compare_feasibility

    baseline, rl, frozen = _inputs(tmp_path)
    malformed = {
        "world_model": "not-a-sha256",
        "classifier": "b" * 64,
    }
    frozen["frozen_hashes_before"] = malformed
    frozen["frozen_hashes_after"] = dict(malformed)
    verdict = compare_feasibility(
        _write(tmp_path / "baseline.json", baseline),
        _write(tmp_path / "rl.json", rl),
        _write(tmp_path / "frozen.json", frozen),
    )

    assert verdict["passed"] is False
    assert verdict["checks"]["frozen_models_unchanged"] is False


def test_comparator_rejects_wrong_checkpoint_kind_or_path(tmp_path: Path) -> None:
    from dreamervla.diagnostics.compare_frozen_rl_eval import compare_feasibility

    baseline, rl, frozen = _inputs(tmp_path)
    baseline["ckpt_kind"] = "dreamer"
    rl["ckpt_path"] = str(tmp_path / "other.ckpt")
    verdict = compare_feasibility(
        _write(tmp_path / "baseline.json", baseline),
        _write(tmp_path / "rl.json", rl),
        _write(tmp_path / "frozen.json", frozen),
    )

    assert verdict["passed"] is False
    assert verdict["checks"]["checkpoint_kinds_match"] is False
    assert verdict["checks"]["rl_checkpoint_matches_summary"] is False


def test_comparator_rejects_evaluated_component_hash_mismatch(tmp_path: Path) -> None:
    from dreamervla.diagnostics.compare_frozen_rl_eval import compare_feasibility

    baseline, rl, frozen = _inputs(tmp_path)
    rl["checkpoint_state_hashes"]["policy"] = "e" * 64
    verdict = compare_feasibility(
        _write(tmp_path / "baseline.json", baseline),
        _write(tmp_path / "rl.json", rl),
        _write(tmp_path / "frozen.json", frozen),
    )

    assert verdict["passed"] is False
    assert verdict["checks"]["evaluated_states_match_summary"] is False
