from __future__ import annotations

import subprocess
import sys

import pytest

from dreamervla.workers.cotrain.placement import build_manual_cotrain_placement


@pytest.mark.parametrize("ngpu", [0, 1, 2, 3, 4, 5])
def test_manual_cotrain_placement_supports_zero_to_five_gpus(ngpu: int) -> None:
    plan = build_manual_cotrain_placement(ngpu)
    assert plan.ngpu == ngpu
    assert plan.real_env_ranks == [0]
    assert len(plan.env_specs) == max(1, ngpu)
    assert plan.learner_spec.kind == "learner"
    assert plan.rollout_specs
    assert plan.actor_specs


def test_gpu_placement_matches_manual_notes_for_five_gpus() -> None:
    plan = build_manual_cotrain_placement(5)

    assert plan.env_specs[0].role == "real_env"
    assert plan.env_specs[0].gpu_ids == [0]
    assert [spec.gpu_ids for spec in plan.env_specs[1:]] == [[1], [2], [3], [4]]
    assert [spec.role for spec in plan.env_specs[1:]] == ["wm_env"] * 4
    assert plan.learner_spec.gpu_ids == [0]
    assert [spec.gpu_ids for spec in plan.rollout_specs] == [[0], [1], [2], [3], [4]]
    assert [spec.gpu_ids for spec in plan.actor_specs] == [[1], [2], [3], [4]]


def test_one_gpu_placement_keeps_actor_spec_on_gpu_zero() -> None:
    plan = build_manual_cotrain_placement(1)

    assert plan.env_specs[0].role == "real_env"
    assert plan.env_specs[0].gpu_ids == [0]
    assert plan.learner_spec.gpu_ids == [0]
    assert [spec.gpu_ids for spec in plan.rollout_specs] == [[0]]
    assert [spec.gpu_ids for spec in plan.actor_specs] == [[0]]


def test_zero_gpu_placement_is_cpu_target_topology() -> None:
    plan = build_manual_cotrain_placement(0)

    assert plan.env_specs[0].role == "real_env"
    assert plan.env_specs[0].gpu_ids == []
    assert plan.learner_spec.gpu_ids == []
    assert plan.actor_specs[0].gpu_ids == []
    assert plan.rollout_specs[0].gpu_ids == []
    assert plan.actor_fsdp_strategy == "none"


def test_negative_gpu_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="ngpu must be >= 0"):
        build_manual_cotrain_placement(-1)


def test_placement_import_does_not_load_message_dependencies() -> None:
    code = "\n".join(
        [
            "import sys",
            "import dreamervla.workers.cotrain.placement",
            "assert 'dreamervla.workers.cotrain.messages' not in sys.modules",
            "assert 'numpy' not in sys.modules",
            "assert 'torch' not in sys.modules",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
