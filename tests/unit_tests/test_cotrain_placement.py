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
    assert plan.env_specs[0].gpu_ids == []
    assert [spec.gpu_ids for spec in plan.env_specs[1:]] == [[1], [2], [3], [4]]
    assert [spec.role for spec in plan.env_specs[1:]] == ["wm_env"] * 4
    assert plan.learner_spec.gpu_ids == [0]
    assert [spec.gpu_ids for spec in plan.rollout_specs] == [[0], [1], [2], [3], [4]]
    assert [spec.gpu_ids for spec in plan.actor_specs] == [
        [0], [1], [2], [3], [4]
    ]


def test_one_gpu_placement_keeps_actor_spec_on_gpu_zero() -> None:
    plan = build_manual_cotrain_placement(1)

    assert plan.env_specs[0].role == "real_env"
    assert plan.env_specs[0].gpu_ids == []
    assert plan.learner_spec.gpu_ids == [0]
    assert [spec.gpu_ids for spec in plan.rollout_specs] == [[0]]
    assert [spec.gpu_ids for spec in plan.actor_specs] == [[0]]


def test_frozen_policy_placement_uses_all_eight_gpus_without_real_or_learner() -> None:
    plan = build_manual_cotrain_placement(
        8,
        real_env_workers=0,
        include_learner=False,
    )

    assert plan.real_env_ranks == []
    assert plan.wm_env_ranks == list(range(8))
    assert [spec.gpu_ids for spec in plan.env_specs] == [[gpu] for gpu in range(8)]
    assert [spec.gpu_ids for spec in plan.rollout_specs] == [
        [gpu] for gpu in range(8)
    ]
    assert [spec.gpu_ids for spec in plan.actor_specs] == [[gpu] for gpu in range(8)]
    assert plan.learner_spec is None


def test_mainline_and_frozen_routes_share_all_eight_actor_ranks() -> None:
    mainline = build_manual_cotrain_placement(
        8,
        real_env_workers=4,
        include_learner=True,
    )
    frozen = build_manual_cotrain_placement(
        8,
        real_env_workers=0,
        include_learner=False,
    )

    expected = [[gpu] for gpu in range(8)]
    assert [spec.gpu_ids for spec in mainline.actor_specs] == expected
    assert [spec.gpu_ids for spec in frozen.actor_specs] == expected
    assert mainline.learner_spec is not None
    assert mainline.learner_spec.gpu_ids == [0]
    assert frozen.learner_spec is None


def test_manual_cotrain_placement_honors_component_gpu_groups() -> None:
    plan = build_manual_cotrain_placement(
        7,
        real_env_workers=4,
        component_gpu_groups={
            "env": [[0], [1], [2], [3], [4]],
            "rollout": [[5], [5], [5], [5], [5]],
            "actor": [[6]],
            "learner": [[6]],
        },
    )

    assert [spec.role for spec in plan.env_specs] == [
        "real_env",
        "real_env",
        "real_env",
        "real_env",
        "wm_env",
    ]
    assert [spec.gpu_ids for spec in plan.env_specs] == [[0], [1], [2], [3], [4]]
    assert [spec.gpu_ids for spec in plan.rollout_specs] == [[5]] * 5
    assert [spec.gpu_ids for spec in plan.actor_specs] == [[6]]
    assert plan.learner_spec.gpu_ids == [6]


def test_manual_cotrain_placement_defaults_learner_to_actor_component() -> None:
    plan = build_manual_cotrain_placement(
        7,
        real_env_workers=4,
        component_gpu_groups={
            "env": [[0], [1], [2], [3], [4]],
            "rollout": [[5], [5], [5], [5], [5]],
            "actor": [[6]],
        },
    )

    assert plan.learner_spec.gpu_ids == [6]


def test_manual_cotrain_placement_rejects_rollout_component_that_misses_env_ranks() -> None:
    with pytest.raises(ValueError, match="rollout.*cover.*env"):
        build_manual_cotrain_placement(
            7,
            real_env_workers=4,
            component_gpu_groups={
                "env": [[0], [1], [2], [3], [4]],
                "rollout": [[5]],
                "actor": [[6]],
            },
        )


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
