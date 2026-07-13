from dreamervla.workers.cotrain.placement import build_manual_cotrain_placement


def test_eight_gpu_cotrain_placement_keeps_all_models_resident() -> None:
    plan = build_manual_cotrain_placement(8, real_env_workers=1)

    real = [spec for spec in plan.env_specs if spec.role == "real_env"]
    imagined = [spec for spec in plan.env_specs if spec.role == "wm_env"]
    assert [spec.gpu_ids for spec in real] == [[]]
    assert [spec.gpu_ids for spec in imagined] == [[i] for i in range(1, 8)]
    assert [spec.gpu_ids for spec in plan.rollout_specs] == [[i] for i in range(8)]
    assert [spec.gpu_ids for spec in plan.actor_specs] == [[i] for i in range(8)]
    assert plan.learner_spec is not None
    assert plan.learner_spec.gpu_ids == [0]
