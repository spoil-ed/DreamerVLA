from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra.core.override_parser.overrides_parser import OverridesParser

import dreamervla.launchers.frozen_model_cotrain_ray as launcher
import dreamervla.launchers.manual_cotrain_vla_eval as periodic_eval
from dreamervla.launchers.frozen_model_cotrain_ray import build_launch


def _save_classifier_checkpoint(path: Path, *, threshold: float = 0.45) -> None:
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "threshold": threshold,
            "config": {"classifier": {"hidden_dim": 1}},
        },
        path,
    )


def test_frozen_ray_launcher_builds_one_command_for_eight_gpus(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    run_root = tmp_path / "run"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(run_root))

    launch = build_launch(
        [
            "manual_cotrain.global_steps=12",
        ]
    )

    assert launch.visible_gpus == tuple(str(gpu) for gpu in range(8))
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "experiment=dreamervla_frozen_models_rl_ray" in launch.command
    assert (
        f"init.world_model_state_ckpt={json.dumps(str(wm.resolve()))}"
        in launch.command
    )
    assert (
        f"init.classifier_state_ckpt={json.dumps(str(classifier.resolve()))}"
        in launch.command
    )
    assert f"training.out_dir={json.dumps(str(run_root.resolve()))}" in launch.command
    assert "manual_cotrain.ngpu=8" in launch.command
    assert "cluster.num_gpus=8" in launch.command
    assert "algorithm.lumos.classifier_threshold=0.45" in launch.command
    assert launch.command[-1] == "manual_cotrain.global_steps=12"
    assert launch.resume is False
    assert launch.periodic_eval.interval_global_steps == 0
    assert launch.periodic_eval.include_initial is False
    assert launch.periodic_eval.task_ids == tuple(range(10))
    assert launch.periodic_eval.num_episodes_per_task == 10
    assert launch.periodic_eval.num_envs == 10
    assert launch.periodic_eval.base_vla_ckpt == (
        Path(__file__).resolve().parents[2]
        / "data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
    ).resolve()


def test_frozen_ray_launcher_quotes_hydra_checkpoint_paths_containing_equals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm_step=00004000-loss=0.097758.ckpt"
    classifier = tmp_path / "best_window_f10.9711_th0.45.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(tmp_path / "run=quoted"))

    launch = build_launch([])

    parsed = OverridesParser.create().parse_overrides(overrides=launch.command[3:])
    values = {override.get_key_element(): override.value() for override in parsed}

    assert values["init.world_model_state_ckpt"] == str(wm.resolve())
    assert values["training.out_dir"] == str((tmp_path / "run=quoted").resolve())


@pytest.mark.parametrize(
    "experiment",
    [
        "dreamervla_frozen_models_rl_ray_eval",
        "dreamervla_wmcls_cotrain_ray_eval",
    ],
)
def test_frozen_ray_launcher_resume_is_one_composable_command_with_policy_checkpoint(
    experiment: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4,5,6,7,8,9")
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    run_root = tmp_path / "run"
    resume = run_root / "checkpoints" / "manual_cotrain_step_500" / "manual_cotrain.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    resume.parent.mkdir(parents=True)
    resume.touch()
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RESUME_CKPT", str(resume))

    launch = build_launch([f"experiment={experiment}"])
    cfg = launcher._compose_training_config(launch.command)

    assert launch.resume is True
    assert launch.out_dir == run_root.resolve()
    assert (
        f"++manual_cotrain.resume_ckpt={json.dumps(str(resume.resolve()))}"
        in launch.command
    )
    assert "training.resume=true" in launch.command
    assert str(cfg.manual_cotrain.resume_ckpt) == str(resume.resolve())


def test_frozen_ray_launcher_resume_infers_checkpoint_run_even_if_run_root_env_is_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUN_ROOT", str(tmp_path / "stale-stage-root"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    run_root = tmp_path / "frozen-run"
    resume = run_root / "checkpoints" / "manual_cotrain_step_500" / "manual_cotrain.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    resume.parent.mkdir(parents=True)
    resume.touch()
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RESUME_CKPT", str(resume))

    launch = build_launch([])

    assert launch.out_dir == run_root.resolve()


def test_frozen_ray_launcher_rejects_non_eight_gpu_visibility(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))

    with pytest.raises(ValueError, match="exactly 8"):
        build_launch([])


def test_frozen_ray_launcher_rejects_duplicate_visible_gpu_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,6")
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))

    with pytest.raises(ValueError, match="distinct"):
        build_launch([])


def test_frozen_ray_launcher_resolves_completed_stage_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm_run = tmp_path / "wm-run"
    classifier_run = tmp_path / "classifier-run"
    wm_run.mkdir()
    classifier_run.mkdir()
    selected_wm = tmp_path / "selected-wm.ckpt"
    selected_classifier = tmp_path / "selected-classifier.ckpt"
    selected_wm.touch()
    selected_classifier.touch()
    monkeypatch.setattr(
        launcher,
        "select_available_world_model_checkpoint",
        lambda path: selected_wm,
    )
    monkeypatch.setattr(
        launcher,
        "select_available_classifier_checkpoint",
        lambda path: selected_classifier,
    )
    monkeypatch.setattr(
        launcher,
        "resolve_available_classifier_threshold",
        lambda path, default=0.5: 0.45,
    )
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm_run))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier_run))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(tmp_path / "out"))

    launch = build_launch([])

    assert (
        f"init.world_model_state_ckpt={json.dumps(str(selected_wm))}"
        in launch.command
    )
    assert (
        f"init.classifier_state_ckpt={json.dumps(str(selected_classifier))}"
        in launch.command
    )


def test_frozen_ray_launcher_loads_classifier_final_with_best_sibling_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    wm.touch()
    checkpoints = tmp_path / "classifier-run" / "checkpoints"
    checkpoints.mkdir(parents=True)
    final = checkpoints / "final.ckpt"
    best = checkpoints / "best_window_f10.9711_th0.45.ckpt"
    torch.save(
        {
            "cfg": {"classifier": {"hidden_dim": 1}},
            "state_dicts": {"model": {"weight": torch.ones(1)}},
            "pickles": {},
        },
        final,
    )
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "threshold": 0.45,
            "f1": 0.9711,
            "step": 500,
            "config": {"classifier": {"hidden_dim": 1}},
        },
        best,
    )
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(final))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(tmp_path / "out"))

    launch = build_launch([])

    assert (
        f"init.classifier_state_ckpt={json.dumps(str(final.resolve()))}"
        in launch.command
    )
    assert "algorithm.lumos.classifier_threshold=0.45" in launch.command


def test_frozen_ray_launcher_rejects_positional_component_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("WORLD_MODEL_CKPT", raising=False)
    monkeypatch.delenv("CLASSIFIER_CKPT", raising=False)

    with pytest.raises(ValueError, match="key=value"):
        build_launch([str(tmp_path / "wm.ckpt"), str(tmp_path / "classifier.ckpt")])


def test_frozen_ray_launcher_requires_explicit_checkpoint_assignments(
    monkeypatch,
) -> None:
    monkeypatch.delenv("WORLD_MODEL_CKPT", raising=False)
    monkeypatch.delenv("CLASSIFIER_CKPT", raising=False)

    with pytest.raises(ValueError, match="WORLD_MODEL_CKPT=/path"):
        build_launch([])


def test_frozen_ray_periodic_eval_schedule_is_step_zero_then_strict_tens() -> None:
    assert launcher.periodic_eval_steps(
        start_step=0,
        target_step=25,
        interval=10,
        include_initial=True,
    ) == [0, 10, 20]
    assert launcher.periodic_eval_steps(
        start_step=10,
        target_step=25,
        interval=10,
        include_initial=True,
    ) == [10, 20]


def test_eval_launcher_materializes_hydra_global_steps_for_segmentation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(tmp_path / "run"))

    launch = build_launch(
        ["experiment=dreamervla_frozen_models_rl_ray_eval"]
    )

    assert periodic_eval.manual_cotrain_target_step(launch.command) == 20000
    assert launch.command.count("manual_cotrain.global_steps=20000") == 1


def test_frozen_ray_periodic_eval_commands_use_base_vla_then_policy_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    run_root = tmp_path / "run=periodic"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(run_root))
    launch = build_launch(
        [
            "experiment=dreamervla_frozen_models_rl_ray_eval",
            "manual_cotrain.global_steps=20",
        ]
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        if "dreamervla.train" in command:
            target = int(
                next(
                    item.split("=", 1)[1]
                    for item in command
                    if item.startswith("manual_cotrain.global_steps=")
                )
            )
            ckpt = (
                run_root
                / "checkpoints"
                / f"manual_cotrain_step_{target}"
                / "manual_cotrain.ckpt"
            )
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"global_step": target, "state_dicts": {"policy": {}}}, ckpt)
        else:
            out_dir = Path(
                json.loads(
                    next(
                        item.split("=", 1)[1]
                        for item in command
                        if item.startswith("out_dir=")
                    )
                )
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "eval_libero_metrics.json").write_text(
                json.dumps(
                    {
                        "eval_success_rate": 0.5,
                        "eval_total_episodes": 100,
                        "eval_tasks": 10,
                        **{
                            f"eval_task_{task_id}_success_rate": 0.5
                            for task_id in range(10)
                        },
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(periodic_eval.subprocess, "run", fake_run)

    assert launcher.execute_launch(launch) == 0

    eval_calls = [call for call in calls if "dreamervla.launchers.train" in call]
    train_calls = [call for call in calls if "dreamervla.train" in call]
    assert [
        next(item for item in call if item.startswith("eval.ckpt_kind="))
        for call in eval_calls
    ] == [
        "eval.ckpt_kind=vla",
        "eval.ckpt_kind=vla_policy",
        "eval.ckpt_kind=vla_policy",
    ]
    assert [
        int(
            next(
                item.split("=", 1)[1]
                for item in call
                if item.startswith("manual_cotrain.global_steps=")
            )
        )
        for call in train_calls
    ] == [10, 20]
    assert all("eval.num_episodes_per_task=10" in call for call in eval_calls)
    assert all("eval.num_envs=10" in call for call in eval_calls)
    assert all("eval.task_ids=[0,1,2,3,4,5,6,7,8,9]" in call for call in eval_calls)
    assert all("eval.cotrain_diagnostics=true" not in call for call in eval_calls)
    assert any(
        item.startswith("++manual_cotrain.resume_ckpt=")
        and "manual_cotrain_step_10" in item
        for item in train_calls[1]
    )
    summary = json.loads((run_root / "eval/eval_summary.json").read_text())
    assert [record["global_step"] for record in summary["records"]] == [0, 10, 20]
    assert all(record["eval_total_episodes"] == 100 for record in summary["records"])


def test_learned_wmcls_policy_eval_enables_fixed_read_only_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    launch = build_launch([])
    spec = replace(launch.periodic_eval, learner_updates_enabled=True)

    learned = periodic_eval.vla_eval_command(
        "python",
        spec,
        global_step=1,
        policy_ckpt=tmp_path / "manual_cotrain.ckpt",
        out_dir=tmp_path / "eval",
    )
    baseline = periodic_eval.vla_eval_command(
        "python",
        spec,
        global_step=0,
        policy_ckpt=None,
        out_dir=tmp_path / "eval0",
    )

    assert "eval.cotrain_diagnostics=true" in learned
    assert "eval.cotrain_expected_trajectories=100" in learned
    assert "eval.cotrain_diagnostics=true" not in baseline


@pytest.mark.parametrize(
    "experiment",
    [
        "dreamervla_wmcls_cotrain_ray_eval",
        "dreamervla_frozen_models_rl_ray_eval",
    ],
)
def test_periodic_eval_resume_segment_composes_for_supported_recipe_schemas(
    experiment: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("COTRAIN_RESUME_CKPT", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    run_root = tmp_path / "run"
    resume_ckpt = (
        run_root
        / "checkpoints"
        / "manual_cotrain_step_10"
        / "manual_cotrain.ckpt"
    )
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(run_root))
    launch = build_launch(
        [
            f"experiment={experiment}",
            "manual_cotrain.global_steps=20",
        ]
    )

    segment = periodic_eval.segment_train_command(
        launch.command,
        target_step=20,
        checkpoint_every=10,
        run_root=run_root,
        resume_ckpt=resume_ckpt,
        learner_updates_enabled=launch.periodic_eval.learner_updates_enabled,
    )
    cfg = launcher._compose_training_config(segment)

    assert str(cfg.manual_cotrain.resume_ckpt) == str(resume_ckpt)
    assert sum(
        item.startswith("++manual_cotrain.resume_ckpt=") for item in segment
    ) == 1


def test_wmcls_eval_recipe_enables_learner_and_uses_same_eval_protocol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(tmp_path / "run"))

    launch = build_launch(
        [
            "experiment=dreamervla_wmcls_cotrain_ray_eval",
            "manual_cotrain.global_steps=20",
        ]
    )
    cfg = launcher._compose_training_config(launch.command)

    assert launch.periodic_eval.interval_global_steps == 10
    assert launch.periodic_eval.include_initial is True
    assert launch.periodic_eval.num_envs == 25
    assert cfg.manual_cotrain.learner_updates_enabled is True
    assert cfg.manual_cotrain.staged_policy_update is True
    assert cfg.manual_cotrain.real_env_enabled is True
    assert cfg.manual_cotrain.real_env_workers == 1
    assert cfg.manual_cotrain.real_rollout_target_trajectories == 32
    assert cfg.replay.seed is None
    assert cfg.manual_cotrain.env_rollout_timeout_s == 5400
    assert cfg.learner is not None


def test_wmcls_debug_oneclick_uses_ten_single_step_eval_segments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    _save_classifier_checkpoint(classifier)
    monkeypatch.setenv("WORLD_MODEL_CKPT", str(wm))
    monkeypatch.setenv("CLASSIFIER_CKPT", str(classifier))
    monkeypatch.setenv("COTRAIN_RUN_ROOT", str(tmp_path / "run"))

    launch = build_launch(
        [
            "experiment=dreamervla_wmcls_cotrain_ray_eval",
            "manual_cotrain.global_steps=20000",
            "++training.debug=true",
        ]
    )
    cfg = launcher._compose_training_config(launch.command)

    assert periodic_eval.manual_cotrain_target_step(launch.command) == 10
    assert launch.periodic_eval.interval_global_steps == 1
    assert cfg.manual_cotrain.checkpoint_every == 1
