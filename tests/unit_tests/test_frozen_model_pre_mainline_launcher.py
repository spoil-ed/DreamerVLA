from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def _config(tmp_path: Path, **overrides):
    from dreamervla.utils.hydra_config import script_config

    items = [f"run_root={tmp_path}", "python=python", "dry_run=true"]
    items.extend(f"{key}={value}" for key, value in overrides.items())
    return script_config("frozen_model_pre_mainline", items)


def test_complete_plan_contains_every_stage_without_collected_rollouts(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import build_pipeline_plan

    plan = build_pipeline_plan(_config(tmp_path))
    rendered = "\n".join(" ".join(command) for command in plan.commands)

    assert "experiment=wm_official_upper_bound" in rendered
    assert "experiment=classifier_official_upper_bound" in rendered
    assert "experiment=dreamervla_frozen_models_rl" in rendered
    assert rendered.count("experiment=eval_libero_vla") == 2
    assert "dreamervla.diagnostics.compare_frozen_rl_eval" in rendered
    assert "collected_rollouts" not in rendered
    assert f"init.world_model_state_ckpt={tmp_path}/summary/selected_wm.ckpt" in rendered
    assert f"init.classifier_state_ckpt={tmp_path}/summary/selected_classifier.ckpt" in rendered


def test_plan_uses_identical_protocol_for_baseline_and_rl(tmp_path: Path) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import build_pipeline_plan

    plan = build_pipeline_plan(_config(tmp_path))
    baseline = plan.eval_baseline_cmd
    rl = plan.eval_rl_cmd
    protocol_prefixes = (
        "eval.task_suite_name=",
        "eval.task_ids=",
        "eval.num_episodes_per_task=",
        "eval.num_envs=",
        "eval.seed=",
        "eval.num_steps_wait=",
        "eval.action_steps=",
        "eval.max_steps=",
        "eval.enumerate_all_init_states=",
        "eval.scheme=",
        "eval.reconfigure_per_episode=",
    )
    for prefix in protocol_prefixes:
        assert [value for value in baseline if value.startswith(prefix)] == [
            value for value in rl if value.startswith(prefix)
        ]


def test_world_model_selector_prefers_lowest_ranked_loss(tmp_path: Path) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        select_world_model_checkpoint,
    )

    wm_root = tmp_path / "wm"
    ranked = wm_root / "ckpt" / "warmup_topk" / "wm"
    ranked.mkdir(parents=True)
    state = {"weight": torch.ones(1)}
    component_config = {"_target_": "test.WorldModel", "hidden_dim": 1}
    final = wm_root / "ckpt" / "wm_warmup.ckpt"
    torch.save(
        {
            "world_model": state,
            "complete": True,
            "global_step": 2,
            "warmup_component": "wm",
            "warmup_step": 2,
            "warmup_total_steps": 2,
            "config": {"world_model": component_config},
        },
        final,
    )
    worse = ranked / "wm_step=00000001-loss=0.400000.ckpt"
    better = ranked / "wm_step=00000002-loss=0.200000.ckpt"
    for path, step, loss in ((worse, 1, 0.4), (better, 2, 0.2)):
        torch.save(
            {
                "world_model": state,
                "warmup_component": "wm",
                "warmup_step": step,
                "warmup_total_steps": 2,
                "complete": False,
                "metrics": {"loss": loss},
                "config": {"world_model": component_config},
            },
            path,
        )

    assert select_world_model_checkpoint(wm_root) == better.resolve()


def test_world_model_selector_requires_complete_canonical_checkpoint(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        select_world_model_checkpoint,
    )

    ranked = tmp_path / "wm" / "ckpt" / "warmup_topk" / "wm"
    ranked.mkdir(parents=True)
    torch.save(
        {
            "world_model": {"weight": torch.ones(1)},
            "warmup_component": "wm",
            "warmup_step": 1,
            "warmup_total_steps": 2,
            "complete": False,
            "metrics": {"loss": 0.2},
            "config": {"world_model": {"hidden_dim": 1}},
        },
        ranked / "wm_step=00000001-loss=0.200000.ckpt",
    )

    with pytest.raises(FileNotFoundError, match="complete WM checkpoint"):
        select_world_model_checkpoint(tmp_path / "wm")


def test_available_world_model_selector_accepts_incomplete_ranked_checkpoint(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        select_available_world_model_checkpoint,
    )

    ranked = tmp_path / "wm" / "ckpt" / "warmup_topk" / "wm"
    ranked.mkdir(parents=True)
    state = {"weight": torch.ones(1)}
    component_config = {"_target_": "test.WorldModel", "hidden_dim": 1}
    worse = ranked / "wm_step=00003000-loss=0.200000.ckpt"
    better = ranked / "wm_step=00004000-loss=0.097758.ckpt"
    for path, step, loss in ((worse, 3000, 0.2), (better, 4000, 0.097758)):
        torch.save(
            {
                "world_model": state,
                "warmup_component": "wm",
                "warmup_step": step,
                "warmup_total_steps": 23160,
                "complete": False,
                "metrics": {"loss": loss},
                "config": {"world_model": component_config},
            },
            path,
        )

    assert select_available_world_model_checkpoint(tmp_path / "wm") == better.resolve()


def test_available_world_model_selector_uses_latest_progress_without_topk(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        select_available_world_model_checkpoint,
    )

    progress = tmp_path / "wm" / "ckpt" / "warmup_progress"
    progress.mkdir(parents=True)
    state = {"weight": torch.ones(1)}
    component_config = {"_target_": "test.WorldModel", "hidden_dim": 1}
    older = progress / "wm_step_00000500.ckpt"
    latest = progress / "wm_step_00001000.ckpt"
    for path, step in ((older, 500), (latest, 1000)):
        torch.save(
            {
                "world_model": state,
                "warmup_component": "wm",
                "warmup_step": step,
                "warmup_total_steps": 23160,
                "complete": False,
                "metrics": {"loss": 0.2},
                "config": {"world_model": component_config},
            },
            path,
        )

    assert select_available_world_model_checkpoint(tmp_path / "wm") == latest.resolve()


def test_classifier_selector_requires_heldout_window_best_checkpoint(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        select_classifier_checkpoint,
    )

    classifier_root = tmp_path / "classifier"
    checkpoint = classifier_root / "checkpoints" / "best_window.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "threshold": 0.6,
            "f1": 0.8,
            "step": 4,
            "config": {"classifier": {"hidden_dim": 1}},
            "extra": {"val_window": {"best_f1": 0.8, "best_thresh": 0.6}},
        },
        checkpoint,
    )
    (classifier_root / "summary.json").write_text(
        json.dumps(
            {
                "best_window_ckpt_path": str(checkpoint),
                "best_window_f1": 0.8,
                "total_steps": 4,
            }
        ),
        encoding="utf-8",
    )

    assert select_classifier_checkpoint(classifier_root) == checkpoint.resolve()


def _save_available_classifier_checkpoint(
    path: Path,
    *,
    f1: float,
    threshold: float,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "threshold": threshold,
            "f1": f1,
            "step": step,
            "config": {"classifier": {"hidden_dim": 1}},
            "extra": {
                "val_window": {"best_f1": f1, "best_thresh": threshold}
            },
        },
        path,
    )


def test_available_classifier_selector_prefers_highest_window_f1_without_summary(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        select_available_classifier_checkpoint,
    )

    checkpoints = tmp_path / "classifier" / "checkpoints"
    lower = checkpoints / "best_window_f10.9524_th0.05.ckpt"
    higher = checkpoints / "best_window_f10.9711_th0.45.ckpt"
    _save_available_classifier_checkpoint(lower, f1=0.9524, threshold=0.05, step=250)
    _save_available_classifier_checkpoint(higher, f1=0.9711, threshold=0.45, step=500)

    assert (
        select_available_classifier_checkpoint(tmp_path / "classifier")
        == higher.resolve()
    )


@pytest.mark.parametrize("fallback_name", ["final.ckpt", "latest.ckpt"])
def test_available_classifier_selector_accepts_runner_checkpoint_fallback(
    tmp_path: Path,
    fallback_name: str,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        select_available_classifier_checkpoint,
    )

    checkpoint = tmp_path / "classifier" / "checkpoints" / fallback_name
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "cfg": {"classifier": {"hidden_dim": 1}},
            "state_dicts": {"model": {"weight": torch.ones(1)}},
            "pickles": {},
        },
        checkpoint,
    )

    assert (
        select_available_classifier_checkpoint(tmp_path / "classifier")
        == checkpoint.resolve()
    )


def test_available_classifier_threshold_uses_best_window_for_runner_checkpoint(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        resolve_available_classifier_threshold,
    )

    checkpoints = tmp_path / "classifier" / "checkpoints"
    best = checkpoints / "best_window_f10.9711_th0.45.ckpt"
    final = checkpoints / "final.ckpt"
    _save_available_classifier_checkpoint(best, f1=0.9711, threshold=0.45, step=500)
    torch.save(
        {
            "cfg": {"classifier": {"hidden_dim": 1}},
            "state_dicts": {"model": {"weight": torch.ones(1)}},
            "pickles": {},
        },
        final,
    )

    assert resolve_available_classifier_threshold(final) == pytest.approx(0.45)


def test_available_classifier_threshold_uses_explicit_default_without_calibration(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        resolve_available_classifier_threshold,
    )

    latest = tmp_path / "classifier" / "checkpoints" / "latest.ckpt"
    latest.parent.mkdir(parents=True)
    torch.save(
        {
            "cfg": {"classifier": {"hidden_dim": 1}},
            "state_dicts": {"model": {"weight": torch.ones(1)}},
            "pickles": {},
        },
        latest,
    )

    assert resolve_available_classifier_threshold(latest, default=0.5) == 0.5


def test_classifier_selector_rejects_checkpoint_outside_stage_root(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        select_classifier_checkpoint,
    )

    classifier_root = tmp_path / "classifier"
    classifier_root.mkdir()
    external = tmp_path / "external.ckpt"
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "threshold": 0.6,
            "f1": 0.8,
            "step": 4,
            "config": {"classifier": {"hidden_dim": 1}},
        },
        external,
    )
    (classifier_root / "summary.json").write_text(
        json.dumps({"best_window_ckpt_path": str(external)}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside classifier stage"):
        select_classifier_checkpoint(classifier_root)


def test_rl_stage_refreshes_stale_selection_links(tmp_path: Path) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import (
        build_pipeline_plan,
        refresh_upstream_selections,
    )

    plan = build_pipeline_plan(_config(tmp_path))
    wm_path = tmp_path / "wm" / "ckpt" / "wm_warmup.ckpt"
    wm_path.parent.mkdir(parents=True)
    torch.save(
        {
            "world_model": {"weight": torch.ones(1)},
            "complete": True,
            "global_step": 2,
            "warmup_component": "wm",
            "warmup_step": 2,
            "warmup_total_steps": 2,
            "config": {"world_model": {"hidden_dim": 1}},
        },
        wm_path,
    )
    classifier_path = tmp_path / "classifier" / "checkpoints" / "best_window.ckpt"
    classifier_path.parent.mkdir(parents=True)
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "threshold": 0.6,
            "f1": 0.8,
            "step": 4,
            "config": {"classifier": {"hidden_dim": 1}},
            "extra": {"val_window": {"best_f1": 0.8, "best_thresh": 0.6}},
        },
        classifier_path,
    )
    (tmp_path / "classifier" / "summary.json").write_text(
        json.dumps(
            {
                "best_window_ckpt_path": str(classifier_path),
                "best_window_f1": 0.8,
                "total_steps": 4,
            }
        ),
        encoding="utf-8",
    )
    stale = tmp_path / "stale.ckpt"
    stale.touch()
    plan.summary_dir.mkdir(parents=True)
    plan.selected_wm_ckpt.symlink_to(stale)
    plan.selected_classifier_ckpt.symlink_to(stale)

    refresh_upstream_selections(plan)

    assert plan.selected_wm_ckpt.resolve() == wm_path.resolve()
    assert plan.selected_classifier_ckpt.resolve() == classifier_path.resolve()


def test_dry_run_does_not_launch_subprocess(monkeypatch, tmp_path: Path) -> None:
    import dreamervla.launchers.frozen_model_pre_mainline as launcher

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run launched a subprocess")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    assert launcher.main([f"run_root={tmp_path}", "dry_run=true"]) == 0


@pytest.mark.parametrize(
    ("section", "override"),
    [
        ("wm_overrides", "training.out_dir=/tmp/elsewhere"),
        ("wm_overrides", "pre_mainline.official_task_ids=[0]"),
        ("classifier_overrides", "task=openvla_onetraj_libero_object"),
        ("classifier_overrides", "task.hdf5_reward_dir=/tmp/not-official"),
        ("rl_overrides", "init.world_model_state_ckpt=/tmp/other.ckpt"),
        ("eval_overrides", "eval.ckpt_path=/tmp/other.ckpt"),
        ("eval_overrides", "eval.seed=999"),
        ("eval_overrides", "eval.dreamer_policy_source=init"),
        ("eval_overrides", "eval.require_strict_component_load=false"),
        ("eval_overrides", "init.vla_ckpt_path=/tmp/other-vla"),
    ],
)
def test_plan_rejects_overrides_of_causal_boundaries(
    tmp_path: Path,
    section: str,
    override: str,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import build_pipeline_plan

    cfg = _config(tmp_path)
    cfg[section] = [override]

    with pytest.raises(ValueError, match="reserved"):
        build_pipeline_plan(cfg)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("eval.task_ids=[0]", "requires eval.task_ids"),
        ("eval.max_tasks=1", "requires eval.max_tasks"),
        (
            "tasks.goal.baseline_ckpt=/tmp/not-the-canonical-baseline",
            "canonical one-trajectory baseline",
        ),
    ],
)
def test_plan_rejects_top_level_proof_contract_overrides(
    tmp_path: Path,
    override: str,
    message: str,
) -> None:
    from dreamervla.launchers.frozen_model_pre_mainline import build_pipeline_plan
    from dreamervla.utils.hydra_config import script_config

    cfg = script_config(
        "frozen_model_pre_mainline",
        [f"run_root={tmp_path}", "dry_run=true", override],
    )

    with pytest.raises(ValueError, match=message):
        build_pipeline_plan(cfg)
