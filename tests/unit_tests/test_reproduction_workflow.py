from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from dreamervla.runtime.common.reproduction import (
    ReproductionError,
    atomic_write_json,
    decide_stage,
    select_metric_checkpoint,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _compose_reproduction(name: str):
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs" / "scripts"),
        version_base=None,
    ):
        cfg = compose(config_name=f"reproduce/{name}")
    OmegaConf.resolve(cfg)
    return cfg


def test_sha256_file_hashes_file_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"dreamervla")

    assert sha256_file(path) == hashlib.sha256(b"dreamervla").hexdigest()


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    atomic_write_json(path, {"schema_version": 1, "status": "complete"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "status": "complete",
    }
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_select_metric_checkpoint_uses_minimum_loss(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    for name in (
        "epoch=0001-loss=0.400000.ckpt",
        "epoch=0002-loss=0.200000.ckpt",
        "epoch=0003-loss=0.300000.ckpt",
    ):
        (root / name).write_bytes(name.encode())

    selected = select_metric_checkpoint(root, metric_name="loss", mode="min")

    assert selected.path.name == "epoch=0002-loss=0.200000.ckpt"
    assert selected.epoch == 2
    assert selected.value == pytest.approx(0.2)
    assert selected.sha256 == sha256_file(selected.path)


def test_select_metric_checkpoint_uses_maximum_f1_and_latest_tie(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    for name in (
        "epoch=0003-f1=0.800000.ckpt",
        "epoch=0007-f1=0.900000.ckpt",
        "epoch=0008-f1=0.900000.ckpt",
    ):
        (root / name).write_bytes(name.encode())

    selected = select_metric_checkpoint(root, metric_name="f1", mode="max")

    assert selected.path.name == "epoch=0008-f1=0.900000.ckpt"


def test_select_metric_checkpoint_rejects_missing_candidates(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "latest.ckpt").touch()

    with pytest.raises(ReproductionError, match="no loss metric checkpoints"):
        select_metric_checkpoint(root, metric_name="loss", mode="min")


def test_decide_stage_starts_fresh_when_run_root_is_absent(tmp_path: Path) -> None:
    decision = decide_stage({}, stage="world_model", run_root=tmp_path / "world_model", budget=30)

    assert decision.action == "fresh"
    assert decision.resume_source is None


def test_decide_stage_resumes_when_latest_exists(tmp_path: Path) -> None:
    run_root = tmp_path / "world_model"
    latest = run_root / "checkpoints" / "latest.ckpt"
    latest.parent.mkdir(parents=True)
    latest.touch()

    decision = decide_stage({}, stage="world_model", run_root=run_root, budget=30)

    assert decision.action == "resume"
    assert decision.resume_source == run_root.resolve()


def test_decide_stage_skips_valid_completed_stage(tmp_path: Path) -> None:
    selected = tmp_path / "world_model" / "checkpoints" / "epoch=0030-loss=0.2.ckpt"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"wm")
    state = {
        "stages": {
            "world_model": {
                "status": "completed",
                "budget": 30,
                "selected_checkpoint": str(selected),
                "sha256": sha256_file(selected),
            }
        }
    }

    decision = decide_stage(
        state, stage="world_model", run_root=tmp_path / "world_model", budget=30
    )

    assert decision.action == "skip"
    assert decision.selected_checkpoint == selected.resolve()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("budget", 100, "budget mismatch"),
        ("sha256", "0" * 64, "hash mismatch"),
    ],
)
def test_decide_stage_rejects_completed_state_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    selected = tmp_path / "run" / "checkpoints" / "epoch=0001-loss=0.2.ckpt"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"checkpoint")
    record: dict[str, object] = {
        "status": "completed",
        "budget": 30,
        "selected_checkpoint": str(selected),
        "sha256": sha256_file(selected),
    }
    record[field] = value

    with pytest.raises(ReproductionError, match=match):
        decide_stage(
            {"stages": {"world_model": record}},
            stage="world_model",
            run_root=tmp_path / "run",
            budget=30,
        )


def test_prepare_reproduction_config_pins_public_assets_and_hardware() -> None:
    cfg = _compose_reproduction("prepare_assets")

    assert cfg.profile.task == "libero_goal"
    assert cfg.profile.num_gpus == 8
    assert cfg.profile.gpu_name == "NVIDIA H100 80GB HBM3"
    assert cfg.assets.openvla.repo == "Haozhan72/Openvla-oft-SFT-libero-goal-traj1"
    assert cfg.assets.openvla.revision == "d20e1d447dfd87c0daa121b0739e2a379f7fe334"
    assert cfg.assets.libero.repo == "yifengzhu-hf/LIBERO-datasets"
    assert cfg.preprocess.ngpu == 8
    assert str(cfg.preprocess.gpus) == "0,1,2,3,4,5,6,7"


def test_verify_third_party_honors_configured_source_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dreamervla.launchers import reproduce as module

    cfg = _compose_reproduction("prepare_assets")
    cfg.third_party_root = str(tmp_path)
    for key in cfg.third_party:
        cfg.third_party[key] = "a"
    visited: list[Path] = []

    def fake_git_revision(path: Path) -> str:
        visited.append(path)
        return "a" * 40

    monkeypatch.setattr(module, "_git_revision", fake_git_revision)

    revisions = module._verify_third_party(cfg)

    assert set(revisions) == set(cfg.third_party)
    assert visited
    assert all(path.parent == tmp_path for path in visited)


def test_prepare_assets_runs_commands_with_verified_openvla_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dreamervla.launchers import reproduce as module

    cfg = _compose_reproduction("prepare_assets")
    cfg.data_root = str(tmp_path / "data")
    cfg.third_party_root = str(tmp_path / "sources")
    cfg.assets.openvla.target = str(tmp_path / "model")
    cfg.assets.libero.target = str(tmp_path / "dataset")
    captured_envs: list[dict[str, str]] = []

    def fake_run(command, *, env, dry_run):
        captured_envs.append(dict(env))

    monkeypatch.setattr(module, "_run", fake_run)

    module._prepare_assets(
        module.ReproductionWorkflow(
            config_name="reproduce/prepare_assets",
            cfg=cfg,
            dry_run=True,
        )
    )

    assert captured_envs
    assert all(
        env["OPENVLA_OFT_ROOT"] == str((tmp_path / "sources" / "openvla-oft").resolve())
        for env in captured_envs
    )


def test_train_reproduction_config_has_release_budgets_and_selection() -> None:
    cfg = _compose_reproduction("train_dreamer")

    assert cfg.profile.task == "libero_goal"
    assert cfg.stages.world_model.budget == 30
    assert cfg.stages.world_model.budget_key == "training.warmup_replay_epochs"
    assert cfg.stages.world_model.selection.metric_name == "loss"
    assert cfg.stages.world_model.selection.mode == "min"
    assert cfg.stages.classifier.budget == 8
    assert cfg.stages.classifier.selection.metric_name == "f1"
    assert cfg.stages.classifier.selection.mode == "max"
    assert cfg.stages.dreamer.budget == 20000
    assert cfg.stages.dreamer.experiment == "openvla_libero"
    assert cfg.frozen_assertions.manual_cotrain.learner_updates_enabled is False
    assert cfg.frozen_assertions.manual_cotrain.training_mode == "failure_imagined_rl"


def test_aggressive_reproduction_config_uses_external_components_and_isolated_state() -> None:
    cfg = _compose_reproduction("train_dreamer_aggressive")

    assert cfg.profile.id == "cu124-h100-libero-goal-aggressive-v1"
    assert cfg.require_component_checkpoints is True
    assert cfg.component_checkpoints.world_model is None
    assert cfg.component_checkpoints.classifier is None
    assert list(cfg.stages) == ["dreamer"]
    assert cfg.stages.dreamer.experiment == "openvla_libero_aggressive"
    assert cfg.stages.dreamer.budget == 20
    assert str(cfg.state_path).endswith("training_state_aggressive.json")
    assert str(cfg.output_root).endswith("libero_goal/openvla_libero_aggressive")
    assert cfg.frozen_assertions.manual_cotrain.initial_condition_selector == "episode_start"


def test_success_sft_probe_reproduction_uses_external_components_and_one_step() -> None:
    cfg = _compose_reproduction("train_dreamer_success_sft_probe")

    assert cfg.require_component_checkpoints is True
    assert list(cfg.stages) == ["dreamer"]
    assert cfg.stages.dreamer.experiment == "openvla_libero_success_sft_probe"
    assert cfg.stages.dreamer.budget == 1
    assert cfg.frozen_assertions.manual_cotrain.training_mode == "imagined_success_sft"


def test_build_workflow_accepts_hydra_overrides() -> None:
    from dreamervla.launchers.reproduce import build_workflow

    workflow = build_workflow(
        [
            "--config-name",
            "reproduce/train_dreamer",
            "dry_run=true",
            "profile.num_gpus=4",
        ]
    )

    assert workflow.config_name == "reproduce/train_dreamer"
    assert workflow.dry_run is True
    assert workflow.cfg.profile.num_gpus == 4


def test_build_workflow_accepts_public_aggressive_config_and_checkpoint_pair(
    tmp_path: Path,
) -> None:
    from dreamervla.launchers.reproduce import build_workflow

    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    classifier.touch()

    workflow = build_workflow(
        [
            "--config",
            "reproduce/train_dreamer_aggressive",
            "--wm_ckpt",
            str(wm),
            "--cls_ckpt",
            str(classifier),
            "dry_run=true",
        ]
    )

    assert workflow.config_name == "reproduce/train_dreamer_aggressive"
    assert Path(workflow.cfg.component_checkpoints.world_model) == wm.resolve()
    assert Path(workflow.cfg.component_checkpoints.classifier) == classifier.resolve()


def test_build_workflow_rejects_partial_component_checkpoint_pair(tmp_path: Path) -> None:
    from dreamervla.launchers.reproduce import build_workflow

    wm = tmp_path / "wm.ckpt"
    wm.touch()

    with pytest.raises(ValueError, match="--wm_ckpt and --cls_ckpt must be supplied together"):
        build_workflow(
            [
                "--config",
                "reproduce/train_dreamer_aggressive",
                "--wm_ckpt",
                str(wm),
            ]
        )


def test_aggressive_reproduction_dry_run_launches_only_dreamer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dreamervla.launchers import reproduce as module

    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    classifier.touch()
    workflow = module.build_workflow(
        [
            "--config",
            "reproduce/train_dreamer_aggressive",
            "--wm_ckpt",
            str(wm),
            "--cls_ckpt",
            str(classifier),
            "dry_run=true",
        ]
    )
    commands: list[tuple[str, ...]] = []

    def capture(command, *, env, dry_run):
        del env
        assert dry_run is True
        commands.append(tuple(command))

    monkeypatch.setattr(module, "_run", capture)

    module._train_dreamer(workflow)

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--config") + 1] == "openvla_libero_aggressive"
    assert command[command.index("--wm_ckpt") + 1] == str(wm.resolve())
    assert command[command.index("--cls_ckpt") + 1] == str(classifier.resolve())
    assert "manual_cotrain.global_steps=20" in command


def test_build_stage_command_constructs_fresh_wm_command(tmp_path: Path) -> None:
    from dreamervla.launchers.reproduce import build_stage_command

    cfg = _compose_reproduction("train_dreamer")
    cfg.output_root = str(tmp_path / "outputs")
    cfg.stages.world_model.run_root = str(tmp_path / "outputs" / "world_model")

    command = build_stage_command(
        cfg,
        "world_model",
        action="fresh",
        selected_checkpoints={},
    )

    assert command[:4] == (
        "bash",
        str(PROJECT_ROOT / "scripts/experiments/world_model_training/train.sh"),
        "--config",
        "dreamer-wm",
    )
    assert "training.warmup_replay_epochs=30" in command
    assert f"training.out_dir={tmp_path / 'outputs' / 'world_model'}" in command
    assert "ngpu=8" in command
    assert "gpus=0,1,2,3,4,5,6,7" in command


def test_build_stage_command_appends_stage_specific_hydra_overrides(tmp_path: Path) -> None:
    from dreamervla.launchers.reproduce import build_stage_command

    cfg = _compose_reproduction("train_dreamer")
    cfg.stages.world_model.run_root = str(tmp_path / "world_model")
    with open_dict(cfg.stages.world_model):
        cfg.stages.world_model.overrides = [
            "profile=smoke",
            "training.warmup_replay_max_steps=1",
        ]

    command = build_stage_command(
        cfg,
        "world_model",
        action="fresh",
        selected_checkpoints={},
    )

    assert command[-2:] == (
        "profile=smoke",
        "training.warmup_replay_max_steps=1",
    )


def test_build_stage_command_constructs_resumable_frozen_dreamer(tmp_path: Path) -> None:
    from dreamervla.launchers.reproduce import build_stage_command

    cfg = _compose_reproduction("train_dreamer")
    run_root = tmp_path / "dreamer"
    cfg.stages.dreamer.run_root = str(run_root)
    wm = tmp_path / "wm.ckpt"
    classifier = tmp_path / "classifier.ckpt"
    wm.touch()
    classifier.touch()

    command = build_stage_command(
        cfg,
        "dreamer",
        action="resume",
        selected_checkpoints={"world_model": wm, "classifier": classifier},
    )

    assert "--resume" in command
    assert str(run_root) in command
    assert "--wm_ckpt" in command
    assert str(wm) in command
    assert "--cls_ckpt" in command
    assert str(classifier) in command
    assert "manual_cotrain.global_steps=20000" in command
    assert not any(item.startswith("training.out_dir=") for item in command)


def test_libero_download_command_pins_repository_revision_and_target() -> None:
    from dreamervla.launchers.reproduce import build_libero_download_command

    cfg = _compose_reproduction("prepare_assets")

    command = build_libero_download_command(cfg)

    assert command[:3] == ("python", "-m", "dreamervla.preprocess.download_libero")
    assert command[command.index("--repo") + 1] == cfg.assets.libero.repo
    assert command[command.index("--revision") + 1] == cfg.assets.libero.revision
    assert command[command.index("--suite") + 1] == "libero_goal"
    assert command[command.index("--target") + 1] == str(Path(cfg.assets.libero.target).resolve())


def test_openvla_validation_rejects_a_different_git_revision(tmp_path: Path) -> None:
    from dreamervla.launchers.reproduce import _valid_openvla

    model_root = tmp_path / "model"
    model_root.mkdir()
    subprocess.run(["git", "init", "-q", str(model_root)], check=True)
    subprocess.run(
        ["git", "-C", str(model_root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(model_root), "config", "user.name", "DreamerVLA Test"],
        check=True,
    )
    for name in ("config.json", "dataset_statistics.json", "tokenizer_config.json"):
        (model_root / name).write_text("{}\n", encoding="utf-8")
    (model_root / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    (model_root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    subprocess.run(["git", "-C", str(model_root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(model_root), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(model_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cfg = OmegaConf.create(
        {
            "assets": {
                "openvla": {
                    "target": str(model_root),
                    "revision": revision,
                    "required_files": [
                        "config.json",
                        "dataset_statistics.json",
                        "tokenizer_config.json",
                    ],
                }
            }
        }
    )

    assert _valid_openvla(cfg)
    cfg.assets.openvla.revision = "0" * 40
    assert not _valid_openvla(cfg)


def test_download_libero_persists_the_pinned_source_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dreamervla.preprocess import download_libero as module

    target = tmp_path / "libero_goal"
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        target.mkdir(parents=True)
        (target / "demo.hdf5").write_bytes(b"hdf5")
        return str(tmp_path)

    monkeypatch.setattr(module, "snapshot_download", fake_snapshot_download)

    module.download_libero(
        repo="owner/libero",
        revision="a" * 40,
        suite="libero_goal",
        target=target,
    )

    assert calls == [
        {
            "repo_id": "owner/libero",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "local_dir": str(tmp_path),
            "allow_patterns": "libero_goal/*",
        }
    ]
    assert json.loads((target / ".dreamervla-source.json").read_text(encoding="utf-8")) == {
        "repo": "owner/libero",
        "revision": "a" * 40,
        "suite": "libero_goal",
    }
