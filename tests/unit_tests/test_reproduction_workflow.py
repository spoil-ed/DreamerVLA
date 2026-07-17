from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.runtime.reproduction import (
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


def test_reproduction_shell_scripts_are_thin_python_entrypoints() -> None:
    scripts = PROJECT_ROOT / "scripts" / "reproduce"

    for name, config in (
        ("01_prepare_assets.sh", "reproduce/prepare_assets"),
        ("02_train_dreamer.sh", "reproduce/train_dreamer"),
    ):
        text = (scripts / name).read_text(encoding="utf-8")
        assert "exec python -m dreamervla.launchers.reproduce" in text
        assert f"--config-name {config}" in text
        assert "for " not in text
        assert "case " not in text


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


def test_dockerfile_pins_runtime_source_and_complete_third_party_install() -> None:
    text = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04" in text
    assert "WORKDIR /opt/dreamervla" in text
    assert "DVLA_DATA_ROOT=/data" in text
    assert "LIBERO_CONFIG_PATH=/tmp/dreamervla-libero" in text
    assert "LIBERO_CONFIG_PATH=/data/.libero" not in text
    assert "bash scripts/install_env.sh" in text
    assert "INSTALL_OPENVLA_OFT_THIRD_PARTY=true" in text
    assert "python -m dreamervla.diagnostics.verify_install" in text
    assert ".dreamervla-image.json" in text
    assert 'CMD ["/bin/bash"]' in text


def test_dockerfile_caches_dependencies_before_copying_full_source() -> None:
    text = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    install_index = text.index("bash scripts/install_env.sh")
    full_copy_index = text.index("COPY . /opt/dreamervla")
    dependency_prefix = text[:install_index]

    for required in (
        "COPY pyproject.toml requirements.txt",
        "COPY scripts/install_env.sh",
        "COPY scripts/install/",
        "COPY configs/scripts/install/",
        "COPY dreamervla/__init__.py dreamervla/config_resolvers.py",
        "COPY dreamervla/launchers/__init__.py dreamervla/launchers/workflow.py",
        "COPY dreamervla/diagnostics/__init__.py dreamervla/diagnostics/verify_install.py",
    ):
        assert required in dependency_prefix
    assert "COPY README.md" not in dependency_prefix
    assert full_copy_index > install_index
    assert text.count("COPY . /opt/dreamervla") == 1
    for dynamic_arg in ("DVLA_GIT_COMMIT", "DVLA_IMAGE_VERSION", "DVLA_BUILD_TIME"):
        assert text.index(f"ARG {dynamic_arg}") > full_copy_index
    assert text.index("org.opencontainers.image.revision") > full_copy_index
    assert text.rindex("python -m dreamervla.diagnostics.verify_install") > full_copy_index


def test_dockerfile_uses_cpu_rendering_for_build_time_import_checks() -> None:
    text = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "MUJOCO_GL=osmesa" in text
    assert "PYOPENGL_PLATFORM=osmesa" in text


def test_container_install_pins_third_party_runtime_compatibility() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    third_party_install = (PROJECT_ROOT / "scripts" / "install" / "40_third_party.sh").read_text(
        encoding="utf-8"
    )

    assert "mujoco==3.8.0" in requirements
    assert "protobuf==4.25.9" in requirements
    assert "jsonlines==4.0.0" in third_party_install
    assert "tensorflow_metadata==1.17.3" in third_party_install


def test_dockerignore_excludes_runtime_state_but_keeps_source() -> None:
    entries = {
        line.strip()
        for line in (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".git" in entries
    assert ".worktrees" in entries
    assert "data" in entries
    assert "third_party" in entries
    assert "dreamervla" not in entries
    assert "configs" not in entries


def test_public_docs_register_reproduction_commands() -> None:
    guide = (PROJECT_ROOT / "docs" / "docker_reproduction.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    readme_zh = (PROJECT_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")

    for text in (guide, readme, readme_zh):
        assert "spoil/dreamervla:cu124-h100-v1" in text
        assert "scripts/reproduce/01_prepare_assets.sh" in text
        assert "scripts/reproduce/02_train_dreamer.sh" in text

    assert "[中文](README.zh-CN.md)" in readme
    assert "[English](README.md)" in readme_zh
    for text in (readme, readme_zh):
        assert "scripts/install_env.sh" in text
        assert "scripts/install/60_verify.sh" in text
        assert "DVLA_DATA_ROOT" in text
        assert "WM: 30 epochs" in text
        assert "CLS: 8 epochs" in text
        assert "Dreamer: 20,000 steps" in text
    assert "automatically resumes" in readme
    assert "自动续训" in readme_zh

    assert "WM 30" in guide
    assert "CLS 8" in guide
    assert "20,000" in guide
    assert "third_party" in guide
    assert "自动续训" in guide


def test_docker_publish_workflow_uses_secrets_and_release_tags() -> None:
    text = (PROJECT_ROOT / ".github/workflows/docker-publish.yml").read_text(encoding="utf-8")

    assert "docker/login-action" in text
    assert "secrets.DOCKERHUB_USERNAME" in text
    assert "secrets.DOCKERHUB_TOKEN" in text
    assert "spoil/dreamervla" not in text
    assert "cu124-h100-v1" in text
    assert "sha-" in text
    assert "push:" in text
    assert "cache-from: type=gha" in text
    assert "cache-to: type=gha,mode=max" in text
