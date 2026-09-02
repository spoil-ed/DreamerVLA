from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner, _source_tree_sha256


class _ConcreteRunner(BaseRunner):
    def execute(self) -> None:
        return None

    def run(self) -> object:
        return None

    def teardown(self) -> None:
        return None


class _EvalOnlyRunner(_ConcreteRunner):
    checkpoint_output_enabled = False


class _RecordingRunner(_ConcreteRunner):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.loaded_checkpoint: Path | None = None

    def load_checkpoint(self, path=None, **_kwargs):
        self.loaded_checkpoint = Path(path)
        return {}


def test_source_tree_hash_ignores_untracked_runtime_directories(tmp_path: Path) -> None:
    source = tmp_path / "dreamervla"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n", encoding="utf-8")
    runtime = tmp_path / ".rlinf-runtime"
    runtime.mkdir()
    (runtime / "cache.bin").write_bytes(b"first")
    environment = tmp_path / "environments" / "test"
    virtualenv = environment / ".venv"
    virtualenv.mkdir(parents=True)
    (environment / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (virtualenv / "large-runtime-library.so").write_bytes(b"first")

    _source_tree_sha256.cache_clear()
    initial = _source_tree_sha256(tmp_path)
    (runtime / "cache.bin").write_bytes(b"second")
    (virtualenv / "large-runtime-library.so").write_bytes(b"second")
    _source_tree_sha256.cache_clear()
    assert _source_tree_sha256(tmp_path) == initial

    (environment / "pyproject.toml").write_text("[project]\nname = 'changed'\n", encoding="utf-8")
    _source_tree_sha256.cache_clear()
    assert _source_tree_sha256(tmp_path) != initial

    initial = _source_tree_sha256(tmp_path)
    (source / "module.py").write_text("value = 2\n", encoding="utf-8")
    _source_tree_sha256.cache_clear()
    assert _source_tree_sha256(tmp_path) != initial


def test_base_runner_setup_waits_for_rank_zero_artifacts(tmp_path: Path) -> None:
    class _Distributed:
        is_distributed = True
        world_size = 8

        def __init__(self, *, is_main_process: bool) -> None:
            self.is_main_process = is_main_process
            self.rank = 0 if is_main_process else 1
            self.local_rank = self.rank
            self.object_barrier_calls = 0

        def object_barrier(self) -> None:
            self.object_barrier_calls += 1

    for is_main_process in (True, False):
        out_dir = tmp_path / f"run-{is_main_process}"
        runner = _ConcreteRunner(
            OmegaConf.create(
                {
                    "seed": 7,
                    "training": {
                        "out_dir": str(out_dir),
                        "distributed_strategy": "ddp",
                    },
                }
            )
        )
        runner.distributed = _Distributed(is_main_process=is_main_process)

        runner.setup()

        assert (out_dir / "run_manifest.json").is_file() is is_main_process
        assert runner.distributed.object_barrier_calls == 1


def test_base_runner_uses_rlinf_style_run_artifact_dirs(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    runner = _ConcreteRunner(OmegaConf.create({"training": {"out_dir": str(out_dir)}}))

    assert runner.get_run_dir() == out_dir.resolve()
    assert runner.get_log_dir() == out_dir.resolve() / "logs"
    assert runner.get_checkpoint_dir() == out_dir.resolve() / "checkpoints"
    assert runner.get_tensorboard_dir() == out_dir.resolve() / "tensorboard"
    assert runner.get_wandb_dir() == out_dir.resolve() / "wandb"
    assert runner.get_video_dir("eval") == out_dir.resolve() / "video" / "eval"
    assert runner.get_hf_checkpoint_path() == out_dir.resolve() / "checkpoint_hf"
    assert (
        runner.get_global_step_checkpoint_dir(12)
        == out_dir.resolve() / "checkpoints" / "global_step_12"
    )
    assert (
        runner.get_component_checkpoint_dir("actor", step=12)
        == out_dir.resolve() / "checkpoints" / "global_step_12" / "actor"
    )


def test_base_runner_prefers_new_checkpoint_dir_but_resumes_compat_latest(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    compat_latest = out_dir / "ckpt" / "latest.ckpt"
    compat_latest.parent.mkdir(parents=True)
    compat_latest.write_bytes(b"legacy_payload")
    runner = _ConcreteRunner(OmegaConf.create({"training": {"out_dir": str(out_dir)}}))

    assert runner.get_checkpoint_path("latest") == out_dir / "checkpoints" / "latest.ckpt"
    assert runner.get_checkpoint_path("latest", prefer_existing=True) == compat_latest.resolve()

    new_latest = out_dir / "checkpoints" / "latest.ckpt"
    new_latest.parent.mkdir(parents=True)
    new_latest.write_bytes(b"new")
    assert runner.get_checkpoint_path("latest", prefer_existing=True) == new_latest


def test_base_runner_resume_reuses_checkpoint_owning_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "dreamer-wm" / "20260714_120000"
    checkpoint = run_dir / "checkpoints" / "latest.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    cfg = OmegaConf.create(
        {
            "training": {
                "out_dir": str(tmp_path / "fresh"),
                "resume": True,
                "resume_dir": str(checkpoint),
                "resume_path": str(checkpoint),
            }
        }
    )

    runner = _ConcreteRunner(cfg)

    assert runner.get_run_dir() == run_dir.resolve()


def test_base_runner_resume_directory_loads_canonical_latest(tmp_path: Path) -> None:
    run_dir = tmp_path / "dreamer-wm" / "20260714_120000"
    checkpoint = run_dir / "checkpoints" / "latest.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    cfg = OmegaConf.create(
        {
            "training": {
                "out_dir": str(tmp_path / "fresh"),
                "resume": True,
                "resume_dir": str(run_dir),
                "resume_path": None,
            }
        }
    )
    runner = _RecordingRunner(cfg)

    runner.resume()

    assert runner.loaded_checkpoint == checkpoint.resolve()


def test_base_runner_setup_writes_only_shallow_run_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "training": {
                "out_dir": str(out_dir),
                "distributed_strategy": "ddp",
            },
            "runner": {
                "logger": {
                    "logger_backends": ["tensorboard", "wandb"],
                }
            },
        }
    )
    runner = _ConcreteRunner(cfg)

    runner.setup()

    manifest_path = out_dir / "run_manifest.json"
    assert manifest_path.is_file()
    assert (out_dir / ".hydra" / "resolved_config.yaml").is_file()
    assert (out_dir / "checkpoints").is_dir()
    for absent_artifact in (
        "resolved_config.yaml",
        "logs",
        "tensorboard",
        "wandb",
        "video",
        "diagnostics",
    ):
        assert not (out_dir / absent_artifact).exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "created_at_utc",
        "runner",
        "distributed",
        "logging",
        "config",
        "git",
        "source",
    }
    assert manifest["schema_version"] == 3
    assert manifest["runner"]["class"] == "_ConcreteRunner"
    assert manifest["runner"]["name"] == "base"
    assert manifest["runner"]["family"] == "runner"
    assert manifest["runner"]["status"] == "abstract"
    assert manifest["logging"]["backends"] == ["tensorboard", "wandb"]
    assert manifest["distributed"]["strategy"] == "ddp"
    assert len(manifest["config"]["sha256"]) == 64
    assert "git" in manifest


def test_eval_only_runner_setup_does_not_create_checkpoint_directory(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "eval" / "libero_goal"
    runner = _EvalOnlyRunner(
        OmegaConf.create(
            {
                "seed": 7,
                "training": {"out_dir": str(out_dir)},
            }
        )
    )

    runner.setup()

    assert (out_dir / "run_manifest.json").is_file()
    assert not (out_dir / "checkpoints").exists()
    assert not (out_dir / "checkpoint_hf").exists()


def test_libero_evaluation_runner_disables_checkpoint_output() -> None:
    from dreamervla.runners.libero_vla_evaluation_runner import LIBEROVLAEvaluationRunner

    assert LIBEROVLAEvaluationRunner.checkpoint_output_enabled is False


def test_base_runner_metric_logger_keeps_tensorboard_artifacts_shallow(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(out_dir)},
            "runner": {"logger": {"logger_backends": ["tensorboard"]}},
        }
    )
    runner = _ConcreteRunner(cfg)

    runner.log_metrics({"train/loss": 1.0}, step=0)
    runner.finish_metric_logger()

    tensorboard_dir = out_dir / "tensorboard"
    assert any(path.name.startswith("events.out.tfevents") for path in tensorboard_dir.iterdir())
    assert not (tensorboard_dir / "config.yaml").exists()
