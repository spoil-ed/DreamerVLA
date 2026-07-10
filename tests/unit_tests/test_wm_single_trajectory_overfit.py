from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from dreamervla.diagnostics import wm_single_trajectory_overfit as diag


def test_epoch_batches_visit_each_window_once() -> None:
    starts = diag.sliding_window_starts(episode_len=12, sequence_len=5)
    batches = list(
        diag.iter_epoch_batches(
            starts,
            batch_size=3,
            rng=np.random.default_rng(7),
        )
    )

    visited = np.concatenate(batches)
    assert sorted(visited.tolist()) == starts.tolist()
    assert len(set(visited.tolist())) == len(starts)


def test_convergence_requires_consecutive_threshold_passes() -> None:
    tracker = diag.ConvergenceTracker(
        mse_threshold=0.03,
        cosine_threshold=0.95,
        required_passes=3,
    )

    assert tracker.observe(mse=0.02, cosine_similarity=0.96) is False
    assert tracker.observe(mse=0.04, cosine_similarity=0.97) is False
    assert tracker.streak == 0
    assert tracker.observe(mse=0.02, cosine_similarity=0.96) is False
    assert tracker.observe(mse=0.01, cosine_similarity=0.97) is False
    assert tracker.observe(mse=0.01, cosine_similarity=0.98) is True


class _TinyWorldModel(torch.nn.Module):
    num_hist = 1
    chunk_size = 1

    def __init__(self) -> None:
        super().__init__()
        self.prediction = torch.nn.Parameter(torch.zeros(2))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target = batch["obs_embedding"][:, 1, 0, :]
        prediction = self.prediction[None].expand_as(target)
        mse = torch.nn.functional.mse_loss(prediction, target)
        cosine = torch.nn.functional.cosine_similarity(
            prediction,
            target,
            dim=-1,
        ).mean()
        return {
            "_loss": mse,
            "hidden_mse": mse.detach(),
            "hidden_cosine_loss": (1.0 - cosine).detach(),
        }


def _tiny_episode() -> diag.EpisodeArrays:
    return diag.EpisodeArrays(
        hidden=np.ones((6, 1, 2), dtype=np.float32),
        lang=np.zeros((4,), dtype=np.float32),
        actions=np.zeros((6, 7), dtype=np.float32),
        rewards=np.zeros((6,), dtype=np.float32),
        proprio=np.zeros((6, 8), dtype=np.float32),
    )


def test_run_overfit_converges_and_writes_checkpoints(tmp_path: Path) -> None:
    settings = diag.RunSettings(
        max_epochs=20,
        batch_size=2,
        lr=0.2,
        grad_clip=1.0,
        eval_every=1,
        mse_threshold=0.03,
        cosine_threshold=0.95,
        required_passes=2,
        seed=3,
    )

    summary = diag.run_overfit(
        model=_TinyWorldModel(),
        episode=_tiny_episode(),
        settings=settings,
        out_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert summary["status"] == "converged"
    assert summary["best_hidden_mse"] <= 0.03
    assert summary["best_cosine_similarity"] >= 0.95
    assert (tmp_path / "checkpoints" / "best.ckpt").is_file()
    assert (tmp_path / "checkpoints" / "final.ckpt").is_file()


def test_run_overfit_reports_not_converged_at_epoch_limit(tmp_path: Path) -> None:
    settings = diag.RunSettings(
        max_epochs=1,
        batch_size=2,
        lr=0.0,
        grad_clip=1.0,
        eval_every=1,
        mse_threshold=0.001,
        cosine_threshold=0.99,
        required_passes=2,
        seed=3,
    )

    summary = diag.run_overfit(
        model=_TinyWorldModel(),
        episode=_tiny_episode(),
        settings=settings,
        out_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert summary["status"] == "not_converged"
    assert summary["epochs_completed"] == 1
    assert summary["success_streak"] == 0


def _write_episode_fixture(tmp_path: Path) -> tuple[Path, Path]:
    hidden_path = tmp_path / "hidden.hdf5"
    raw_path = tmp_path / "raw.hdf5"
    with h5py.File(hidden_path, "w") as hidden_file:
        demo = hidden_file.create_group("data/demo_0")
        demo.create_dataset(
            "obs_embedding",
            data=np.ones((6, 1, 2), dtype=np.float32),
        )
        demo.create_dataset("lang_emb", data=np.zeros((4,), dtype=np.float32))
    with h5py.File(raw_path, "w") as raw_file:
        demo = raw_file.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.zeros((6, 7), dtype=np.float32))
        demo.create_dataset("rewards", data=np.zeros((6,), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("ee_pos", data=np.zeros((6, 3), dtype=np.float32))
        obs.create_dataset("ee_ori", data=np.zeros((6, 4), dtype=np.float32))
        obs.create_dataset(
            "gripper_states",
            data=np.zeros((6, 1), dtype=np.float32),
        )
    return hidden_path, raw_path


def test_load_episode_reads_matching_hidden_and_raw_demo(tmp_path: Path) -> None:
    hidden_path, raw_path = _write_episode_fixture(tmp_path)

    episode = diag.load_episode(hidden_path, raw_path, "demo_0")

    assert episode.hidden.shape == (6, 1, 2)
    assert episode.actions.shape == (6, 7)
    assert episode.proprio.shape == (6, 8)


def test_build_plan_uses_random_initialization_and_convergence_defaults(
    tmp_path: Path,
) -> None:
    hidden_path, raw_path = _write_episode_fixture(tmp_path)
    args = diag.parse_args(
        [
            "--hidden-hdf5",
            str(hidden_path),
            "--raw-hdf5",
            str(raw_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    plan = diag.build_plan(args)

    assert plan["initialization"] == "random"
    assert plan["max_epochs"] == 200
    assert plan["eval_every"] == 5
    assert plan["mse_threshold"] == 0.03
    assert plan["cosine_threshold"] == 0.95
    assert plan["required_passes"] == 3
    assert plan["hidden_hdf5"] == str(hidden_path)
    assert plan["raw_hdf5"] == str(raw_path)


def test_main_dry_run_does_not_create_output(
    tmp_path: Path,
    capsys,
) -> None:
    hidden_path, raw_path = _write_episode_fixture(tmp_path)
    out_dir = tmp_path / "out"

    exit_code = diag.main(
        [
            "--hidden-hdf5",
            str(hidden_path),
            "--raw-hdf5",
            str(raw_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["initialization"] == "random"
    assert out_dir.exists() is False


def test_experiment_launcher_is_thin_and_dry_run_safe() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "experiments" / "wm_single_trajectory_overfit.sh"

    text = script.read_text(encoding="utf-8")

    assert "dreamervla.diagnostics.wm_single_trajectory_overfit" in text
    assert '"$@"' in text
    assert "--run" not in text.split('"$@"')[0]


def test_plot_curves_writes_nonempty_png(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    records = [
        {
            "event": "eval",
            "epoch": 0,
            "hidden_mse": 1.0,
            "cosine_similarity": 0.0,
        },
        {"event": "train_epoch", "epoch": 1, "train_loss": 0.4},
        {
            "event": "eval",
            "epoch": 1,
            "hidden_mse": 0.02,
            "cosine_similarity": 0.96,
        },
    ]
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output_path = tmp_path / "overfit_curves.png"

    diag._plot_curves(
        metrics_path,
        output_path,
        mse_threshold=0.03,
        cosine_threshold=0.95,
    )

    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
