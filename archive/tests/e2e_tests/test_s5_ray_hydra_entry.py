from __future__ import annotations

import ray


def test_ray_cotrain_synthetic_experiment_runs_through_train_entry(tmp_path) -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from dreamervla.train import run

    if ray.is_initialized():
        ray.shutdown()

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=online_cotrain_ray_synthetic",
                f"training.out_dir={tmp_path}",
            ],
        )

    run(cfg)

    assert (tmp_path / "resolved_config.yaml").is_file()
    assert (tmp_path / "run_manifest.json").is_file()
    assert not ray.is_initialized()
