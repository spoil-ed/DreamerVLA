from __future__ import annotations

import os
from pathlib import Path

import pytest

CKPT = (
    Path(os.environ.get("DVLA_DATA_ROOT", "data"))
    / "checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
)
pytestmark = pytest.mark.skipif(
    not CKPT.is_dir() or os.environ.get("DVLA_GPU_E2E") != "1",
    reason="needs real OFT ckpt + LIBERO + GPU; set DVLA_GPU_E2E=1 to run",
)


def test_ray_coldstart_real_oft_matches_nonray_schema(tmp_path) -> None:
    import h5py
    import ray
    from hydra import compose, initialize_config_dir

    from dreamervla.train import run

    if ray.is_initialized():
        ray.shutdown()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=collect_rollouts_ray",
                "collect.task_ids=[0]",
                "collect.episodes_per_task=2",
                "collect.episode_horizon=64",
                f"task.openvla_oft.hdf5_reward_dir={tmp_path / 'reward'}",
                f"task.openvla_oft.action_hidden_dir={tmp_path / 'hidden'}",
                f"training.out_dir={tmp_path / 'run'}",
            ],
        )
    run(cfg)

    sidecar = next((tmp_path / "hidden").glob("*.hdf5"))
    with h5py.File(sidecar, "r") as handle:
        demo0 = handle["data"]["demo_0"]["obs_embedding"]
        assert demo0.shape[1] == 229376
        assert str(demo0.dtype) == "float16"
    assert (tmp_path / "hidden" / "preprocess_config.json").is_file()
    assert not ray.is_initialized()
