from __future__ import annotations

import json
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


def test_collection_real_oft_matches_sidecar_schema(tmp_path) -> None:
    import h5py
    import ray
    from hydra import compose, initialize_config_dir

    from dreamervla.runners import RolloutCollectionRunner
    from dreamervla.runtime.oft_collect import load_policy, vla_hidden_token_spec
    from dreamervla.train import run

    if ray.is_initialized():
        ray.shutdown()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=collect_rollouts",
                "collect.task_ids=[0]",
                "collect.episodes_per_task=2",
                "collect.episode_horizon=64",
                f"task.openvla_oft.hdf5_reward_dir={tmp_path / 'reward'}",
                f"task.openvla_oft.hidden_token_dir={tmp_path / 'hidden'}",
                f"training.out_dir={tmp_path / 'run'}",
            ],
        )
    run(cfg)

    preprocess_cfg = json.loads((tmp_path / "hidden" / "preprocess_config.json").read_text())
    sidecar = next((tmp_path / "hidden").glob("*.hdf5"))
    with h5py.File(sidecar, "r") as handle:
        demo0 = handle["data"]["demo_0"]["obs_embedding"]
        assert preprocess_cfg["obs_hidden_source"] == "hidden_token"
        plan = RolloutCollectionRunner(cfg).build_oft_worker_plan()
        policy = load_policy(dict(plan["collect"], _rank=0), 0)
        spec = vla_hidden_token_spec(
            policy.vla,
            plan["inference"]["decoder"]["kwargs"]["image_keys"],
        )
        assert tuple(demo0.shape[1:]) == (spec["token_count"], spec["token_dim"])
        assert str(demo0.dtype) == "float16"
    assert preprocess_cfg["hidden_key"] == "obs_embedding"
    assert not ray.is_initialized()
