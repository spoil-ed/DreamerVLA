from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import pytest
import ray


@pytest.mark.skipif(
    os.environ.get("DVLA_REAL_OFT_COLLECT_SMOKE") != "1",
    reason=(
        "set DVLA_REAL_OFT_COLLECT_SMOKE=1 plus DVLA_OFT_CKPT to run the real "
        "OpenVLA-OFT/LIBERO Ray collection smoke"
    ),
)
def test_ray_real_oft_collect_writes_reward_and_matching_sidecar(tmp_path) -> None:
    from hydra import compose, initialize_config_dir

    from dreamervla.runners.cold_start_ray_collect_runner import ColdStartRayCollectRunner

    oft_ckpt = os.environ.get("DVLA_OFT_CKPT")
    if not oft_ckpt:
        pytest.skip("DVLA_OFT_CKPT is required")
    if ray.is_initialized():
        ray.shutdown()

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=collect_rollouts_ray",
                f"task.openvla_oft.ckpt_path={oft_ckpt}",
                f"task.openvla_oft.input_token_dir={hidden_dir}",
                f"task.openvla_oft.hdf5_reward_dir={reward_dir}",
                "collect.task_ids=[0]",
                "collect.episodes_per_task=1",
                "collect.envs_per_gpu=1",
                "rollout.target_episodes=1",
                "rollout.max_steps=300",
                "env.num_workers=1",
            ],
        )

    history = ColdStartRayCollectRunner(cfg).run()

    assert history["rollout/episodes"] == 1
    reward_path = reward_dir / "ray_shard_000.hdf5"
    hidden_path = hidden_dir / "ray_shard_000.hdf5"
    preprocess_path = hidden_dir / "preprocess_config.json"
    assert reward_path.is_file()
    assert hidden_path.is_file()
    assert preprocess_path.is_file()

    preprocess = json.loads(preprocess_path.read_text())
    hidden_key = preprocess["hidden_key"]
    with h5py.File(hidden_path, "r") as hidden_f:
        demo = hidden_f["data"]["demo_0"]
        assert hidden_key in demo
        assert tuple(demo[hidden_key].shape[1:]) == (256, 4096)
    assert preprocess["obs_hidden_source"] == "input_token_embedding"
    assert preprocess["token_count"] == 256
    assert preprocess["token_dim"] == 4096
    assert not ray.is_initialized()
