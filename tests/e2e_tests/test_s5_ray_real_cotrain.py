from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import ray


@pytest.mark.skipif(
    os.environ.get("DVLA_REAL_RAY_COTRAIN_SMOKE") != "1",
    reason=(
        "set DVLA_REAL_RAY_COTRAIN_SMOKE=1 plus DVLA_RYNNVLA_CKPT and "
        "DVLA_DREAMERVLA_WARMUP_CKPT to run the real LIBERO/RynnVLA Ray smoke"
    ),
)
def test_ray_real_cotrain_smoke_reports_component_losses(tmp_path) -> None:
    from hydra import compose, initialize_config_dir

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    vla_ckpt = os.environ.get("DVLA_RYNNVLA_CKPT")
    warmup_ckpt = os.environ.get("DVLA_DREAMERVLA_WARMUP_CKPT")
    if not vla_ckpt or not warmup_ckpt:
        pytest.skip("DVLA_RYNNVLA_CKPT and DVLA_DREAMERVLA_WARMUP_CKPT are required")
    if ray.is_initialized():
        ray.shutdown()

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=online_cotrain_ray_oft",
                f"training.out_dir={tmp_path / 'run'}",
                f"init.vla_ckpt_path={vla_ckpt}",
                f"init.warmup_ckpt_path={warmup_ckpt}",
                "env.num_workers=1",
                "env.cfg.kwargs.max_steps=8",
                "rollout.steps=16",
                "rollout.min_replay_episodes=1",
                "replay.cfg.sequence_length=8",
                "ray_data.sequence_length=8",
                "learner.train_cfg.batch_size=1",
                "learner.train_cfg.classifier_batch_size=1",
                "learner.model_cfg.classifier.kwargs.window=1",
                "learner.train_cfg.algorithm_cfg.lumos.classifier_min_steps=1",
            ],
        )

    history = OnlineCotrainRayRunner(cfg).run()

    for key in ("wm/loss", "cls/loss", "rl/actor_loss"):
        assert key in history
        assert math.isfinite(float(history[key]))
    assert history["train/learner_updates"] >= 1
    assert not ray.is_initialized()
