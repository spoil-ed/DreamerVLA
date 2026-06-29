from __future__ import annotations

import json

from dreamervla.diagnostics.benchmark_manual_workers import (
    GpuSampler,
    SyntheticBatchWMEnv,
    build_rollout_result,
    build_synthetic_observation,
    main,
    run_wm_env_direct_benchmark,
)
from dreamervla.workers.cotrain.messages import ObservationMsg, RolloutResultMsg


def test_builds_manual_cotrain_shaped_messages() -> None:
    obs = build_synthetic_observation(
        env_rank=3,
        slot_id=2,
        step=5,
        latent_dim=4,
        lang_dim=2,
        proprio_dim=3,
    )

    assert isinstance(obs, ObservationMsg)
    assert obs.key == "3:2"
    assert obs.obs["obs_embedding"].shape == (4,)
    assert obs.obs["lang_emb"].shape == (2,)
    assert obs.obs["proprio"].shape == (3,)

    result = build_rollout_result(obs, action_dim=7, chunk_size=8)

    assert isinstance(result, RolloutResultMsg)
    assert result.key == obs.key
    assert result.actions.shape == (8, 7)
    assert result.forward_inputs["hidden"].shape == (1, 4)
    assert result.forward_inputs["action"].shape == (1, 8, 7)
    assert result.forward_inputs["lang_emb"].shape == (2,)


def test_runs_wm_env_direct_benchmark_with_batch_step(tmp_path) -> None:
    output = tmp_path / "bench.json"
    metrics = run_wm_env_direct_benchmark(
        env_cfg={
            "target": "dreamervla.diagnostics.benchmark_manual_workers:SyntheticBatchWMEnv",
            "kwargs": {
                "num_envs": 2,
                "latent_dim": 4,
                "action_dim": 7,
                "lang_dim": 2,
                "proprio_dim": 3,
                "horizon": 32,
            },
        },
        num_slots=2,
        chunk_steps=3,
        action_dim=7,
        chunk_size=8,
        latent_dim=4,
        lang_dim=2,
        proprio_dim=3,
        output_json=output,
    )

    assert SyntheticBatchWMEnv.__name__ in metrics["worker/env_class"]
    assert metrics["worker/component"] == "wm-env-direct"
    assert metrics["worker/chunk_steps"] == 3
    assert metrics["worker/slot_count"] == 2
    assert metrics["env/wm_env/model_forwards"] == 24
    assert metrics["env/wm_env/batch_size_sum"] == 48
    assert metrics["env/wm_env/batch_size_avg"] == 2
    assert metrics["env/wm_env/batch_size_min"] == 2
    assert metrics["env/wm_env/batch_size_max"] == 2
    assert metrics["worker/trajectory_shards"] == 6
    assert json.loads(output.read_text())["worker/component"] == "wm-env-direct"


def test_gpu_sampler_reports_long_zero_utilization_runs() -> None:
    sampler = GpuSampler(interval_s=0.25)
    sampler.samples = [
        {"index": 0, "util_gpu": 0, "memory_used_mb": 100},
        {"index": 0, "util_gpu": 0, "memory_used_mb": 100},
        {"index": 0, "util_gpu": 30, "memory_used_mb": 110},
        {"index": 0, "util_gpu": 0, "memory_used_mb": 120},
    ]

    metrics = sampler.summary()

    assert metrics["gpu/0/util_zero_run_max_samples"] == 2
    assert metrics["gpu/0/util_zero_run_max_s"] == 0.5


def test_cli_runs_all_worker_benchmarks_to_one_json(tmp_path, capsys) -> None:
    output = tmp_path / "suite.json"

    main(
        [
            "--component",
            "all",
            "--profile",
            "tiny",
            "--device",
            "cpu",
            "--num-slots",
            "2",
            "--chunk-steps",
            "1",
            "--chunk-size",
            "2",
            "--latent-dim",
            "4",
            "--action-dim",
            "3",
            "--output-json",
            str(output),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text())

    assert printed["worker/component"] == "all"
    assert saved["worker/component"] == "all"
    assert saved["wm-env"]["worker/component"] == "wm-env-direct"
    assert saved["rollout"]["worker/component"] == "rollout-direct"
    assert saved["pair"]["worker/component"] == "rollout-wm-env-pair-direct"
