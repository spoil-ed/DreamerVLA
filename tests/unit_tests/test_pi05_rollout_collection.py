from __future__ import annotations

from pathlib import Path

import numpy as np


def test_pi05_collection_plan_is_target_injected_and_rlinf_aligned() -> None:
    from hydra import compose, initialize_config_dir

    from dreamervla.runners import RolloutCollectionRunner

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=collect_rollouts_pi05"])

    plan = RolloutCollectionRunner(cfg).build_vla_worker_plan()
    manifest = plan["dump"]["preprocess_config"]
    assert plan["inference"]["decoder"]["target"].endswith("pi05_rollout:Pi05RolloutBundle")
    assert plan["inference"]["action_steps"] == 10
    assert manifest["policy_family"] == "pi05"
    assert manifest["obs_hidden_source"] == "pi05_prefix_output"
    assert manifest["obs_embedding_shape"] == [768, 2048]
    assert manifest["alignment_source"] == "rlinf_rlt_prefix"
    assert plan["collect"]["episode_horizon"] == 512
    assert plan["collect"]["episodes_per_task"] == 150
    assert plan["dump"]["num_workers"] == 8
    assert plan["dump"]["write_hidden_sidecar"] is False
    assert plan["inference"]["emit_hidden_sidecar"] is False
    assert plan["dump"]["compression"] == {
        "codec": "gzip",
        "level": 1,
        "shuffle": True,
        "time_chunk": 1,
    }


def test_pi05_dump_workers_follow_multi_gpu_inference_count() -> None:
    from hydra import compose, initialize_config_dir

    from dreamervla.runners import RolloutCollectionRunner

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=collect_rollouts_pi05",
                "collect.num_inference_workers=8",
                "env.num_workers=32",
            ],
        )

    plan = RolloutCollectionRunner(cfg).build_vla_worker_plan()
    assert plan["dump"]["num_workers"] == 8


def test_pi05_prefix_extractor_uses_two_libero_cameras_and_state() -> None:
    from dreamervla.workers.inference.pi05_rollout import Pi05PrefixExtractor

    extractor = Pi05PrefixExtractor(
        base_image_key="agentview_rgb",
        wrist_image_key="eye_in_hand_rgb",
        state_key="state",
        rotate_images_180=True,
    )
    base = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    wrist = base + 1
    prepared = extractor.prepare(
        {
            "agentview_rgb": base,
            "eye_in_hand_rgb": wrist,
            "state": np.arange(8, dtype=np.float32),
        },
        "pick up the object",
    )

    np.testing.assert_array_equal(prepared["observation/image"], base[::-1, ::-1])
    np.testing.assert_array_equal(prepared["observation/wrist_image"], wrist[::-1, ::-1])
    assert prepared["observation/state"].shape == (8,)
    assert prepared["prompt"] == "pick up the object"


def test_pi05_rollout_bundle_returns_env_actions_and_prefix(monkeypatch) -> None:
    import torch

    from dreamervla.workers.inference import pi05_rollout

    class _Policy:
        def __init__(self, **_kwargs) -> None:
            return None

        def eval(self):
            return self

        def to(self, _device):
            return self

        def infer_batch_with_prefix(self, observations):
            batch = len(observations)
            return torch.zeros(batch, 10, 7), torch.zeros(batch, 768, 2048)

    monkeypatch.setattr(pi05_rollout, "Pi05Policy", _Policy)
    bundle = pi05_rollout.Pi05RolloutBundle(model_path="unused", device="cpu")
    result = bundle.predict_batch([{"prompt": "one"}, {"prompt": "two"}])

    assert bundle.actions_are_env_ready is True
    assert len(result) == 2
    assert result[0][0].shape == (10, 7)
    assert result[0][1].shape == (768, 2048)
    assert result[0][1].dtype == torch.float32


def test_pi05_policy_loads_sft_delta_from_run_root(tmp_path) -> None:
    import torch

    from dreamervla.models.embodiment.pi05.policy import Pi05Policy

    policy = Pi05Policy.__new__(Pi05Policy)
    torch.nn.Module.__init__(policy)
    policy.register_parameter("delta", torch.nn.Parameter(torch.zeros(2)))
    checkpoint = tmp_path / "checkpoints" / "latest.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"state_dicts": {"policy": {"delta": torch.ones(2)}}}, checkpoint)

    policy.load_sft_delta(str(tmp_path))

    torch.testing.assert_close(policy.delta, torch.ones(2))


def test_pi05_observation_latent_schema_rejects_non_rlinf_geometry() -> None:
    import pytest

    from dreamervla.runtime.observation_latent import ObservationLatentSpec

    spec = ObservationLatentSpec(
        policy_family="pi05",
        obs_hidden_source="pi05_prefix_output",
        action_head_type="flow_matching",
        token_count=512,
        token_dim=2048,
        chunk_size=10,
        include_state=True,
        prefix_selection="image_only",
        alignment_source="rlinf_rlt_prefix",
    )
    with pytest.raises(ValueError, match="expected 768"):
        spec.preprocess_config()


def test_pi05_prefix_sidecar_round_trip(tmp_path) -> None:
    import h5py

    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
    from dreamervla.runtime.observation_latent import ObservationLatentSpec

    spec = ObservationLatentSpec(
        policy_family="pi05",
        obs_hidden_source="pi05_prefix_output",
        action_head_type="flow_matching",
        token_count=768,
        token_dim=2048,
        chunk_size=10,
        include_state=True,
        prompt_style="openpi",
        rotate_images_180=True,
        prefix_selection="image_only",
        alignment_source="rlinf_rlt_prefix",
    )
    zeros = np.zeros
    step = {
        "actions": zeros(7),
        "rewards": 0.0,
        "sparse_rewards": 0,
        "dones": 1,
        "robot_states": zeros(9),
        "states": zeros(8),
        "obs": {
            "agentview_rgb": zeros((2, 2, 3), dtype=np.uint8),
            "eye_in_hand_rgb": zeros((2, 2, 3), dtype=np.uint8),
            "ee_pos": zeros(3),
            "ee_ori": zeros(3),
            "ee_states": zeros(6),
            "gripper_states": zeros(2),
            "joint_states": zeros(7),
        },
        "obs_embedding": zeros((768, 2048), dtype=np.float16),
    }
    with RolloutDumpWriter(tmp_path / "reward", tmp_path / "latent", "pi05.hdf5") as writer:
        writer.write_demo(0, [step], preprocess_config=spec.preprocess_config())

    with h5py.File(tmp_path / "latent" / "pi05.hdf5", "r") as handle:
        assert handle["data/demo_0/obs_embedding"].shape == (1, 768, 2048)
