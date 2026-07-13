from __future__ import annotations


def test_oft_collect_common_exposes_shared_helpers() -> None:
    from dreamervla.runners.oft_collect_common import (
        assert_policy_mode_matches,
        load_policy,
        make_preprocess_config,
        resolve_num_images_in_input,
    )

    for fn in (
        load_policy,
        make_preprocess_config,
        assert_policy_mode_matches,
        resolve_num_images_in_input,
    ):
        assert callable(fn)


def test_oft_collect_policy_device_accepts_cpu_sentinel() -> None:
    import torch

    from dreamervla.runners.oft_collect_common import _policy_device_from_id

    assert _policy_device_from_id(-1) == torch.device("cpu")
    assert _policy_device_from_id("cpu") == torch.device("cpu")
    assert _policy_device_from_id(0) == torch.device("cuda:0")
    assert _policy_device_from_id("cuda:2") == torch.device("cuda:2")


def test_vla_hidden_token_spec_derives_loaded_policy_geometry() -> None:
    from dreamervla.runners.oft_collect_common import vla_hidden_token_spec

    class _VisionBackbone:
        per_image = 128
        views = 1

        def get_num_patches(self) -> int:
            return self.per_image

        def get_num_images_in_input(self) -> int:
            return self.views

    class _VLA:
        vision_backbone = _VisionBackbone()
        token_dim = 1024

    spec = vla_hidden_token_spec(_VLA(), ["agentview_rgb"])
    assert spec["per_image"] == _VisionBackbone.per_image
    assert spec["patches_per_image"] == _VisionBackbone.per_image
    assert spec["views"] == _VisionBackbone.views
    assert spec["num_images_in_input"] == _VisionBackbone.views
    assert spec["token_dim"] == _VLA.token_dim
    assert spec["token_count"] == _VisionBackbone.per_image * _VisionBackbone.views
    assert spec["flat_dim"] == spec["token_count"] * _VLA.token_dim

    import pytest

    with pytest.raises(ValueError, match="image_keys"):
        vla_hidden_token_spec(_VLA(), ["agentview_rgb", "eye_in_hand_rgb"])


def test_runner_builds_bundle_cfg_from_central_config(tmp_path) -> None:
    from dreamervla.runners.cold_start_ray_collect_runner import ColdStartRayCollectRunner

    cfg = {
        "mode": "oft",
        "collect": {
            "num_images_in_input": 1,
            "episode_horizon": 8,
            "envs_per_gpu": 2,
            "memory_fraction": 0.8,
            "episodes_per_task": 2,
            "task_ids": [0],
            "policy_mode": "discrete",
        },
        "task": {
            "suite": "libero_goal",
            "action_dim": 7,
            "image_resolution": 256,
            "image_keys": ["agentview_rgb"],
            "openvla_oft": {
                "ckpt_path": str(tmp_path / "ckpt"),
                "dataset_statistics_key": "libero_goal_no_noops",
                "hdf5_reward_dir": str(tmp_path / "reward"),
                "hidden_token_dir": str(tmp_path / "hidden"),
                "hidden_token": {
                    "expected_action_head_type": "oft_discrete_token",
                    "expected_include_state": False,
                    "expected_obs_hidden_source": "hidden_token",
                    "expected_prompt_style": "vla_policy",
                    "expected_history": 1,
                    "expected_rotate_images_180": True,
                    "time_horizon": 8,
                    "token_count": 256,
                    "token_dim": 4096,
                    "wm_obs_dim": 1_048_576,
                    "patches_per_image": 256,
                    "chunk_size": 8,
                },
            },
        },
    }
    plan = ColdStartRayCollectRunner(cfg).build_oft_worker_plan()
    assert plan["inference"]["decoder"]["target"].endswith("oft_rollout:OFTRolloutBundle")
    assert (
        plan["inference"]["action_steps"]
        == cfg["task"]["openvla_oft"]["hidden_token"]["chunk_size"]
    )
    assert plan["inference"]["decoder"]["kwargs"]["history"] == 1
    assert (
        plan["inference"]["decoder"]["kwargs"]["obs_hidden_source"]
        == "hidden_token"
    )
    assert plan["inference"]["decoder"]["kwargs"]["image_keys"] == ["agentview_rgb"]
    env_kwargs = plan["env"]["kwargs"]
    assert env_kwargs["history_length"] == 1
    assert env_kwargs["include_state"] is False
    assert env_kwargs["action_head_type"] == "oft_discrete_token"
    assert env_kwargs["validate_canonical"] is False
    assert plan["dump"]["preprocess_config"]["hidden_key"] == "obs_embedding"
    assert plan["dump"]["preprocess_config"]["action_head_type"] == "oft_discrete_token"
    assert (
        plan["dump"]["preprocess_config"]["obs_hidden_source"]
        == "hidden_token"
    )
    assert plan["dump"]["preprocess_config"]["num_images_in_input"] == 1


def test_oft_collect_plan_respects_cpu_inference_device_override(tmp_path) -> None:
    from dreamervla.runners.cold_start_ray_collect_runner import ColdStartRayCollectRunner

    cfg = {
        "mode": "oft",
        "inference": {"device": "cpu"},
        "collect": {
            "num_images_in_input": 1,
            "episode_horizon": 8,
            "envs_per_gpu": 1,
            "memory_fraction": 0.8,
            "episodes_per_task": 1,
            "task_ids": [0],
            "policy_mode": "discrete",
        },
        "task": {
            "suite": "libero_goal",
            "action_dim": 7,
            "image_resolution": 256,
            "image_keys": ["agentview_rgb"],
            "openvla_oft": {
                "ckpt_path": str(tmp_path / "ckpt"),
                "dataset_statistics_key": "libero_goal_no_noops",
                "hdf5_reward_dir": str(tmp_path / "reward"),
                "hidden_token_dir": str(tmp_path / "hidden"),
                "hidden_token": {
                    "expected_action_head_type": "oft_discrete_token",
                    "expected_include_state": False,
                    "expected_obs_hidden_source": "hidden_token",
                    "expected_prompt_style": "vla_policy",
                    "expected_history": 1,
                    "expected_rotate_images_180": True,
                    "time_horizon": 8,
                    "token_count": 256,
                    "token_dim": 4096,
                    "wm_obs_dim": 1_048_576,
                    "patches_per_image": 256,
                    "chunk_size": 8,
                },
            },
        },
    }

    plan = ColdStartRayCollectRunner(cfg).build_oft_worker_plan()

    assert plan["inference"]["device"] == "cpu"
    assert plan["inference"]["decoder"]["kwargs"]["device"] == "cpu"


def test_collect_rollouts_ray_experiment_composes() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=collect_rollouts_ray"])

    assert cfg._target_.endswith("ColdStartRayCollectRunner")
    assert cfg.mode == "oft"
    assert cfg.collect.num_images_in_input == 1


def test_collect_egl_render_pool_defaults_to_non_inference_gpus() -> None:
    from dreamervla.runners.cold_start_ray_collect_runner import (
        _ensure_collect_render_device_pool,
    )

    class _Cluster:
        num_gpus = 4

    env_cfg = {"render_backend": "egl"}

    resolved = _ensure_collect_render_device_pool(
        env_cfg,
        _Cluster(),
        collect_cfg={"gpu_id": 0, "num_inference_workers": 2},
    )

    assert resolved["render_devices"] == [2, 3]
    assert "render_devices" not in env_cfg


def test_collect_egl_render_pool_rejects_inference_overlap_without_spare_gpu() -> None:
    import pytest

    from dreamervla.runners.cold_start_ray_collect_runner import (
        _ensure_collect_render_device_pool,
    )

    class _Cluster:
        num_gpus = 1

    with pytest.raises(ValueError, match="render_backend=osmesa"):
        _ensure_collect_render_device_pool(
            {"render_backend": "egl"},
            _Cluster(),
            collect_cfg={"gpu_id": 0, "num_inference_workers": 1},
        )


def test_collect_egl_render_pool_preserves_explicit_pool() -> None:
    from dreamervla.runners.cold_start_ray_collect_runner import (
        _ensure_collect_render_device_pool,
    )

    class _Cluster:
        num_gpus = 3

    env_cfg = {"render_backend": "egl", "render_devices": [7]}

    resolved = _ensure_collect_render_device_pool(
        env_cfg,
        _Cluster(),
        collect_cfg={"gpu_id": 0, "num_inference_workers": 1},
    )

    assert resolved["render_devices"] == [7]


def test_ray_task_scheduler_expands_all_and_reserves_round_robin() -> None:
    from dreamervla.runners.cold_start_ray_collect_runner import (
        _next_ray_task_id,
        _resolve_ray_task_ids,
    )

    task_ids = _resolve_ray_task_ids("all", num_tasks=3, suite="libero_goal")
    assert task_ids == [0, 1, 2]

    counts = {task_id: 0 for task_id in task_ids}
    assigned = [
        _next_ray_task_id(task_ids, counts, episodes_per_task=2)
        for _ in range(7)
    ]
    # (task_id, scheduled_index): each task's index advances 0 then 1 across the rounds.
    assert assigned == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), None]
    assert counts == {0: 2, 1: 2, 2: 2}


def test_ray_start_episode_id_adds_resume_offset_to_scheduled_index() -> None:
    from dreamervla.runners.cold_start_ray_collect_runner import _ray_start_episode_id

    # fresh: episode_id == scheduled index (distinct init_states 0,1,2,...)
    assert [_ray_start_episode_id({}, 0, i) for i in range(3)] == [0, 1, 2]
    # resume: task 0 already has 5 on disk -> continue at 5,6,7 (no init_state re-collect)
    assert [_ray_start_episode_id({0: 5}, 0, i) for i in range(3)] == [5, 6, 7]


def test_ray_dump_step_records_episode_id() -> None:
    from dreamervla.runners.cold_start_ray_collect_runner import _build_oft_dump_step

    class _Env:
        def full_record(self):
            import numpy as np

            return {
                "agentview_rgb": np.zeros((256, 256, 3), "uint8"),
                "eye_in_hand_rgb": np.zeros((256, 256, 3), "uint8"),
                "ee_pos": np.zeros(3), "ee_ori": np.zeros(3), "ee_states": np.zeros(6),
                "gripper_states": np.zeros(2), "joint_states": np.zeros(7),
                "robot_states": np.zeros(9), "states": np.zeros(45),
            }

    import numpy as np

    step = _build_oft_dump_step(
        _Env(), {}, np.ones(7), 0.0, True, False,
        {"task_id": 1, "episode_id": 4, "init_state_index": 4, "success": True},
        np.zeros(8, "float16"),
    )
    assert step["task_id"] == 1
    assert step["episode_id"] == 4
    assert step["init_state_index"] == 4


def test_wait_worker_results_batches_ray_get(monkeypatch) -> None:
    import ray

    from dreamervla.runners.cold_start_ray_collect_runner import _wait_worker_results

    calls = []

    class _Result:
        def __init__(self, *refs) -> None:
            self.refs = list(refs)

    def fake_get(refs):
        calls.append(list(refs))
        return [f"value:{ref}" for ref in refs]

    monkeypatch.setattr(ray, "get", fake_get)

    out = _wait_worker_results([_Result("a"), _Result("b", "c")])

    assert out == ["value:a", "value:b", "value:c"]
    assert calls == [["a", "b", "c"]]
