from unittest import mock

import pytest

from dreamervla.runners.collect_parallel_rollouts import (
    _REQUIRED_COLLECT_KEYS,
    _assert_gpu_free_memory,
    _assert_policy_mode_matches,
    _collect_vectorized_path,
    _make_preprocess_config,
    _require_keys,
    _resolve_task_ids,
)


def test_gpu_preflight_raises_when_target_gpu_is_occupied():
    # 6 GB free, need 18 GB -> clear, GPU-named error instead of a silent OOM.
    with mock.patch(
        "dreamervla.runners.collect_parallel_rollouts.torch.cuda.mem_get_info",
        return_value=(6 * 1024**3, 80 * 1024**3),
    ):
        with pytest.raises(RuntimeError, match="GPU 3 has only"):
            _assert_gpu_free_memory(3, 18.0, rank=3)


def test_gpu_preflight_passes_with_enough_free_memory():
    with mock.patch(
        "dreamervla.runners.collect_parallel_rollouts.torch.cuda.mem_get_info",
        return_value=(40 * 1024**3, 80 * 1024**3),
    ):
        _assert_gpu_free_memory(0, 18.0, rank=0)  # no raise


def test_gpu_preflight_disabled_when_threshold_zero():
    # min_free_gb=0 disables the check (never calls mem_get_info).
    _assert_gpu_free_memory(0, 0.0, rank=0)


def _discrete_cfg() -> dict:
    # Mirrors what CollectRolloutsRunner._build_collect_cfg produces for the
    # cold-start (discrete / h1 / no-state) task, plus loader-set _policy_mode.
    return {
        "model_path": "data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1",
        "unnorm_key": "libero_goal_no_noops",
        "task_suite_name": "libero_goal",
        "expected_history": 1,
        "num_images_in_input": 2,
        "expected_action_head_type": "oft_discrete_token",
        "expected_include_state": False,
        "expected_obs_hidden_source": "action_query",
        "expected_prompt_style": "vla_policy",
        "expected_rotate_images_180": True,
        "time_horizon": 8,
        "token_dim": 4096,
        "action_dim": 7,
        "chunk_size": 8,
        "resolution": 256,
        # set by _load_policy (auto-detect); discrete => no proprio
        "_policy_mode": "discrete",
        "_use_proprio": False,
    }


def test_make_preprocess_config_is_config_driven_discrete():
    cfg = _discrete_cfg()
    pc = _make_preprocess_config(cfg)
    assert pc["history"] == 1
    assert pc["num_images_in_input"] == 2
    assert pc["action_head_type"] == "oft_discrete_token"
    assert pc["include_state"] is False
    assert pc["time_horizon"] == 8
    assert pc["obs_hidden_source"] == "action_query"
    assert pc["prompt_style"] == "vla_policy"
    assert pc["rotate_images_180"] is True
    assert pc["hidden_key"] == "obs_embedding"


def test_make_preprocess_config_l1_h2_variant():
    cfg = _discrete_cfg()
    cfg.update(
        expected_history=2,
        num_images_in_input=4,
        expected_action_head_type="oft_l1_regression",
        expected_include_state=True,
        _policy_mode="l1",
        _use_proprio=True,
    )
    pc = _make_preprocess_config(cfg)
    assert pc["history"] == 2
    assert pc["num_images_in_input"] == 4
    assert pc["action_head_type"] == "oft_l1_regression"
    assert pc["include_state"] is True


def test_make_preprocess_config_missing_key_raises():
    cfg = _discrete_cfg()
    del cfg["expected_history"]
    with pytest.raises(KeyError):
        _make_preprocess_config(cfg)


def test_assert_policy_mode_matches_ok():
    _assert_policy_mode_matches(_discrete_cfg())  # no raise


def test_assert_policy_mode_matches_head_mismatch():
    cfg = _discrete_cfg()
    cfg["_policy_mode"] = "l1"  # detected L1 but task expects discrete
    with pytest.raises(ValueError):
        _assert_policy_mode_matches(cfg)


def test_assert_policy_mode_matches_state_mismatch():
    cfg = _discrete_cfg()
    cfg["_use_proprio"] = True  # detected proprio but task expects no-state
    with pytest.raises(ValueError):
        _assert_policy_mode_matches(cfg)


def test_require_keys_reports_missing():
    with pytest.raises(KeyError):
        _require_keys({})


def test_require_keys_passes_when_complete():
    cfg = {k: 0 for k in _REQUIRED_COLLECT_KEYS}
    _require_keys(cfg)  # no raise


@pytest.mark.parametrize(
    "task_ids,expected",
    [
        ("all", [0, 1, 2]),
        ("0,2", [0, 2]),
        (1, [1]),
        ([0, 1], [0, 1]),
    ],
)
def test_resolve_task_ids(task_ids, expected):
    assert _resolve_task_ids(task_ids, num_tasks=3) == expected


def test_build_work_list_fresh_collection_starts_at_zero():
    from dreamervla.runners.collect_parallel_rollouts import _build_work_list

    assert _build_work_list([0, 1], 2, {}) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_build_work_list_resume_continues_from_collected_count():
    from dreamervla.runners.collect_parallel_rollouts import _build_work_list

    # task 0 already has 2 episodes -> continue at 2,3; task 1 has 1 -> continue at 1,2.
    # No init_state (episode_id) is re-collected.
    assert _build_work_list([0, 1], 2, {0: 2, 1: 1}) == [(0, 2), (0, 3), (1, 1), (1, 2)]


def test_build_resume_work_list_reaches_target_without_recollecting_complete_ids():
    from dreamervla.runners.collect_parallel_rollouts import _build_resume_work_list

    assert _build_resume_work_list([0], 4, {0: {0, 2}}) == [
        (0, 1),
        (0, 3),
    ]


def test_collect_rollouts_missing_keys_raises_before_gpu():
    from dreamervla.runners.collect_parallel_rollouts import collect_rollouts

    # Empty cfg must fail at _require_keys, before any CUDA/model work.
    with pytest.raises(KeyError):
        collect_rollouts({}, rank=0, world_size=1, local_rank=0)


def test_vectorized_path_threads_obs_hidden_source_to_decoder(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeDecoder:
        def __init__(self, policy, unnorm_key, obs_hidden_source="action_query", image_keys=None):
            captured["obs_hidden_source"] = obs_hidden_source
            captured["image_keys"] = list(image_keys or [])

        def predict_batch(self, preps):
            return []

    class FakeVecEnv:
        def __init__(self, *, num_envs, cfg_kwargs, env_vars):
            self.num_envs = num_envs

        def set_task(self, task_ids, env_ids=None):
            return ["fake task"]

        def close(self):
            return None

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_collect_vectorized(*args, **kwargs):
        return 0

    monkeypatch.setattr(
        "dreamervla.runners.rollout_hidden_extractor.OFTBatchedDecoder",
        FakeDecoder,
    )
    monkeypatch.setattr(
        "dreamervla.runners.vec_rollout_env.VecRolloutEnv",
        FakeVecEnv,
    )
    monkeypatch.setattr(
        "dreamervla.runners.vectorized_collect.collect_vectorized",
        fake_collect_vectorized,
    )
    monkeypatch.setattr(
        "dreamervla.runners.collect_parallel_rollouts._make_dump_writer",
        lambda *args, **kwargs: FakeWriter(),
    )

    _collect_vectorized_path(
        policy=object(),
        extractor=object(),
        unnorm_key="libero_goal_no_noops",
        env_cfg_kwargs={},
        num_envs=1,
        my_work=[(0, 0)],
        episode_horizon=4,
        reward_dir=tmp_path / "reward",
        hidden_dir=tmp_path / "hidden",
        shard_name="shard_000.hdf5",
        shard_prefix="shard",
        shard_idx=0,
        demos_per_shard=0,
        preprocess_config={"chunk_size": 8},
        task_suite_name="libero_goal",
        rank=0,
        world_size=1,
        progress_dir=None,
        history=1,
        rotate_images_180=True,
        image_keys=["agentview_rgb"],
        obs_hidden_source="input_token_embedding",
    )

    assert captured["obs_hidden_source"] == "input_token_embedding"
    assert captured["image_keys"] == ["agentview_rgb"]
