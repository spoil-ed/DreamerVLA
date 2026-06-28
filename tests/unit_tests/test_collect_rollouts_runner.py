import numpy as np
import torch
from omegaconf import OmegaConf


def test_collect_rollouts_experiment_composes_and_validates():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from dreamervla.config import validate_cfg

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=collect_rollouts_onetraj",
                "task=openvla_onetraj_coldstart_libero",
            ],
        )
    OmegaConf.resolve(cfg)
    validate_cfg(cfg)

    assert cfg._target_ == "dreamervla.runners.CollectRolloutsRunner"
    oft = cfg.task.openvla_oft
    assert str(oft.ckpt_path).endswith("Openvla-oft-SFT-libero-goal-traj1")
    assert oft.expected_action_head_type == "oft_discrete_token"
    assert oft.expected_include_state is False
    assert oft.expected_obs_hidden_source == "action_query"
    assert int(oft.expected_history) == 1
    assert int(oft.time_horizon) == 8
    assert str(oft.action_hidden_dir).endswith("_oft_legacy_action_hidden_vla_policy_h1")
    assert "OpenVLA_Onetraj_LIBERO_libero_goal" in str(oft.hdf5_reward_dir)
    assert cfg.collect.envs_per_gpu == 1


def _fake_cfg():
    return OmegaConf.create(
        {
            "_target_": "dreamervla.runners.CollectRolloutsRunner",
            "task": {
                "suite": "libero_goal",
                "action_dim": 7,
                "image_resolution": 256,
                "image_keys": ["agentview_rgb", "eye_in_hand_rgb"],
                "openvla_oft": {
                    "ckpt_path": "data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1",
                    "dataset_statistics_key": "libero_goal_no_noops",
                    "hdf5_reward_dir": "data/processed_data/X/no_noops_t_256_remaining_reward",
                    "action_hidden_dir": "data/processed_data/X/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h1",
                    "expected_action_head_type": "oft_discrete_token",
                    "expected_include_state": False,
                    "expected_obs_hidden_source": "input_token_embedding",
                    "expected_prompt_style": "vla_policy",
                    "expected_rotate_images_180": True,
                    "expected_history": 1,
                    "time_horizon": 8,
                    "token_dim": 4096,
                    "chunk_size": 8,
                },
            },
            "collect": {
                "policy_mode": "auto",
                "task_ids": "all",
                "episodes_per_task": 2,
                "episode_horizon": 64,
                "envs_per_gpu": 1,
                "memory_fraction": 0.7,
            },
        }
    )


def test_build_collect_cfg_maps_task_and_collect():
    from dreamervla.runners import CollectRolloutsRunner

    runner = CollectRolloutsRunner(_fake_cfg())
    cc = runner._build_collect_cfg()

    assert cc["model_path"].endswith("Openvla-oft-SFT-libero-goal-traj1")
    assert cc["unnorm_key"] == "libero_goal_no_noops"
    assert cc["reward_dir"].endswith("no_noops_t_256_remaining_reward")
    assert cc["hidden_dir"].endswith("_oft_legacy_action_hidden_vla_policy_h1")
    assert cc["expected_history"] == 1
    # OFT single-view default from the central collect config (NOT len(image_keys)=2):
    # the checkpoint does not persist num_images_in_input and the discrete VLA wants 1.
    assert cc["num_images_in_input"] == 1
    assert cc["expected_action_head_type"] == "oft_discrete_token"
    assert cc["expected_include_state"] is False
    assert cc["time_horizon"] == 8
    assert cc["resolution"] == 256  # task.image_resolution, NOT image_size
    assert cc["task_suite_name"] == "libero_goal"
    assert cc["task_ids"] == "all"
    assert cc["envs_per_gpu"] == 1
    assert cc["memory_fraction"] == _fake_cfg().collect.memory_fraction
    assert cc["demos_per_shard"] == 0  # default: one shard per rank
    # every required key present
    from dreamervla.runners.collect_parallel_rollouts import _require_keys
    _require_keys(cc)


def test_build_collect_cfg_forwards_demos_per_shard():
    from dreamervla.runners import CollectRolloutsRunner

    cfg = _fake_cfg()
    cfg.collect.demos_per_shard = 25
    cc = CollectRolloutsRunner(cfg)._build_collect_cfg()
    assert cc["demos_per_shard"] == 25


def test_build_collect_cfg_forwards_num_inference_workers():
    from dreamervla.runners import CollectRolloutsRunner

    cfg = _fake_cfg()
    cfg.collect.num_inference_workers = 2
    cc = CollectRolloutsRunner(cfg)._build_collect_cfg()
    assert cc["num_inference_workers"] == 2


def _record(t: int) -> dict:
    return {
        "agentview_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "eye_in_hand_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "ee_pos": np.array([0, 0, t], dtype=np.float64),
        "ee_ori": np.zeros(3, dtype=np.float64),
        "ee_states": np.zeros(6, dtype=np.float64),
        "gripper_states": np.zeros(2, dtype=np.float64),
        "joint_states": np.zeros(7, dtype=np.float64),
        "robot_states": np.zeros(9, dtype=np.float64),
        "states": np.zeros(11, dtype=np.float64),
    }


class _EpisodeEnv:
    def __init__(self, *, done_after: int):
        self.t = 0
        self.done_after = done_after
        self.actions: list[np.ndarray] = []

    def reset(self, *, episode_id: int, task_id: int):
        self.t = 0
        return {}, {}

    def full_record(self):
        return _record(self.t)

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.t += 1
        done = self.t >= self.done_after
        return {}, 0.0, done, False, {
            "success": done,
            "wm_action": np.asarray(action, dtype=np.float64),
        }


class _EpisodeExtractor:
    def __init__(self):
        self.calls = 0

    def reset(self):
        pass

    def step(self, obs, task_description):
        base = self.calls * 10
        self.calls += 1
        return (
            [np.array([base + j, 0, 0, 0, 0, 0, 0.9], dtype=np.float64) for j in range(3)],
            torch.zeros(16, dtype=torch.float16),
        )


class _DecodeResult:
    def __init__(self, action_chunk, hidden_state, lang_emb=None):
        self.action_chunk = action_chunk
        self.hidden_state = hidden_state
        self.lang_emb = lang_emb

    def __iter__(self):
        yield self.action_chunk
        yield self.hidden_state


class _LangEpisodeExtractor(_EpisodeExtractor):
    def step(self, obs, task_description):
        action_chunk, hidden_state = super().step(obs, task_description)
        return _DecodeResult(
            action_chunk,
            hidden_state,
            lang_emb=np.full(8, float(self.calls), dtype=np.float32),
        )


def test_single_episode_executes_action_chunk_open_loop(monkeypatch):
    import dreamervla.runners.collect_parallel_rollouts as mod
    import dreamervla.runners.oft_collect_common as occ

    # process_action is now applied inside the shared oft_open_loop_action; patch it
    # at its real call site so the open-loop action SEQUENCE check stays independent
    # of the gripper transform.
    monkeypatch.setattr(occ, "process_action", lambda action: np.asarray(action, dtype=np.float64))
    env = _EpisodeEnv(done_after=5)
    extractor = _EpisodeExtractor()

    steps = mod._run_episode(
        env=env,
        extractor=extractor,
        task_description="task0",
        episode_id=0,
        episode_horizon=5,
        task_id=0,
        rank=0,
        action_steps=3,
    )

    assert len(steps) == 5
    assert [int(action[0]) for action in env.actions] == [0, 1, 2, 30, 31]


def test_single_episode_records_language_embedding(monkeypatch):
    import dreamervla.runners.collect_parallel_rollouts as mod
    import dreamervla.runners.oft_collect_common as occ

    monkeypatch.setattr(occ, "process_action", lambda action: np.asarray(action, dtype=np.float64))
    env = _EpisodeEnv(done_after=2)
    extractor = _LangEpisodeExtractor()

    steps = mod._run_episode(
        env=env,
        extractor=extractor,
        task_description="task0",
        episode_id=0,
        episode_horizon=2,
        task_id=0,
        rank=0,
        action_steps=1,
    )

    assert np.array_equal(steps[0]["lang_emb"], np.full(8, 1.0, dtype=np.float32))
