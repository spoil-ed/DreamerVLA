from __future__ import annotations

import numpy as np
import torch


def test_online_cotrain_runner_has_extracted_methods():
    from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner

    assert hasattr(OnlineCotrainRunner, "_build_components")
    assert hasattr(OnlineCotrainRunner, "_online_cotrain_loop")


def test_online_cotrain_actor_update_uses_registry():
    import inspect

    import dreamervla.runners.online_cotrain_runner as mod

    loop_src = inspect.getsource(mod.OnlineCotrainRunner._online_cotrain_loop)
    burst_src = inspect.getsource(mod.OnlineCotrainRunner._run_training_bursts)

    assert "get_actor_update_route" in loop_src
    assert "actor_update_route.step_fn" in burst_src
    assert "dino_lumos_step(" not in burst_src
    assert "build_rollout_progress_metrics" in burst_src
    helper_src = inspect.getsource(mod.build_rollout_progress_metrics)
    assert "rollout/episodes" in helper_src
    assert "rollout/successes" in helper_src
    assert "rollout/success_rate_valid" in helper_src


def test_trainable_classifier_preserves_hydra_target(monkeypatch):
    from omegaconf import OmegaConf

    import dreamervla.runners.online_cotrain_runner as mod

    runner = mod.OnlineCotrainRunner.__new__(mod.OnlineCotrainRunner)
    runner.device = torch.device("cpu")
    runner.distributed = _FakeDistributed()

    monkeypatch.setattr(
        mod,
        "build_classifier",
        lambda cfg: torch.nn.Linear(int(cfg["latent_dim"]), 2),
    )
    monkeypatch.setattr(mod, "build_optimizer", lambda module, cfg: object())
    cfg = OmegaConf.create(
        {
            "algorithm": {"lumos": {"classifier_threshold": 0.5}},
            "classifier": {
                "_target_": "tests.fake.CustomSuccessVerifier",
                "latent_dim": 3,
                "window": 2,
            },
            "world_model": {"obs_dim": 3},
            "init": {"classifier_state_ckpt": None},
            "optim": {"classifier": {"lr": 1.0e-4}},
        }
    )

    runner._build_trainable_classifier(cfg)

    assert runner._classifier_target == "tests.fake.CustomSuccessVerifier"
    assert runner._classifier_cls_kwargs == {"latent_dim": 3, "window": 2}


def test_classifier_warmup_hf_sidecar_uses_config_target(tmp_path, monkeypatch):
    import dreamervla.runners.online_cotrain_pipeline_runner as mod

    captured: dict[str, object] = {}
    runner = mod.OnlineCotrainPipelineRunner.__new__(mod.OnlineCotrainPipelineRunner)
    runner.global_step = 0
    runner.classifier = torch.nn.Linear(3, 2)
    runner.classifier_threshold = 0.5
    runner._classifier_target = "tests.fake.CustomSuccessVerifier"
    runner._classifier_cls_kwargs = {"latent_dim": 3, "window": 2}
    runner.checkpoint_save_torch = lambda: False
    runner.checkpoint_save_hf = lambda: True
    runner._cls_warmup_hf_dir = lambda: tmp_path / "classifier_hf"

    def fake_save(module, save_dir, *, target, init_args):
        captured["module"] = module
        captured["save_dir"] = save_dir
        captured["target"] = target
        captured["init_args"] = init_args

    monkeypatch.setattr(mod, "save_module_pretrained", fake_save)

    runner._save_cls_warmup()

    assert captured["target"] == "tests.fake.CustomSuccessVerifier"
    assert captured["init_args"] == {"latent_dim": 3, "window": 2}


# --------------------------------------------------------------------------
# Local fixture helpers (kept local on purpose: do NOT import from
# tests.runners.test_offline_seed — that path is fragile / wrong). The shape
# below mirrors what the collector / RolloutDumpWriter writes.
# --------------------------------------------------------------------------
def _demo_steps(T, success, emb_dim=16):
    steps = []
    for t in range(T):
        steps.append({
            "actions": np.full(7, t, np.float64),
            "rewards": np.float32(0.0),
            "sparse_rewards": np.uint8(1 if (success and t == T - 1) else 0),
            "dones": np.uint8(1 if t == T - 1 else 0),
            "robot_states": np.zeros(9, np.float64),
            "states": np.zeros(5, np.float64),
            "obs": {
                "agentview_rgb": np.zeros((256, 256, 3), np.uint8),
                "eye_in_hand_rgb": np.zeros((256, 256, 3), np.uint8),
                "ee_pos": np.zeros(3, np.float64), "ee_ori": np.zeros(3, np.float64),
                "ee_states": np.zeros(6, np.float64), "gripper_states": np.zeros(2, np.float64),
                "joint_states": np.zeros(7, np.float64),
            },
            "obs_embedding": np.full(emb_dim, t, np.float16),
        })
    return steps


def _seeded_replay(tmp_path, emb_dim=16, seq_len=4):
    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
    from dreamervla.runners.offline_seed import seed_replay_from_offline
    from dreamervla.runners.online_replay import OnlineReplay

    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    with RolloutDumpWriter(rdir, hdir, "r0_shard.hdf5") as w:
        for i in range(4):
            w.write_demo(index=i, steps=_demo_steps(8, success=(i % 2 == 0), emb_dim=emb_dim),
                         task_id=0, episode_id=i)
    replay = OnlineReplay(capacity=10_000, sequence_length=seq_len, task_ids=(0,), rank=0)
    seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir, default_task_id=0)
    return replay


def test_offline_warmup_steps_update_modules(tmp_path, monkeypatch):
    # Use a fake WM/classifier + recording step fns to assert the warmup loops
    # call the existing step functions N times against the seeded buffer.
    import dreamervla.runners.online_cotrain_pipeline_runner as mod

    replay = _seeded_replay(tmp_path)
    calls = {"wm": 0, "cls": 0}

    def fake_wm_step(**kw):
        assert kw["batch"] is not None
        calls["wm"] += 1
        return {"loss": 0.1}

    def fake_cls_step(**kw):
        assert kw["replay"] is replay
        calls["cls"] += 1
        return {"loss": 0.2, "acc": 0.5, "f1": 0.0}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)
    monkeypatch.setattr(mod, "online_classifier_update_step", fake_cls_step)

    runner = mod.OnlineCotrainPipelineRunner.__new__(mod.OnlineCotrainPipelineRunner)
    runner.device = torch.device("cpu")
    runner.global_step = 0
    runner._build_wm_pretrain_batch = lambda b: {
        "images": torch.zeros(1), "obs_embedding": torch.zeros(1), "actions": torch.zeros(1)
    }
    # world_model needs .train() (warmup puts it in train mode, like the online
    # loop does); the step fns themselves are faked so these are otherwise inert.
    runner.world_model = torch.nn.Module()
    runner.world_model_optimizer = object()
    runner.policy = object()
    runner.classifier = torch.nn.Module()
    runner.classifier_optimizer = object()
    runner._cls_window = 4

    runner._offline_warmup_wm(replay, steps=3, batch_size=2, optim_cfg=None)
    runner._offline_warmup_classifier(replay, steps=5, batch_size=2, early_neg_stride=8, grad_clip=1.0)
    assert calls == {"wm": 3, "cls": 5}


# --------------------------------------------------------------------------
# run() orchestration tests — no models / no LIBERO. We monkeypatch the heavy
# pieces (component build, seeding, warmup loops, online loop) and assert run()
# wires them together in the right order and writes the split warmup ckpts.
# --------------------------------------------------------------------------
def _orchestration_cfg(tmp_path, *, resume=False, total_env_steps=None):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "training": {
            "out_dir": str(tmp_path),
            "debug": True,
            "resume": resume,
            # orchestration uses fake nn.Linear modules; HF export needs real
            # target/init_args, so pin torch-only (the test asserts the .ckpt).
            "checkpoint_format": "torch",
        },
        "offline_warmup": {
            "data_dir": str(tmp_path / "offline_data"),
            "hidden_dir": str(tmp_path / "offline_hidden"),
            "debug_wm_warmup_steps": 2,
            "debug_classifier_warmup_steps": 2,
        },
        # run() reads optim.grad_clip_norm; the real config always supplies optim.
        "optim": {"grad_clip_norm": 1.0},
    })
    if total_env_steps is not None:
        OmegaConf.update(cfg, "online_rollout.total_env_steps", int(total_env_steps), force_add=True)
        # cfg sets debug=True, so _apply_debug_overrides swaps in the debug knob;
        # pin it too or the requested step count is overwritten by the fallback.
        OmegaConf.update(cfg, "online_rollout.debug_total_env_steps", int(total_env_steps), force_add=True)
    return cfg


class _FakeDistributed:
    rank = 0
    world_size = 1
    is_main_process = True

    def wrap_trainable_module(self, module):
        return module


def _make_orchestration_runner(
    tmp_path,
    monkeypatch,
    calls,
    *,
    resume=False,
    total_env_steps=None,
    cfg_updates=None,
    seed_capture=None,
):
    import dreamervla.runners.online_cotrain_pipeline_runner as mod

    runner = mod.OnlineCotrainPipelineRunner.__new__(mod.OnlineCotrainPipelineRunner)
    runner.cfg = _orchestration_cfg(tmp_path, resume=resume, total_env_steps=total_env_steps)
    if cfg_updates:
        from omegaconf import OmegaConf

        for key, value in cfg_updates.items():
            OmegaConf.update(runner.cfg, key, value, force_add=True)
    runner.config = runner.cfg
    runner._output_dir = str(tmp_path)
    runner.global_step = 0
    runner.distributed = _FakeDistributed()
    runner.device = torch.device("cpu")

    # run() now identifies the collected dump before loading models; create a minimal
    # shard in each offline dir so that existence check passes (seeding itself is faked).
    for sub in ("offline_data", "offline_hidden"):
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "shard_000.hdf5").touch()

    def fake_build_components(self, cfg):
        calls.append("build")
        self.world_model = torch.nn.Linear(2, 2)
        self.policy = None
        self.critic = None
        self.classifier = torch.nn.Linear(2, 2)
        self.classifier_threshold = 0.5

    def fake_seed(
        replay,
        *,
        data_dir,
        hidden_dir,
        default_task_id=None,
        max_episodes_per_task=None,
    ):
        calls.append("seed")
        if seed_capture is not None:
            seed_capture["capacity_mode"] = replay.capacity_mode
            seed_capture["capacity"] = replay.capacity
            seed_capture["max_episodes_per_task"] = max_episodes_per_task
        # add a tiny real episode so num_transitions > 0 (run() guards on it)
        episode = [
            {"image": np.zeros((4, 4, 3), np.uint8), "obs_embedding": np.zeros(8, np.float32),
             "reward": 0.0, "done": 0.0, "is_last": 0.0, "is_terminal": 0.0,
             "wm_action": np.zeros(7, np.float32), "task_id": 0, "success": True}
            for _ in range(replay.sequence_length + 1)
        ]
        replay.add_episode(episode)
        return 1

    def fake_wm_warmup(self, replay, *, steps, batch_size, optim_cfg):
        calls.append("wm_warmup")
        return 0.0  # run() formats the returned loss into the warmup banner

    def fake_cls_warmup(self, replay, *, steps, batch_size, early_neg_stride, grad_clip):
        calls.append("cls_warmup")
        return 0.0  # run() formats the returned acc into the warmup banner

    def fake_online_loop(self, cfg):
        calls.append("online")
        return []

    monkeypatch.setattr(mod.OnlineCotrainPipelineRunner, "_build_components", fake_build_components)
    monkeypatch.setattr(mod, "seed_replay_from_offline", fake_seed)
    monkeypatch.setattr(mod.OnlineCotrainPipelineRunner, "_offline_warmup_wm", fake_wm_warmup)
    monkeypatch.setattr(mod.OnlineCotrainPipelineRunner, "_offline_warmup_classifier", fake_cls_warmup)
    monkeypatch.setattr(mod.OnlineCotrainPipelineRunner, "_online_cotrain_loop", fake_online_loop)
    # wrap _save_* so we can record their order while still writing the files
    real_save_wm = mod.OnlineCotrainPipelineRunner._save_wm_warmup
    real_save_cls = mod.OnlineCotrainPipelineRunner._save_cls_warmup

    def save_wm(self):
        calls.append("save_wm")
        real_save_wm(self)

    def save_cls(self):
        calls.append("save_cls")
        real_save_cls(self)

    monkeypatch.setattr(mod.OnlineCotrainPipelineRunner, "_save_wm_warmup", save_wm)
    monkeypatch.setattr(mod.OnlineCotrainPipelineRunner, "_save_cls_warmup", save_cls)
    return runner


def test_run_orchestrates_seed_warmup_split_ckpt_online(tmp_path, monkeypatch):
    import os

    calls: list[str] = []
    runner = _make_orchestration_runner(tmp_path, monkeypatch, calls, total_env_steps=1)

    history = runner.run()

    assert history == []
    # order: build -> seed -> wm_warmup -> save_wm -> cls_warmup -> save_cls -> online
    assert calls == [
        "build", "seed", "wm_warmup", "save_wm", "cls_warmup", "save_cls", "online"
    ]
    assert os.path.exists(os.path.join(str(tmp_path), "ckpt", "wm_warmup.ckpt"))
    assert os.path.exists(os.path.join(str(tmp_path), "ckpt", "classifier_warmup.ckpt"))


def test_run_passes_replay_capacity_mode_and_seed_cap_from_hydra(tmp_path, monkeypatch):
    calls: list[str] = []
    seed_capture: dict[str, object] = {}
    runner = _make_orchestration_runner(
        tmp_path,
        monkeypatch,
        calls,
        total_env_steps=0,
        cfg_updates={
            "online_rollout.buffer_size": 321,
            "online_rollout.replay_capacity_mode": "total_sharded",
            "offline_warmup.max_episodes_per_task": 7,
        },
        seed_capture=seed_capture,
    )

    runner.run()

    assert seed_capture == {
        "capacity_mode": "total_sharded",
        "capacity": 321,
        "max_episodes_per_task": 7,
    }


def test_run_fails_fast_when_collected_dump_missing(tmp_path, monkeypatch):
    """No collected shards + no warmup-ckpt resume -> identify-before-load error,
    raised BEFORE _build_components loads the heavy WM/encoder/classifier."""
    import shutil

    import pytest

    calls: list[str] = []
    runner = _make_orchestration_runner(tmp_path, monkeypatch, calls, total_env_steps=1)
    # Remove the offline dirs the helper created so the dump looks un-collected.
    for sub in ("offline_data", "offline_hidden"):
        shutil.rmtree(tmp_path / sub)

    with pytest.raises(FileNotFoundError, match="cold-start collection"):
        runner.run()
    assert "build" not in calls  # never reached the model load


def test_run_stops_after_warmup_when_total_env_steps_zero(tmp_path, monkeypatch):
    import os

    calls: list[str] = []
    runner = _make_orchestration_runner(tmp_path, monkeypatch, calls, total_env_steps=0)

    history = runner.run()

    assert history == []
    assert calls == [
        "build", "seed", "wm_warmup", "save_wm", "cls_warmup", "save_cls"
    ]
    assert os.path.exists(os.path.join(str(tmp_path), "ckpt", "wm_warmup.ckpt"))
    assert os.path.exists(os.path.join(str(tmp_path), "ckpt", "classifier_warmup.ckpt"))


def test_warmup_only_component_build_skips_rollout_encoder(monkeypatch):
    from omegaconf import OmegaConf

    import dreamervla.runners.online_cotrain_runner as mod

    runner = mod.OnlineCotrainRunner.__new__(mod.OnlineCotrainRunner)
    runner.device = torch.device("cpu")
    runner.distributed = _FakeDistributed()
    calls: list[str] = []

    def fail_if_encoder_cfg(_cfg):
        raise AssertionError("warmup-only pipeline must not build the rollout encoder")

    def fake_instantiate(cfg):
        calls.append(str(OmegaConf.select(cfg, "_target_", default="unknown")))
        return torch.nn.Linear(2, 2)

    def fake_build_classifier(self, cfg):
        self.classifier = torch.nn.Linear(2, 2)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.1)
        self.classifier_threshold = 0.5
        self._cls_window = 4

    monkeypatch.setattr(runner, "_build_frozen_encoder_cfg", fail_if_encoder_cfg)
    monkeypatch.setattr(mod.hydra.utils, "instantiate", fake_instantiate)
    monkeypatch.setattr(mod.OnlineCotrainRunner, "_build_trainable_classifier", fake_build_classifier)

    cfg = OmegaConf.create({
        "online_rollout": {"total_env_steps": 0},
        "world_model": {"_target_": "world_model"},
        "policy": {"_target_": "policy"},
        "critic": {"_target_": "critic"},
        "algorithm": {},
        "optim": {
            "world_model": {"name": "adam", "lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8},
            "policy": {"name": "adam", "lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8},
            "critic": {"name": "adam", "lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8},
        },
        "init": {"world_model_state_ckpt": None},
    })

    runner._build_components(cfg)

    assert runner.encoder is None
    assert runner.processor is None
    assert calls == ["world_model", "policy", "critic"]


def test_online_cotrain_env_preserves_input_token_discrete_contract(monkeypatch):
    from omegaconf import OmegaConf

    import dreamervla.runners.online_cotrain_runner as mod

    runner = mod.OnlineCotrainRunner.__new__(mod.OnlineCotrainRunner)
    runner.distributed = _FakeDistributed()
    captured: dict[str, object] = {}

    def fake_env_factory(cfg):
        captured.update(dict(cfg))
        return object()

    monkeypatch.setattr(mod, "default_env_factory", fake_env_factory)

    cfg = OmegaConf.create(
        {
            "seed": 7,
            "env": {
                "_target_": "tests.fake.Env",
                "task_suite_name": "libero_goal",
                "task_ids": [0],
                "episode_horizon": 64,
                "history_length": 1,
                "include_state": False,
                "vla_rotate_180": True,
                "obs_hidden_source": "input_token_embedding",
                "action_head_type": "oft_discrete_token",
            },
        }
    )

    runner._build_env(cfg)

    assert captured["_target_"] == "tests.fake.Env"
    assert captured["obs_hidden_source"] == "input_token_embedding"
    assert captured["action_head_type"] == "oft_discrete_token"
    assert captured["history_length"] == 1
    assert captured["include_state"] is False


def test_online_env_validation_accepts_oft_discrete_input_token_contract():
    from dreamervla.envs.train_env import (
        DreamerVLAOnlineTrainEnv,
        DreamerVLAOnlineTrainEnvConfig,
    )

    env = DreamerVLAOnlineTrainEnv.__new__(DreamerVLAOnlineTrainEnv)
    env.cfg = DreamerVLAOnlineTrainEnvConfig(
        history_length=1,
        include_state=False,
        obs_hidden_source="input_token_embedding",
        action_head_type="oft_discrete_token",
    )

    env._validate_canonical_config()


def test_backbone_rollout_uses_oft_input_token_extractor(monkeypatch):
    import dreamervla.runners.online_cotrain_runner as mod

    runner = mod.OnlineCotrainRunner.__new__(mod.OnlineCotrainRunner)
    runner.device = torch.device("cpu")
    runner._latent_type = "backbone_latent"
    calls: list[str] = []

    class FakeExtractor:
        def reset(self):
            calls.append("reset")

        def step(self, obs, task_description):
            calls.append(f"step:{task_description}")
            return [], torch.arange(4, dtype=torch.float16)

    class FakeWorldModel:
        def __call__(self, batch):
            if batch["mode"] == "encode_latent":
                calls.append(f"encode:{tuple(batch['hidden'].shape)}")
                return {"hidden": batch["hidden"]}
            if batch["mode"] == "actor_input":
                calls.append("actor_input")
                return torch.zeros(1, 6)
            raise AssertionError(batch["mode"])

    class FakePolicy:
        def __call__(self, batch):
            calls.append(f"policy:{tuple(batch['hidden'].shape)}")
            return torch.zeros(1, 2, 7), torch.zeros(1), {}

    def fail_rynn_input_token(*_args, **_kwargs):
        raise AssertionError("OFT backbone rollout must not call Rynn input-token extraction")

    runner._oft_input_token_extractor = FakeExtractor()
    monkeypatch.setattr(mod, "obs_to_input_token_embedding", fail_rynn_input_token)

    action, obs_embedding, latent = runner._rollout_action(
        FakeWorldModel(),
        FakePolicy(),
        processor=None,
        obs={"is_first": True, "task_description": "Pick up the block"},
        latent=None,
        prev_action=torch.zeros(1, 7),
        target_token_id=10004,
    )

    assert action.shape == (7,)
    assert obs_embedding.shape == (1, 4)
    assert latent["hidden"].shape == (1, 4)
    assert calls == [
        "reset",
        "step:Pick up the block",
        "encode:(1, 4)",
        "actor_input",
        "policy:(1, 6)",
    ]


def test_run_resume_skips_seed_and_warmups_when_ckpts_exist(tmp_path, monkeypatch):
    import os

    # Pre-create both warmup ckpts with minimal valid payloads.
    ckpt_dir = os.path.join(str(tmp_path), "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {"global_step": 7, "world_model": torch.nn.Linear(2, 2).state_dict()},
        os.path.join(ckpt_dir, "wm_warmup.ckpt"),
    )
    torch.save(
        {"global_step": 9, "classifier": torch.nn.Linear(2, 2).state_dict(),
         "classifier_threshold": 0.42},
        os.path.join(ckpt_dir, "classifier_warmup.ckpt"),
    )

    calls: list[str] = []
    runner = _make_orchestration_runner(
        tmp_path,
        monkeypatch,
        calls,
        resume=True,
        total_env_steps=1,
    )

    history = runner.run()

    assert history == []
    # build always runs; seeding + both warmups + both saves are skipped (ckpts loaded).
    assert "seed" not in calls
    assert "wm_warmup" not in calls
    assert "cls_warmup" not in calls
    assert "save_wm" not in calls
    assert "save_cls" not in calls
    assert calls == ["build", "online"]
    # threshold restored from the cls warmup ckpt
    assert runner.classifier_threshold == 0.42
