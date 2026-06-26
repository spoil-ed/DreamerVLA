from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


def test_online_cotrain_runner_has_extracted_methods():
    from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner

    assert hasattr(OnlineCotrainRunner, "_build_components")
    assert hasattr(OnlineCotrainRunner, "_online_cotrain_loop")


def test_wm_pretrain_batch_omits_images_when_action_hidden_exists():
    from dreamervla.runners.online_cotrain_pipeline_runner import (
        OnlineCotrainPipelineRunner,
    )

    runner = OnlineCotrainPipelineRunner.__new__(OnlineCotrainPipelineRunner)
    batch = {
        "images": torch.zeros(2, 24, 6, 64, 64),
        "obs_embedding": torch.zeros(2, 24, 224, 1024),
        "actions": torch.zeros(2, 24, 7),
        "current_actions": torch.zeros(2, 24, 7),
        "rewards": torch.zeros(2, 24),
        "dones": torch.zeros(2, 24),
        "is_first": torch.zeros(2, 24, dtype=torch.bool),
        "task_ids": torch.zeros(2, dtype=torch.long),
    }

    wm_batch = runner._build_wm_pretrain_batch(batch)

    assert wm_batch is not None
    assert "obs_embedding" in wm_batch
    assert "images" not in wm_batch


def test_wm_pretrain_batch_accepts_action_hidden_without_images():
    from dreamervla.runners.online_cotrain_pipeline_runner import (
        OnlineCotrainPipelineRunner,
    )

    runner = OnlineCotrainPipelineRunner.__new__(OnlineCotrainPipelineRunner)
    batch = {
        "obs_embedding": torch.zeros(2, 24, 224, 1024, dtype=torch.float16),
        "actions": torch.zeros(2, 24, 7),
        "current_actions": torch.zeros(2, 24, 7),
        "rewards": torch.zeros(2, 24),
        "dones": torch.zeros(2, 24),
        "is_first": torch.zeros(2, 24, dtype=torch.bool),
        "task_ids": torch.zeros(2, dtype=torch.long),
    }

    wm_batch = runner._build_wm_pretrain_batch(batch)

    assert wm_batch is not None
    assert wm_batch["obs_embedding"].dtype == torch.float16
    assert "images" not in wm_batch


def test_world_model_pretrain_step_uses_configured_bf16_autocast():
    from omegaconf import OmegaConf

    from dreamervla.algorithms.dreamervla import world_model_pretrain_step

    class AutocastRecordingWM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.autocast_seen = False

        def forward(self, batch):
            del batch
            self.autocast_seen = torch.is_autocast_enabled("cpu")
            loss = (self.weight * 2.0).square()
            return {"_loss": loss, "loss": loss.detach()}

    wm = AutocastRecordingWM()
    optimizer = torch.optim.SGD(wm.parameters(), lr=0.01)

    world_model_pretrain_step(
        policy=torch.nn.Identity(),
        world_model=wm,
        optimizer=optimizer,
        batch={"obs_embedding": torch.zeros(1, 1)},
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create({"precision": "bf16", "grad_clip_norm": 1.0}),
    )

    assert wm.autocast_seen is True


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
    assert "rollout/recent_success_rate" in helper_src
    assert "rollout/avg_success_rate" not in helper_src


def test_training_bursts_episode_trigger_waits_for_completed_episode(monkeypatch):
    import dreamervla.runners.online_cotrain_runner as mod

    runner = mod.OnlineCotrainRunner.__new__(mod.OnlineCotrainRunner)
    runner.device = torch.device("cpu")
    runner.distributed = _FakeDistributed()
    runner.global_step = 0
    calls: list[int] = []

    def fake_stats(*_args, **_kwargs):
        calls.append(1)
        return {}, False, False

    monkeypatch.setattr(mod, "get_replay_task_stats_global", fake_stats)
    knobs = {
        "min_replay": 0,
        "min_eps": 0,
        "is_dist": False,
        "train_trigger": "episode_end",
        "train_every": 1,
        "updates_per_train": 1,
    }

    assert runner._run_training_bursts(
        1,
        10,
        replay=object(),
        env_task_ids=(0,),
        knobs=knobs,
        counters={},
        history=[],
        episode_added=False,
    ) is False
    assert calls == []

    assert runner._run_training_bursts(
        2,
        10,
        replay=object(),
        env_task_ids=(0,),
        knobs=knobs,
        counters={},
        history=[],
        episode_added=True,
    ) is False
    assert calls == [1]


def test_training_bursts_env_step_trigger_keeps_train_every_gate(monkeypatch):
    import dreamervla.runners.online_cotrain_runner as mod

    runner = mod.OnlineCotrainRunner.__new__(mod.OnlineCotrainRunner)
    runner.device = torch.device("cpu")
    runner.distributed = _FakeDistributed()
    runner.global_step = 0
    calls: list[int] = []

    def fake_stats(*_args, **_kwargs):
        calls.append(1)
        return {}, False, False

    monkeypatch.setattr(mod, "get_replay_task_stats_global", fake_stats)
    knobs = {
        "min_replay": 0,
        "min_eps": 0,
        "is_dist": False,
        "train_trigger": "env_step",
        "train_every": 4,
        "updates_per_train": 1,
    }

    runner._run_training_bursts(
        3,
        10,
        replay=object(),
        env_task_ids=(0,),
        knobs=knobs,
        counters={},
        history=[],
        episode_added=True,
    )
    runner._run_training_bursts(
        4,
        10,
        replay=object(),
        env_task_ids=(0,),
        knobs=knobs,
        counters={},
        history=[],
        episode_added=False,
    )
    assert calls == [1]


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


def test_task_conditioning_validation_is_disabled_by_default():
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_runner import validate_task_conditioning_cfg

    validate_task_conditioning_cfg(
        OmegaConf.create({}),
        world_model=torch.nn.Linear(1, 1),
        classifier=torch.nn.Linear(1, 1),
    )


def test_task_conditioning_validation_fails_without_module_support():
    import pytest
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_runner import validate_task_conditioning_cfg

    cfg = OmegaConf.create(
        {"task_conditioning": {"enabled": True, "num_tasks": 10, "embedding_dim": 64}}
    )

    with pytest.raises(ValueError, match="lack task-conditioning support"):
        validate_task_conditioning_cfg(
            cfg,
            world_model=torch.nn.Linear(1, 1),
            classifier=torch.nn.Linear(1, 1),
        )


def test_task_conditioning_validation_accepts_capable_modules():
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_runner import validate_task_conditioning_cfg

    class Capable(torch.nn.Linear):
        supports_task_conditioning = True

    cfg = OmegaConf.create(
        {"task_conditioning": {"enabled": True, "num_tasks": 10, "embedding_dim": 64}}
    )

    validate_task_conditioning_cfg(
        cfg,
        world_model=Capable(1, 1),
        classifier=Capable(1, 1),
    )


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
    logged = []
    progress = []

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
    runner.log_metrics = lambda metrics, step: logged.append((dict(metrics), int(step)))
    runner.console_progress = (
        lambda current, total, desc, **kwargs: progress.append(
            (int(current), int(total), str(desc), kwargs.get("unit"))
        )
    )

    runner._offline_warmup_wm(replay, steps=3, batch_size=2, optim_cfg=None)
    runner._offline_warmup_classifier(replay, steps=5, batch_size=2, early_neg_stride=8, grad_clip=1.0)
    assert calls == {"wm": 3, "cls": 5}
    assert progress == [
        (1, 3, "wm-warmup", "update"),
        (2, 3, "wm-warmup", "update"),
        (3, 3, "wm-warmup", "update"),
        (1, 5, "classifier-warmup", "update"),
        (2, 5, "classifier-warmup", "update"),
        (3, 5, "classifier-warmup", "update"),
        (4, 5, "classifier-warmup", "update"),
        (5, 5, "classifier-warmup", "update"),
    ]
    logged_keys = {key for metrics, _step in logged for key in metrics}
    assert "train/wm_warmup_loss" in logged_keys
    assert "train/classifier_warmup_loss" in logged_keys
    assert "train/classifier_warmup_acc" in logged_keys


def test_offline_warmup_wm_samples_without_images(monkeypatch):
    import dreamervla.runners.online_cotrain_pipeline_runner as mod

    sample_kwargs = []

    class Replay:
        def sample(self, batch_size, **kwargs):
            sample_kwargs.append((int(batch_size), dict(kwargs)))
            return {
                "obs_embedding": torch.zeros(2, 3, 4, dtype=torch.float16),
                "actions": torch.zeros(2, 3, 7),
                "rewards": torch.zeros(2, 3),
                "dones": torch.zeros(2, 3),
                "is_first": torch.zeros(2, 3, dtype=torch.bool),
            }

    def fake_wm_step(**kw):
        assert "images" not in kw["batch"]
        return {"loss": 0.1}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)

    runner = mod.OnlineCotrainPipelineRunner.__new__(mod.OnlineCotrainPipelineRunner)
    runner.device = torch.device("cpu")
    runner.world_model = torch.nn.Module()
    runner.world_model_optimizer = object()
    runner.policy = object()
    runner._build_wm_pretrain_batch = lambda b: b
    runner._log_replay_warmup_metrics = lambda *_args, **_kwargs: None
    runner.console_progress = lambda *_args, **_kwargs: None

    runner._offline_warmup_wm(Replay(), steps=1, batch_size=2, optim_cfg=None)

    assert sample_kwargs == [(2, {"include_images": False})]


def test_offline_warmup_alternating_interleaves_wm_and_classifier(tmp_path, monkeypatch):
    import dreamervla.runners.online_cotrain_pipeline_runner as mod

    replay = _seeded_replay(tmp_path)
    calls = []
    logged = []
    progress = []

    def fake_wm_step(**kw):
        assert kw["batch"] is not None
        calls.append("wm")
        return {"loss": float(len(calls))}

    def fake_cls_step(**kw):
        assert kw["replay"] is replay
        calls.append("cls")
        return {"loss": 0.2, "acc": 0.5, "f1": 0.25, "pos_frac": 0.5}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)
    monkeypatch.setattr(mod, "online_classifier_update_step", fake_cls_step)

    runner = mod.OnlineCotrainPipelineRunner.__new__(mod.OnlineCotrainPipelineRunner)
    runner.device = torch.device("cpu")
    runner._build_wm_pretrain_batch = lambda b: {
        "images": torch.zeros(1),
        "obs_embedding": torch.zeros(1),
        "actions": torch.zeros(1),
    }
    runner.world_model = torch.nn.Module()
    runner.world_model_optimizer = object()
    runner.policy = object()
    runner.classifier = torch.nn.Module()
    runner.classifier_optimizer = object()
    runner._log_replay_warmup_metrics = lambda metrics, step: logged.append((dict(metrics), int(step)))
    runner.console_progress = (
        lambda current, total, desc, **kwargs: progress.append(
            (int(current), int(total), str(desc), kwargs.get("unit"))
        )
    )

    wm_last, cls_last = runner._offline_warmup_alternating(
        replay,
        wm_steps=2,
        cls_steps=3,
        wm_batch_size=2,
        cls_batch_size=2,
        optim_cfg=None,
        early_neg_stride=8,
        grad_clip=1.0,
    )

    assert calls == ["wm", "cls", "wm", "cls", "cls"]
    assert progress == [
        (1, 3, "replay-warmup", "update"),
        (2, 3, "replay-warmup", "update"),
        (3, 3, "replay-warmup", "update"),
    ]
    assert wm_last == 3.0
    assert cls_last == 0.5
    assert [step for _metrics, step in logged] == [0, 1, 2]
    logged_keys = {key for metrics, _step in logged for key in metrics}
    assert "train/classifier_warmup_loss" in logged_keys
    assert "train/classifier_warmup_f1" in logged_keys
    assert "train/classifier_warmup_pos_frac" in logged_keys


def test_warmup_replay_epochs_derive_steps_from_sampleable_windows(tmp_path):
    from dreamervla.runners.online_cotrain_pipeline_runner import OnlineCotrainPipelineRunner

    replay = _seeded_replay(tmp_path, seq_len=4)

    assert replay.sampleable_window_count() == 12
    assert OnlineCotrainPipelineRunner._steps_for_replay_epochs(
        replay,
        replay_epochs=2,
        batch_size=6,
    ) == 4


def test_warmup_replay_epochs_use_classifier_window_count_for_classifier(tmp_path):
    from dreamervla.runners.online_cotrain_pipeline_runner import OnlineCotrainPipelineRunner

    replay = _seeded_replay(tmp_path, seq_len=4)

    assert replay.sampleable_window_count() == 12
    assert replay.classifier_window_count(window=2, chunk_size=2) == 4
    assert OnlineCotrainPipelineRunner._resolve_warmup_steps(
        replay,
        wm_steps=1200,
        cls_steps=1200,
        replay_epochs=2,
        replay_max_steps=0,
        wm_batch_size=6,
        cls_batch_size=4,
        cls_window=2,
        cls_chunk_size=2,
    ) == (4, 2)


def test_warmup_replay_epochs_cap_to_configured_budget(tmp_path):
    from dreamervla.runners.online_cotrain_pipeline_runner import OnlineCotrainPipelineRunner

    replay = _seeded_replay(tmp_path, seq_len=4)

    assert replay.sampleable_window_count() == 12
    assert OnlineCotrainPipelineRunner._resolve_warmup_steps(
        replay,
        wm_steps=1200,
        cls_steps=1200,
        replay_epochs=1,
        replay_max_steps=4,
        wm_batch_size=2,
        cls_batch_size=1,
        cls_window=2,
        cls_chunk_size=2,
    ) == (4, 4)
    assert OnlineCotrainPipelineRunner._resolve_warmup_steps(
        replay,
        wm_steps=1200,
        cls_steps=1200,
        replay_epochs=0,
        replay_max_steps=4,
        wm_batch_size=2,
        cls_batch_size=2,
        cls_window=2,
        cls_chunk_size=2,
    ) == (1200, 1200)


def test_debug_overrides_disable_replay_epoch_and_lumos_bounds():
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_pipeline_runner import OnlineCotrainPipelineRunner

    cfg = OmegaConf.create(
        {
            "training": {
                "debug": True,
                "warmup_replay_epochs": 1,
                "wm_warmup_steps": 1200,
                "classifier_warmup_steps": 1200,
            },
            "offline_warmup": {
                "debug_wm_warmup_steps": 2,
                "debug_classifier_warmup_steps": 2,
            },
            "algorithm": {
                "debug_ppo_rollouts_per_start": 2,
                "ppo_rollouts_per_start": 4,
                "lumos": {
                    "ppo_rollouts_per_start_min": 4,
                    "ppo_rollouts_per_start_max": 16,
                },
            },
        }
    )

    OnlineCotrainPipelineRunner._apply_debug_overrides(cfg)

    assert OmegaConf.select(cfg, "training.warmup_replay_epochs") == 0
    assert OmegaConf.select(cfg, "training.wm_warmup_steps") == 2
    assert OmegaConf.select(cfg, "training.classifier_warmup_steps") == 2
    assert OmegaConf.select(cfg, "algorithm.ppo_rollouts_per_start") == 2
    assert OmegaConf.select(cfg, "algorithm.lumos.ppo_rollouts_per_start_min") == 2
    assert OmegaConf.select(cfg, "algorithm.lumos.ppo_rollouts_per_start_max") == 2


def test_task_conditioned_classifier_receives_replay_task_ids():
    from dreamervla.runners.online_dreamervla import online_classifier_update_step

    class Replay:
        def sample_classifier_windows(
            self,
            batch_size: int,
            *,
            window: int,
            chunk_size: int,
            chunk_pool: str,
            early_neg_stride: int,
        ) -> dict[str, torch.Tensor]:
            del chunk_size, chunk_pool, early_neg_stride
            return {
                "windows": torch.ones(int(batch_size), int(window), 4),
                "labels": torch.tensor([1, 0], dtype=torch.long),
                "task_ids": torch.tensor([3, 4], dtype=torch.long),
            }

    class TaskAwareClassifier(torch.nn.Module):
        supports_task_conditioning = True

        def __init__(self) -> None:
            super().__init__()
            self.cfg = SimpleNamespace(window=2, chunk_size=1, chunk_pool="last")
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen_task_ids: torch.Tensor | None = None

        def forward(
            self,
            windows: torch.Tensor,
            *,
            task_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del windows
            assert task_ids is not None
            self.seen_task_ids = task_ids.detach().cpu()
            logits = torch.stack(
                [-self.weight.expand(task_ids.shape[0]), self.weight.expand(task_ids.shape[0])],
                dim=-1,
            )
            return logits

    classifier = TaskAwareClassifier()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)

    online_classifier_update_step(
        classifier=classifier,
        optimizer=optimizer,
        replay=Replay(),
        device=torch.device("cpu"),
        batch_size=2,
        early_neg_stride=8,
        grad_clip=1.0,
    )

    assert classifier.seen_task_ids is not None
    assert classifier.seen_task_ids.tolist() == [3, 4]


def test_world_model_metrics_namespace_includes_hidden_losses():
    from dreamervla.algorithms.dreamervla import namespaced_world_model_metrics

    assert namespaced_world_model_metrics(
        {
            "loss": 1.0,
            "hidden_rec_loss": 2.0,
            "hidden_cosine_loss": 3.0,
            "full_hidden_rec_loss": 4.0,
            "full_hidden_cosine_loss": 5.0,
            "ignored": 6.0,
        }
    ) == {
        "wm/loss": 1.0,
        "wm/hidden_rec_loss": 2.0,
        "wm/hidden_cosine_loss": 3.0,
        "wm/full_hidden_rec_loss": 4.0,
        "wm/full_hidden_cosine_loss": 5.0,
    }


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


def test_online_cotrain_loop_passes_full_ready_gates(monkeypatch):
    import dreamervla.runners.online_cotrain_runner as mod

    captured = {}
    runner = mod.OnlineCotrainRunner.__new__(mod.OnlineCotrainRunner)
    runner.device = torch.device("cpu")
    runner.distributed = _FakeDistributed()
    runner.global_step = 0

    class Replay:
        num_transitions = 16

    def fake_global_ready(replay, **kwargs):
        del replay
        captured.update(kwargs)
        return {}, False, False

    monkeypatch.setattr(mod, "get_replay_task_stats_global", fake_global_ready)

    stop = runner._run_training_bursts(
        env_step=1,
        total_env_steps=1,
        replay=Replay(),
        env_task_ids=(0,),
        knobs={
            "train_trigger": "episode_end",
            "updates_per_episode": 1,
            "updates_per_train": 1,
            "train_every": 1,
            "min_replay": 12,
            "min_eps": 1,
            "min_sampleable_windows": 9,
            "require_classifier_evidence": True,
            "is_dist": False,
            "batch_size": 1,
            "max_train_updates": 0,
            "warmup_steps": 0,
        },
        counters={"n_episodes": 1, "n_success": 0},
        history=[],
        episode_added=True,
    )

    assert stop is False
    assert captured["min_sampleable_windows"] == 9
    assert captured["require_classifier_evidence"] is True


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

    def fake_alternating_warmup(
        self,
        replay,
        *,
        wm_steps,
        cls_steps,
        wm_batch_size,
        cls_batch_size,
        optim_cfg,
        early_neg_stride,
        grad_clip,
    ):
        calls.append("alternating_warmup")
        return 0.0, 0.0

    def fake_online_loop(self, cfg):
        calls.append("online")
        return []

    monkeypatch.setattr(mod.OnlineCotrainPipelineRunner, "_build_components", fake_build_components)
    monkeypatch.setattr(mod, "seed_replay_from_offline", fake_seed)
    monkeypatch.setattr(mod.OnlineCotrainPipelineRunner, "_offline_warmup_wm", fake_wm_warmup)
    monkeypatch.setattr(mod.OnlineCotrainPipelineRunner, "_offline_warmup_classifier", fake_cls_warmup)
    monkeypatch.setattr(mod.OnlineCotrainPipelineRunner, "_offline_warmup_alternating", fake_alternating_warmup)
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


def test_run_orchestrates_seed_warmup_split_ckpt_online(tmp_path, monkeypatch, capsys):
    import os

    calls: list[str] = []
    runner = _make_orchestration_runner(tmp_path, monkeypatch, calls, total_env_steps=1)

    history = runner.run()

    assert history == []
    out = capsys.readouterr().out
    assert "[pipeline][replay] loading offline shards" in out
    assert "[pipeline][replay] loaded complete episodes=1" in out
    assert "[pipeline][warmup] resolved replay warmup" in out
    # order: build -> seed -> WM warmup/checkpoint -> classifier warmup/checkpoint -> online
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


def test_release_pipeline_warmup_uses_all_collected_episodes_by_default():
    from hydra import compose, initialize_config_dir

    config_dir = "/mnt/data/spoil/workspace/DreamerVLA/configs"
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=online_cotrain_pipeline_oft_action_hidden",
                "task=openvla_onetraj_coldstart_libero",
            ],
        )

    assert cfg.offline_warmup.max_episodes_per_task is None


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
    assert calls == ["build", "seed", "wm_warmup", "save_wm", "cls_warmup", "save_cls"]
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


def test_single_env_rollout_executes_full_chunk_and_clears_on_reset(monkeypatch):
    from omegaconf import OmegaConf

    import dreamervla.runners.online_cotrain_runner as mod

    class FakeReplay:
        num_transitions = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def add_episode(self, _episode, *, source="online"):
            assert source == "online"
            return None

    class FakeEnv:
        def __init__(self):
            self.episode_step = 0
            self.actions: list[np.ndarray] = []

        def reset(self):
            self.episode_step = 0
            return {"is_first": True, "task_description": "task"}, {}

        def step(self, action):
            self.actions.append(np.asarray(action, dtype=np.float32).copy())
            self.episode_step += 1
            done = self.episode_step >= 2
            return (
                {"is_first": False, "task_description": "task"},
                1.0 if done else 0.0,
                done,
                False,
                {},
            )

        def make_transition(self, *_args, **_kwargs):
            return {}

        def close(self):
            pass

    class FakeExtractor:
        def reset(self):
            pass

        def step(self, _obs, _desc):
            return [], torch.zeros(4)

    class FakeWorldModel:
        def __call__(self, batch):
            if batch["mode"] == "encode_latent":
                return {"hidden": batch["hidden"]}
            if batch["mode"] == "observe_next":
                return {"hidden": batch["hidden"], "prev": batch["actions"]}
            if batch["mode"] == "actor_input":
                return torch.zeros(1, 6)
            raise AssertionError(batch["mode"])

    class FakePolicy:
        def __call__(self, _batch):
            first = torch.full((1, 1, 7), 0.25)
            second = torch.full((1, 1, 7), 0.75)
            return torch.cat([first, second], dim=1), torch.zeros(1), {}

    fake_env = FakeEnv()
    monkeypatch.setattr(mod, "OnlineReplay", FakeReplay)
    runner = mod.OnlineCotrainRunner.__new__(mod.OnlineCotrainRunner)
    runner.device = torch.device("cpu")
    runner.distributed = _FakeDistributed()
    runner.processor = None
    runner.world_model = FakeWorldModel()
    runner.policy = FakePolicy()
    runner._latent_type = "action_hidden"
    runner._oft_action_hidden_extractor = FakeExtractor()
    runner._build_env = lambda _cfg: fake_env
    runner.resume = lambda: None
    runner.console_progress = lambda *_args, **_kwargs: None
    runner.console_record_success = lambda *_args, **_kwargs: None
    runner._save_cotrain_ckpt = lambda: None

    cfg = OmegaConf.create(
        {
            "algorithm": {"update_type": "LUMOS"},
            "dataloader": {"batch_size": 1},
            "env": {"episode_horizon": 2, "task_ids": [0]},
            "online_rollout": {
                "buffer_size": 10,
                "max_train_updates": 0,
                "min_episodes_per_task": 99,
                "min_replay": 99,
                "num_envs": 1,
                "render_backend": "osmesa",
                "sequence_length": 2,
                "total_env_steps": 3,
                "train_trigger": "episode_end",
            },
            "optim": {},
            "training": {
                "checkpoint_every": 0,
                "train_actor_after_warmup": False,
                "train_classifier_inline": False,
                "warmup_steps": 0,
            },
        }
    )

    runner._online_cotrain_loop(cfg)

    assert len(fake_env.actions) == 3
    np.testing.assert_array_equal(fake_env.actions[0], np.full(7, 0.25, np.float32))
    np.testing.assert_array_equal(fake_env.actions[1], np.full(7, 0.75, np.float32))
    np.testing.assert_array_equal(fake_env.actions[2], np.full(7, 0.25, np.float32))


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
