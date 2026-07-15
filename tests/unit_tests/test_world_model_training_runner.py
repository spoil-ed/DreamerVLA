from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def test_world_model_training_common_has_extracted_methods():
    from dreamervla.runtime.world_model_training_common import _WorldModelTrainingCommon

    assert hasattr(_WorldModelTrainingCommon, "_build_components")
    assert not hasattr(_WorldModelTrainingCommon, "_online_cotrain_loop")


def test_wm_pretrain_batch_omits_images_when_hidden_token_exist():
    from dreamervla.runners.world_model_training_runner import (
        WorldModelTrainingRunner,
    )

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
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


def test_wm_pretrain_batch_accepts_hidden_token_without_images():
    from dreamervla.runners.world_model_training_runner import (
        WorldModelTrainingRunner,
    )

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    batch = {
        "obs_embedding": torch.zeros(2, 24, 224, 1024, dtype=torch.float16),
        "actions": torch.zeros(2, 24, 7),
        "current_actions": torch.zeros(2, 24, 7),
        "rewards": torch.zeros(2, 24),
        "dones": torch.zeros(2, 24),
        "is_first": torch.zeros(2, 24, dtype=torch.bool),
        "task_ids": torch.zeros(2, dtype=torch.long),
        "proprio": torch.zeros(2, 24, 8),
        "lang_emb": torch.zeros(2, 4096),
    }

    wm_batch = runner._build_wm_pretrain_batch(batch)

    assert wm_batch is not None
    assert wm_batch["obs_embedding"].dtype == torch.float16
    assert wm_batch["proprio"].shape == (2, 24, 8)
    assert wm_batch["lang_emb"].shape == (2, 4096)
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


def test_world_model_pretrain_step_forwards_condition_sidecars():
    from omegaconf import OmegaConf

    from dreamervla.algorithms.dreamervla import world_model_pretrain_step

    class SidecarRecordingWM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen: dict[str, torch.Tensor] = {}

        def forward(self, batch):
            self.seen = dict(batch)
            loss = self.weight.square()
            return {"_loss": loss, "loss": loss.detach()}

    wm = SidecarRecordingWM()
    optimizer = torch.optim.SGD(wm.parameters(), lr=0.01)
    proprio = torch.zeros(2, 12, 8)
    lang_emb = torch.zeros(2, 4096)

    world_model_pretrain_step(
        policy=torch.nn.Identity(),
        world_model=wm,
        optimizer=optimizer,
        batch={
            "obs_embedding": torch.zeros(2, 12, 256, 4096),
            "actions": torch.zeros(2, 12, 7),
            "proprio": proprio,
            "lang_emb": lang_emb,
        },
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create({"precision": "fp32", "grad_clip_norm": 1.0}),
    )

    assert wm.seen["proprio"] is proprio
    assert wm.seen["lang_emb"] is lang_emb


def test_world_model_pretrain_step_preserves_chunk_hidden_mse_metrics():
    from omegaconf import OmegaConf

    from dreamervla.algorithms.dreamervla import world_model_pretrain_step

    class ChunkMetricWM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, batch):
            del batch
            loss = self.weight.square()
            return {
                "_loss": loss,
                "loss": loss.detach(),
                "hidden_mse": loss.detach() + 1.0,
                "next_latent_mse": loss.detach() + 2.0,
                "hidden_cosine_loss": loss.detach() + 3.0,
                "rollout_cosine_similarity": loss.detach() * 0.0 + 0.75,
            }

    wm = ChunkMetricWM()
    optimizer = torch.optim.SGD(wm.parameters(), lr=0.01)

    metrics = world_model_pretrain_step(
        policy=torch.nn.Identity(),
        world_model=wm,
        optimizer=optimizer,
        batch={"obs_embedding": torch.zeros(1, 1)},
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create({"precision": "fp32", "grad_clip_norm": 1.0}),
    )

    assert metrics["hidden_mse"] == 2.0
    assert metrics["next_latent_mse"] == 3.0
    assert metrics["hidden_rec_loss"] == 2.0
    assert metrics["rollout_cosine_similarity"] == 0.75


def test_world_model_pretrain_step_populates_optional_profile_timings():
    from omegaconf import OmegaConf

    from dreamervla.algorithms.dreamervla import world_model_pretrain_step

    class TimedWM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, batch):
            del batch
            loss = self.weight.square()
            return {"_loss": loss, "loss": loss.detach()}

    wm = TimedWM()
    optimizer = torch.optim.SGD(wm.parameters(), lr=0.01)
    timings: dict[str, float] = {}

    world_model_pretrain_step(
        policy=torch.nn.Identity(),
        world_model=wm,
        optimizer=optimizer,
        batch={"obs_embedding": torch.zeros(1, 1)},
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create({"precision": "fp32", "grad_clip_norm": 1.0}),
        profile_timings=timings,
    )

    assert {
        "h2d",
        "forward",
        "backward",
        "grad_clip",
        "optimizer",
        "metrics",
    }.issubset(timings)
    assert all(timings[key] >= 0.0 for key in timings)


def test_world_model_warmup_can_skip_unused_per_loss_metric_transfers():
    from omegaconf import OmegaConf

    from dreamervla.algorithms.dreamervla import world_model_pretrain_step

    class MetricHeavyWM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, batch):
            del batch
            loss = self.weight.square()
            return {
                "_loss": loss,
                "loss": loss.detach(),
                "transition_loss": loss.detach() + 1.0,
                "reward_loss": loss.detach() + 2.0,
            }

    wm = MetricHeavyWM()
    optimizer = torch.optim.SGD(wm.parameters(), lr=0.01)

    metrics = world_model_pretrain_step(
        policy=torch.nn.Identity(),
        world_model=wm,
        optimizer=optimizer,
        batch={"obs_embedding": torch.zeros(1, 1)},
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create({"precision": "fp32", "grad_clip_norm": 1.0}),
        metrics_mode="loss_only",
    )

    assert set(metrics) == {"loss", "grad_norm"}


def test_world_model_warmup_can_defer_loss_device_to_host_transfer():
    from omegaconf import OmegaConf

    from dreamervla.algorithms.dreamervla import world_model_pretrain_step

    class TinyWM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, batch):
            del batch
            loss = self.weight.square()
            return {"_loss": loss, "loss": loss.detach()}

    wm = TinyWM()
    metrics = world_model_pretrain_step(
        policy=torch.nn.Identity(),
        world_model=wm,
        optimizer=torch.optim.SGD(wm.parameters(), lr=0.01),
        batch={"obs_embedding": torch.zeros(1, 1)},
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create({"precision": "fp32", "grad_clip_norm": 1.0}),
        metrics_mode="loss_tensor",
    )

    assert isinstance(metrics["loss"], torch.Tensor)
    assert isinstance(metrics["grad_norm"], torch.Tensor)


def test_world_model_warmup_keeps_detached_cosine_diagnostics_on_device():
    from omegaconf import OmegaConf

    from dreamervla.algorithms.dreamervla import world_model_pretrain_step

    class TinyWM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, batch):
            del batch
            loss = self.weight.square()
            return {
                "_loss": loss,
                "loss": loss.detach(),
                "one_step_cosine_similarity": loss.detach() * 0.0 + 0.75,
                "persistence_cosine_similarity": loss.detach() * 0.0 + 0.5,
                "chunk_cosine_similarity": loss.detach() * 0.0 + 0.625,
                "rollout_cosine_similarity": loss.detach() * 0.0 + 0.375,
            }

    wm = TinyWM()
    metrics = world_model_pretrain_step(
        policy=torch.nn.Identity(),
        world_model=wm,
        optimizer=torch.optim.SGD(wm.parameters(), lr=0.01),
        batch={"obs_embedding": torch.zeros(1, 1)},
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create({"precision": "fp32", "grad_clip_norm": 1.0}),
        metrics_mode="loss_tensor",
    )

    assert set(metrics) == {
        "loss",
        "grad_norm",
        "one_step_cosine_similarity",
        "persistence_cosine_similarity",
        "chunk_cosine_similarity",
        "rollout_cosine_similarity",
    }
    assert abs(metrics["one_step_cosine_similarity"].item() - 0.75) < 1.0e-7
    assert all(not value.requires_grad for value in metrics.values())


def test_dreamer_wm_progress_status_reports_all_prediction_horizons():
    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    status = WorldModelTrainingRunner._progress_status(
        {
            "loss": 0.1234567,
            "grad_norm": 1.25,
            "learning_rate": 1.0e-4,
            "one_step_cosine_similarity": 0.9876543,
            "chunk_cosine_similarity": 0.5,
            "rollout_cosine_similarity": 0.375,
            "persistence_cosine_similarity": 0.25,
        },
        global_step=39,
    )

    assert status == (
        "global_step=39 loss=0.123457 grad_norm=1.250000 lr=1.000e-04 "
        "one_step_cos=0.987654 "
        "chunk_cos=0.500000 rollout_cos=0.375000 persistence_cos=0.250000"
    )


def test_dreamer_wm_optimizer_diagnostics_report_adam_state():
    from omegaconf import OmegaConf

    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    runner.cfg = OmegaConf.create({"training": {"wm_diagnostics_every": 1}})
    runner.world_model = torch.nn.Linear(3, 2)
    runner.world_model_optimizer = torch.optim.AdamW(
        runner.world_model.parameters(),
        lr=1.0e-4,
    )
    loss = runner.world_model(torch.ones(2, 3)).square().mean()
    loss.backward()
    runner.world_model_optimizer.step()

    metrics = runner._wm_optimizer_diagnostics()

    assert runner._wm_diagnostics_every() == 1
    assert metrics["parameter_norm"].item() > 0.0
    assert metrics["optimizer_exp_avg_norm"].item() > 0.0
    assert metrics["optimizer_exp_avg_sq_norm"].item() > 0.0
    assert metrics["optimizer_step"].item() == 1.0


def test_dreamer_wm_replay_budget_uses_dino_style_epoch_progress():
    from omegaconf import OmegaConf

    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    class Replay:
        @staticmethod
        def sampleable_window_count():
            return 40_000

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    runner.cfg = OmegaConf.create({"training": {"warmup_replay_epochs": 100}})
    runner.distributed = SimpleNamespace(world_size=8)

    assert runner._wm_warmup_progress(
        Replay(),
        step_index=230,
        total_steps=31_300,
        batch_size=16,
    ) == (231, 313, "dreamer-wm epoch 1/100", "step")
    assert runner._wm_warmup_progress(
        Replay(),
        step_index=313,
        total_steps=31_300,
        batch_size=16,
    ) == (1, 313, "dreamer-wm epoch 2/100", "step")


def test_trainable_classifier_preserves_hydra_target(monkeypatch):
    from omegaconf import OmegaConf

    import dreamervla.runtime.world_model_training_common as mod

    runner = mod._WorldModelTrainingCommon.__new__(mod._WorldModelTrainingCommon)
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


def test_trainable_classifier_restores_swept_threshold_from_ckpt(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    import dreamervla.runtime.world_model_training_common as mod

    runner = mod._WorldModelTrainingCommon.__new__(mod._WorldModelTrainingCommon)
    runner.device = torch.device("cpu")
    runner.distributed = _FakeDistributed()
    classifier = torch.nn.Linear(3, 2)
    ckpt = tmp_path / "classifier.ckpt"
    torch.save({"model": classifier.state_dict(), "threshold": 0.73}, ckpt)

    monkeypatch.setattr(mod, "build_classifier", lambda cfg: torch.nn.Linear(3, 2))
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
            "init": {"classifier_state_ckpt": str(ckpt)},
            "optim": {"classifier": {"lr": 1.0e-4}},
        }
    )

    runner._build_trainable_classifier(cfg)

    assert runner.classifier_threshold == 0.73


def test_task_conditioning_validation_is_disabled_by_default():
    from omegaconf import OmegaConf

    from dreamervla.runtime.world_model_training_common import validate_task_conditioning_cfg

    validate_task_conditioning_cfg(
        OmegaConf.create({}),
        world_model=torch.nn.Linear(1, 1),
        classifier=torch.nn.Linear(1, 1),
    )


def test_task_conditioning_validation_fails_without_module_support():
    import pytest
    from omegaconf import OmegaConf

    from dreamervla.runtime.world_model_training_common import validate_task_conditioning_cfg

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

    from dreamervla.runtime.world_model_training_common import validate_task_conditioning_cfg

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
    import dreamervla.runners.world_model_training_runner as mod

    captured: dict[str, object] = {}
    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
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
_HIDDEN_TOKEN_PREPROCESS_CONFIG = {
    "action_head_type": "oft_discrete_token",
    "obs_hidden_source": "hidden_token",
    "hidden_key": "obs_embedding",
    "token_count": 256,
    "token_dim": 4096,
    "hidden_dim": 1_048_576,
    "obs_embedding_shape": [256, 4096],
    "hidden_storage_format": "tokenized",
    "num_images_in_input": 1,
    "patches_per_image": 256,
    "history": 1,
    "include_state": False,
    "sidecar_schema_version": 1,
    "required_demo_datasets": ["obs_embedding"],
}


def _demo_steps(T, success):
    steps = []
    for t in range(T):
        steps.append(
            {
                "actions": np.full(7, t, np.float64),
                "rewards": np.float32(0.0),
                "sparse_rewards": np.uint8(1 if (success and t == T - 1) else 0),
                "dones": np.uint8(1 if t == T - 1 else 0),
                "robot_states": np.zeros(9, np.float64),
                "states": np.zeros(5, np.float64),
                "obs": {
                    "agentview_rgb": np.zeros((256, 256, 3), np.uint8),
                    "eye_in_hand_rgb": np.zeros((256, 256, 3), np.uint8),
                    "ee_pos": np.zeros(3, np.float64),
                    "ee_ori": np.zeros(3, np.float64),
                    "ee_states": np.zeros(6, np.float64),
                    "gripper_states": np.zeros(2, np.float64),
                    "joint_states": np.zeros(7, np.float64),
                },
                "obs_embedding": np.broadcast_to(np.asarray(t, dtype=np.float16), (256, 4096)),
            }
        )
    return steps


def _seeded_replay(tmp_path, seq_len=4):
    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
    from dreamervla.runtime.offline_seed import seed_replay_from_offline
    from dreamervla.runtime.online_replay import OnlineReplay

    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    with RolloutDumpWriter(rdir, hdir, "r0_shard.hdf5") as w:
        for i in range(4):
            w.write_demo(
                index=i,
                steps=_demo_steps(8, success=(i % 2 == 0)),
                preprocess_config=_HIDDEN_TOKEN_PREPROCESS_CONFIG,
                task_id=0,
                episode_id=i,
            )
    replay = OnlineReplay(capacity=10_000, sequence_length=seq_len, task_ids=(0,), rank=0)
    seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir, default_task_id=0)
    return replay


def test_offline_warmup_steps_update_modules(tmp_path, monkeypatch):
    # Use a fake WM/classifier + recording step fns to assert the warmup loops
    # call the existing step functions N times against the seeded buffer.
    import dreamervla.runners.world_model_training_runner as mod

    replay = _seeded_replay(tmp_path)
    calls = {"wm": 0, "cls": 0}
    cls_kwargs = []
    logged = []
    progress = []

    def fake_wm_step(**kw):
        assert kw["batch"] is not None
        calls["wm"] += 1
        return {"loss": 0.1}

    def fake_cls_step(**kw):
        assert kw["replay"] is replay
        cls_kwargs.append(
            (
                kw.get("loss_type"),
                kw.get("sampling_protocol"),
                kw.get("balance_batches"),
            )
        )
        calls["cls"] += 1
        return {"loss": 0.2, "acc": 0.5, "f1": 0.0}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)
    monkeypatch.setattr(mod, "online_classifier_update_step", fake_cls_step)

    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
    runner.device = torch.device("cpu")
    runner.global_step = 0
    runner._build_wm_pretrain_batch = lambda b: {
        "images": torch.zeros(1),
        "obs_embedding": torch.zeros(1),
        "actions": torch.zeros(1),
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
    runner.console_progress = lambda current, total, desc, **kwargs: progress.append(
        (int(current), int(total), str(desc), kwargs.get("unit"))
    )

    runner._offline_warmup_wm(
        replay,
        steps=3,
        batch_size=2,
        optim_cfg=None,
    )
    runner._offline_warmup_classifier(
        replay,
        steps=5,
        batch_size=2,
        early_neg_stride=8,
        grad_clip=1.0,
        loss_type="bce",
        sampling_protocol="wmpo",
        balance_batches=True,
        log_step_offset=3,
    )
    assert calls == {"wm": 3, "cls": 5}
    assert cls_kwargs == [("bce", "wmpo", True)] * 5
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
    assert [step for _metrics, step in logged] == list(range(8))
    assert "train/wm_warmup_loss" in logged_keys
    assert "train/classifier_warmup_loss" in logged_keys
    assert "train/classifier_warmup_acc" in logged_keys


def test_offline_warmup_wm_samples_without_images(monkeypatch):
    import dreamervla.runners.world_model_training_runner as mod

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

    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
    runner.device = torch.device("cpu")
    runner.world_model = torch.nn.Module()
    runner.world_model_optimizer = object()
    runner.policy = object()
    runner._build_wm_pretrain_batch = lambda b: b
    runner._log_replay_warmup_metrics = lambda *_args, **_kwargs: None
    runner.console_progress = lambda *_args, **_kwargs: None

    runner._offline_warmup_wm(Replay(), steps=1, batch_size=2, optim_cfg=None)

    assert sample_kwargs == [(2, {"include_images": False})]


def test_offline_warmup_wm_profiles_configured_initial_steps(monkeypatch):
    from omegaconf import OmegaConf

    import dreamervla.runners.world_model_training_runner as mod

    profile_keys_by_step = []
    logged = []

    class Replay:
        def sample(self, batch_size, **kwargs):
            del batch_size, kwargs
            return {
                "obs_embedding": torch.zeros(2, 3, 4, dtype=torch.float16),
                "actions": torch.zeros(2, 3, 7),
                "rewards": torch.zeros(2, 3),
                "dones": torch.zeros(2, 3),
                "is_first": torch.zeros(2, 3, dtype=torch.bool),
            }

    def fake_wm_step(**kw):
        timings = kw.get("profile_timings")
        if timings is not None:
            timings["h2d"] = 0.001
            timings["forward"] = 0.002
            timings["backward"] = 0.003
            timings["grad_clip"] = 0.004
            timings["optimizer"] = 0.005
            timings["metrics"] = 0.006
            profile_keys_by_step.append(set(timings))
        else:
            profile_keys_by_step.append(set())
        return {"loss": 0.1}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)

    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
    runner.cfg = OmegaConf.create({"training": {"wm_profile_steps": 1}})
    runner.device = torch.device("cpu")
    runner.world_model = torch.nn.Module()
    runner.world_model_optimizer = object()
    runner.policy = object()
    runner._build_wm_pretrain_batch = lambda b: b
    runner._log_replay_warmup_metrics = lambda metrics, **kwargs: logged.append(
        (dict(metrics), kwargs["step"])
    )
    runner.console_progress = lambda *_args, **_kwargs: None

    runner._offline_warmup_wm(Replay(), steps=2, batch_size=2, optim_cfg=None)

    assert profile_keys_by_step == [
        {
            "data_wait",
            "sample",
            "batch_build",
            "h2d",
            "forward",
            "backward",
            "grad_clip",
            "optimizer",
            "metrics",
        },
        set(),
    ]
    time_metrics = {key for metrics, _step in logged for key in metrics}
    assert "time/wm_warmup_sample_ms" in time_metrics
    assert "time/wm_warmup_forward_ms" in time_metrics
    assert "time/wm_warmup_total_ms" in time_metrics

    runner._offline_warmup_wm(
        Replay(), steps=3, start_step=2, batch_size=2, optim_cfg=None
    )
    assert profile_keys_by_step[-1] == set()


def test_offline_warmup_wm_profiles_every_step_when_configured_negative(monkeypatch):
    from omegaconf import OmegaConf

    import dreamervla.runners.world_model_training_runner as mod

    profile_enabled = []

    class Replay:
        def sample(self, batch_size, **kwargs):
            del batch_size, kwargs
            return {
                "obs_embedding": torch.zeros(2, 3, 4, dtype=torch.float16),
                "actions": torch.zeros(2, 3, 7),
                "rewards": torch.zeros(2, 3),
                "dones": torch.zeros(2, 3),
                "is_first": torch.zeros(2, 3, dtype=torch.bool),
            }

    def fake_wm_step(**kwargs):
        timings = kwargs.get("profile_timings")
        profile_enabled.append(timings is not None)
        if timings is not None:
            timings["forward"] = 0.001
        return {"loss": 0.1}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)

    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
    runner.cfg = OmegaConf.create({"training": {"wm_profile_steps": -1}})
    runner.device = torch.device("cpu")
    runner.world_model = torch.nn.Module()
    runner.world_model_optimizer = object()
    runner.policy = object()
    runner._build_wm_pretrain_batch = lambda batch: batch
    runner._log_replay_warmup_metrics = lambda *_args, **_kwargs: None
    runner.console_progress = lambda *_args, **_kwargs: None

    runner._offline_warmup_wm(Replay(), steps=3, batch_size=2, optim_cfg=None)

    assert profile_enabled == [True, True, True]


def test_offline_warmup_wm_loss_progress_uses_rank_zero_event_printer(monkeypatch, capsys):
    from omegaconf import OmegaConf

    import dreamervla.runners.world_model_training_runner as mod

    class Replay:
        def sample(self, batch_size, **kwargs):
            del batch_size, kwargs
            return {
                "obs_embedding": torch.zeros(2, 3, 4, dtype=torch.float16),
                "actions": torch.zeros(2, 3, 7),
                "rewards": torch.zeros(2, 3),
                "dones": torch.zeros(2, 3),
                "is_first": torch.zeros(2, 3, dtype=torch.bool),
            }

    monkeypatch.setattr(
        mod,
        "world_model_pretrain_step",
        lambda **_kwargs: {"loss": 0.1},
    )

    events = []
    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
    runner.cfg = OmegaConf.create({"training": {"replay_warmup_log_every": 1}})
    runner.device = torch.device("cpu")
    runner.world_model = torch.nn.Module()
    runner.world_model_optimizer = object()
    runner.policy = object()
    runner._build_wm_pretrain_batch = lambda batch: batch
    runner._log_replay_warmup_metrics = lambda *_args, **_kwargs: None
    runner._print_pipeline_event = lambda message: events.append(message)
    runner.console_progress = lambda *_args, **_kwargs: None

    runner._offline_warmup_wm(Replay(), steps=1, batch_size=2, optim_cfg=None)

    assert events == ["[pipeline][wm-warmup] step=0/1 loss=0.1000"]
    assert "[pipeline][wm-warmup]" not in capsys.readouterr().out


def test_offline_warmup_classifier_progress_uses_rank_zero_event_printer(monkeypatch, capsys):
    from omegaconf import OmegaConf

    import dreamervla.runners.world_model_training_runner as mod

    monkeypatch.setattr(
        mod,
        "online_classifier_update_step",
        lambda **_kwargs: {"loss": 0.2, "acc": 0.75, "f1": 0.5, "pos_frac": 0.25},
    )

    events = []
    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
    runner.cfg = OmegaConf.create({"training": {"replay_warmup_log_every": 1}})
    runner.device = torch.device("cpu")
    runner.classifier = torch.nn.Module()
    runner.classifier_optimizer = object()
    runner._log_replay_warmup_metrics = lambda *_args, **_kwargs: None
    runner._print_pipeline_event = lambda message: events.append(message)
    runner.console_progress = lambda *_args, **_kwargs: None

    runner._offline_warmup_classifier(
        object(),
        steps=1,
        batch_size=2,
        early_neg_stride=1,
        grad_clip=1.0,
    )

    assert events == ["[pipeline][cls-warmup] step=0/1 loss=0.2000 acc=0.750 f1=0.500 pos=0.250"]
    assert "[pipeline][cls-warmup]" not in capsys.readouterr().out


def test_offline_warmup_wm_uses_configured_prefetch_workers(monkeypatch):
    from omegaconf import OmegaConf

    import dreamervla.runners.world_model_training_runner as mod

    executor_workers = []
    submitted = []
    seen_batches = []

    class FakeFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class FakeExecutor:
        def __init__(self, *, max_workers):
            executor_workers.append(int(max_workers))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            submitted.append((fn.__name__, args, kwargs))
            return FakeFuture(fn(*args, **kwargs))

    monkeypatch.setattr(mod, "ThreadPoolExecutor", FakeExecutor)

    class Replay:
        def __init__(self):
            self.calls = 0

        def sample(self, batch_size, **kwargs):
            self.calls += 1
            return {
                "obs_embedding": torch.full((2, 3, 4), float(self.calls)),
                "actions": torch.zeros(2, 3, 7),
                "rewards": torch.zeros(2, 3),
                "dones": torch.zeros(2, 3),
                "is_first": torch.zeros(2, 3, dtype=torch.bool),
            }

    def fake_wm_step(**kw):
        seen_batches.append(float(kw["batch"]["obs_embedding"][0, 0, 0]))
        return {"loss": 0.1}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)

    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
    runner.cfg = OmegaConf.create({"training": {"wm_prefetch_workers": 2}})
    runner.device = torch.device("cpu")
    runner.world_model = torch.nn.Module()
    runner.world_model_optimizer = object()
    runner.policy = object()
    runner._build_wm_pretrain_batch = lambda b: b
    runner._log_replay_warmup_metrics = lambda *_args, **_kwargs: None
    runner.console_progress = lambda *_args, **_kwargs: None

    runner._offline_warmup_wm(Replay(), steps=3, batch_size=2, optim_cfg=None)

    assert executor_workers == [2]
    assert [item[0] for item in submitted] == ["_sample_wm_pretrain_batch"] * 3
    assert seen_batches == [1.0, 2.0, 3.0]


def test_offline_warmup_alternating_interleaves_wm_and_classifier(tmp_path, monkeypatch):
    import dreamervla.runners.world_model_training_runner as mod

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

    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
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
    runner._log_replay_warmup_metrics = lambda metrics, step: logged.append(
        (dict(metrics), int(step))
    )
    runner.console_progress = lambda current, total, desc, **kwargs: progress.append(
        (int(current), int(total), str(desc), kwargs.get("unit"))
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
    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    replay = _seeded_replay(tmp_path, seq_len=4)

    assert replay.sampleable_window_count() == 12
    assert (
        WorldModelTrainingRunner._steps_for_replay_epochs(
            replay,
            replay_epochs=2,
            batch_size=6,
        )
        == 4
    )


def test_warmup_replay_epochs_use_classifier_window_count_for_classifier(tmp_path):
    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    replay = _seeded_replay(tmp_path, seq_len=4)

    assert replay.sampleable_window_count() == 12
    assert replay.classifier_window_count(window=2, chunk_size=2) == 4
    assert WorldModelTrainingRunner._resolve_warmup_steps(
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


def test_warmup_replay_epochs_keep_explicit_zero_classifier_disabled(tmp_path):
    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    replay = _seeded_replay(tmp_path, seq_len=4)

    assert WorldModelTrainingRunner._resolve_warmup_steps(
        replay,
        wm_steps=1200,
        cls_steps=0,
        replay_epochs=2,
        replay_max_steps=0,
        wm_batch_size=6,
        cls_batch_size=1,
        cls_window=2,
        cls_chunk_size=2,
    ) == (4, 0)


def test_warmup_replay_epochs_cap_to_configured_budget(tmp_path):
    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    replay = _seeded_replay(tmp_path, seq_len=4)

    assert replay.sampleable_window_count() == 12
    assert WorldModelTrainingRunner._resolve_warmup_steps(
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
    assert WorldModelTrainingRunner._resolve_warmup_steps(
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


def test_debug_profile_owns_offline_warmup_budget():
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=wm_full_dataset_train", "profile=debug"],
        )

    assert cfg.profile.name == "debug"
    assert cfg.training.wm_warmup_steps == 2
    assert cfg.training.classifier_warmup_steps == 2
    assert cfg.training.warmup_replay_epochs == 0
    assert cfg.dataloader.batch_size == 2


def test_task_conditioned_classifier_receives_replay_task_ids():
    from dreamervla.runtime.classifier_update import online_classifier_update_step

    class Replay:
        def sample_classifier_windows(
            self,
            batch_size: int,
            *,
            window: int,
            chunk_size: int,
            chunk_pool: str,
            early_neg_stride: int,
            sampling_protocol: str = "lumos",
            balance_batches: bool = False,
        ) -> dict[str, torch.Tensor]:
            del chunk_size, chunk_pool, early_neg_stride, sampling_protocol, balance_batches
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
            "one_step_cosine_similarity": 0.9,
            "persistence_cosine_similarity": 0.8,
            "chunk_cosine_similarity": 0.7,
            "rollout_cosine_similarity": 0.6,
            "ignored": 6.0,
        }
    ) == {
        "wm/loss": 1.0,
        "wm/hidden_rec_loss": 2.0,
        "wm/hidden_cosine_loss": 3.0,
        "wm/full_hidden_rec_loss": 4.0,
        "wm/full_hidden_cosine_loss": 5.0,
        "wm/one_step_cosine_similarity": 0.9,
        "wm/persistence_cosine_similarity": 0.8,
        "wm/chunk_cosine_similarity": 0.7,
        "wm/rollout_cosine_similarity": 0.6,
    }


def test_world_model_metrics_namespace_aliases_chunk_hidden_mse():
    from dreamervla.algorithms.dreamervla import namespaced_world_model_metrics

    assert namespaced_world_model_metrics(
        {
            "loss": 1.0,
            "hidden_mse": 2.0,
            "next_latent_mse": 3.0,
            "reward_loss": 4.0,
            "hidden_pred_norm": 5.0,
            "hidden_target_norm": 6.0,
            "grad_norm": 7.0,
            "ignored": 8.0,
        }
    ) == {
        "wm/loss": 1.0,
        "wm/hidden_rec_loss": 2.0,
        "wm/hidden_mse": 2.0,
        "wm/next_latent_mse": 3.0,
        "wm/reward_loss": 4.0,
        "wm/hidden_pred_norm": 5.0,
        "wm/hidden_target_norm": 6.0,
        "wm/grad_norm": 7.0,
    }


# --------------------------------------------------------------------------
# run() orchestration tests — no models / no LIBERO. We monkeypatch the heavy
# pieces (component build, seeding, and warmup loops) and assert run()
# wires them together in the right order and writes the split warmup ckpts.
# --------------------------------------------------------------------------
def _orchestration_cfg(tmp_path, *, resume=False):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "training": {
                "out_dir": str(tmp_path),
                "debug": True,
                "resume": resume,
                "wm_warmup_steps": 2,
                "classifier_warmup_steps": 2,
                "warmup_replay_epochs": 0,
                # orchestration uses fake nn.Linear modules; HF export needs real
                # target/init_args, so pin torch-only (the test asserts the .ckpt).
                "checkpoint_format": "torch",
            },
            "offline_warmup": {
                "data_dir": str(tmp_path / "offline_data"),
                "hidden_dir": str(tmp_path / "offline_hidden"),
            },
            # run() reads optim.grad_clip_norm; the real config always supplies optim.
            "optim": {"grad_clip_norm": 1.0},
            # Warmup checkpoints persist the Hydra construction contract.
            "world_model": {
                "_target_": "torch.nn.Linear",
                "in_features": 2,
                "out_features": 2,
            },
        }
    )
    return cfg


class _FakeDistributed:
    rank = 0
    world_size = 1
    is_main_process = True

    def wrap_trainable_module(self, module, **_kwargs):
        return module


def _make_orchestration_runner(
    tmp_path,
    monkeypatch,
    calls,
    *,
    resume=False,
    cfg_updates=None,
    seed_capture=None,
):
    import dreamervla.runners.world_model_training_runner as mod

    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
    runner.cfg = _orchestration_cfg(tmp_path, resume=resume)
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
        self.world_model_optimizer = torch.optim.AdamW(self.world_model.parameters(), lr=1e-3)
        self.policy = None
        self.critic = None
        self.classifier = torch.nn.Linear(2, 2)
        self.classifier_optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=1e-3)
        self.classifier_threshold = 0.5

    def fake_seed(
        replay,
        *,
        data_dir,
        hidden_dir,
        default_task_id=None,
        infer_task_id_from_shard=False,
        max_episodes_per_task=None,
        require_reference_complete=True,
    ):
        del (
            data_dir,
            hidden_dir,
            default_task_id,
            infer_task_id_from_shard,
            require_reference_complete,
        )
        calls.append("seed")
        if seed_capture is not None:
            seed_capture["capacity_mode"] = replay.capacity_mode
            seed_capture["capacity"] = replay.capacity
            seed_capture["max_episodes_per_task"] = max_episodes_per_task
        # add a tiny real episode so num_transitions > 0 (run() guards on it)
        episode = [
            {
                "image": np.zeros((4, 4, 3), np.uint8),
                "obs_embedding": np.zeros(8, np.float32),
                "reward": 0.0,
                "done": 0.0,
                "is_last": 0.0,
                "is_terminal": 0.0,
                "wm_action": np.zeros(7, np.float32),
                "task_id": 0,
                "success": True,
            }
            for _ in range(replay.sequence_length + 1)
        ]
        replay.add_episode(episode)
        return 1

    def fake_wm_warmup(self, replay, *, steps, batch_size, optim_cfg, **_kwargs):
        calls.append("wm_warmup")
        return 0.0  # run() formats the returned loss into the warmup banner

    def fake_cls_warmup(self, replay, *, steps, batch_size, early_neg_stride, grad_clip, **_kwargs):
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

    monkeypatch.setattr(mod.WorldModelTrainingRunner, "_build_components", fake_build_components)
    monkeypatch.setattr(mod, "seed_replay_from_offline", fake_seed)
    monkeypatch.setattr(mod.WorldModelTrainingRunner, "_offline_warmup_wm", fake_wm_warmup)
    monkeypatch.setattr(mod.WorldModelTrainingRunner, "_offline_warmup_classifier", fake_cls_warmup)
    monkeypatch.setattr(
        mod.WorldModelTrainingRunner, "_offline_warmup_alternating", fake_alternating_warmup
    )
    # wrap _save_* so we can record their order while still writing the files
    real_save_wm = mod.WorldModelTrainingRunner._save_wm_warmup
    real_save_cls = mod.WorldModelTrainingRunner._save_cls_warmup

    def save_wm(
        self,
        *,
        completed_steps,
        completed_epochs=1,
        metrics=None,
        topk_manager=None,
        steps_per_epoch=None,
    ):
        calls.append("save_wm")
        real_save_wm(
            self,
            completed_steps=completed_steps,
            completed_epochs=completed_epochs,
            metrics=metrics,
            topk_manager=topk_manager,
            steps_per_epoch=steps_per_epoch,
        )

    def save_cls(
        self,
        *,
        completed_steps=0,
        completed_epochs=1,
        metrics=None,
        topk_manager=None,
        steps_per_epoch=None,
    ):
        calls.append("save_cls")
        real_save_cls(
            self,
            completed_steps=completed_steps,
            completed_epochs=completed_epochs,
            metrics=metrics,
            topk_manager=topk_manager,
            steps_per_epoch=steps_per_epoch,
        )

    monkeypatch.setattr(mod.WorldModelTrainingRunner, "_save_wm_warmup", save_wm)
    monkeypatch.setattr(mod.WorldModelTrainingRunner, "_save_cls_warmup", save_cls)
    return runner


def test_run_orchestrates_seed_warmup_and_split_checkpoints(tmp_path, monkeypatch, capsys):
    import os

    calls: list[str] = []
    runner = _make_orchestration_runner(tmp_path, monkeypatch, calls)

    history = runner.run()

    assert history == []
    out = capsys.readouterr().out
    assert "[pipeline][replay] loading offline shards" in out
    assert "[pipeline][replay] loaded complete episodes=1" in out
    assert "[pipeline][warmup] resolved replay warmup" in out
    assert calls == ["build", "seed", "wm_warmup", "save_wm", "cls_warmup", "save_cls"]
    assert os.path.exists(os.path.join(str(tmp_path), "checkpoints", "wm_warmup.ckpt"))
    assert os.path.exists(os.path.join(str(tmp_path), "checkpoints", "classifier_warmup.ckpt"))
    wm_payload = torch.load(
        tmp_path / "checkpoints" / "wm_warmup.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert wm_payload["warmup_step"] == 2
    assert wm_payload["warmup_epoch"] == 1
    assert wm_payload["complete"] is True
    assert {"world_model", "world_model_optimizer"}.issubset(wm_payload["state_dicts"])


def test_run_passes_replay_capacity_mode_and_seed_cap_from_hydra(tmp_path, monkeypatch):
    calls: list[str] = []
    seed_capture: dict[str, object] = {}
    runner = _make_orchestration_runner(
        tmp_path,
        monkeypatch,
        calls,
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

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=wm_full_dataset_train",
                "task=openvla_onetraj_coldstart_libero",
            ],
        )

    from omegaconf import OmegaConf

    assert OmegaConf.select(cfg, "offline_warmup.max_episodes_per_task") is None


def test_run_fails_fast_when_collected_dump_missing(tmp_path, monkeypatch):
    """No collected shards + no warmup-ckpt resume -> identify-before-load error,
    raised BEFORE _build_components loads the heavy WM/encoder/classifier."""
    import shutil

    import pytest

    calls: list[str] = []
    runner = _make_orchestration_runner(tmp_path, monkeypatch, calls)
    # Remove the offline dirs the helper created so the dump looks un-collected.
    for sub in ("offline_data", "offline_hidden"):
        shutil.rmtree(tmp_path / sub)

    with pytest.raises(FileNotFoundError, match="cold-start collection"):
        runner.run()
    assert "build" not in calls  # never reached the model load


def test_wm_only_resume_sets_restored_wm_metric_axis_before_logging(
    tmp_path, monkeypatch
):
    calls: list[str] = []
    runner = _make_orchestration_runner(
        tmp_path,
        monkeypatch,
        calls,
        resume=True,
        cfg_updates={
            "training.classifier_warmup_steps": 0,
        },
    )
    runner._metric_logger = None
    runner._metric_resume_step = None
    resume_setter_calls: list[int] = []

    def set_metric_resume_step(step: int) -> None:
        assert runner._metric_logger is None
        resume_setter_calls.append(int(step))
        runner._metric_resume_step = int(step)

    runner.set_metric_resume_step = set_metric_resume_step
    runner._load_latest_wm_warmup_progress = lambda **_kwargs: {
        "epoch": 0,
        "step": 1,
        "complete": False,
    }

    runner.run()

    assert runner._metric_logger is None
    assert runner._metric_resume_step == 1
    assert resume_setter_calls == [1]


def test_classifier_resume_offsets_metric_axis_by_total_wm_steps(
    tmp_path, monkeypatch
):
    import dreamervla.runners.world_model_training_runner as mod

    calls: list[str] = []
    runner = _make_orchestration_runner(
        tmp_path,
        monkeypatch,
        calls,
        resume=True,
    )
    runner._metric_logger = None
    runner._metric_resume_step = None
    resume_setter_calls: list[int] = []

    def set_metric_resume_step(step: int) -> None:
        assert runner._metric_logger is None
        resume_setter_calls.append(int(step))
        runner._metric_resume_step = int(step)

    runner.set_metric_resume_step = set_metric_resume_step
    runner._canonical_warmup_is_complete = lambda component: component == "wm"
    wm_checkpoint = tmp_path / "checkpoints" / "wm_warmup.ckpt"
    wm_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    wm_checkpoint.touch()
    runner._existing_warmup_checkpoint = lambda name: (
        wm_checkpoint if name == "wm_warmup.ckpt" else None
    )
    runner._load_wm_warmup_checkpoint = lambda *_args, **_kwargs: {
        "epoch": 1,
        "step": 2,
        "complete": True,
    }
    runner._load_latest_cls_warmup_progress = lambda **_kwargs: {
        "epoch": 0,
        "step": 1,
        "complete": False,
    }
    runner._latest_warmup_progress_path = lambda component: (
        tmp_path / "classifier-progress.ckpt" if component == "classifier" else None
    )
    monkeypatch.setattr(mod, "load_runner_payload", lambda _path: {"global_step": 2})

    runner.run()

    assert runner._metric_logger is None
    assert runner._metric_resume_step == 3
    assert resume_setter_calls == [3]


def test_classifier_resume_starts_at_zero_after_complete_wm_without_cls_checkpoint(
    tmp_path, monkeypatch
):
    import dreamervla.runners.world_model_training_runner as mod

    calls: list[str] = []
    runner = _make_orchestration_runner(tmp_path, monkeypatch, calls, resume=True)
    runner._metric_logger = None
    runner._metric_resume_step = None
    runner._canonical_warmup_is_complete = lambda component: component == "wm"
    wm_checkpoint = tmp_path / "checkpoints" / "wm_warmup.ckpt"
    wm_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    wm_checkpoint.touch()
    runner._existing_warmup_checkpoint = lambda name: (
        wm_checkpoint if name == "wm_warmup.ckpt" else None
    )
    wm_restore_rng: list[bool] = []
    runner._load_wm_warmup_checkpoint = lambda *_args, **kwargs: (
        wm_restore_rng.append(bool(kwargs["restore_rng"]))
        or {"epoch": 1, "step": 2, "complete": True}
    )
    cls_start: dict[str, int] = {}

    def run_classifier(_replay, **kwargs):
        cls_start.update(
            start_step=int(kwargs["start_step"]),
            start_epoch=int(kwargs["start_epoch"]),
        )
        return 0.0

    runner._run_cls_warmup_epochs = run_classifier
    resume_steps: list[int] = []
    runner.set_metric_resume_step = lambda step: resume_steps.append(int(step))
    monkeypatch.setattr(mod, "load_runner_payload", lambda _path: {"global_step": 2})

    runner.run()

    assert wm_restore_rng == [True]
    assert cls_start == {"start_step": 0, "start_epoch": 0}
    assert resume_steps == [2]


def test_classifier_starts_at_zero_after_resumed_wm_finishes_without_cls_checkpoint(
    tmp_path, monkeypatch
):
    calls: list[str] = []
    runner = _make_orchestration_runner(tmp_path, monkeypatch, calls, resume=True)
    runner._metric_logger = None
    runner._metric_resume_step = None
    runner._canonical_warmup_is_complete = lambda _component: False
    runner._load_latest_wm_warmup_progress = lambda **_kwargs: {
        "epoch": 0,
        "step": 1,
        "complete": False,
    }
    cls_start: dict[str, int] = {}

    def run_classifier(_replay, **kwargs):
        cls_start.update(
            start_step=int(kwargs["start_step"]),
            start_epoch=int(kwargs["start_epoch"]),
        )
        return 0.0

    runner._run_cls_warmup_epochs = run_classifier
    resume_steps: list[int] = []
    runner.set_metric_resume_step = lambda step: resume_steps.append(int(step))

    runner.run()

    assert cls_start == {"start_step": 0, "start_epoch": 0}
    assert resume_steps == [1]


def test_offline_warmup_requires_every_hydra_declared_task(
    tmp_path,
    monkeypatch,
):
    import pytest

    calls: list[str] = []
    runner = _make_orchestration_runner(
        tmp_path,
        monkeypatch,
        calls,
        cfg_updates={"offline_warmup.required_task_ids": [0, 1]},
    )

    with pytest.raises(RuntimeError, match=r"required task IDs \[1\]"):
        runner.run()

    assert calls == ["build", "seed"]


def test_wm_only_run_never_calibrates_or_checkpoints_classifier(tmp_path, monkeypatch):
    calls: list[str] = []
    runner = _make_orchestration_runner(
        tmp_path,
        monkeypatch,
        calls,
        cfg_updates={
            "training.classifier_warmup_steps": 0,
            "algorithm.lumos.calibrate_threshold": True,
            "algorithm.lumos.classifier_min_val_f1": 0.9,
        },
    )

    history = runner.run()

    assert history == []
    assert calls == ["build", "seed", "wm_warmup", "save_wm"]
    assert not (tmp_path / "checkpoints" / "classifier_warmup.ckpt").exists()


def test_wm_warmup_checkpoint_atomically_overwrites_canonical_path(tmp_path):
    from omegaconf import OmegaConf

    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    runner._output_dir = str(tmp_path)
    runner.global_step = 0
    runner.cfg = OmegaConf.create(
        {
            "world_model": {
                "_target_": "torch.nn.Linear",
                "in_features": 2,
                "out_features": 2,
            }
        }
    )
    runner.world_model = torch.nn.Linear(2, 2)
    runner.world_model_optimizer = torch.optim.AdamW(runner.world_model.parameters(), lr=1e-3)

    with torch.no_grad():
        runner.world_model.weight.fill_(3.0)
    first = runner._save_wm_warmup_checkpoint(
        step=7,
        epoch=1,
        complete=False,
        metrics={"loss": 0.25},
        steps_per_epoch=7,
        total_steps=20,
    )
    with torch.no_grad():
        runner.world_model.weight.fill_(9.0)
    second = runner._save_wm_warmup_checkpoint(
        step=14,
        epoch=2,
        complete=False,
        metrics={"loss": 0.2},
        steps_per_epoch=7,
        total_steps=20,
    )

    assert first == second == tmp_path / "checkpoints" / "wm_warmup.ckpt"
    assert not (tmp_path / "checkpoints" / "warmup_progress").exists()
    payload = torch.load(second, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2
    assert payload["component"] == "wm"
    assert payload["warmup_epoch"] == 2
    assert payload["warmup_step"] == 14
    assert payload["warmup_steps_per_epoch"] == 7
    assert payload["warmup_total_steps"] == 20
    assert payload["complete"] is False
    assert {"world_model", "world_model_optimizer"}.issubset(payload["state_dicts"])
    assert payload["rng_by_rank"]
    from dreamervla.utils.component_checkpoint import load_component_checkpoint

    loaded = load_component_checkpoint(second, "world_model")
    assert set(loaded.state_dict) == set(runner.world_model.state_dict())


def test_wm_warmup_resume_restores_next_epoch_and_requires_optimizer(tmp_path):
    import pytest
    from omegaconf import OmegaConf

    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    runner._output_dir = str(tmp_path)
    runner.global_step = 0
    runner.cfg = OmegaConf.create({"world_model": {}})
    runner.world_model = torch.nn.Linear(2, 2)
    runner.world_model_optimizer = torch.optim.AdamW(runner.world_model.parameters(), lr=1e-3)
    path = runner._save_wm_warmup_checkpoint(
        step=14,
        epoch=2,
        complete=False,
        metrics={"loss": 0.2},
        steps_per_epoch=7,
        total_steps=14,
    )

    restored = runner._load_wm_warmup_checkpoint(path, strict=True)
    assert restored == {"epoch": 2, "step": 14, "complete": False}
    assert restored["epoch"] + 1 == 3

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["warmup_step"] = 13
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="progress mismatch"):
        runner._load_wm_warmup_checkpoint(path, strict=True)
    payload["warmup_step"] = 14
    del payload["state_dicts"]["world_model_optimizer"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="world_model_optimizer"):
        runner._load_wm_warmup_checkpoint(path, strict=True)

    payload["state_dicts"]["world_model_optimizer"] = (
        runner.world_model_optimizer.state_dict()
    )
    payload["complete"] = True
    payload["warmup_epoch"] = 99
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="epoch mismatch"):
        runner._load_wm_warmup_checkpoint(path, strict=True)

    payload["format_version"] = 3
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format_version=3"):
        runner._load_wm_warmup_checkpoint(path, strict=True)


def test_strict_warmup_resume_rejects_hf_only_without_torch_progress(tmp_path):
    import pytest
    from omegaconf import OmegaConf

    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    runner._output_dir = str(tmp_path)
    runner.cfg = OmegaConf.create({})
    (tmp_path / "checkpoints" / "wm_warmup_hf").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="no WM warmup checkpoint"):
        runner._load_latest_wm_warmup_progress(
            steps_per_epoch=5, total_steps=10
        )


def test_classifier_warmup_checkpoint_atomically_overwrites_canonical_path(tmp_path):
    from omegaconf import OmegaConf

    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    runner._output_dir = str(tmp_path)
    runner.cfg = OmegaConf.create({})
    runner.global_step = 0
    runner.classifier = torch.nn.Linear(2, 2)
    runner.classifier_optimizer = torch.optim.AdamW(runner.classifier.parameters(), lr=1e-3)
    runner.classifier_threshold = 0.42
    runner.best_classifier_f1 = 0.8
    runner.best_classifier_ckpt_path = "best.ckpt"

    first = runner._save_cls_warmup_checkpoint(
        step=5,
        epoch=1,
        complete=False,
        metrics={"f1": 0.8},
        steps_per_epoch=5,
        total_steps=10,
    )
    second = runner._save_cls_warmup_checkpoint(
        step=10,
        epoch=2,
        complete=True,
        metrics={"f1": 0.9},
        steps_per_epoch=5,
        total_steps=10,
    )

    assert first == second == tmp_path / "checkpoints" / "classifier_warmup.ckpt"
    assert not (tmp_path / "checkpoints" / "warmup_progress").exists()
    payload = torch.load(second, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2
    assert payload["component"] == "classifier"
    assert payload["warmup_epoch"] == 2
    assert payload["warmup_step"] == 10
    assert payload["warmup_steps_per_epoch"] == 5
    assert payload["warmup_total_steps"] == 10
    assert payload["complete"] is True
    assert payload["classifier_threshold"] == 0.42
    assert payload["best_metric"] == 0.8
    assert payload["best_checkpoint_path"] == "best.ckpt"
    assert {"classifier", "classifier_optimizer"}.issubset(payload["state_dicts"])
    assert payload["rng_by_rank"]


def test_legacy_warmup_progress_remains_readable(tmp_path):
    from omegaconf import OmegaConf

    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    runner._output_dir = str(tmp_path)
    runner.global_step = 0
    runner.cfg = OmegaConf.create({"world_model": {}})
    runner.world_model = torch.nn.Linear(2, 2)
    runner.world_model_optimizer = torch.optim.AdamW(runner.world_model.parameters(), lr=1e-3)
    legacy = tmp_path / "ckpt" / "warmup_progress" / "wm_step_00000007.ckpt"
    legacy.parent.mkdir(parents=True)
    torch.save(
        {
            "world_model": runner.world_model.state_dict(),
            "world_model_optimizer": runner.world_model_optimizer.state_dict(),
            "warmup_step": 7,
            "complete": False,
        },
        legacy,
    )

    restored = runner._load_latest_wm_warmup_progress(
        steps_per_epoch=5, total_steps=20
    )
    assert restored == {"epoch": 1, "step": 7, "complete": False}


def test_warmup_topk_checkpoint_keeps_best_metric_values(tmp_path):
    import pytest
    from omegaconf import OmegaConf

    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    runner._output_dir = str(tmp_path)
    runner.cfg = OmegaConf.create({})
    runner.global_step = 0
    runner.classifier = torch.nn.Linear(2, 2)
    runner.classifier_optimizer = torch.optim.AdamW(runner.classifier.parameters(), lr=1e-3)
    runner.classifier_threshold = 0.5

    topk = runner._make_warmup_topk_manager(component="classifier", k=2)
    for step, f1 in [(1, 0.10), (2, 0.70), (3, 0.30), (4, 0.90)]:
        runner._save_cls_warmup_checkpoint(
            step=step,
            epoch=step,
            complete=False,
            metrics={"loss": 1.0 - f1, "acc": f1, "f1": f1, "pos_frac": 0.5},
            topk_manager=topk,
            steps_per_epoch=1,
            total_steps=10,
        )

    names = sorted(
        p.name for p in (tmp_path / "checkpoints" / "warmup_topk" / "classifier").glob("*.ckpt")
    )
    assert len(names) == 2
    assert any("step=00000002" in name and "f1=0.700000" in name for name in names)
    assert any("step=00000004" in name and "f1=0.900000" in name for name in names)
    assert not any("step=00000001" in name for name in names)
    assert not any("step=00000003" in name for name in names)

    resumed = runner._make_warmup_topk_manager(component="classifier", k=2)
    runner._save_cls_warmup_checkpoint(
        step=5,
        epoch=5,
        complete=False,
        metrics={"loss": 0.05, "acc": 0.95, "f1": 0.95, "pos_frac": 0.5},
        topk_manager=resumed,
        steps_per_epoch=1,
        total_steps=10,
    )
    resumed_names = list(
        (tmp_path / "checkpoints" / "warmup_topk" / "classifier").glob("*.ckpt")
    )
    assert len(resumed_names) == 2
    assert any("f1=0.950000" in path.name for path in resumed_names)

    future = resumed_names[0]
    future_payload = torch.load(future, map_location="cpu", weights_only=False)
    future_payload["format_version"] = 3
    torch.save(future_payload, future)
    with pytest.raises(ValueError, match="format_version=3"):
        runner._make_warmup_topk_manager(component="classifier", k=2)


def test_partial_final_warmup_keeps_completed_epoch_floor_and_final_metrics():
    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    calls: list[tuple[int, int]] = []
    saved: dict[str, object] = {}
    runner._offline_warmup_wm = lambda _replay, *, steps, start_step, **_kwargs: (
        calls.append((start_step, steps)) or float(steps)
    )
    runner._save_wm_warmup_checkpoint = lambda **_kwargs: None
    runner._save_wm_warmup = lambda **kwargs: saved.update(kwargs)

    runner._run_wm_warmup_epochs(
        object(),
        total_steps=12,
        steps_per_epoch=5,
        start_step=0,
        start_epoch=0,
        batch_size=1,
        optim_cfg=None,
        checkpoint_every_epochs=1,
        topk_manager=object(),
    )

    assert calls == [(0, 5), (5, 10), (10, 12)]
    assert saved["completed_epochs"] == 2
    assert saved["completed_steps"] == 12
    assert saved["metrics"] == {"loss": 12.0}
    assert saved["topk_manager"] is not None


def test_classifier_final_topk_uses_f1_not_accuracy():
    from dreamervla.runners.world_model_training_runner import WorldModelTrainingRunner

    runner = WorldModelTrainingRunner.__new__(WorldModelTrainingRunner)
    saved: dict[str, object] = {}

    def train(*_args, **_kwargs):
        runner._last_classifier_warmup_metrics = {"acc": 0.2, "f1": 0.8, "loss": 0.4}
        return 0.2

    runner._offline_warmup_classifier = train
    runner._save_cls_warmup_checkpoint = lambda **_kwargs: None
    runner._save_cls_warmup = lambda **kwargs: saved.update(kwargs)

    runner._run_cls_warmup_epochs(
        object(),
        total_steps=5,
        steps_per_epoch=5,
        start_step=0,
        start_epoch=0,
        batch_size=1,
        early_neg_stride=1,
        grad_clip=1.0,
        loss_type=None,
        sampling_protocol="lumos",
        balance_batches=False,
        log_step_offset=0,
        checkpoint_every_epochs=1,
        topk_manager=object(),
        calibration_kwargs={},
    )

    assert saved["metrics"]["acc"] == 0.2
    assert saved["metrics"]["f1"] == 0.8
    assert saved["topk_manager"] is not None


def test_warmup_only_component_build_skips_rollout_encoder(monkeypatch):
    from omegaconf import OmegaConf

    import dreamervla.runtime.world_model_training_common as mod

    runner = mod._WorldModelTrainingCommon.__new__(mod._WorldModelTrainingCommon)
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
    monkeypatch.setattr(
        mod._WorldModelTrainingCommon, "_build_trainable_classifier", fake_build_classifier
    )

    cfg = OmegaConf.create(
        {
            "online_rollout": {"total_env_steps": 0},
            "world_model": {"_target_": "world_model"},
            "policy": {"_target_": "policy"},
            "critic": {"_target_": "critic"},
            "algorithm": {},
            "optim": {
                "param_precision": "fp32",
                "precision": "fp32",
                "world_model": {
                    "name": "adam",
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
                "policy": {
                    "name": "adam",
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
                "critic": {
                    "name": "adam",
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            },
            "init": {"world_model_state_ckpt": None},
        }
    )

    runner._build_components(cfg)

    assert runner.encoder is None
    assert runner.processor is None
    assert calls == ["world_model"]


def test_run_resume_skips_seed_and_warmups_when_ckpts_exist(tmp_path, monkeypatch):
    import os

    # Pre-create both warmup ckpts with minimal valid payloads.
    ckpt_dir = os.path.join(str(tmp_path), "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    saved_world_model = torch.nn.Linear(2, 2)
    saved_world_model_optimizer = torch.optim.AdamW(saved_world_model.parameters(), lr=1e-3)
    saved_world_model(torch.ones(1, 2)).sum().backward()
    saved_world_model_optimizer.step()
    torch.save(
        {
            "global_step": 7,
            "world_model": saved_world_model.state_dict(),
            "world_model_optimizer": saved_world_model_optimizer.state_dict(),
        },
        os.path.join(ckpt_dir, "wm_warmup.ckpt"),
    )
    saved_classifier = torch.nn.Linear(2, 2)
    saved_classifier_optimizer = torch.optim.AdamW(saved_classifier.parameters(), lr=1e-3)
    saved_classifier(torch.ones(1, 2)).sum().backward()
    saved_classifier_optimizer.step()
    torch.save(
        {
            "global_step": 9,
            "classifier": saved_classifier.state_dict(),
            "classifier_optimizer": saved_classifier_optimizer.state_dict(),
            "classifier_threshold": 0.42,
        },
        os.path.join(ckpt_dir, "classifier_warmup.ckpt"),
    )

    calls: list[str] = []
    runner = _make_orchestration_runner(
        tmp_path,
        monkeypatch,
        calls,
        resume=True,
    )

    history = runner.run()

    assert history == []
    # build always runs; seeding + both warmups + both saves are skipped (ckpts loaded).
    assert "seed" not in calls
    assert "wm_warmup" not in calls
    assert "cls_warmup" not in calls
    assert "save_wm" not in calls
    assert "save_cls" not in calls
    assert calls == ["build"]
    assert runner.world_model_optimizer.state_dict()["state"]
    assert runner.classifier_optimizer.state_dict()["state"]
    # threshold restored from the cls warmup ckpt
    assert runner.classifier_threshold == 0.42


# ---------------------------------------------------------------------------
# B1/B2: warmup threshold calibration + held-out validation gate
# ---------------------------------------------------------------------------


def test_sweep_metrics_is_exported_and_picks_separating_threshold():
    # B1 Step 1: the sweep must live in the shared classifier_metrics module
    # (so the cotrain pipeline does not depend on the classifier runner) and
    # select a threshold that perfectly separates a linearly-separable set.
    from dreamervla.runners.success_classifier_training_runner import _sweep_metrics
    from dreamervla.runtime.classifier_metrics import sweep_threshold_metrics

    assert _sweep_metrics is sweep_threshold_metrics

    probs = np.array([0.1, 0.2, 0.8, 0.9])
    ys = np.array([0, 0, 1, 1])
    out = sweep_threshold_metrics(probs, ys, np.linspace(0.1, 0.9, 9), "val")
    assert out["best_f1"] == 1.0
    assert 0.2 < out["best_thresh"] <= 0.8


class _FakeSeparableReplay:
    """Returns a fixed classifier-window batch encoding its own labels."""

    def __init__(self, labels):
        self._labels = torch.tensor(labels, dtype=torch.int64)

    def sample_classifier_windows(self, batch_size, **kwargs):
        ys = self._labels
        windows = (ys.float() * 2.0 - 1.0).unsqueeze(1)  # -1 / +1
        return {"windows": windows, "labels": ys}


class _FakeSeparableClassifier(torch.nn.Module):
    """logits = [-x, x] so softmax[:,1] is high for +1 windows, low for -1."""

    cfg = SimpleNamespace(window=4, chunk_size=1, chunk_pool="last")

    def forward(self, windows, **kwargs):
        x = windows[:, 0]
        return torch.stack([-x, x], dim=1)


class _FakeConstantClassifier(torch.nn.Module):
    """Always emits the same low P(success) regardless of input (bad model)."""

    cfg = SimpleNamespace(window=4, chunk_size=1, chunk_pool="last")

    def forward(self, windows, **kwargs):
        n = windows.shape[0]
        # logits [+1, -1] -> P(success)=softmax[:,1] ~= 0.12 for every sample
        return torch.stack([torch.ones(n), -torch.ones(n)], dim=1)


def _make_warmup_runner(monkeypatch, classifier):
    import dreamervla.runners.world_model_training_runner as mod

    def fake_cls_step(**kw):
        return {"loss": 0.2, "acc": 0.5, "f1": 0.0, "pos_frac": 0.5}

    monkeypatch.setattr(mod, "online_classifier_update_step", fake_cls_step)

    logged = []
    runner = mod.WorldModelTrainingRunner.__new__(mod.WorldModelTrainingRunner)
    runner.device = torch.device("cpu")
    runner.classifier = classifier
    runner.classifier_optimizer = object()
    runner.classifier_threshold = 0.5
    runner.log_metrics = lambda metrics, step: logged.append((dict(metrics), int(step)))
    runner.console_progress = lambda *a, **k: None
    return runner, logged


def test_offline_warmup_calibration_default_off_preserves_threshold(monkeypatch):
    runner, logged = _make_warmup_runner(monkeypatch, _FakeSeparableClassifier())
    replay = _FakeSeparableReplay([0, 0, 1, 1])

    runner._offline_warmup_classifier(
        replay, steps=1, batch_size=2, early_neg_stride=8, grad_clip=1.0
    )

    # default path: threshold untouched, no eval/* calibration metrics emitted.
    assert runner.classifier_threshold == 0.5
    keys = {k for metrics, _ in logged for k in metrics}
    assert not any(k.startswith("eval/classifier_warmup") for k in keys)


def test_offline_warmup_calibrates_threshold_when_enabled(monkeypatch):
    runner, logged = _make_warmup_runner(monkeypatch, _FakeSeparableClassifier())
    replay = _FakeSeparableReplay([0, 0, 1, 1])

    runner._offline_warmup_classifier(
        replay,
        steps=1,
        batch_size=2,
        early_neg_stride=8,
        grad_clip=1.0,
        calibrate=True,
        val_num_batches=2,
    )

    # threshold moved off the 0.5 default into the separating band.
    assert runner.classifier_threshold != 0.5
    probs_low, probs_high = 0.11, 0.89  # softmax([-(-1),-1]) etc.
    assert probs_low < runner.classifier_threshold < probs_high
    metric_map = {k: v for metrics, _ in logged for k, v in metrics.items()}
    assert metric_map["eval/classifier_warmup_best_f1"] == 1.0
    assert metric_map["eval/classifier_warmup_best_thresh"] == runner.classifier_threshold


def test_offline_warmup_val_gate_raises_below_min_f1(monkeypatch):
    runner, _ = _make_warmup_runner(monkeypatch, _FakeConstantClassifier())
    replay = _FakeSeparableReplay([0, 0, 1, 1])

    import pytest

    with pytest.raises(RuntimeError, match="val F1"):
        runner._offline_warmup_classifier(
            replay,
            steps=1,
            batch_size=2,
            early_neg_stride=8,
            grad_clip=1.0,
            min_val_f1=0.9,
        )


def test_offline_warmup_val_gate_passes_and_logs(monkeypatch):
    runner, logged = _make_warmup_runner(monkeypatch, _FakeSeparableClassifier())
    replay = _FakeSeparableReplay([0, 0, 1, 1])

    runner._offline_warmup_classifier(
        replay,
        steps=1,
        batch_size=2,
        early_neg_stride=8,
        grad_clip=1.0,
        min_val_f1=0.5,
    )

    metric_map = {k: v for metrics, _ in logged for k, v in metrics.items()}
    assert metric_map["eval/classifier_warmup_val_f1"] == 1.0
    # gate-only run does not calibrate: threshold stays at the 0.5 default.
    assert runner.classifier_threshold == 0.5
