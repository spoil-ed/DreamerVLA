from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dreamervla.algorithms.registry import ActorUpdateRoute, register_actor_update_route
from dreamervla.runners.frozen_model_policy_runner import FrozenModelPolicyRunner
from dreamervla.utils.frozen_components import module_state_sha256
from dreamervla.workers.actor._test_models import (
    TinyLumosWorldModel,
    TinySuccessClassifier,
)


def _policy_update_step(**kwargs: Any) -> dict[str, float]:
    world_model = kwargs["chunk_world_model"]
    classifier = kwargs["classifier"]
    policy = kwargs["policy"]
    optimizer = kwargs["actor_optimizer"]
    assert not world_model.training
    assert not classifier.training
    assert not any(parameter.requires_grad for parameter in world_model.parameters())
    assert not any(parameter.requires_grad for parameter in classifier.parameters())
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.sum() for parameter in policy.parameters())
    loss.backward()
    optimizer.step()
    return {
        "actor_loss": float(loss.detach()),
        "actor_grad_norm": 1.0,
        "ppo_step_applied": 1.0,
        "LUMOS/success_rate": 0.5,
    }


def _mutating_update_step(**kwargs: Any) -> dict[str, float]:
    metrics = _policy_update_step(**kwargs)
    with torch.no_grad():
        next(kwargs["chunk_world_model"].parameters()).add_(1.0)
    return metrics


def _no_policy_update_step(**_kwargs: Any) -> dict[str, float]:
    return {"ppo_step_applied": 0.0, "actor_grad_norm": 0.0}


for route in (
    ActorUpdateRoute(
        name="FROZEN_TEST_UPDATE",
        step_fn=_policy_update_step,
        world_model_arg="chunk_world_model",
        requires_classifier=True,
    ),
    ActorUpdateRoute(
        name="FROZEN_TEST_MUTATE_WM",
        step_fn=_mutating_update_step,
        world_model_arg="chunk_world_model",
        requires_classifier=True,
    ),
    ActorUpdateRoute(
        name="FROZEN_TEST_NO_STEP",
        step_fn=_no_policy_update_step,
        world_model_arg="chunk_world_model",
        requires_classifier=True,
    ),
):
    register_actor_update_route(route)


def _seed_tiny_replay(replay, **_kwargs: Any) -> int:
    for episode_index in range(2):
        episode = []
        for step in range(2):
            terminal = bool(step == 1 and episode_index == 0)
            episode.append(
                {
                    "obs_embedding": np.full((4,), step + episode_index, np.float32),
                    "wm_action": np.zeros((7,), dtype=np.float32),
                    "reward": float(terminal),
                    "done": float(step == 1),
                    "is_last": float(step == 1),
                    "is_terminal": float(terminal),
                    "success": terminal,
                    "task_id": 0,
                }
            )
        assert replay.add_episode(episode, source="coldstart") is not None
    return 2


def _config(tmp_path: Path, *, update_type: str) -> Any:
    world_model = TinyLumosWorldModel(hidden_dim=4, action_dim=7)
    classifier = TinySuccessClassifier(hidden_dim=4, window=2)
    wm_path = tmp_path / "wm.ckpt"
    cls_path = tmp_path / "classifier.ckpt"
    world_model_cfg = {
        "_target_": "dreamervla.workers.actor._test_models.TinyLumosWorldModel",
        "hidden_dim": 4,
        "action_dim": 7,
    }
    classifier_cfg = {
        "_target_": "dreamervla.workers.actor._test_models.TinySuccessClassifier",
        "hidden_dim": 4,
        "window": 2,
    }
    torch.save(
        {
            "world_model": world_model.state_dict(),
            "config": {"world_model": world_model_cfg},
        },
        wm_path,
    )
    torch.save(
        {
            "model": classifier.state_dict(),
            "threshold": 0.6,
            "f1": 0.9,
            "config": {"classifier": classifier_cfg},
        },
        cls_path,
    )
    return OmegaConf.create(
        {
            "_target_": "dreamervla.runners.FrozenModelPolicyRunner",
            "seed": 3,
            "training": {
                "out_dir": str(tmp_path / "run"),
                "device": "cpu",
                "num_updates": 1,
                "checkpoint_every": 1,
                "require_policy_update": True,
                "resume": False,
            },
            "init": {
                "world_model_state_ckpt": str(wm_path),
                "classifier_state_ckpt": str(cls_path),
            },
            "official_replay": {
                "data_dir": str(tmp_path / "official_reward"),
                "hidden_dir": str(tmp_path / "official_hidden"),
                "task_id": 0,
                "infer_task_id_from_shard": False,
                "capacity": 16,
                "sequence_length": 2,
                "capacity_mode": "total_sharded",
                "task_balanced": True,
                "rank": 0,
                "replay_sampling": {"enabled": False},
                "task_ids": [0],
                "max_episodes_per_task": None,
            },
            "world_model": world_model_cfg,
            "classifier": classifier_cfg,
            "policy": {
                "_target_": "dreamervla.workers.actor._test_models.TinyLumosPolicy",
                "hidden_dim": 4,
                "action_dim": 7,
                "chunk_size": 1,
            },
            "algorithm": {
                "update_type": update_type,
                "kl_coef": 0.0,
                "actor_bc_to_ref_scale": 0.0,
            },
            "optim": {
                "grad_clip_norm": 1.0,
                "zero_grad_set_to_none": True,
                "policy": {
                    "name": "adam",
                    "lr": 1.0e-2,
                    "weight_decay": 0.0,
                },
            },
            "dataloader": {"batch_size": 1},
            "runner": {"logger": {"logger_backends": []}},
        }
    )


def test_runner_updates_only_policy_and_preserves_frozen_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dreamervla.runners.frozen_model_policy_runner.seed_replay_from_offline",
        _seed_tiny_replay,
    )
    runner = FrozenModelPolicyRunner(_config(tmp_path, update_type="FROZEN_TEST_UPDATE"))
    runner.setup()
    initial_policy_hash = module_state_sha256(runner.policy)
    frozen_before = dict(runner.frozen_state_hashes)

    summary = runner.execute()

    assert module_state_sha256(runner.policy) != initial_policy_hash
    assert runner.frozen_state_hashes == frozen_before
    assert summary["applied_policy_steps"] == 1
    assert not hasattr(runner, "world_model_optimizer")
    assert not hasattr(runner, "classifier_optimizer")
    assert (tmp_path / "run" / "checkpoints" / "baseline.ckpt").is_file()
    assert (tmp_path / "run" / "checkpoints" / "final.ckpt").is_file()
    assert (tmp_path / "run" / "frozen_rl_summary.json").is_file()


def test_runner_fails_when_actor_route_mutates_world_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dreamervla.runners.frozen_model_policy_runner.seed_replay_from_offline",
        _seed_tiny_replay,
    )
    runner = FrozenModelPolicyRunner(_config(tmp_path, update_type="FROZEN_TEST_MUTATE_WM"))
    runner.setup()

    with pytest.raises(RuntimeError, match="world_model state changed"):
        runner.execute()


def test_runner_fails_when_policy_never_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dreamervla.runners.frozen_model_policy_runner.seed_replay_from_offline",
        _seed_tiny_replay,
    )
    runner = FrozenModelPolicyRunner(_config(tmp_path, update_type="FROZEN_TEST_NO_STEP"))
    runner.setup()

    with pytest.raises(RuntimeError, match="no policy optimizer step"):
        runner.execute()


def test_runner_resumes_policy_progress_without_changing_frozen_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dreamervla.runners.frozen_model_policy_runner.seed_replay_from_offline",
        _seed_tiny_replay,
    )
    first_cfg = _config(tmp_path, update_type="FROZEN_TEST_UPDATE")
    first = FrozenModelPolicyRunner(first_cfg)
    first.setup()
    first.execute()
    frozen_hashes = dict(first.frozen_state_hashes)
    initial_policy_hash = first.policy_initial_hash

    resumed_cfg = _config(tmp_path, update_type="FROZEN_TEST_UPDATE")
    OmegaConf.update(resumed_cfg, "training.num_updates", 2)
    OmegaConf.update(resumed_cfg, "training.resume", True)
    OmegaConf.update(resumed_cfg, "training.resume_dir", str(tmp_path / "run"))
    resumed = FrozenModelPolicyRunner(resumed_cfg)
    resumed.setup()

    assert resumed.global_step == 1
    assert resumed.applied_policy_steps == 1
    assert resumed.policy_initial_hash == initial_policy_hash
    assert resumed.frozen_state_hashes == frozen_hashes
    assert first.replay is not None
    assert resumed.replay is not None
    assert first.replay.task_sample_cursor == 1
    assert resumed.replay.task_sample_cursor == 1

    summary = resumed.execute()

    assert summary["total_updates"] == 2
    assert summary["applied_policy_steps"] == 2
    assert summary["frozen_hashes_before"] == summary["frozen_hashes_after"]


def test_runner_resume_rejects_empty_rng_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dreamervla.runners.frozen_model_policy_runner.seed_replay_from_offline",
        _seed_tiny_replay,
    )
    first = FrozenModelPolicyRunner(_config(tmp_path, update_type="FROZEN_TEST_UPDATE"))
    first.setup()
    first.execute()
    latest = tmp_path / "run" / "checkpoints" / "latest.ckpt"
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    payload["pickles"]["rng_state"] = pickle.dumps({})
    broken = tmp_path / "empty_rng.ckpt"
    torch.save(payload, broken)

    resumed_cfg = _config(tmp_path, update_type="FROZEN_TEST_UPDATE")
    OmegaConf.update(resumed_cfg, "training.num_updates", 2)
    OmegaConf.update(resumed_cfg, "training.resume", True)
    OmegaConf.update(resumed_cfg, "training.resume_dir", str(broken))

    with pytest.raises(RuntimeError, match="strict RNG"):
        FrozenModelPolicyRunner(resumed_cfg).setup()


@pytest.mark.parametrize("missing_state", ["policy", "policy_optimizer"])
def test_runner_resume_rejects_incomplete_state_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_state: str,
) -> None:
    monkeypatch.setattr(
        "dreamervla.runners.frozen_model_policy_runner.seed_replay_from_offline",
        _seed_tiny_replay,
    )
    cfg = _config(tmp_path, update_type="FROZEN_TEST_UPDATE")
    first = FrozenModelPolicyRunner(cfg)
    first.setup()
    first.execute()
    latest = tmp_path / "run" / "checkpoints" / "latest.ckpt"
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    payload["state_dicts"].pop(missing_state)
    broken = tmp_path / f"missing_{missing_state}.ckpt"
    torch.save(payload, broken)

    resumed_cfg = _config(tmp_path, update_type="FROZEN_TEST_UPDATE")
    OmegaConf.update(resumed_cfg, "training.num_updates", 2)
    OmegaConf.update(resumed_cfg, "training.resume", True)
    OmegaConf.update(resumed_cfg, "training.resume_dir", str(broken))

    with pytest.raises(RuntimeError, match=f"missing.*{missing_state}"):
        FrozenModelPolicyRunner(resumed_cfg).setup()


def test_runner_resume_rejects_changed_objective_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dreamervla.runners.frozen_model_policy_runner.seed_replay_from_offline",
        _seed_tiny_replay,
    )
    first = FrozenModelPolicyRunner(_config(tmp_path, update_type="FROZEN_TEST_UPDATE"))
    first.setup()
    first.execute()

    resumed_cfg = _config(tmp_path, update_type="FROZEN_TEST_UPDATE")
    OmegaConf.update(resumed_cfg, "seed", 4)
    OmegaConf.update(resumed_cfg, "training.num_updates", 2)
    OmegaConf.update(resumed_cfg, "training.resume", True)
    OmegaConf.update(resumed_cfg, "training.resume_dir", str(tmp_path / "run"))

    with pytest.raises(RuntimeError, match="resume contract"):
        FrozenModelPolicyRunner(resumed_cfg).setup()


def test_runner_rejects_checkpoint_component_config_mismatch(tmp_path: Path) -> None:
    cfg = _config(tmp_path, update_type="FROZEN_TEST_UPDATE")
    wm_path = Path(str(cfg.init.world_model_state_ckpt))
    payload = torch.load(wm_path, map_location="cpu", weights_only=False)
    payload["config"]["world_model"]["hidden_dim"] = 99
    torch.save(payload, wm_path)

    with pytest.raises(ValueError, match="world_model.*config"):
        FrozenModelPolicyRunner(cfg).setup()


def test_runner_requires_sampleable_replay_for_every_configured_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dreamervla.runners.frozen_model_policy_runner.seed_replay_from_offline",
        _seed_tiny_replay,
    )
    cfg = _config(tmp_path, update_type="FROZEN_TEST_UPDATE")
    OmegaConf.update(cfg, "official_replay.task_ids", [0, 1])

    with pytest.raises(RuntimeError, match=r"task IDs \[1\]"):
        FrozenModelPolicyRunner(cfg).setup()


def test_runner_rejects_replay_capacity_that_evicts_official_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dreamervla.runners.frozen_model_policy_runner.seed_replay_from_offline",
        _seed_tiny_replay,
    )
    cfg = _config(tmp_path, update_type="FROZEN_TEST_UPDATE")
    OmegaConf.update(cfg, "official_replay.capacity", 2)

    with pytest.raises(RuntimeError, match="did not retain every.*episode"):
        FrozenModelPolicyRunner(cfg).setup()


def test_runner_rejects_multi_process_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    runner = FrozenModelPolicyRunner(_config(tmp_path, update_type="FROZEN_TEST_UPDATE"))

    with pytest.raises(RuntimeError, match="single process"):
        runner.setup()


def test_runner_rejects_invalid_classifier_threshold(tmp_path: Path) -> None:
    cfg = _config(tmp_path, update_type="FROZEN_TEST_UPDATE")
    classifier_path = Path(str(cfg.init.classifier_state_ckpt))
    payload = torch.load(classifier_path, map_location="cpu", weights_only=False)
    payload["threshold"] = 1.5
    torch.save(payload, classifier_path)
    runner = FrozenModelPolicyRunner(cfg)

    with pytest.raises(ValueError, match="threshold must be finite"):
        runner.setup()
