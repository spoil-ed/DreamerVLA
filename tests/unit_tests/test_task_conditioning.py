from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _latent_success_classifier_cls():
    path = (
        Path(__file__).resolve().parents[2]
        / "dreamervla"
        / "models"
        / "reward"
        / "latent_success_classifier.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dreamervla_latent_success_classifier_for_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LatentSuccessClassifier


def test_latent_success_classifier_default_has_no_task_conditioning() -> None:
    LatentSuccessClassifier = _latent_success_classifier_cls()
    model = LatentSuccessClassifier(
        latent_dim=4,
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="linear",
    )

    logits = model(torch.zeros(3, 2, 4))

    assert logits.shape == (3, 2)
    assert model.supports_task_conditioning is False


def test_latent_success_classifier_uses_task_ids_when_enabled() -> None:
    LatentSuccessClassifier = _latent_success_classifier_cls()
    model = LatentSuccessClassifier(
        latent_dim=4,
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="linear",
        task_conditioning={
            "enabled": True,
            "num_tasks": 3,
            "embedding_dim": 4,
        },
    )
    windows = torch.zeros(2, 2, 4)

    logits_a = model(windows, task_ids=torch.tensor([0, 1]))
    logits_b = model(windows, task_ids=torch.tensor([1, 0]))

    assert model.supports_task_conditioning is True
    assert logits_a.shape == (2, 2)
    assert not torch.allclose(logits_a, logits_b)


def test_latent_success_classifier_uses_dino_wm_style_proprio_language_tokens() -> None:
    LatentSuccessClassifier = _latent_success_classifier_cls()
    model = LatentSuccessClassifier(
        latent_dim=9,
        token_dim=4,
        token_count=2,
        token_pool="mean",
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="linear",
        proprio_dim=3,
        proprio_emb_dim=2,
        lang_dim=5,
        lang_emb_dim=3,
    )

    windows = torch.zeros(2, 2, 2, 4)
    proprio = torch.randn(2, 2, 3)
    lang = torch.randn(2, 5)
    logits_a = model(windows, proprio=proprio, lang_emb=lang)
    logits_b = model(windows, proprio=proprio, lang_emb=lang.flip(0))

    assert model.supports_proprio_conditioning is True
    assert model.supports_language_conditioning is True
    assert model.obs_token_dim == 6
    assert model.state_token_dim == 9
    assert logits_a.shape == (2, 2)
    assert not torch.allclose(logits_a, logits_b)


def test_latent_success_classifier_predict_success_threads_proprio_language() -> None:
    LatentSuccessClassifier = _latent_success_classifier_cls()
    model = LatentSuccessClassifier(
        latent_dim=9,
        token_dim=4,
        token_count=2,
        token_pool="mean",
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="linear",
        proprio_dim=3,
        proprio_emb_dim=2,
        lang_dim=5,
        lang_emb_dim=3,
        granularity="chunk",
        chunk_size=2,
        chunk_pool="last",
    )

    video = torch.zeros(2, 6, 2, 4)
    proprio = torch.randn(2, 6, 3)
    lang = torch.randn(2, 5)
    out = model.predict_success(
        video,
        threshold=0.0,
        min_steps=0,
        proprio=proprio,
        lang_emb=lang,
    )

    assert set(out) == {"complete", "finish_step", "score", "score_step"}
    assert out["complete"].shape == (2,)


def test_latent_success_classifier_messages_use_role_based_wm_wording() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "dreamervla"
        / "models"
        / "reward"
        / "latent_success_classifier.py"
    ).read_text(encoding="utf-8")
    assert "DINO-WM" not in source


def test_chunk_wm_declares_task_conditioning_support_when_enabled() -> None:
    from dreamervla.models.world_model.dino_wm_chunk import ChunkAwareDinoWMWorldModel

    wm = ChunkAwareDinoWMWorldModel(
        chunk_size=2,
        obs_dim=8,
        action_dim=7,
        token_count=2,
        token_dim=4,
        action_emb_dim=2,
        num_action_repeat=1,
        model_dim=6,
        depth=1,
        heads=2,
        dim_head=2,
        mlp_dim=16,
        num_hist=2,
        chunk_rollout_chunks=1,
        task_conditioning={
            "enabled": True,
            "num_tasks": 3,
            "embedding_dim": 4,
        },
    )

    assert wm.supports_task_conditioning is True
    batch = {
        "obs_embedding": torch.zeros(2, 4, 8),
        "actions": torch.zeros(2, 4, 7),
        "rewards": torch.zeros(2, 4),
        "dones": torch.zeros(2, 4),
        "is_first": torch.zeros(2, 4, dtype=torch.bool),
        "task_ids": torch.tensor([0, 1]),
    }
    out = wm(batch)
    assert "loss" in out or "_loss" in out


def test_validate_task_conditioning_rejects_missing_capability() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_runner import validate_task_conditioning_cfg

    cfg = OmegaConf.create(
        {
            "task_conditioning": {
                "enabled": True,
                "num_tasks": 2,
                "embedding_dim": 4,
            }
        }
    )

    class NoSupport:
        supports_task_conditioning = False

    try:
        validate_task_conditioning_cfg(
            cfg, world_model=NoSupport(), classifier=NoSupport()
        )
    except ValueError as exc:
        assert "lack task-conditioning support" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_task_conditioning_accepts_default_off() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_runner import validate_task_conditioning_cfg

    cfg = OmegaConf.create({"task_conditioning": {"enabled": False}})

    class NoSupport:
        supports_task_conditioning = False

    validate_task_conditioning_cfg(
        cfg, world_model=NoSupport(), classifier=NoSupport()
    )
