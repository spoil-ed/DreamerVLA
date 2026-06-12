from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from dreamer_vla.dataset.libero_pixel_rynn_hidden_sequence_dataset import (
    LIBEROPixelRynnHiddenSequenceDataset,
)
from dreamer_vla.models.actor import (
    LatentToActionHiddenActor,
    RynnVLAActionHiddenActor,
    VLAActionHeadActor,
)
from dreamer_vla.models.reward import LatentSuccessClassifier
from dreamer_vla.models.world_model.rynn_dino_wm import RynnDinoWMWorldModel
from dreamer_vla.runners.dreamer_vla_runner import DreamerVLARunner
from dreamer_vla.runners.eval_libero_vla_runner import EvalLiberoVLARunner


def test_rynn_hidden_sidecar_validates_action_head_type(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "model_path": "/tmp/model",
                "encoder_state_ckpt": "/tmp/encoder.ckpt",
                "time_horizon": 5,
                "action_head_type": "legacy",
                "save_actor_sequence": True,
            }
        ),
        encoding="utf-8",
    )
    dataset = LIBEROPixelRynnHiddenSequenceDataset.__new__(LIBEROPixelRynnHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    with pytest.raises(ValueError, match="action_head_type mismatch"):
        dataset._validate_hidden_sidecar(
            expected_model_path=None,
            expected_encoder_state_ckpt=None,
            expected_time_horizon=5,
            expected_action_head_type="old_head",
            require_preprocess_config=True,
        )


def test_rynn_hidden_sidecar_requires_expected_path_metadata(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "time_horizon": 5,
                "action_head_type": "legacy",
            }
        ),
        encoding="utf-8",
    )
    dataset = LIBEROPixelRynnHiddenSequenceDataset.__new__(LIBEROPixelRynnHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    with pytest.raises(ValueError, match="model_path mismatch"):
        dataset._validate_hidden_sidecar(
            expected_model_path="/tmp/model",
            expected_encoder_state_ckpt=None,
            expected_time_horizon=5,
            expected_action_head_type="legacy",
            require_preprocess_config=True,
        )

    with pytest.raises(ValueError, match="encoder_state_ckpt mismatch"):
        dataset._validate_hidden_sidecar(
            expected_model_path=None,
            expected_encoder_state_ckpt="/tmp/encoder.ckpt",
            expected_time_horizon=5,
            expected_action_head_type="legacy",
            require_preprocess_config=True,
        )


def test_rynn_hidden_sidecar_accepts_legacy_ckpts_checkpoint_alias(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "model_path": "/old/workspace/DreamerVLA/data/ckpts/VLA_model_256/libero_goal",
                "time_horizon": 5,
                "action_head_type": "legacy",
            }
        ),
        encoding="utf-8",
    )
    dataset = LIBEROPixelRynnHiddenSequenceDataset.__new__(LIBEROPixelRynnHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    dataset._validate_hidden_sidecar(
        expected_model_path="data/checkpoints/VLA_model_256/libero_goal",
        expected_encoder_state_ckpt=None,
        expected_time_horizon=5,
        expected_action_head_type="legacy",
        require_preprocess_config=True,
    )


def test_vla_action_head_actor_uses_rynnvla_action_tokens() -> None:
    actor = VLAActionHeadActor(
        hidden_dim=16,
        action_dim=3,
        time_horizon=4,
        vla_hidden_size=16,
        hidden_size_factor=0.25,
        num_encoder_layers=1,
        adapter_type="identity",
        action_head_type="legacy",
    )

    chunk = actor({"mode": "sample", "hidden": torch.randn(2, 16), "deterministic": True})[2][
        "action_chunk"
    ]

    assert actor.action_token_embeddings.weight.shape == (1, 4 * 3 * 16)
    assert chunk.shape == (2, 4, 3)


def test_vla_action_head_actor_rejects_ckpt_without_action_head(tmp_path) -> None:
    path = tmp_path / "vla_without_action_head.ckpt"
    torch.save({"state_dicts": {"encoder": {"backbone.other.weight": torch.ones(1)}}}, path)

    with pytest.raises(RuntimeError, match="action_head"):
        VLAActionHeadActor(
            hidden_dim=16,
            action_dim=3,
            time_horizon=4,
            vla_hidden_size=16,
            hidden_size_factor=0.25,
            num_encoder_layers=1,
            adapter_type="identity",
            action_head_type="legacy",
            init_action_head_ckpt=str(path),
        )


def test_dreamer_eval_keeps_rynnvla_action_hidden_tokens_for_wm() -> None:
    workspace = EvalLiberoVLARunner.__new__(EvalLiberoVLARunner)
    workspace.cfg = OmegaConf.create(
        {
            "eval": {"obs_hidden_source": "action_query", "target_token_id": 10004},
            "encoder": {"action_head_type": "legacy"},
        }
    )

    hidden_states = torch.randn(1, 3, 6)
    input_ids = torch.tensor([[11, 12, 13, 10004]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    action_hidden = torch.arange(1 * 5 * 4, dtype=torch.float32).reshape(1, 5, 4)

    class DummyEncoder:
        def extract_action_hidden(self, **kwargs):
            assert kwargs["hidden_states"] is hidden_states
            assert kwargs["input_ids"] is input_ids
            assert kwargs["attention_mask"] is attention_mask
            assert kwargs["target_token_id"] == 10004
            return action_hidden

    workspace.encoder = DummyEncoder()
    workspace._wm_io_mode = lambda: "hidden"  # type: ignore[method-assign]
    workspace._encode_hidden_sequence_from_tokenized = lambda _tokens: (  # type: ignore[method-assign]
        hidden_states,
        input_ids,
        attention_mask,
    )

    obs_embedding = workspace._obs_embedding_for_wm([[11, 12, 13]])

    assert obs_embedding.shape == (1, 5, 4)
    assert obs_embedding.tolist() == action_hidden.tolist()


def test_dreamer_eval_accepts_plain_dict_checkpoint_cfg() -> None:
    cfg = EvalLiberoVLARunner._checkpoint_cfg_from_payload(
        {
            "cfg": {
                "training": {"out_dir": "/tmp/eval"},
                "eval": {"task_suite_name": "libero_goal"},
            }
        }
    )

    assert OmegaConf.select(cfg, "training.out_dir") == "/tmp/eval"
    assert OmegaConf.select(cfg, "eval.task_suite_name") == "libero_goal"


def test_rynnvla_action_hidden_actor_decodes_flattened_action_hidden() -> None:
    actor = RynnVLAActionHiddenActor(
        action_hidden_dim=4,
        action_dim=3,
        time_horizon=5,
        adapter_type="identity",
    )
    assert actor.hidden_dim == 5 * 3 * 4

    action, log_prob, extra = actor(
        {
            "mode": "sample",
            "hidden": torch.randn(2, actor.hidden_dim),
            "deterministic": True,
        }
    )

    assert action.shape == (2, 3)
    assert log_prob.shape == (2,)
    assert extra["action_chunk"].shape == (2, 5, 3)


def test_latent_to_action_hidden_actor_bridges_tokenized_input_latents() -> None:
    actor = LatentToActionHiddenActor(
        hidden_dim=6 * 4,
        source_token_count=6,
        source_token_dim=4,
        action_hidden_dim=8,
        action_dim=2,
        time_horizon=3,
        bridge_hidden_dim=16,
        num_bridge_layers=1,
        num_bridge_heads=2,
        freeze_output_projection=False,
    )

    action, log_prob, extra = actor(
        {
            "mode": "sample",
            "hidden": torch.randn(2, 6, 4),
            "deterministic": True,
            "return_chunk": True,
        }
    )

    assert action.shape == (2, 3, 2)
    assert log_prob.shape == (2,)
    assert extra["action_hidden"].shape == (2, 6, 8)
    assert extra["action_chunk"].shape == (2, 3, 2)


def test_latent_to_action_hidden_actor_accepts_flat_latents() -> None:
    actor = LatentToActionHiddenActor(
        hidden_dim=5 * 4,
        source_token_count=5,
        source_token_dim=4,
        action_hidden_dim=8,
        action_dim=2,
        time_horizon=3,
        bridge_hidden_dim=16,
        num_bridge_layers=1,
        num_bridge_heads=2,
        adapter_type="identity",
    )

    action, _, _ = actor(
        {
            "mode": "sample",
            "hidden": torch.randn(2, 5 * 4),
            "deterministic": True,
            "return_chunk": True,
        }
    )

    assert action.shape == (2, 3, 2)


def test_rynn_dino_wm_derives_flat_action_hidden_dimensions() -> None:
    model = RynnDinoWMWorldModel(
        obs_dim=None,
        action_dim=3,
        token_count=None,
        token_dim=4,
        time_horizon=5,
        model_dim=16,
        depth=1,
        heads=4,
        mlp_dim=32,
        max_seq_len=8,
    )

    assert model.token_count == 5 * 3
    assert model.obs_dim == 5 * 3 * 4


def test_rynn_dino_wm_accepts_tokenized_action_hidden_without_flattening() -> None:
    model = RynnDinoWMWorldModel(
        obs_dim=None,
        action_dim=3,
        token_count=None,
        token_dim=4,
        time_horizon=5,
        model_dim=16,
        depth=1,
        heads=4,
        mlp_dim=32,
        max_seq_len=8,
    )
    tokens = torch.randn(2, 15, 4)
    flat = tokens.reshape(2, -1)

    assert model.obs_to_tokens(tokens).shape == (2, 1, 15, 4)
    assert torch.allclose(model.obs_to_tokens(tokens)[:, 0], tokens)
    assert torch.allclose(model.obs_to_tokens(flat)[:, 0], tokens)


def test_rynn_dino_wm_encode_latent_preserves_action_hidden_tokens() -> None:
    model = RynnDinoWMWorldModel(
        obs_dim=None,
        action_dim=3,
        token_count=None,
        token_dim=4,
        time_horizon=5,
        model_dim=16,
        depth=1,
        heads=4,
        mlp_dim=32,
        max_seq_len=8,
        num_hist=2,
    )
    tokens = torch.randn(2, 15, 4)

    latent = model.encode_latent(tokens)

    assert latent["hidden"].shape == (2, 15, 4)
    assert latent["history"].shape == (2, 2, 15, 4)
    assert model.actor_input(latent).shape == (2, 15, 4)
    assert model.critic_input(latent).shape == (2, 4)


def test_latent_success_classifier_derives_latent_dim() -> None:
    classifier = LatentSuccessClassifier(
        latent_dim=None,
        action_dim=3,
        time_horizon=5,
        token_dim=4,
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="transformer",
    )

    assert classifier.cfg.latent_dim == 5 * 3 * 4


def test_latent_success_classifier_accepts_tokenized_windows() -> None:
    classifier = LatentSuccessClassifier(
        latent_dim=None,
        action_dim=3,
        time_horizon=5,
        token_dim=4,
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="linear",
    )

    logits = classifier(torch.randn(3, 2, 15, 4))

    assert logits.shape == (3, 2)


def test_latent_success_classifier_can_mean_pool_tokenized_frames() -> None:
    classifier = LatentSuccessClassifier(
        latent_dim=None,
        action_dim=3,
        time_horizon=5,
        token_dim=4,
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="transformer",
        token_pool="mean",
    )

    logits = classifier(torch.randn(3, 2, 64, 4))

    assert classifier.cfg.latent_dim == 4
    assert logits.shape == (3, 2)


def test_rynnvla_action_hidden_actor_loads_vla_output_projection(tmp_path) -> None:
    source = RynnVLAActionHiddenActor(
        action_hidden_dim=4,
        action_dim=3,
        time_horizon=5,
        adapter_type="identity",
    )
    ckpt = {
        "state_dicts": {
            "encoder": {
                f"backbone.action_head.output_projection.{key}": value.detach().clone()
                for key, value in source.output_projection.state_dict().items()
            }
        }
    }
    path = tmp_path / "vla.ckpt"
    torch.save(ckpt, path)

    actor = RynnVLAActionHiddenActor(
        action_hidden_dim=4,
        action_dim=3,
        time_horizon=5,
        adapter_type="identity",
        init_action_head_ckpt=str(path),
    )

    for key, value in source.output_projection.state_dict().items():
        assert torch.equal(actor.output_projection.state_dict()[key], value)


def test_rynnvla_action_hidden_actor_rejects_ckpt_without_output_projection(
    tmp_path,
) -> None:
    path = tmp_path / "vla_without_projection.ckpt"
    torch.save({"state_dicts": {"encoder": {"backbone.other.weight": torch.ones(1)}}}, path)

    with pytest.raises(RuntimeError, match="output_projection"):
        RynnVLAActionHiddenActor(
            action_hidden_dim=4,
            action_dim=3,
            time_horizon=5,
            adapter_type="identity",
            init_action_head_ckpt=str(path),
        )


def test_dreamervla_init_loader_filters_and_remaps_compatible_state() -> None:
    class TinyWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.reward_head = nn.Module()
            self.reward_head.net = nn.Module()
            self.reward_head.net.net = nn.Linear(2, 1)
            self.keep = nn.Linear(2, 2)

    class DummyDistributed:
        def __init__(self) -> None:
            self.rank0_only_args: list[bool] = []
            self.is_main_process = False

        def model_state_dict_context(self, _module: nn.Module, rank0_only: bool = True):
            self.rank0_only_args.append(rank0_only)

            class Ctx:
                def __enter__(self):
                    return None

                def __exit__(self, *_exc):
                    return False

            return Ctx()

    workspace = DreamerVLARunner.__new__(DreamerVLARunner)
    workspace.world_model = nn.Module()
    workspace.world_model.module = TinyWorldModel()
    workspace.distributed = DummyDistributed()

    target = workspace.world_model.module
    original_keep_weight = target.keep.weight.detach().clone()
    reward_weight = torch.full_like(target.reward_head.net.net.weight, 0.25)
    reward_bias = torch.full_like(target.reward_head.net.net.bias, -0.5)

    workspace._load_compatible_module_state(
        "world_model",
        {
            "reward_head.net.weight": reward_weight,
            "reward_head.net.bias": reward_bias,
            "keep.weight": torch.zeros(3, 3),
        },
    )

    assert workspace.distributed.rank0_only_args == [False]
    assert torch.equal(target.reward_head.net.net.weight, reward_weight)
    assert torch.equal(target.reward_head.net.net.bias, reward_bias)
    assert torch.equal(target.keep.weight, original_keep_weight)
