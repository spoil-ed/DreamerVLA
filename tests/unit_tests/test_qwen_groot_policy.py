from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from dreamervla.models.embodiment import QwenGR00TPolicy
from dreamervla.models.embodiment.qwen_groot import GR00TActionHead, QwenBackbone


class _TinyVLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(1, 2, bias=False)

    def build_qwenvl_inputs(self, images, instructions):
        del images
        values = torch.arange(len(instructions) * 3, dtype=torch.float32).reshape(-1, 3)
        return {
            "input_ids": values,
            "attention_mask": torch.ones_like(values, dtype=torch.long),
        }

    def forward(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        hidden = self.projection(input_ids.unsqueeze(-1))
        return SimpleNamespace(hidden_states=[hidden - 1.0, hidden])


def _head_config() -> dict:
    return {
        "variant": "Tiny",
        "action_dim": 2,
        "state_dim": 2,
        "action_horizon": 2,
        "hidden_size": 2,
        "input_embedding_dim": 2,
        "backbone_embedding_dim": 2,
        "max_num_embodiments": 1,
        "add_pos_embed": False,
        "use_vlln": False,
        "use_alternate_vl_dit": False,
        "vl_self_attention_cfg": {"num_layers": 0},
        "repeated_diffusion_steps": 1,
        "num_inference_timesteps": 1,
        "diffusion_model_cfg": {
            "num_attention_heads": 1,
            "attention_head_dim": 2,
            "output_dim": 2,
            "num_layers": 1,
            "cross_attention_dim": 2,
            "positional_embeddings": None,
            "dropout": 0.0,
            "final_dropout": False,
        },
    }


def _policy() -> QwenGR00TPolicy:
    backbone = QwenBackbone(vlm=_TinyVLM(), input_cameras=["front"], include_state=True)
    action_head = GR00TActionHead(**_head_config())
    return QwenGR00TPolicy(
        input_cameras=["front"],
        camera_keys=["front"],
        action_horizon=2,
        action_dim=2,
        state_dim=2,
        include_state=True,
        freeze_backbone=False,
        freeze_state_encoder=False,
        action_min=[-2.0, 0.0],
        action_max=[2.0, 1.0],
        action_normalized_mask=[True, False],
        backbone=backbone,
        action_head=action_head,
    )


def _batch() -> dict:
    return {
        "prompt_text": ["Task A", "Task B"],
        "images": [
            [np.zeros((2, 2, 3), dtype=np.uint8)],
            [np.ones((2, 2, 3), dtype=np.uint8)],
        ],
        "state": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "action": torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ]
        ),
    }


def test_qwen_groot_forward_backward_and_predict_shapes() -> None:
    policy = _policy()
    torch.manual_seed(101)
    loss = policy({"mode": "sft", "data": _batch()})

    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.action_head.action_decoder.layer2.weight.grad is not None

    policy.eval()
    torch.manual_seed(202)
    predicted = policy({"mode": "predict", "data": _batch()})
    assert predicted["normalized_actions"].shape == (2, 2, 2)
    assert np.isfinite(predicted["normalized_actions"]).all()


def test_qwen_groot_retains_sipai_action_head_state_dict_keys() -> None:
    head = GR00TActionHead(**_head_config())
    keys = set(head.state_dict())

    assert len(keys) == 36
    assert "model.transformer_blocks.0.attn1.to_q.weight" in keys
    assert "action_encoder.layer1.weight" in keys
    assert "action_decoder.layer2.weight" in keys
    clone = GR00TActionHead(**_head_config())
    incompatible = clone.load_state_dict(head.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_qwen_groot_aligns_bfloat16_backbone_to_float32_action_head() -> None:
    head = GR00TActionHead(**_head_config()).eval()
    encoded = {
        "last_hidden": torch.randn(1, 3, 2, dtype=torch.bfloat16),
        "encoder_attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "image_mask": torch.tensor([[True, False, False]]),
        "state": torch.randn(1, 2, dtype=torch.bfloat16),
    }

    output = head.predict_action(encoded)

    assert output["normalized_actions"].dtype == np.float32
    assert output["normalized_actions"].shape == (1, 2, 2)
    assert np.isfinite(output["normalized_actions"]).all()


def test_qwen_groot_checkpoint_round_trip(tmp_path) -> None:
    policy = _policy()
    path = tmp_path / "sipai-export.pt"
    torch.save({"model": policy.state_dict()}, path)

    restored = _policy()
    restored.load_sipai_checkpoint(str(path))
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_qwen_groot_libero_action_normalization_round_trip() -> None:
    policy = _policy()
    actions = torch.tensor([[[0.0, 0.75], [1.0, 0.25]]])

    normalized = policy.normalize_actions(actions)
    restored = policy.denormalize_actions(normalized)

    torch.testing.assert_close(restored, actions)
    torch.testing.assert_close(normalized[..., 1], actions[..., 1])


def test_qwen_groot_hydra_component_config_composes() -> None:
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["VLA=qwen_groot"])

    assert cfg.VLA._target_ == "dreamervla.models.embodiment.QwenGR00TPolicy"
    assert cfg.VLA.action_horizon == 8
    assert cfg.VLA.action_head_cfg.variant == "DiT-L"
