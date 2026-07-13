from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from dreamervla.models.embodiment.openvla_oft_policy import OpenVLAOFTPolicy


class _FakeVisionBackbone:
    def __init__(self, token_count: int) -> None:
        self.token_count = int(token_count)

    def get_num_patches(self) -> int:
        return self.token_count

    def get_num_images_in_input(self) -> int:
        return 1


class _FunctionalEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.01))
        self.width = int(width)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        basis = torch.arange(self.width, device=token_ids.device, dtype=torch.float32)
        return torch.sin(token_ids.float().unsqueeze(-1) * self.scale + basis * 0.001)


class _FakeLanguageModel(nn.Module):
    def __init__(self, width: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, vocab_size, bias=False)

    def forward(self, *, inputs_embeds: torch.Tensor, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            logits=self.proj(inputs_embeds),
            hidden_states=(inputs_embeds,),
        )


class _FakeVLA(nn.Module):
    def __init__(self, *, token_count: int = 256, token_dim: int = 4096) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=int(token_dim))
        self.vision_backbone = _FakeVisionBackbone(token_count)
        self.vocab_size = 16
        self.bin_centers = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        self.embedding = _FunctionalEmbedding(token_dim)
        self.vision_scale = nn.Parameter(torch.tensor(0.5))
        self.language_model = _FakeLanguageModel(token_dim, self.vocab_size)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def _prepare_input_for_action_prediction(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_ids = torch.zeros(
            (input_ids.shape[0], 56), dtype=input_ids.dtype, device=input_ids.device
        )
        return (
            torch.cat([input_ids, action_ids], dim=1),
            torch.cat(
                [attention_mask, torch.ones_like(action_ids, dtype=attention_mask.dtype)],
                dim=1,
            ),
        )

    def _prepare_labels_for_action_prediction(
        self, labels: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        del labels
        out = torch.full_like(input_ids, -100)
        out[:, -56:] = input_ids[:, -56:]
        return out

    def _process_action_masks(self, labels: torch.Tensor) -> torch.Tensor:
        return labels != -100

    def _process_vision_features(
        self,
        pixel_values: torch.Tensor,
        language_embeddings: torch.Tensor,
        *,
        use_film: bool,
    ) -> torch.Tensor:
        del language_embeddings, use_film
        base = pixel_values.float().mean(dim=tuple(range(1, pixel_values.ndim)))
        return (
            base[:, None, None]
            * self.vision_scale
            * torch.ones(
                (
                    pixel_values.shape[0],
                    self.vision_backbone.get_num_patches(),
                    self.config.hidden_size,
                ),
                device=pixel_values.device,
            )
        )

    def _build_multimodal_attention(
        self,
        input_embeddings: torch.Tensor,
        projected: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat([input_embeddings[:, :1], projected, input_embeddings[:, 1:]], dim=1)
        vision_mask = torch.ones(
            (attention_mask.shape[0], projected.shape[1]),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        mask = torch.cat([attention_mask[:, :1], vision_mask, attention_mask[:, 1:]], dim=1)
        return embeddings, mask

    def _unnormalize_actions(self, actions, unnorm_key):
        assert unnorm_key == "fake_libero"
        return actions


def _policy() -> OpenVLAOFTPolicy:
    policy = OpenVLAOFTPolicy.from_modules(
        vla=_FakeVLA(),
        action_head=None,
        action_tokenizer=object(),
        num_patches=256,
    )
    policy.unnorm_key = "fake_libero"
    policy.action_dim = 7
    policy.time_horizon = 8
    return policy


def test_policy_geometry_is_derived_from_the_loaded_vla() -> None:
    policy = OpenVLAOFTPolicy.from_modules(
        vla=_FakeVLA(token_count=4, token_dim=12),
        action_head=None,
        action_tokenizer=object(),
        num_patches=4,
    )

    output = policy.forward_action_tokens(
        input_ids=torch.tensor([[1, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        pixel_values=torch.ones((1, 3, 2, 2)),
    )

    assert policy.token_count == 4
    assert policy.token_dim == 12
    assert output.projected_tokens.shape == (1, 4, 12)
    assert output.action_logits.shape == (1, 56, 8)


def test_raw_and_projected_token_paths_share_native_action_decoder() -> None:
    policy = _policy()
    input_ids = torch.tensor([[1, 7, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)

    raw = policy.forward_action_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )
    latent = policy.forward_action_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        projected_tokens=raw.projected_tokens.detach(),
    )

    assert raw.projected_tokens.shape == (1, 256, 4096)
    assert raw.action_logits.shape == (1, 56, 8)
    torch.testing.assert_close(latent.action_logits, raw.action_logits)
    torch.testing.assert_close(latent.language_embedding, raw.language_embedding)


def test_native_action_forward_keeps_encoder_and_actor_differentiable() -> None:
    policy = _policy()
    output = policy.forward_action_tokens(
        input_ids=torch.tensor([[1, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        pixel_values=torch.ones((1, 3, 2, 2)),
    )
    labels = output.action_logits.detach().argmax(dim=-1)
    loss, metrics = policy.action_token_loss(output, labels)
    loss.backward()

    assert policy.vla.vision_scale.grad is not None
    assert policy.vla.language_model.proj.weight.grad is not None
    assert metrics["action_token_count"] == 56.0


def test_native_action_forward_requires_exactly_one_visual_source() -> None:
    policy = _policy()
    common = {
        "input_ids": torch.tensor([[1, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }

    for kwargs in (
        {},
        {"pixel_values": torch.ones((1, 3, 2, 2)), "projected_tokens": torch.ones((1, 256, 4096))},
    ):
        try:
            policy.forward_action_tokens(**common, **kwargs)
        except ValueError as exc:
            assert "exactly one" in str(exc)
        else:
            raise AssertionError("expected the visual-source contract to reject the call")


def test_actor_interface_reuses_exact_native_action_tokens_for_evaluation() -> None:
    policy = _policy()
    sample_batch = {
        "mode": "sample",
        "hidden": torch.ones((2, 256, 4096)),
        "input_ids": torch.tensor([[1, 3], [1, 4]], dtype=torch.long),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
        "return_chunk": True,
        "deterministic": True,
        "logprob_type": "token_level",
    }

    actions, old_logprob, extra = policy(sample_batch)
    new_logprob, entropy, evaluated = policy(
        {
            **sample_batch,
            "mode": "evaluate",
            "action": actions,
            "action_token_ids": extra["action_token_ids"],
        }
    )

    assert actions.shape == (2, 8, 7)
    assert old_logprob.shape == (2, 8, 7)
    assert extra["action_token_ids"].shape == (2, 8, 7)
    torch.testing.assert_close(new_logprob, old_logprob)
    assert entropy.shape == old_logprob.shape
    torch.testing.assert_close(evaluated["action_token_ids"], extra["action_token_ids"])


def test_staged_modes_expose_encoder_only_sft_and_raw_reencoding() -> None:
    policy = _policy()
    common = {
        "pixel_values": torch.ones((2, 3, 2, 2)),
        "input_ids": torch.tensor([[1, 3], [1, 4]], dtype=torch.long),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
    }
    native = policy.forward_action_tokens(**common)
    labels = native.action_token_ids[native.action_logits.argmax(dim=-1)]

    loss, _, sft_extra = policy({"mode": "encoder_sft", **common, "action_token_ids": labels})
    encoded, _, encode_extra = policy({"mode": "encode_raw", **common})

    assert loss.ndim == 0
    torch.testing.assert_close(sft_extra["hidden"], native.projected_tokens)
    torch.testing.assert_close(encoded, native.projected_tokens)
    torch.testing.assert_close(encode_extra["hidden"], native.projected_tokens)
    assert "vla.vision_scale" in policy.encoder_parameter_names()
    assert all("language_model" not in name for name in policy.encoder_parameter_names())
