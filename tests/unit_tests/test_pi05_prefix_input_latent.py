from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg
from dreamervla.models.embodiment.pi05.prefix_input import (
    PI05_IMAGE_TOKEN_COUNT,
    PI05_PREFIX_INPUT_TOKEN_DIM,
    PI05_TEXT_TOKEN_COUNT,
    Pi05PrefixInputLatent,
    build_prefix_input_latent,
)


def _prefix_parts(batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    image = torch.ones(batch_size, PI05_IMAGE_TOKEN_COUNT, PI05_PREFIX_INPUT_TOKEN_DIM)
    language = torch.full(
        (batch_size, PI05_TEXT_TOKEN_COUNT, PI05_PREFIX_INPUT_TOKEN_DIM),
        2.0,
    )
    image_mask = torch.ones(batch_size, PI05_IMAGE_TOKEN_COUNT, dtype=torch.bool)
    language_mask = torch.zeros(batch_size, PI05_TEXT_TOKEN_COUNT, dtype=torch.bool)
    language_mask[:, :7] = True
    return image, language, image_mask, language_mask


def test_prefix_input_exact_shape_mask_and_position_ids() -> None:
    image, language, image_mask, language_mask = _prefix_parts()

    prefix = build_prefix_input_latent(
        image,
        language,
        image_mask,
        language_mask,
        text_mode="exact",
    )

    assert prefix.latent.shape == (1, 968, 2048)
    assert prefix.attention_mask.shape == (1, 968)
    torch.testing.assert_close(prefix.latent[:, :768], image)
    torch.testing.assert_close(prefix.latent[:, 768:], language)
    torch.testing.assert_close(prefix.attention_mask[:, 768:], language_mask)
    assert prefix.position_ids[0, 767].item() == 767
    assert prefix.position_ids[0, 774].item() == 774
    assert prefix.position_ids[0, -1].item() == 774


def test_prefix_input_masked_keeps_geometry_and_removes_text() -> None:
    image, language, image_mask, language_mask = _prefix_parts()

    prefix = build_prefix_input_latent(
        image,
        language,
        image_mask,
        language_mask,
        text_mode="masked",
    )

    assert prefix.latent.shape == (1, 968, 2048)
    assert torch.count_nonzero(prefix.latent[:, 768:]).item() == 0
    assert torch.count_nonzero(prefix.attention_mask[:, 768:]).item() == 0
    torch.testing.assert_close(prefix.latent[:, :768], image)


def test_native_pi05_action_inference_prefills_prefix_and_uses_flow_expert() -> None:
    from dreamervla.models.embodiment.pi05.policy import Pi05Policy

    class _Prefill:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.paligemma = SimpleNamespace(
                language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="unset"))
            )

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            return (None, None), "native-kv-cache"

    class _Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(action_horizon=10, action_dim=32)
            self.paligemma_with_expert = _Prefill()
            self.cache_seen = None

        @staticmethod
        def _prepare_attention_masks_4d(mask):
            return mask[:, None]

        @staticmethod
        def sample_noise(shape, device):
            return torch.zeros(shape, device=device)

        def denoise_step(self, state, prefix_mask, cache, x_t, timestep):
            del state, prefix_mask, timestep
            self.cache_seen = cache
            return torch.ones_like(x_t)

    policy = Pi05Policy.__new__(Pi05Policy)
    torch.nn.Module.__init__(policy)
    policy.register_parameter("device_anchor", torch.nn.Parameter(torch.zeros(())))
    policy.model = _Model()
    policy.num_steps = 1
    image, language, image_mask, language_mask = _prefix_parts()
    prefix = build_prefix_input_latent(
        image,
        language,
        image_mask,
        language_mask,
        text_mode="exact",
    )

    actions = policy.sample_actions_from_prefix_input(
        prefix,
        state=torch.zeros(1, 32),
        noise=torch.zeros(1, 10, 32),
    )

    assert actions.shape == (1, 10, 32)
    torch.testing.assert_close(actions, -torch.ones_like(actions))
    assert policy.model.cache_seen == "native-kv-cache"
    call = policy.model.paligemma_with_expert.calls[0]
    assert call["use_cache"] is True
    assert call["inputs_embeds"][0].shape == (1, 968, 2048)
    torch.testing.assert_close(call["position_ids"], prefix.position_ids)
    assert call["attention_mask"][0, 0, 0, 768]
    assert not call["attention_mask"][0, 0, 0, -1]
    assert not call["attention_mask"][0, 0, -1].any()


def _bare_prefix_wm(*, wm_use_text: bool = True, policy_use_text: bool = True):
    from dreamervla.models.embodiment.world_model.wm_pi05_prefix_input import (
        Pi05PrefixInputWorldModel,
    )

    wm = Pi05PrefixInputWorldModel.__new__(Pi05PrefixInputWorldModel)
    torch.nn.Module.__init__(wm)
    wm.image_token_count = 768
    wm.text_token_count = 200
    wm.token_count = 968
    wm.token_dim = 2048
    wm.text_mode = "exact"
    wm.wm_use_text = wm_use_text
    wm.policy_use_text = policy_use_text
    wm.hidden_loss_scale = 1.0
    wm.cosine_loss_scale = 0.0

    def obs_to_tokens(_self, value):
        return value if value.ndim == 4 else value[:, None]

    wm.obs_to_tokens = MethodType(obs_to_tokens, wm)
    return wm


def test_wm_use_text_false_masks_wm_but_preserves_exact_policy_language() -> None:
    wm = _bare_prefix_wm(wm_use_text=False)
    raw = torch.ones(1, 1, 968, 2048)
    mask = torch.ones(1, 1, 968, dtype=torch.bool)
    mask[..., -3:] = False

    wm_tokens, exact, canonical_mask = wm._prepare_wm_observation(raw, mask)

    assert torch.count_nonzero(wm_tokens[..., 768:, :]).item() == 0
    assert torch.count_nonzero(exact[..., 768:, :]).item() > 0
    torch.testing.assert_close(canonical_mask, mask)


def test_policy_use_text_false_recomposes_masked_native_prefix() -> None:
    wm = _bare_prefix_wm(policy_use_text=False)
    hidden = torch.ones(1, 968, 2048)

    def latent_hidden(_self, _latent):
        return hidden

    wm._latent_hidden = MethodType(latent_hidden, wm)
    prefix = wm.policy_prefix_input(
        {
            "hidden": hidden,
            "lang": torch.full((1, 200, 2048), 2.0),
            "prefix_attention_mask": torch.ones(1, 968, dtype=torch.bool),
        }
    )

    assert prefix.text_mode == "masked"
    assert torch.count_nonzero(prefix.latent[:, 768:]).item() == 0
    assert torch.count_nonzero(prefix.attention_mask[:, 768:]).item() == 0


def test_prefix_input_wm_loss_ignores_language_slots() -> None:
    wm = _bare_prefix_wm()
    prediction = torch.zeros(1, 1, 968, 2048)
    target = prediction.clone()
    target[..., 768:, :] = 100.0

    loss, mse, cosine = wm._hidden_loss_terms(prediction, target)

    torch.testing.assert_close(loss, torch.zeros(()))
    torch.testing.assert_close(mse, torch.zeros(()))
    torch.testing.assert_close(cosine, torch.ones(()))
    target[..., 0, 0] = 1.0
    changed_loss, changed_mse, _ = wm._hidden_loss_terms(prediction, target)
    assert changed_loss > 0
    assert changed_mse > 0


def test_prefix_input_hydra_route_and_geometry_validation() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=wm_pi05_prefix_input_train"],
        )
    OmegaConf.resolve(cfg)

    assert cfg.world_model._target_.endswith("Pi05PrefixInputWorldModel")
    assert cfg.world_model.token_count == 968
    assert cfg.world_model.image_token_count == 768
    assert cfg.world_model.text_token_count == 200
    assert cfg.world_model.text_mode == "exact"
    assert cfg.world_model.wm_use_text is True
    assert cfg.world_model.policy_use_text is True
    assert cfg.world_model.predict_text_tokens is False

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        invalid = compose(config_name="train", overrides=["experiment=pi05_libero_sft"])
    invalid.task.prefix_input_latent.token_count = 967
    with pytest.raises(ValueError, match=r"768 \+ 200 != 967"):
        validate_cfg(invalid, world_size=1)


def test_prefix_input_hydra_route_passes_full_validation(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=wm_pi05_prefix_input_train"],
        )
    cfg.offline_warmup.data_dir = str(tmp_path)

    validate_cfg(cfg, world_size=8)


def test_prefix_input_latent_sidecar_contract_has_distinct_source() -> None:
    from dreamervla.runtime.observation_latent import ObservationLatentSpec

    spec = ObservationLatentSpec(
        policy_family="pi05",
        obs_hidden_source="prefix_input_latent",
        action_head_type="flow_matching",
        token_count=968,
        token_dim=2048,
        image_token_count=768,
        text_token_count=200,
        chunk_size=10,
        include_state=True,
        prefix_selection="image_language",
        alignment_source="openpi_prefix_input",
        text_mode="exact",
        wm_use_text=True,
        policy_use_text=True,
        predict_text_tokens=False,
        prefix_attention_mask_key="prefix_attention_mask",
    )

    manifest = spec.preprocess_config()

    assert manifest["obs_hidden_source"] == "prefix_input_latent"
    assert manifest["obs_embedding_shape"] == [968, 2048]
    assert manifest["sidecar_schema_version"] == 3


def test_prefix_input_bundle_rejects_wrong_geometry() -> None:
    with pytest.raises(ValueError, match=r"\[B,968,2048\]"):
        Pi05PrefixInputLatent(
            latent=torch.zeros(1, 768, 2048),
            attention_mask=torch.ones(1, 768, dtype=torch.bool),
            position_ids=torch.arange(768)[None],
            text_mode="exact",
        )
