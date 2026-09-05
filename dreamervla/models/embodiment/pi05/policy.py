"""RLinf-aligned π0.5 policy built on the official OpenPI model."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils._pytree import tree_map

from dreamervla.models.embodiment.pi05.openpi_config import (
    PI05_LIBERO_CONFIG_NAME,
    get_pi05_libero_config,
)
from dreamervla.models.embodiment.pi05.prefix_input import (
    PI05_IMAGE_TOKEN_COUNT,
    PI05_TEXT_TOKEN_COUNT,
    Pi05ImagePrefixLatent,
    Pi05PrefixInputLatent,
    build_prefix_input_latent,
    prefix_attention_matrix,
)
from dreamervla.models.embodiment.pi05.pytree import register_pytree_dataclasses
from dreamervla.utils.openpi_imports import ensure_openpi_on_path


def _freeze_unused_continuous_action_parameters(model: nn.Module) -> int:
    """Freeze the OpenPI language head that flow-matching SFT never executes."""

    try:
        lm_head = model.paligemma_with_expert.gemma_expert.lm_head
    except AttributeError as exc:
        raise RuntimeError("π0.5 model is missing gemma_expert.lm_head") from exc
    frozen = 0
    for parameter in lm_head.parameters():
        parameter.requires_grad = False
        frozen += parameter.numel()
    return frozen


class Pi05Policy(nn.Module):
    """Official OpenPI π0.5 with RLinf-aligned, SFT-only training semantics.

    The OpenPI construction, freeze boundary, transforms, SFT reduction, and
    FSDP wrap targets follow RLinf's OpenPI action model. DreamerVLA owns only
    the runner lifecycle and its repository-wide checkpoint/logging contract.
    """

    policy_family = "pi05"
    preserve_parameter_dtypes = True
    alignment_source = "RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py:sft_forward"

    def __init__(
        self,
        model_path: str,
        assets_path: str | None = None,
        config_name: str = PI05_LIBERO_CONFIG_NAME,
        batch_size: int = 256,
        num_workers: int = 2,
        seed: int = 0,
        learning_rate: float = 5e-5,
        lr_warmup_steps: int = 1000,
        total_training_steps: int = 30_000,
        action_chunk: int = 10,
        action_dim: int = 7,
        num_steps: int = 5,
        train_expert_only: bool = True,
        add_value_head: bool = False,
        rotate_images_180: bool = True,
    ) -> None:
        super().__init__()
        if add_value_head:
            raise ValueError("π0.5 SFT does not construct an RL/PPO value head")
        checkpoint = Path(model_path).expanduser().resolve()
        assets_checkpoint = (
            checkpoint if assets_path is None else Path(assets_path).expanduser().resolve()
        )
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"π0.5 checkpoint is missing {weights}")

        ensure_openpi_on_path()
        from openpi import transforms
        from openpi.training import checkpoints

        if str(config_name) != PI05_LIBERO_CONFIG_NAME:
            raise ValueError(f"π0.5 policy requires {PI05_LIBERO_CONFIG_NAME}; got {config_name!r}")
        train_config = get_pi05_libero_config(
            model_path=str(checkpoint),
            assets_path=str(assets_checkpoint),
            batch_size=int(batch_size),
            action_horizon=int(action_chunk),
            num_workers=int(num_workers),
            seed=int(seed),
            learning_rate=float(learning_rate),
            lr_warmup_steps=int(lr_warmup_steps),
            total_training_steps=int(total_training_steps),
        )
        if not bool(getattr(train_config.model, "pi05", False)):
            raise ValueError(f"OpenPI config {config_name!r} is not a π0.5 model")
        if int(train_config.model.action_horizon) != int(action_chunk):
            raise ValueError(
                "π0.5 action_chunk must match the OpenPI action horizon: "
                f"{action_chunk} != {train_config.model.action_horizon}"
            )
        if int(action_dim) > int(train_config.model.action_dim):
            raise ValueError("environment action_dim exceeds the OpenPI padded action dimension")

        model = train_config.model.load_pytorch(train_config, str(weights))
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        if train_expert_only:
            # This is RLinf's freeze_vlm boundary: the vision/language prefix is
            # frozen while Gemma expert and action projections remain trainable.
            model.paligemma_with_expert.paligemma.eval()
            for parameter in model.paligemma_with_expert.paligemma.parameters():
                parameter.requires_grad = False
        self.frozen_unused_parameters = _freeze_unused_continuous_action_parameters(model)
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        self.model = model
        self.action_chunk = int(action_chunk)
        self.action_dim = int(action_dim)
        self.num_steps = int(num_steps)
        self.rotate_images_180 = bool(rotate_images_180)

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        if data_config.asset_id is None:
            raise ValueError("π0.5 LIBERO config has no normalization asset id")
        # RLinf checkpoints put the asset-id directory directly below the model
        # root (rather than below an additional assets/ directory).
        norm_stats = checkpoints.load_norm_stats(assets_checkpoint, data_config.asset_id)
        self._input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = transforms.compose(
            [
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.data_transforms.outputs,
            ]
        )
        for name, module in self.named_modules():
            module._fsdp_wrap_name = name.rsplit(".", maxsplit=1)[-1]

    @property
    def _no_split_modules(self) -> list[str]:
        """RLinf's train-expert-only OpenPI FSDP class targets."""

        return [
            "GemmaDecoderLayer",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaRotaryEmbedding",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        """RLinf's standalone OpenPI projection wrap targets."""

        return [
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            "time_mlp_in",
            "time_mlp_out",
        ]

    def get_fsdp_wrap_module_list(self) -> list[nn.Module]:
        """Resolve RLinf's class/name policy for DreamerVLA's FSDP helper."""

        class_names = set(self._no_split_modules)
        module_names = set(self._no_split_names)
        return [
            module
            for module in self.modules()
            if module is not self
            and (
                type(module).__name__ in class_names
                or getattr(module, "_fsdp_wrap_name", None) in module_names
            )
        ]

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run the SFT loss or deterministic LIBERO sampling path."""

        mode = str(batch.get("mode", "sft")).lower()
        if mode == "sft":
            return self.sft_forward(
                batch["data"],
                use_action_chunk_loss=bool(batch.get("use_action_chunk_loss", False)),
            )
        if mode == "sample":
            observations = batch.get("observations")
            if not isinstance(observations, list):
                raise TypeError("π0.5 sample mode requires observations as a list")
            return torch.stack([self.infer_one(observation) for observation in observations])
        if mode == "sample_prefix_input":
            prefix = batch.get("prefix_input")
            state = batch.get("state")
            if not isinstance(prefix, Pi05PrefixInputLatent):
                raise TypeError("sample_prefix_input requires Pi05PrefixInputLatent")
            if not isinstance(state, torch.Tensor):
                raise TypeError("sample_prefix_input requires tensor state")
            return self.sample_actions_from_prefix_input(
                prefix,
                state=state,
                noise=batch.get("noise"),
            )
        raise ValueError(
            f"π0.5 SFT policy mode must be sft, sample, or sample_prefix_input; got {mode!r}"
        )

    def sft_forward(self, data: Any, *, use_action_chunk_loss: bool = False) -> torch.Tensor:
        """Match RLinf's OpenPI SFT batch conversion and scalar reduction."""

        if isinstance(data, tuple):
            observation, actions = data
        else:
            observation = data["observation"]
            actions = data["actions"]

        device = next(self.parameters()).device
        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda value: (
                torch.as_tensor(value, device=device).contiguous().clone()
                if value is not None
                else None
            ),
            observation,
        )
        actions = torch.as_tensor(actions, device=device, dtype=torch.float32)
        loss = self.model(observation, actions)
        if use_action_chunk_loss:
            loss = loss[:, : self.action_chunk, : self.action_dim]
        return loss.mean()

    @torch.no_grad()
    def infer_one(self, observation: dict[str, Any]) -> torch.Tensor:
        """Transform one raw LIBERO observation and sample one action chunk."""
        actions, _prefix = self.infer_batch_with_prefix([observation])
        return actions[0]

    @torch.no_grad()
    def encode_observation_prefix_bundle(self, observation: Any) -> Pi05ImagePrefixLatent:
        """Encode an OpenPI loader observation into the RLinf image prefix.

        Unlike :meth:`infer_batch_with_prefix`, this path does not run the
        flow-matching expert. It is intended for frozen feature consumers such
        as the pixel decoder and produces the exact normalized-WM source tensor
        shape ``[B,768,2048]``.
        """

        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        device = next(self.parameters()).device
        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda value: (
                torch.as_tensor(value, device=device).contiguous() if value is not None else None
            ),
            observation,
        )
        images, img_masks, lang_tokens, lang_masks, _state = self.model._preprocess_observation(
            observation,
            train=False,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        attention_mask = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
            "eager"
        )
        (prefix_output, _), _ = self.model.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        image_token_count = int(prefix_output.shape[1] - lang_tokens.shape[1])
        image_prefix = prefix_output[:, :image_token_count].detach()
        if tuple(image_prefix.shape[1:]) != (768, 2048):
            raise ValueError(
                "RLinf-aligned π0.5 image prefix must be [768,2048], got "
                f"{tuple(image_prefix.shape[1:])}"
            )
        image_mask = prefix_pad_masks[:, :image_token_count].to(dtype=torch.bool)
        return Pi05ImagePrefixLatent(latent=image_prefix, attention_mask=image_mask)

    @torch.no_grad()
    def encode_observation_prefix(self, observation: Any) -> torch.Tensor:
        """Return only the image-prefix tensor for legacy feature consumers."""

        return self.encode_observation_prefix_bundle(observation).latent

    @torch.no_grad()
    def encode_raw_observation_prefix_batch(
        self,
        observations: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Transform raw OpenPI dictionaries and encode only their image prefix."""

        if not observations:
            raise ValueError("π0.5 prefix encoding requires at least one observation")
        from openpi.models import model as openpi_model

        transformed = [self._input_transform(copy.deepcopy(item)) for item in observations]
        device = next(self.parameters()).device
        tensor_inputs = tree_map(
            lambda *values: torch.stack(
                [torch.from_numpy(np.asarray(value)) for value in values], dim=0
            ).to(device),
            *transformed,
        )
        model_observation = openpi_model.Observation.from_dict(tensor_inputs)
        return self.encode_observation_prefix(model_observation)

    @torch.no_grad()
    def encode_raw_observation_prefix_bundle_batch(
        self,
        observations: list[dict[str, Any]],
    ) -> Pi05ImagePrefixLatent:
        """Encode image-prefix outputs together with native image-slot masks."""

        if not observations:
            raise ValueError("π0.5 prefix encoding requires at least one observation")
        from openpi.models import model as openpi_model

        transformed = [self._input_transform(copy.deepcopy(item)) for item in observations]
        device = next(self.parameters()).device
        tensor_inputs = tree_map(
            lambda *values: torch.stack(
                [torch.from_numpy(np.asarray(value)) for value in values], dim=0
            ).to(device),
            *transformed,
        )
        model_observation = openpi_model.Observation.from_dict(tensor_inputs)
        return self.encode_observation_prefix_bundle(model_observation)

    @torch.no_grad()
    def encode_observation_prefix_input(
        self,
        observation: Any,
        *,
        text_mode: str = "exact",
    ) -> Pi05PrefixInputLatent:
        """Return native image/language embeddings before PaliGemma prefill."""

        device = next(self.parameters()).device
        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda value: (
                torch.as_tensor(value, device=device).contiguous() if value is not None else None
            ),
            observation,
        )
        images, img_masks, lang_tokens, lang_masks, _state = self.model._preprocess_observation(
            observation,
            train=False,
        )
        prefix_embs, prefix_pad_masks, _prefix_att_masks = self.model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )
        if int(lang_tokens.shape[1]) != PI05_TEXT_TOKEN_COUNT:
            raise ValueError(
                "π0.5 tokenizer must produce exactly 200 language slots, got "
                f"{int(lang_tokens.shape[1])}"
            )
        image_embeddings = prefix_embs[:, :PI05_IMAGE_TOKEN_COUNT]
        language_embeddings = prefix_embs[:, PI05_IMAGE_TOKEN_COUNT:]
        image_mask = prefix_pad_masks[:, :PI05_IMAGE_TOKEN_COUNT]
        language_mask = prefix_pad_masks[:, PI05_IMAGE_TOKEN_COUNT:]
        return build_prefix_input_latent(
            image_embeddings,
            language_embeddings,
            image_mask,
            language_mask,
            text_mode=text_mode,
        )

    @torch.no_grad()
    def encode_raw_observation_prefix_input_batch(
        self,
        observations: list[dict[str, Any]],
        *,
        text_mode: str = "exact",
    ) -> Pi05PrefixInputLatent:
        """Transform raw OpenPI dictionaries into native prefix-input latents."""

        if not observations:
            raise ValueError("π0.5 prefix-input encoding requires at least one observation")
        from openpi.models import model as openpi_model

        transformed = [self._input_transform(copy.deepcopy(item)) for item in observations]
        device = next(self.parameters()).device
        tensor_inputs = tree_map(
            lambda *values: torch.stack(
                [torch.from_numpy(np.asarray(value)) for value in values], dim=0
            ).to(device),
            *transformed,
        )
        model_observation = openpi_model.Observation.from_dict(tensor_inputs)
        return self.encode_observation_prefix_input(model_observation, text_mode=text_mode)

    def _sample_model_actions_from_cache(
        self,
        *,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the unmodified π0.5 action expert and flow denoising loop."""

        device = prefix_pad_masks.device
        batch_size = int(prefix_pad_masks.shape[0])
        action_shape = (batch_size, self.model.config.action_horizon, self.model.config.action_dim)
        x_t = self.model.sample_noise(action_shape, device) if noise is None else noise.to(device)
        if tuple(x_t.shape) != action_shape:
            raise ValueError(
                f"π0.5 action noise must have shape {action_shape}, got {tuple(x_t.shape)}"
            )
        dt = torch.tensor(-1.0 / self.num_steps, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            velocity = self.model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                time.expand(batch_size),
            )
            x_t = x_t + dt * velocity
            time += dt
        return x_t

    @torch.no_grad()
    def sample_actions_from_prefix_input(
        self,
        prefix: Pi05PrefixInputLatent,
        *,
        state: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prefill native PaliGemma from `[B,968,2048]`, then sample actions."""

        device = next(self.parameters()).device
        language_model = self.model.paligemma_with_expert.paligemma.language_model
        embed_tokens = getattr(language_model, "embed_tokens", None)
        embedding_weight = getattr(embed_tokens, "weight", None)
        prefix_dtype = (
            embedding_weight.dtype
            if isinstance(embedding_weight, torch.Tensor)
            else prefix.latent.dtype
        )
        prefix_embs = prefix.latent.to(device=device, dtype=prefix_dtype)
        prefix_pad_masks = prefix.attention_mask.to(device=device, dtype=torch.bool)
        attention_mask = self.model._prepare_attention_masks_4d(
            prefix_attention_matrix(prefix_pad_masks)
        )
        position_ids = prefix.position_ids.to(device=device, dtype=torch.long)
        language_model.config._attn_implementation = "eager"
        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32)
        if state_tensor.ndim != 2 or state_tensor.shape[0] != prefix_embs.shape[0]:
            raise ValueError(
                f"π0.5 prefix-input action state must be [B,D], got {tuple(state_tensor.shape)}"
            )
        return self._sample_model_actions_from_cache(
            state=state_tensor,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            noise=noise,
        )

    @torch.no_grad()
    def infer_batch_with_prefix(
        self, observations: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return RLinf-aligned image-only prefix outputs.

        RLinf's RLT path extracts the PaliGemma *output* after building the prefix
        KV cache, then removes the fixed language-token tail.  For pi0.5 this is
        a ``[B, 768, 2048]`` tensor (three 256-token image slots, including the
        standard padded third slot).  Collection persists this exact tensor as
        the world-model observation rather than introducing another encoder.
        """

        if not observations:
            raise ValueError("π0.5 inference requires at least one observation")
        from openpi.models import model as openpi_model
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        transformed = [self._input_transform(copy.deepcopy(item)) for item in observations]
        device = next(self.parameters()).device
        tensor_inputs = tree_map(
            lambda *values: torch.stack(
                [torch.from_numpy(np.asarray(value)) for value in values], dim=0
            ).to(device),
            *transformed,
        )
        model_observation = openpi_model.Observation.from_dict(tensor_inputs)
        images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(
            model_observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
            "eager"
        )
        (prefix_output, _), past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        batch_size = int(state.shape[0])
        x_t = self._sample_model_actions_from_cache(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
        )

        env_actions: list[torch.Tensor] = []
        for index in range(batch_size):
            output = {
                "state": np.asarray(tensor_inputs["state"][index].detach().cpu()),
                "actions": np.asarray(x_t[index].detach().cpu()),
            }
            item = self._output_transform(output)
            env_actions.append(
                torch.from_numpy(np.asarray(item["actions"], dtype=np.float32)).to(device)
            )

        image_token_count = int(prefix_output.shape[1] - lang_tokens.shape[1])
        image_prefix = prefix_output[:, :image_token_count].detach()
        if tuple(image_prefix.shape[1:]) != (768, 2048):
            raise ValueError(
                "RLinf-aligned π0.5 image prefix must be [768,2048], got "
                f"{tuple(image_prefix.shape[1:])}"
            )
        return torch.stack(env_actions), image_prefix

    def make_extractor(self) -> _Pi05RawExtractor:
        """Expose raw LIBERO inference to the shared evaluation runner."""

        return _Pi05RawExtractor(self)

    def load_sft_delta(self, checkpoint_path: str) -> None:
        """Restore a DreamerVLA π0.5 trainable-parameter checkpoint."""

        from collections.abc import Mapping

        from dreamervla.utils.hf_checkpoint import load_runner_payload
        from dreamervla.utils.run_paths import resolve_resume_checkpoint

        resolved = resolve_resume_checkpoint(checkpoint_path)
        payload = load_runner_payload(resolved)
        policy_state = payload.get("state_dicts", {}).get("policy")
        if not isinstance(policy_state, Mapping) or not policy_state:
            raise RuntimeError(f"{resolved} has no non-empty state_dicts.policy")
        missing, unexpected = self.load_state_dict(dict(policy_state), strict=False)
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        missing_trainable = trainable_names.intersection(missing)
        if missing_trainable or unexpected:
            raise RuntimeError(
                "π0.5 SFT delta mismatch: "
                f"missing_trainable={sorted(missing_trainable)[:5]} "
                f"unexpected={list(unexpected)[:5]}"
            )


class _Pi05RawExtractor:
    actions_are_env_ready = True

    def __init__(self, policy: Pi05Policy) -> None:
        self.policy = policy

    def reset(self) -> None:
        return None

    def step(self, observation: dict[str, Any], task_description: str) -> SimpleNamespace:
        raw = {
            "observation/image": _libero_eval_image(
                observation["agentview_rgb"], rotate_180=self.policy.rotate_images_180
            ),
            "observation/wrist_image": _libero_eval_image(
                observation["eye_in_hand_rgb"], rotate_180=self.policy.rotate_images_180
            ),
            "observation/state": np.asarray(
                observation.get("state", observation.get("proprio")),
                dtype=np.float32,
            ),
            "prompt": str(task_description),
        }
        action_chunk = self.policy.infer_one(raw).detach().cpu().numpy()
        return SimpleNamespace(action_chunk=action_chunk)


def _hwc_uint8(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"π0.5 image must be rank-3, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"π0.5 image must be HWC/CHW RGB, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image, initial=0.0)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _libero_eval_image(value: Any, *, rotate_180: bool) -> np.ndarray:
    image = _hwc_uint8(value)
    if rotate_180:
        image = image[::-1, ::-1]
    return np.ascontiguousarray(image)


__all__ = ["Pi05Policy"]
