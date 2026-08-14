"""Standalone DreamerVLA π0.5 SFT policy built on the official OpenPI model."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils._pytree import tree_map

from dreamervla.models.embodiment.pi05.pytree import register_pytree_dataclasses
from dreamervla.utils.openpi_imports import ensure_openpi_on_path


class Pi05Policy(nn.Module):
    """Official OpenPI π0.5 with RLinf-aligned, SFT-only training semantics.

    RLinf remains the behavioral reference, but is deliberately not imported at
    runtime.  DreamerVLA owns the torch module, DDP lifecycle, optimizer and
    checkpoint while the installed ``rlinf-openpi`` package supplies OpenPI.
    """

    policy_family = "pi05"
    preserve_parameter_dtypes = True
    alignment_source = "RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py:sft_forward"

    def __init__(
        self,
        model_path: str,
        config_name: str = "pi05_libero",
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
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"π0.5 checkpoint is missing {weights}")

        ensure_openpi_on_path()
        from openpi import transforms
        from openpi.training import checkpoints
        from openpi.training import config as training_config

        train_config = training_config.get_config(str(config_name))
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
        # Flow matching consumes the expert hidden states through action_out_proj;
        # its vocabulary projection is never called. RLinf/FSDP can leave this
        # 257152x1024 tensor nominally trainable, but native DDP correctly flags
        # it as unused. Freezing it is behavior-preserving and avoids a 263M-param
        # gradient bucket plus optimizer allocation.
        expert_lm_head = model.paligemma_with_expert.gemma_expert.lm_head
        for parameter in expert_lm_head.parameters():
            parameter.requires_grad = False
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
        norm_stats = checkpoints.load_norm_stats(checkpoint, data_config.asset_id)
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
        raise ValueError(f"π0.5 SFT policy mode must be sft or sample; got {mode!r}")

    def sft_forward(self, data: Any, *, use_action_chunk_loss: bool = False) -> torch.Tensor:
        """Match RLinf's OpenPI SFT batch conversion and scalar reduction."""

        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        if isinstance(data, tuple):
            observation, actions = data
        else:
            observation, actions = data["observation"], data["actions"]

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
    def encode_observation_prefix(self, observation: Any) -> torch.Tensor:
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
        return image_prefix

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
        action_shape = (batch_size, self.model.config.action_horizon, self.model.config.action_dim)
        x_t = self.model.sample_noise(action_shape, device)
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
