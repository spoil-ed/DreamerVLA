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

        from openpi.models import model as openpi_model

        inputs = self._input_transform(copy.deepcopy(observation))
        device = next(self.parameters()).device
        tensor_inputs = tree_map(
            lambda value: torch.from_numpy(np.asarray(value)).to(device)[None, ...],
            inputs,
        )
        model_observation = openpi_model.Observation.from_dict(tensor_inputs)
        actions = self.model.sample_actions(
            device,
            model_observation,
            num_steps=self.num_steps,
        )
        output = tree_map(
            lambda value: np.asarray(value[0].detach().cpu()),
            {"state": tensor_inputs["state"], "actions": actions},
        )
        transformed = self._output_transform(output)
        return torch.from_numpy(np.asarray(transformed["actions"], dtype=np.float32)).to(device)

    def make_extractor(self) -> _Pi05RawExtractor:
        """Expose raw LIBERO inference to the shared evaluation runner."""

        return _Pi05RawExtractor(self)


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
