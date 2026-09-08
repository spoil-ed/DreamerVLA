"""DreamerVLA policy adapter for the migrated SiPAI Qwen-GR00T model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from dreamervla.models.embodiment.protocol import EncoderInputBatch
from dreamervla.models.embodiment.qwen_groot.action_head import GR00TActionHead
from dreamervla.models.embodiment.qwen_groot.backbone import QwenBackbone

SIPAI_LIBERO_ACTION_MIN = (
    -0.9375,
    -0.9375,
    -0.9375,
    -0.25821429,
    -0.375,
    -0.36750001,
    0.0,
)
SIPAI_LIBERO_ACTION_MAX = (
    0.9375,
    0.9375,
    0.9375,
    0.35571429,
    0.375,
    0.375,
    1.0,
)


class QwenGR00TPolicy(nn.Module):
    """Composable Qwen3-VL + GR00T N1.7 action policy.

    The two child names intentionally remain ``backbone`` and ``action_head``
    so an exported SiPAI Qwen-GR00T state dict is directly loadable.
    """

    policy_family = "qwen_groot"
    preserve_parameter_dtypes = True
    alignment_source = "SIPAI@9672af6:QwenBackbone+GR00TActionHead"

    def __init__(
        self,
        model_path: str | None = None,
        *,
        input_cameras: Sequence[str] = ("primary", "left_wrist"),
        camera_keys: Sequence[str] = ("agentview_rgb", "eye_in_hand_rgb"),
        action_horizon: int = 8,
        action_dim: int = 7,
        state_dim: int = 7,
        include_state: bool = False,
        cot_prompt: str | None = (
            "Your task is {instruction}. To identify the key objects for your task. "
            "Locate their bounding boxes in [x1,y1,x2,y2] format."
        ),
        attn_implementation: str = "sdpa",
        freeze_backbone: bool = True,
        freeze_state_encoder: bool = True,
        action_min: Sequence[float] = SIPAI_LIBERO_ACTION_MIN,
        action_max: Sequence[float] = SIPAI_LIBERO_ACTION_MAX,
        action_normalized_mask: Sequence[bool] = (
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ),
        backbone: QwenBackbone | None = None,
        action_head: GR00TActionHead | None = None,
        action_head_cfg: Mapping[str, Any] | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        if len(input_cameras) != len(camera_keys):
            raise ValueError("input_cameras and camera_keys must align one-to-one")
        self.input_cameras = list(input_cameras)
        self.camera_keys = list(camera_keys)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.include_state = bool(include_state)
        self.backbone = backbone or QwenBackbone(
            model_path,
            input_cameras=input_cameras,
            include_state=include_state,
            cot_prompt=cot_prompt,
            attn_implementation=attn_implementation,
        )
        if action_head is None:
            backbone_embedding_dim = self.backbone.hidden_size
            head_overrides = dict(action_head_cfg or {})
            if "vl_self_attention_cfg" not in head_overrides:
                if backbone_embedding_dim % 32:
                    raise ValueError(
                        "Qwen backbone hidden size must be divisible by the SiPAI "
                        "GR00T VL self-attention head count (32), got "
                        f"{backbone_embedding_dim}"
                    )
                # SiPAI's static default is 32 * 64 = 2048, while the current
                # Qwen3-VL-4B checkpoint is 2560 wide. Keep the same 32-head
                # module and derive its head width from the loaded backbone.
                head_overrides["vl_self_attention_cfg"] = {
                    "attention_head_dim": backbone_embedding_dim // 32,
                }
            head_cfg = {
                "variant": "DiT-L",
                "action_dim": self.action_dim,
                "state_dim": int(state_dim),
                "action_horizon": self.action_horizon,
                "backbone_embedding_dim": backbone_embedding_dim,
                **head_overrides,
            }
            action_head = GR00TActionHead(**head_cfg)
        if action_head.action_horizon != self.action_horizon:
            raise ValueError("policy and GR00T action horizons must match")
        if action_head.action_dim != self.action_dim:
            raise ValueError("policy and GR00T action dimensions must match")
        self.action_head = action_head

        self.register_buffer(
            "action_min",
            torch.as_tensor(action_min, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_max",
            torch.as_tensor(action_max, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_normalized_mask",
            torch.as_tensor(action_normalized_mask, dtype=torch.bool),
            persistent=False,
        )
        for name, value in {
            "action_min": self.action_min,
            "action_max": self.action_max,
            "action_normalized_mask": self.action_normalized_mask,
        }.items():
            if value.numel() != self.action_dim:
                raise ValueError(f"{name} must have action_dim={self.action_dim} entries")
        if torch.any(self.action_max <= self.action_min):
            raise ValueError("every action_max entry must exceed action_min")

        if freeze_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        if freeze_state_encoder and self.action_head.state_encoder is not None:
            for parameter in self.action_head.state_encoder.parameters():
                parameter.requires_grad = False
        if checkpoint_path:
            self.load_sipai_checkpoint(checkpoint_path)
        for name, module in self.named_modules():
            module._fsdp_wrap_name = name.rsplit(".", maxsplit=1)[-1]

    @property
    def _no_split_modules(self) -> list[str]:
        return [
            "Qwen3VLDecoderLayer",
            "Qwen3VLVisionBlock",
            "BasicTransformerBlock",
        ]

    def get_fsdp_wrap_module_list(self) -> list[nn.Module]:
        class_names = set(self._no_split_modules)
        return [
            module
            for module in self.modules()
            if module is not self and type(module).__name__ in class_names
        ]

    def train(self, mode: bool = True) -> QwenGR00TPolicy:
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def encode_inputs(
        self,
        batch: EncoderInputBatch | Mapping[str, Any],
        *,
        mode: str,
    ) -> dict[str, Any]:
        return self.backbone.encode(batch, mode=mode)

    def compute_loss(
        self,
        batch: EncoderInputBatch | Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_inputs(batch, mode="loss")
        return self.action_head.compute_loss(encoded, batch=batch)

    @torch.no_grad()
    def predict_action(
        self,
        batch: EncoderInputBatch | Mapping[str, Any],
        *,
        denormalize: bool = False,
    ) -> dict[str, np.ndarray]:
        encoded = self.encode_inputs(batch, mode="predict")
        output = self.action_head.predict_action(encoded, batch=batch)
        if not denormalize:
            return output
        actions = torch.from_numpy(output["normalized_actions"])
        return {"actions": self.denormalize_actions(actions).numpy()}

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor | dict[str, np.ndarray]:
        mode = str(batch.get("mode", "sft")).lower()
        if mode == "sft":
            data = batch.get("data", batch)
            if not isinstance(data, (EncoderInputBatch, Mapping)):
                raise TypeError("Qwen-GR00T SFT data must be EncoderInputBatch or mapping")
            return self.compute_loss(data)["action_loss"]
        if mode == "predict":
            data = batch.get("data", batch)
            if not isinstance(data, (EncoderInputBatch, Mapping)):
                raise TypeError("Qwen-GR00T predict data must be EncoderInputBatch or mapping")
            return self.predict_action(data, denormalize=bool(batch.get("denormalize", False)))
        if mode == "sample":
            observations = batch.get("observations")
            if not isinstance(observations, list):
                raise TypeError("Qwen-GR00T sample mode requires observations as a list")
            return torch.from_numpy(self.infer_raw_batch(observations))
        raise ValueError(f"unsupported Qwen-GR00T policy mode: {mode!r}")

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        minimum = self.action_min.to(device=actions.device, dtype=actions.dtype)
        maximum = self.action_max.to(device=actions.device, dtype=actions.dtype)
        mask = self.action_normalized_mask.to(device=actions.device)
        normalized = 2.0 * (actions - minimum) / (maximum - minimum) - 1.0
        return torch.where(mask, normalized, actions)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        minimum = self.action_min.to(device=actions.device, dtype=actions.dtype)
        maximum = self.action_max.to(device=actions.device, dtype=actions.dtype)
        mask = self.action_normalized_mask.to(device=actions.device)
        denormalized = 0.5 * (actions + 1.0) * (maximum - minimum) + minimum
        return torch.where(mask, denormalized, actions)

    @torch.no_grad()
    def infer_raw_batch(self, observations: list[dict[str, Any]]) -> np.ndarray:
        if not observations:
            raise ValueError("Qwen-GR00T inference requires at least one observation")
        images: list[list[np.ndarray]] = []
        prompts: list[str] = []
        states: list[np.ndarray] = []
        for observation in observations:
            images.append([_hwc_uint8(observation[key]) for key in self.camera_keys])
            prompts.append(str(observation.get("prompt", observation.get("task_description", ""))))
            if self.include_state:
                states.append(
                    np.asarray(
                        observation.get("state", observation.get("proprio")),
                        dtype=np.float32,
                    )
                )
        encoded_batch = EncoderInputBatch(
            prompt_text=prompts,
            conversations=[[] for _ in prompts],
            images=images,
            state=torch.from_numpy(np.stack(states)) if states else None,
        )
        output = self.predict_action(encoded_batch, denormalize=True)
        return output["actions"]

    def make_extractor(self) -> _QwenGR00TRawExtractor:
        return _QwenGR00TRawExtractor(self)

    def load_sipai_checkpoint(self, checkpoint_path: str) -> None:
        """Load an exported SiPAI `.safetensors` or PyTorch model state."""

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SiPAI Qwen-GR00T checkpoint does not exist: {path}")
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(path, device="cpu")
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping):
                raise TypeError("SiPAI checkpoint payload must be a mapping")
            state = payload.get("model", payload.get("state_dict", payload))
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(f"{path} contains no model state")
        incompatible = self.load_state_dict(dict(state), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "strict SiPAI Qwen-GR00T checkpoint load unexpectedly reported "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )

    def load_sft_delta(self, checkpoint_path: str) -> None:
        """Restore a DreamerVLA trainable-parameter checkpoint."""

        from dreamervla.utils.checkpoint.hf_checkpoint import load_runner_payload
        from dreamervla.utils.checkpoint.run_artifacts import resolve_resume_checkpoint

        resolved = resolve_resume_checkpoint(checkpoint_path)
        payload = load_runner_payload(resolved)
        state = payload.get("state_dicts", {}).get("policy")
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(f"{resolved} has no non-empty state_dicts.policy")
        missing, unexpected = self.load_state_dict(dict(state), strict=False)
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        missing_trainable = trainable.intersection(missing)
        if missing_trainable or unexpected:
            raise RuntimeError(
                "Qwen-GR00T SFT delta mismatch: "
                f"missing_trainable={sorted(missing_trainable)[:5]} "
                f"unexpected={list(unexpected)[:5]}"
            )


class _QwenGR00TRawExtractor:
    actions_are_env_ready = True

    def __init__(self, policy: QwenGR00TPolicy) -> None:
        self.policy = policy

    def reset(self) -> None:
        return None

    def step(self, observation: dict[str, Any], task_description: str) -> SimpleNamespace:
        raw = dict(observation)
        raw["task_description"] = task_description
        action_chunk = self.policy.infer_raw_batch([raw])[0]
        return SimpleNamespace(action_chunk=action_chunk)


def _hwc_uint8(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Qwen-GR00T image must be rank-3, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Qwen-GR00T image must be HWC/CHW RGB, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image, initial=0.0)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0)
    return np.ascontiguousarray(image, dtype=np.uint8)


__all__ = [
    "SIPAI_LIBERO_ACTION_MAX",
    "SIPAI_LIBERO_ACTION_MIN",
    "QwenGR00TPolicy",
]
