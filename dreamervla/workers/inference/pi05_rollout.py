"""pi0.5 rollout bundle for producer-neutral Ray collection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from dreamervla.models.embodiment.pi05.policy import Pi05Policy, _libero_eval_image


class Pi05RolloutBundle:
    """Run batched pi0.5 LIBERO inference and expose its image-prefix latent."""

    actions_are_env_ready = True

    def __init__(
        self,
        model_path: str,
        policy_ckpt_path: str | None = None,
        config_name: str = "pi05_libero",
        action_chunk: int = 10,
        action_dim: int = 7,
        num_steps: int = 5,
        rotate_images_180: bool = True,
        base_image_key: str = "agentview_rgb",
        wrist_image_key: str = "eye_in_hand_rgb",
        state_key: str = "state",
        device: str = "cuda",
    ) -> None:
        self._device = str(device)
        self._base_image_key = str(base_image_key)
        self._wrist_image_key = str(wrist_image_key)
        self._state_key = str(state_key)
        self._rotate = bool(rotate_images_180)
        self._policy = Pi05Policy(
            model_path=str(model_path),
            config_name=str(config_name),
            action_chunk=int(action_chunk),
            action_dim=int(action_dim),
            num_steps=int(num_steps),
            train_expert_only=True,
            rotate_images_180=bool(rotate_images_180),
        ).eval()
        if policy_ckpt_path not in (None, ""):
            self._policy.load_sft_delta(str(policy_ckpt_path))
        self.to(device)

    def to(self, device: str) -> Pi05RolloutBundle:
        self._device = str(device)
        self._policy.to(device)
        return self

    def make_extractor(self) -> Pi05PrefixExtractor:
        return Pi05PrefixExtractor(
            base_image_key=self._base_image_key,
            wrist_image_key=self._wrist_image_key,
            state_key=self._state_key,
            rotate_images_180=self._rotate,
        )

    def predict_batch(self, preps: list[dict[str, Any]]) -> list[tuple[np.ndarray, Any]]:
        actions, prefix = self._policy.infer_batch_with_prefix(preps)
        actions_cpu = actions.detach().cpu().numpy()
        # NumPy has no bfloat16 dtype; collection persists float16 after this
        # worker boundary, so expose CPU float32 here for a lossless conversion.
        prefix_cpu = prefix.float().cpu()
        return [(actions_cpu[index], prefix_cpu[index]) for index in range(len(preps))]

    def predict_actions_batch(self, preps: list[dict[str, Any]]) -> list[np.ndarray]:
        """Run policy inference without copying the large prefix tensor to CPU."""

        actions, _prefix = self._policy.infer_batch_with_prefix(preps)
        actions_cpu = actions.detach().cpu().numpy()
        return [actions_cpu[index] for index in range(len(preps))]


class Pi05PrefixExtractor:
    """Convert a raw LIBERO observation to OpenPI's public input dictionary."""

    actions_are_env_ready = True

    def __init__(
        self,
        *,
        base_image_key: str,
        wrist_image_key: str,
        state_key: str,
        rotate_images_180: bool,
    ) -> None:
        self._base_image_key = base_image_key
        self._wrist_image_key = wrist_image_key
        self._state_key = state_key
        self._rotate = bool(rotate_images_180)

    def reset(self) -> None:
        return None

    def prepare(self, observation: dict[str, Any], task_description: str) -> dict[str, Any]:
        state = observation.get(self._state_key, observation.get("proprio"))
        if state is None:
            state = observation.get("robot_states")
        if state is None:
            raise KeyError(
                f"π0.5 observation needs {self._state_key!r}, 'proprio', or 'robot_states'"
            )
        return {
            "observation/image": _libero_eval_image(
                observation[self._base_image_key], rotate_180=self._rotate
            ),
            "observation/wrist_image": _libero_eval_image(
                observation[self._wrist_image_key], rotate_180=self._rotate
            ),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": str(task_description),
        }

    def step(self, observation: dict[str, Any], task_description: str) -> SimpleNamespace:
        raise RuntimeError("Pi05PrefixExtractor.step is owned by the batched rollout bundle")


__all__ = ["Pi05PrefixExtractor", "Pi05RolloutBundle"]
