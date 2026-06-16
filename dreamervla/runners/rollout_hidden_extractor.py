"""OFT action-query hidden extractor for online rollout collection.

The sidecar ``obs_embedding`` field (shape ``(229376,)`` float16 per frame)
is computed by:

1. Stacking ``history`` frames of each camera view, in order
   ``[t-history+1, ..., t]  ×  [view_0, view_1, ...]`` (time-major).
2. Rotating each image 180° if ``rotate_images_180=True``.
3. Resizing each image to 224×224 with TF lanczos3 and applying a centre-crop
   (scale 0.9, BICUBIC resize back) via
   ``experiments.robot.openvla_utils.prepare_images_for_vla``.
4. Running ``vla.predict_action(...)`` which returns
   ``(actions, actions_hidden_states)`` where
   ``actions_hidden_states`` has shape ``(1, 56, 4096)`` = ``(1, ACTION_DIM *
   NUM_ACTIONS_CHUNK, token_dim)`` — these are the last-layer LM hidden states
   at the 56 action-query token positions.
5. Squeezing the batch dimension to ``[56, 4096]``.
6. Reshaping to ``(229376,)`` and casting to float16.

This exactly matches what ``dreamervla/preprocess/preprocess_oft_action_hidden.py``
writes into the ``*_action_hidden_*`` sidecars (``obs_hidden_source="action_query"``).

Hook target:
    The tensor comes directly from the RETURN VALUE of
    ``vla.predict_action(...)`` (second element of the tuple).  No
    ``register_forward_hook`` is required; the Prismatic model exposes the
    action-query hidden states as a first-class output.

    Hook-target location for future swaps (e.g. RynnVLA):
        ``HOOK_TARGET = "predict_action_return[1]"``  — sentinel kept here so
        alternative backends can override at a single site.

Dimension decomposition:
    229376 = 56 × 4096
    56     = NUM_ACTIONS_CHUNK(8) × ACTION_DIM(7)
    4096   = token_dim (LLM hidden size)

History protocol:
    ``OFTRolloutHiddenExtractor`` maintains an internal frame buffer per view.
    On the first call (or after ``reset()``), the buffer is filled by repeating
    the first frame (padding), matching the offline preprocessor's
    ``_history_indices`` which pads with the earliest available frame.
    The collector calls ``reset()`` at episode start and ``step(obs)`` per
    environment step.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

# ── sentinel: the single place to change when swapping hook target ──────────
# For OFT: second return value of vla.predict_action()
# For RynnVLA-legacy: encoder.extract_action_hidden() / obs_to_action_hidden()
HOOK_TARGET = "predict_action_return[1]"  # noqa: S105  (not a secret)


def flatten_action_hidden(h: torch.Tensor) -> torch.Tensor:
    """Flatten an action-query hidden tensor to the sidecar shape.

    Args:
        h: Tensor of shape ``(..., 56, 4096)`` where the leading dims can be
           a batch size of 1 or absent.  Any leading batch dimension of size 1
           is squeezed automatically.

    Returns:
        1-D float16 tensor of shape ``(229376,)``.
    """
    # Squeeze a leading batch dim of 1 so [1,56,4096] and [56,4096] both work.
    if h.ndim == 3 and h.shape[0] == 1:
        h = h.squeeze(0)
    return h.reshape(-1).to(torch.float16)


class OFTRolloutHiddenExtractor:
    """Wraps an ``OpenVLAOFTPolicy`` to capture the action-query hidden states.

    Maintains a per-view frame history buffer so that each call to ``step``
    produces an ``obs_embedding`` that matches the offline sidecar protocol
    (history=2, two camera views, rotate_images_180=True, TF-based centre-crop).

    Args:
        policy: An ``OpenVLAOFTPolicy`` instance loaded via
            ``OpenVLAOFTPolicy`` constructor.
        image_keys: Camera view keys to read from the obs dict, in the same
            order as the offline preprocessor (default:
            ``["agentview_rgb", "eye_in_hand_rgb"]``).
        history: Number of past frames to stack per view (default: 2).
        rotate_images_180: Whether to flip each image 180° before processing
            (default: True, matching the libero_goal sidecar config).
        center_crop: Whether to apply TF-based centre-crop (scale 0.9) after
            resize (default: True, matching the sidecar config).
        unnorm_key: Action unnormalization key matching the training dataset
            (default: ``"libero_goal_no_noops"``).

    Public API::

        extractor.reset()                          # call at episode start
        action_chunk, flat_hidden = extractor.step(obs, task_description)

    where:
        ``action_chunk`` — list of ``np.ndarray`` actions (one per open-loop
                          step), from ``vla.predict_action``
        ``flat_hidden``  — ``torch.Tensor`` shape ``(229376,)`` dtype float16
                          on CPU; numerically equivalent to the offline sidecar
                          ``obs_embedding[t]`` for the same frame.

    The obs dict must contain uint8 ``np.ndarray`` images under each key in
    ``image_keys``, with shape ``(H, W, 3)``.  Optionally it may contain a
    ``"state"`` key with an 8-dim proprio vector (used when
    ``policy.use_proprio`` is True).
    """

    def __init__(
        self,
        policy: Any,
        *,
        image_keys: list[str] | None = None,
        history: int = 2,
        rotate_images_180: bool = True,
        center_crop: bool = True,
        unnorm_key: str = "libero_goal_no_noops",
    ) -> None:
        self._policy = policy
        self._image_keys: list[str] = (
            image_keys if image_keys is not None else ["agentview_rgb", "eye_in_hand_rgb"]
        )
        self._history = max(1, int(history))
        self._rotate_images_180 = bool(rotate_images_180)
        self._center_crop = bool(center_crop)
        self._unnorm_key = unnorm_key
        # Per-view deque of (H, W, 3) uint8 numpy arrays (already rotated).
        # Length is always exactly self._history after the first step.
        self._buffers: dict[str, deque] = {
            key: deque(maxlen=self._history) for key in self._image_keys
        }

    def reset(self) -> None:
        """Clear the history buffer.  Call at the start of every episode."""
        for key in self._image_keys:
            self._buffers[key].clear()

    def _get_history(self, key: str, current_frame: np.ndarray) -> list[np.ndarray]:
        """Return a list of ``self._history`` frames for ``key``.

        The deque is updated in-place with ``current_frame`` (post-rotation).
        If the buffer was empty or short, the earliest frame is padded to fill,
        matching the offline ``_history_indices`` behaviour.
        """
        if not self._buffers[key]:
            # First call: pre-fill with the current frame (padding).
            for _ in range(self._history):
                self._buffers[key].append(current_frame)
        else:
            self._buffers[key].append(current_frame)

        # Pad from the left if still short (shouldn't happen after init).
        frames = list(self._buffers[key])
        while len(frames) < self._history:
            frames = [frames[0]] + frames
        return frames  # length == self._history, oldest first

    def step(
        self,
        obs: dict[str, Any],
        task_description: str,
    ) -> tuple[list[Any], torch.Tensor]:
        """Run one forward pass and return (action_chunk, flat_hidden).

        Args:
            obs: Observation dict.  Must contain uint8 ``np.ndarray`` images
                under each key in ``self._image_keys`` (shape ``(H, W, 3)``).
                Optionally contains ``"state"`` (8-dim float32 array) when
                ``policy.use_proprio`` is True.
            task_description: Natural-language task string.

        Returns:
            Tuple of:
                - action_chunk: list of actions (length = NUM_ACTIONS_CHUNK)
                - flat_hidden: shape ``(229376,)`` float16 tensor on CPU,
                  matching the offline sidecar ``obs_embedding[t]``.
        """
        from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

        ensure_openvla_oft_on_path()

        from experiments.robot.openvla_utils import prepare_images_for_vla

        model = self._policy.vla
        processor = self._policy.processor
        action_head = self._policy.action_head
        proprio_projector = self._policy.proprio_projector
        use_proprio = bool(getattr(self._policy, "use_proprio", False))

        prompt = (
            f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        )

        # ── 1. Build image list: [t-h+1..t] × [view_0, view_1, ...] ──────────
        # This is the same interleaving as the offline preprocessor:
        #   for hidx in _history_indices(index, history):
        #       for key in image_keys:
        #           image_from_hdf5(obs_group, key, hidx, ...)
        all_raw_frames: list[np.ndarray] = []
        history_by_key: dict[str, list[np.ndarray]] = {}
        for key in self._image_keys:
            raw = np.asarray(obs[key], dtype=np.uint8)
            if self._rotate_images_180:
                raw = raw[::-1, ::-1].copy()
            history_by_key[key] = self._get_history(key, raw)

        # Interleave: time-step first, then views — matching offline loop order.
        for t_offset in range(self._history):
            for key in self._image_keys:
                all_raw_frames.append(history_by_key[key][t_offset])

        # ── 2. Preprocess images (TF lanczos3 resize + TF centre-crop) ───────
        cfg_for_prep = type("_Cfg", (), {"center_crop": self._center_crop})()
        processed_images = prepare_images_for_vla(all_raw_frames, cfg_for_prep)

        # ── 3. Build pixel_values tensor: primary + extra views ───────────────
        device = next(model.parameters()).device
        primary_inputs = processor(prompt, processed_images[0]).to(
            device, dtype=torch.bfloat16
        )
        input_ids = primary_inputs["input_ids"]
        attention_mask = primary_inputs["attention_mask"]
        pixel_values = primary_inputs["pixel_values"]

        for img in processed_images[1:]:
            extra = processor(prompt, img).to(device, dtype=torch.bfloat16)
            pixel_values = torch.cat([pixel_values, extra["pixel_values"]], dim=1)

        # ── 4. Proprio (same normalization as offline preprocess) ─────────────
        proprio = None
        if use_proprio and "state" in obs:
            from experiments.robot.openvla_utils import normalize_proprio

            proprio_norm_stats = model.norm_stats[self._unnorm_key]["proprio"]
            proprio = normalize_proprio(obs["state"], proprio_norm_stats)

        # ── 5. Forward pass via predict_action ────────────────────────────────
        with torch.inference_mode():
            actions, actions_hidden_states = model.predict_action(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                unnorm_key=self._unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=action_head,
                use_film=False,
            )

        # actions_hidden_states: (1, 56, 4096) → (229376,) float16
        flat_hidden = flatten_action_hidden(actions_hidden_states.cpu())

        action_chunk = [actions[i] for i in range(len(actions))]
        return action_chunk, flat_hidden

    # ── Backward-compat shim: old callers used __call__(obs, task) ───────────
    def __call__(
        self,
        obs: dict[str, Any],
        task_description: str,
    ) -> tuple[list[Any], torch.Tensor]:
        """Alias for ``step``; initialises history on first call if not reset."""
        return self.step(obs, task_description)
