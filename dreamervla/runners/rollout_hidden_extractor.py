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

Image-preprocessing note:
    The offline sidecars (``*_action_hidden_*``) were built using
    ``dreamervla.preprocess.preprocess_oft_action_hidden._prepare_images_for_vla``
    which resizes via **PIL LANCZOS** and centre-crops via **PIL BICUBIC**.
    This extractor uses ``experiments.robot.openvla_utils.prepare_images_for_vla``
    which resizes via **TF lanczos3** (JPEG-encode-decode roundtrip first) and
    crops via **TF crop_and_resize** (bilinear).  The two pipelines are not
    numerically identical: empirically, against the gold libero_goal sidecar,
    TF prep gives max_abs_err ≤ 0.25 and Pearson r ≥ 0.9996 (8 pairs, demos
    0–1), while PIL prep gives max_abs_err up to 1.93 and r as low as 0.982.
    TF is kept because it is the real-robot deployment path and is far closer
    to the offline gold than PIL.  The residual ~0.25 is therefore a
    **PIL-vs-TF prep difference**, not purely fp16 non-determinism; see the
    consistency gate tolerance comment in the test file.

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
    produces an ``obs_embedding`` consistent with the offline sidecar protocol
    (history=2, two camera views, rotate_images_180=True, TF-based centre-crop;
    see module docstring for the measured residual vs the PIL-built sidecars).

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
        # Lazily built on first step(); reused so the model handles / head mode are resolved once.
        self._decoder: "OFTBatchedDecoder | None" = None

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

    def prepare(
        self,
        obs: dict[str, Any],
        task_description: str,
    ) -> dict[str, Any]:
        """Build VLA model inputs for one observation (updates the history buffer).

        Returns a dict ``{input_ids, attention_mask, pixel_values, proprio}`` ready to
        be stacked across envs and consumed by :func:`batched_forward`.  This is the
        per-env half of ``step``: ``step`` is exactly
        ``batched_forward(policy, [prepare(obs, task)], unnorm_key)[0]``, so single-env
        and batched (step_batch) collection share one inference code path.

        Args:
            obs: Observation dict.  Must contain uint8 ``np.ndarray`` images under each
                key in ``self._image_keys`` (shape ``(H, W, 3)``).  Optionally contains
                ``"state"`` (8-dim float32 array) when ``policy.use_proprio`` is True.
            task_description: Natural-language task string.

        Returns:
            Dict with ``input_ids`` ``(1, L)``, ``attention_mask`` ``(1, L)``,
            ``pixel_values`` ``(1, num_views*C, H, W)`` (all on the model device,
            bfloat16 where applicable), and ``proprio`` (np.ndarray or None).
        """
        from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

        ensure_openvla_oft_on_path()

        from experiments.robot.openvla_utils import prepare_images_for_vla

        model = self._policy.vla
        processor = self._policy.processor
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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "proprio": proprio,
        }

    def step(
        self,
        obs: dict[str, Any],
        task_description: str,
    ) -> tuple[list[Any], torch.Tensor]:
        """Run one forward pass and return (action_chunk, flat_hidden).

        Thin wrapper over :func:`batched_forward` with a single observation, so
        single-env and batched (step_batch) collection share exactly one inference
        code path.

        Args:
            obs: Observation dict (see :meth:`prepare`).
            task_description: Natural-language task string.

        Returns:
            Tuple of:
                - action_chunk: list of actions (length = NUM_ACTIONS_CHUNK)
                - flat_hidden: shape ``(229376,)`` float16 tensor on CPU,
                  matching the offline sidecar ``obs_embedding[t]``.
        """
        if self._decoder is None:
            self._decoder = OFTBatchedDecoder(self._policy, self._unnorm_key)
        prep = self.prepare(obs, task_description)
        return self._decoder.predict_batch([prep])[0]

    # ── Backward-compat shim: old callers used __call__(obs, task) ───────────
    def __call__(
        self,
        obs: dict[str, Any],
        task_description: str,
    ) -> tuple[list[Any], torch.Tensor]:
        """Alias for ``step``; initialises history on first call if not reset."""
        return self.step(obs, task_description)


# ── batched (step_batch) inference ──────────────────────────────────────────
# Feeds K prepared observations through ONE VLA forward.  The upstream OFT
# ``predict_action`` wrapper has two batch==1 assumptions that break for B>1:
#   - modeling_prismatic.py:972  appends a [1,1] token via cat(dim=1)
#   - modeling_prismatic.py:924  reshape(NUM_ACTIONS_CHUNK, ACTION_DIM) drops the batch
# Everything else in the L1-regression path is batch-safe, so we bypass the wrapper
# and call the internals directly with a batched token-append and a (B, chunk, dim)
# reshape.  Verified bit-exact vs ``OFTRolloutHiddenExtractor.step`` at B=1 and
# action-partner-invariant (no cross-batch leakage) by
# scripts/smoke_oft_batched_forward.py.


def _left_pad_batch(
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    pad_token_id: int,
    bos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a list of ``[1, L_i]`` input_ids/masks to ``[B, max L]``, BOS at index 0.

    Real content is right-aligned so the action tokens appended downstream share an
    absolute index across the batch; BOS is forced to absolute index 0 so the
    vision-insert-after-BOS path stays uniform (RLinf's ``PrismaticProcessor`` trick).
    The pad token's value is irrelevant numerically — it is masked out of attention.
    An equal-length batch is returned unchanged.
    """
    batch = len(input_ids_list)
    lengths = [int(x.shape[-1]) for x in input_ids_list]
    max_len = max(lengths)
    ref = input_ids_list[0]
    out_ids = torch.full((batch, max_len), int(pad_token_id), dtype=ref.dtype, device=ref.device)
    out_mask = torch.zeros(
        (batch, max_len), dtype=attention_mask_list[0].dtype, device=ref.device
    )
    for i, (ids, msk, length) in enumerate(zip(input_ids_list, attention_mask_list, lengths)):
        offset = max_len - length
        out_ids[i, offset:] = ids.reshape(-1)
        out_mask[i, offset:] = msk.reshape(-1)
        # Move the (right-aligned) BOS to absolute index 0; mask its old slot as pad.
        out_ids[i, offset] = int(pad_token_id)
        out_mask[i, offset] = 0
        out_ids[i, 0] = int(bos_token_id)
        out_mask[i, 0] = 1
    return out_ids, out_mask


class OFTBatchedDecoder:
    """First-class batched OFT inference (RLinf ``predict_action_batch`` posture, non-invasive).

    Constructed ONCE per policy; caches the model handles, special-token ids, action
    constants, and the head mode.  ``predict_batch(preps)`` runs ONE VLA forward over K
    prepared observations and returns per-env ``(action_chunk, flat_hidden)``.

    It bypasses the upstream ``predict_action`` wrapper — which assumes batch==1 at
    modeling_prismatic.py:972 (token-cat) and :924 (action reshape) — by calling the model's
    batch-safe internals directly, with: a batched trailing-29871 append, left-pad +
    ``position_ids`` for mixed-task (different prompt length) batches, and a
    ``(B, chunk, dim)`` decode reshape.  Verified bit-exact vs
    ``OFTRolloutHiddenExtractor.step`` at B=1.

    Head mode is auto-detected: ``action_head`` present -> L1-regression decode;
    ``action_head is None`` -> discrete (headless / one-trajectory) decode from LM logits.
    The ``obs_embedding`` (action-query hidden) is identical either way; only the action
    decode differs.
    """

    def __init__(self, policy: Any, unnorm_key: str) -> None:
        from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

        ensure_openvla_oft_on_path()
        from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK

        self._model = policy.vla
        self._action_head = policy.action_head  # None => discrete (headless) decode
        self._proprio_projector = policy.proprio_projector
        self._unnorm_key = unnorm_key
        self._action_dim = int(ACTION_DIM)
        self._num_chunks = int(NUM_ACTIONS_CHUNK)
        self._span = int(ACTION_DIM * NUM_ACTIONS_CHUNK)
        self._ignore_index = IGNORE_INDEX
        tok = policy.processor.tokenizer
        self._bos_id = tok.bos_token_id if tok.bos_token_id is not None else 1
        self._pad_id = (
            tok.pad_token_id
            if tok.pad_token_id is not None
            else (tok.eos_token_id if tok.eos_token_id is not None else 0)
        )

    @property
    def is_discrete(self) -> bool:
        """True when headless (no L1 action head; actions decoded from the LM logits)."""
        return self._action_head is None

    def predict_batch(
        self, preps: list[dict[str, Any]]
    ) -> list[tuple[list[Any], torch.Tensor]]:
        """One VLA forward over K preps -> per-env ``(action_chunk, flat_hidden)``.

        Preps MAY have different prompt (``input_ids``) lengths — different tasks: they are
        left-padded to the batch max with ``attention_mask`` + ``position_ids`` so each
        padded sample is numerically equal to computing it alone (block-diagonal attention
        => no cross-sample interaction).  Equal-length batches pad to a no-op.

        Returns, per env, a list of NUM_ACTIONS_CHUNK ``np.ndarray`` actions and a
        ``(229376,)`` float16 CPU ``obs_embedding`` tensor.  Raises if ``preps`` is empty.
        """
        if not preps:
            raise ValueError("predict_batch requires at least one prep")
        input_ids, attention_mask = _left_pad_batch(
            [p["input_ids"] for p in preps],
            [p["attention_mask"] for p in preps],
            self._pad_id,
            self._bos_id,
        )
        pixel_values = torch.cat([p["pixel_values"] for p in preps], dim=0)
        use_proprio = preps[0]["proprio"] is not None
        proprio = (
            np.stack([np.asarray(p["proprio"]).reshape(-1) for p in preps], axis=0)
            if use_proprio
            else None
        )
        actions, hidden = self._forward(input_ids, attention_mask, pixel_values, proprio)
        out: list[tuple[list[Any], torch.Tensor]] = []
        for i in range(len(preps)):
            action_chunk = [actions[i, j] for j in range(actions.shape[1])]
            flat_hidden = flatten_action_hidden(hidden[i : i + 1].cpu())
            out.append((action_chunk, flat_hidden))
        return out

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        proprio: np.ndarray | None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Batched replica of ``predict_action`` + ``_regression_or_discrete_prediction``."""
        model = self._model
        use_proprio = proprio is not None

        # FIX1: batched trailing-29871 ('') append (upstream cats a [1,1] token -> B==1 only)
        if not torch.all(input_ids[:, -1] == 29871):
            pad = torch.full(
                (input_ids.shape[0], 1), 29871, dtype=input_ids.dtype, device=input_ids.device
            )
            input_ids = torch.cat([input_ids, pad], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(pad, dtype=attention_mask.dtype)], dim=1
            )

        labels = input_ids.clone()
        labels[:] = self._ignore_index
        num_prompt_tokens = input_ids.shape[-1] - 1
        input_ids, attention_mask = model._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        labels = model._prepare_labels_for_action_prediction(labels, input_ids)

        # Wrap the whole forward (incl. the action head) in inference_mode, matching how the
        # single-obs path wrapped predict_action; otherwise the head's layer_norm tries to
        # save inference tensors for backward.
        with torch.inference_mode():
            input_embeddings = model.get_input_embeddings()(input_ids)
            all_actions_mask = model._process_action_masks(labels)
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )
            projected = model._process_vision_features(
                pixel_values, language_embeddings, use_film=False
            )
            if use_proprio:
                proprio_t = torch.Tensor(proprio).to(projected.device, dtype=projected.dtype)
                projected = model._process_proprio_features(
                    projected, proprio_t, self._proprio_projector
                )

            num_patches = (
                model.vision_backbone.get_num_patches()
                * model.vision_backbone.get_num_images_in_input()
            )
            if use_proprio:
                num_patches += 1

            input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)
            multimodal_embeddings, multimodal_attention_mask = model._build_multimodal_attention(
                input_embeddings, projected, attention_mask
            )
            # Left-padded (mixed-task) batches: position_ids must skip pad tokens so real
            # tokens get their standalone positions.  No padding (same-task) -> None keeps
            # the forward byte-identical to the un-padded path.
            position_ids = None
            if multimodal_attention_mask is not None and bool(
                (multimodal_attention_mask == 0).any()
            ):
                position_ids = (multimodal_attention_mask.long().cumsum(-1) - 1).clamp_(min=0)
            lm_out = model.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = lm_out.hidden_states[-1]
            start = num_patches + num_prompt_tokens
            actions_hidden_states = last_hidden[:, start : start + self._span, :]  # (B,56,D)
            normalized = self._decode(lm_out, actions_hidden_states, start, input_ids.shape[0])
        actions = model._unnormalize_actions(normalized, self._unnorm_key)
        return actions, actions_hidden_states

    def _decode(
        self,
        lm_out: Any,
        actions_hidden_states: torch.Tensor,
        action_start: int,
        batch_size: int,
    ) -> np.ndarray:
        """Decode normalized actions ``(B, chunk, dim)`` — L1 head, or discrete logits->bins."""
        if self._action_head is not None:
            # L1-regression head: MLP over the action-query hidden states.
            normalized = self._action_head.predict_action(actions_hidden_states)
            return (
                normalized.reshape(batch_size, self._num_chunks, self._action_dim)
                .float()
                .cpu()
                .numpy()
            )  # FIX2: keep the batch dim
        # Discrete (headless / one-trajectory) LM-head: argmax the logits at the action
        # positions -> bin centers.  Mirrors the upstream discrete branch but keeps batch.
        model = self._model
        action_logits = lm_out.logits[:, action_start : action_start + self._span, :]
        predicted_ids = action_logits.argmax(dim=2).cpu().numpy()  # (B, span)
        discretized = model.vocab_size - predicted_ids
        discretized = np.clip(discretized - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
        normalized = model.bin_centers[discretized]  # (B, span)
        return normalized.reshape(batch_size, self._num_chunks, self._action_dim)  # FIX2


def batched_forward(
    policy: Any,
    preps: list[dict[str, Any]],
    unnorm_key: str,
) -> list[tuple[list[Any], torch.Tensor]]:
    """Convenience wrapper: build a one-shot :class:`OFTBatchedDecoder` and run it.

    For repeated inference (collection), construct ``OFTBatchedDecoder`` ONCE and call
    ``predict_batch`` — that resolves the model handles / token ids / head mode a single
    time instead of per forward.  Kept for tests and single-shot callers.

    Raises ``ValueError`` if ``preps`` is empty.
    """
    if not preps:
        raise ValueError("batched_forward requires at least one prep")
    return OFTBatchedDecoder(policy, unnorm_key).predict_batch(preps)
