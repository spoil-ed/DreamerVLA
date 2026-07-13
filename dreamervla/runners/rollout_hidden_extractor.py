"""OpenVLA-OFT hidden-token extractor for online rollout collection.

Each frame emits the one-image projected current-frame vision patch grid as
``obs_embedding [256,4096]``. Persisted sidecars are
``[T,256,4096]``. The model's 56 action positions remain internal
to action decoding and are never returned as the world-model observation.

``OFTRolloutHiddenExtractor`` maintains the configured per-view history buffer.
At episode start it pads history by repeating the first frame, matching offline
preprocessing.  Image preparation follows the deployment path before projected
vision tokens are selected from ``_process_vision_features``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def hidden_token_from_projected(
    projected: torch.Tensor,
    *,
    image_keys: Sequence[str],
    patches_per_image: int,
) -> torch.Tensor:
    """Validate and return the canonical projected current-frame token grid."""

    keys = tuple(image_keys)
    if len(keys) != 1:
        raise ValueError(
            "OpenVLA-OFT hidden-token mainline requires one image; "
            f"got image_keys={keys!r}"
        )
    if int(patches_per_image) != 256:
        raise ValueError(
            "OpenVLA-OFT hidden-token mainline requires patches_per_image=256, "
            f"got {int(patches_per_image)}"
        )
    if projected.ndim != 3 or tuple(projected.shape[1:]) != (256, 4096):
        raise ValueError(
            "projected vision tokens must have shape [B,256,4096], "
            f"got {tuple(projected.shape)}"
        )
    return projected


@dataclass(frozen=True)
class OFTDecodeOutput:
    """Tuple-compatible OFT decode result with optional demo-level sidecars."""

    action_chunk: list[Any]
    hidden_state: torch.Tensor
    lang_emb: torch.Tensor | None = None
    action_token_ids: torch.Tensor | None = None
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    def __iter__(self):
        yield self.action_chunk
        yield self.hidden_state

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        return (self.action_chunk, self.hidden_state)[index]


class OFTRolloutHiddenExtractor:
    """Wrap an ``OpenVLAOFTPolicy`` and capture projected hidden-token embeddings.

    Each call to ``step`` produces the canonical one-image, history-one
    ``obs_embedding`` used by the offline sidecar protocol.

    Args:
        policy: An ``OpenVLAOFTPolicy`` instance loaded via
            ``OpenVLAOFTPolicy`` constructor.
        image_keys: The single camera key to read (default: ``["agentview_rgb"]``).
        history: Fixed to one for the mainline.
        rotate_images_180: Whether to flip each image 180° before processing
            (default: True, matching the libero_goal sidecar config).
        center_crop: Whether to apply TF-based centre-crop (scale 0.9) after
            resize (default: True, matching the sidecar config).
        unnorm_key: Action unnormalization key matching the training dataset
            (default: ``"libero_goal_no_noops"``).

    Public API::

        extractor.reset()                          # call at episode start
        action_chunk, hidden_state = extractor.step(obs, task_description)

    where:
        ``action_chunk`` — list of ``np.ndarray`` actions (one per open-loop
                          step), from ``vla.predict_action``
        ``hidden_state`` — CPU float16 tensor equivalent to sidecar
                          ``obs_embedding[t]`` as ``[N, token_dim]``.
        ``lang_emb`` — available as ``extractor.step(...).lang_emb``; demo-level
                       CPU float16 language embedding matching offline preprocess.

    The obs dict must contain uint8 ``np.ndarray`` images under each key in
    ``image_keys``, with shape ``(H, W, 3)``.
    """

    def __init__(
        self,
        policy: Any,
        *,
        image_keys: list[str] | None = None,
        history: int = 1,
        rotate_images_180: bool = True,
        center_crop: bool = True,
        unnorm_key: str = "libero_goal_no_noops",
        obs_hidden_source: str = "hidden_token",
    ) -> None:
        self._policy = policy
        self._image_keys: list[str] = (
            image_keys if image_keys is not None else ["agentview_rgb"]
        )
        self._history = int(history)
        if len(self._image_keys) != 1 or self._history != 1:
            raise ValueError(
                "OpenVLA-OFT hidden-token mainline requires one image and history=1"
            )
        if bool(getattr(policy, "use_proprio", False)):
            raise ValueError("OpenVLA-OFT hidden-token mainline does not include proprio")
        self._rotate_images_180 = bool(rotate_images_180)
        self._center_crop = bool(center_crop)
        self._unnorm_key = unnorm_key
        self._obs_hidden_source = str(obs_hidden_source)
        if self._obs_hidden_source != "hidden_token":
            raise ValueError(
                "OpenVLA-OFT rollout observations must use "
                "obs_hidden_source='hidden_token'; "
                f"got {self._obs_hidden_source!r}"
            )
        # Per-view deque of (H, W, 3) uint8 numpy arrays (already rotated).
        # Length is always exactly self._history after the first step.
        self._buffers: dict[str, deque] = {
            key: deque(maxlen=self._history) for key in self._image_keys
        }
        # Lazily built on first step(); reused so the model handles / head mode are resolved once.
        self._decoder: OFTBatchedDecoder | None = None
        # Single-slot cache of the prompt's tokenized text tensors, keyed by
        # task_description (the prompt is invariant within a task).  Refreshed
        # when the task changes; see _prompt_text_inputs.
        self._prompt_cache: tuple[str, torch.Tensor, torch.Tensor] | None = None

    def _prompt_text_inputs(
        self, processor: Any, prompt: str, task_description: str, fallback_image: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the prompt's ``(input_ids, attention_mask)``, tokenized once per task.

        The processor's text branch (``processor(prompt, img)["input_ids"/"attention_mask"]``)
        is a pure function of ``prompt`` — itself a pure function of ``task_description`` — so
        it is computed once and cached, then reused while the task is unchanged.  A fresh
        ``task_description`` refreshes the cache.

        On a cache MISS the canonical text tensors are taken from a full
        ``processor(prompt, fallback_image)`` call (the same call path used today), so any
        processor-specific text post-processing (e.g. the dreamervla subclass's left-padded
        BOS normalization) is preserved exactly.  Cached tensors are cloned so downstream
        device moves never mutate the cached copy.
        """
        if self._prompt_cache is not None and self._prompt_cache[0] == task_description:
            return self._prompt_cache[1], self._prompt_cache[2]
        full = processor(prompt, fallback_image)
        input_ids = full["input_ids"].clone()
        attention_mask = full["attention_mask"].clone()
        self._prompt_cache = (task_description, input_ids, attention_mask)
        return input_ids, attention_mask

    @staticmethod
    def _view_pixel_values(processor: Any, image: Any) -> torch.Tensor:
        """Run only the processor's image branch for one view.

        Byte-identical to ``processor(prompt, image)["pixel_values"]`` (the exact line
        ``PrismaticProcessor.__call__`` uses, with the default ``return_tensors="pt"``),
        so the prompt-cache change leaves image numerics untouched while skipping the
        per-view tokenization that the old per-view ``processor(prompt, img)`` repeated.
        """
        return processor.image_processor(image, return_tensors="pt")["pixel_values"]

    def reset(self) -> None:
        """Clear the history buffer.  Call at the start of every episode."""
        for key in self._image_keys:
            self._buffers[key].clear()

    def unnormalize_actions(self, normalized_actions: Any) -> np.ndarray:
        """Map normalized actor outputs with this checkpoint's dataset statistics."""

        actions = self._policy.vla._unnormalize_actions(
            np.asarray(normalized_actions, dtype=np.float32),
            self._unnorm_key,
        )
        return np.asarray(actions, dtype=np.float32)

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
            obs: Observation dict. Must contain a uint8 ``np.ndarray`` image under
                ``agentview_rgb`` with shape ``(H, W, 3)``.
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
        # The prompt's text tokenization is invariant for a fixed task_description, so it is
        # cached and reused across rollout steps; only the image branch runs per view/step.
        # Numerically identical to the old per-view ``processor(prompt, img)`` calls.
        device = next(model.parameters()).device
        input_ids, attention_mask = self._prompt_text_inputs(
            processor, prompt, task_description, processed_images[0]
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pixel_values = torch.cat(
            [self._view_pixel_values(processor, img) for img in processed_images], dim=1
        ).to(device, dtype=torch.bfloat16)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "proprio": None,
        }

    def step(
        self,
        obs: dict[str, Any],
        task_description: str,
    ) -> OFTDecodeOutput:
        """Run one forward pass and return a tuple-compatible decode output.

        Thin wrapper over :func:`batched_forward` with a single observation, so
        single-env and batched (step_batch) collection share exactly one inference
        code path.

        Args:
            obs: Observation dict (see :meth:`prepare`).
            task_description: Natural-language task string.

        Returns:
            Tuple of:
                - action_chunk: list of actions (length = NUM_ACTIONS_CHUNK)
                - hidden_state: CPU float16 tensor matching the offline sidecar
                  ``obs_embedding[t]`` as tokenized ``[N, token_dim]``.
                - lang_emb: CPU float16 language sidecar on the output object.
        """
        if self._decoder is None:
            self._decoder = OFTBatchedDecoder(
                self._policy,
                self._unnorm_key,
                obs_hidden_source=self._obs_hidden_source,
                image_keys=self._image_keys,
            )
        prep = self.prepare(obs, task_description)
        return self._decoder.predict_batch([prep])[0]

# ── batched (step_batch) inference ──────────────────────────────────────────
# Feeds K prepared observations through ONE VLA forward.  The upstream OFT
# ``predict_action`` wrapper has two batch==1 assumptions that break for B>1:
#   - modeling_prismatic.py:972  appends a [1,1] token via cat(dim=1)
#   - modeling_prismatic.py:924  reshape(NUM_ACTIONS_CHUNK, ACTION_DIM) drops the batch
# The discrete headless path is batch-safe once those two assumptions are handled,
# so this module calls the internals directly with a batched token append and a
# ``(B, chunk, dim)`` reshape.


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
    for i, (ids, msk, length) in enumerate(
        zip(input_ids_list, attention_mask_list, lengths, strict=True)
    ):
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
    prepared observations and returns per-env tuple-compatible decode outputs.

    It bypasses the upstream ``predict_action`` wrapper — which assumes batch==1 at
    modeling_prismatic.py:972 (token-cat) and :924 (action reshape) — by calling the model's
    batch-safe internals directly, with: a batched trailing-29871 append, left-pad +
    ``position_ids`` for mixed-task (different prompt length) batches, and a
    ``(B, chunk, dim)`` decode reshape.  Verified bit-exact vs
    ``OFTRolloutHiddenExtractor.step`` at B=1.

    The decoder accepts only the headless one-trajectory checkpoint and decodes
    discrete actions from LM logits.
    """

    def __init__(
        self,
        policy: Any,
        unnorm_key: str,
        obs_hidden_source: str = "hidden_token",
        image_keys: list[str] | None = None,
    ) -> None:
        from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

        ensure_openvla_oft_on_path()
        from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK

        self._model = policy.vla
        self._policy = policy
        if policy.action_head is not None:
            raise ValueError("L1/action-query checkpoints are closed")
        if getattr(policy, "proprio_projector", None) is not None:
            raise ValueError("OpenVLA-OFT hidden-token mainline does not include proprio")
        if bool(getattr(policy, "use_proprio", False)):
            raise ValueError("OpenVLA-OFT hidden-token mainline does not include proprio")
        self._proprio_projector = None
        self._unnorm_key = unnorm_key
        self._obs_hidden_source = str(obs_hidden_source)
        if self._obs_hidden_source != "hidden_token":
            raise ValueError(
                "OpenVLA-OFT rollout observations must use "
                "obs_hidden_source='hidden_token'; "
                f"got {self._obs_hidden_source!r}"
            )
        self._image_keys = list(image_keys) if image_keys is not None else ["agentview_rgb"]
        if self._image_keys != ["agentview_rgb"]:
            raise ValueError(
                "OpenVLA-OFT hidden-token mainline requires image_keys=['agentview_rgb']"
            )
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
        """The only supported decoder mode is discrete."""
        return True

    def predict_batch(
        self, preps: list[dict[str, Any]]
    ) -> list[OFTDecodeOutput]:
        """One VLA forward over K preps -> per-env tuple-compatible decode output.

        Preps MAY have different prompt (``input_ids``) lengths — different tasks: they are
        left-padded to the batch max with ``attention_mask`` + ``position_ids`` so each
        padded sample is numerically equal to computing it alone (block-diagonal attention
        => no cross-sample interaction).  Equal-length batches pad to a no-op.

        Returns, per env, a list of NUM_ACTIONS_CHUNK ``np.ndarray`` actions, a
        CPU float16 ``obs_embedding`` tensor, and a CPU float16 demo-level
        ``lang_emb`` sidecar. Hidden-token sidecars are tokenized
        ``[N, token_dim]``.
        Raises if ``preps`` is empty.
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
        if any(p["proprio"] is not None for p in preps):
            raise ValueError("OpenVLA-OFT hidden-token mainline does not include proprio")
        actions, hidden, lang_emb, action_token_ids = self._forward(
            input_ids, attention_mask, pixel_values, None
        )
        out: list[OFTDecodeOutput] = []
        for i in range(len(preps)):
            action_chunk = [actions[i, j] for j in range(actions.shape[1])]
            hidden_state = hidden[i].detach().cpu().to(torch.float16)
            lang_state = lang_emb[i].detach().cpu().to(torch.float16)
            out.append(
                OFTDecodeOutput(
                    action_chunk,
                    hidden_state,
                    lang_state,
                    action_token_ids=action_token_ids[i].detach().cpu(),
                    input_ids=input_ids[i].detach().cpu(),
                    attention_mask=attention_mask[i].detach().cpu(),
                )
            )
        return out

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        proprio: np.ndarray | None,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched replica of ``predict_action`` + ``_regression_or_discrete_prediction``."""
        model = self._model
        if proprio is not None:
            raise ValueError("OpenVLA-OFT hidden-token mainline does not include proprio")

        with torch.inference_mode():
            native = self._policy.forward_action_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            hidden_token = hidden_token_from_projected(
                native.projected_tokens,
                image_keys=self._image_keys,
                patches_per_image=int(model.vision_backbone.get_num_patches()),
            )
            token_ids = native.full_action_logits.argmax(dim=-1)
            action_classes = self._policy.action_token_ids_to_classes(token_ids)
            normalized = self._policy.action_classes_to_normalized_actions(
                action_classes
            ).reshape(
                input_ids.shape[0], self._num_chunks, self._action_dim
            )
        actions = model._unnormalize_actions(
            normalized.detach().cpu().numpy(), self._unnorm_key
        )
        return actions, hidden_token, native.language_embedding, token_ids.reshape(
            input_ids.shape[0], self._num_chunks, self._action_dim
        )

    def _decode(
        self,
        lm_out: Any,
        action_start: int,
        batch_size: int,
    ) -> np.ndarray:
        """Decode normalized discrete actions as ``(B, chunk, dim)``."""

        # Headless one-trajectory LM: argmax logits at the action
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
) -> list[OFTDecodeOutput]:
    """Convenience wrapper: build a one-shot :class:`OFTBatchedDecoder` and run it.

    For repeated inference (collection), construct ``OFTBatchedDecoder`` ONCE and call
    ``predict_batch`` — that resolves the model handles / token ids / head mode a single
    time instead of per forward.  Kept for tests and single-shot callers.

    Raises ``ValueError`` if ``preps`` is empty.
    """
    if not preps:
        raise ValueError("batched_forward requires at least one prep")
    return OFTBatchedDecoder(policy, unnorm_key).predict_batch(preps)
