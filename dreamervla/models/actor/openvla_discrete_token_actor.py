from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from dreamervla.models.actor._load import extract_state_dict
from dreamervla.models.actor.base_actor import BaseActor
from dreamervla.utils.hf_checkpoint import is_hf_checkpoint, load_hf_prefixed_tensors

logger = logging.getLogger(__name__)


class OpenVLADiscreteTokenActor(BaseActor):
    """OpenVLA discrete action-token policy over predicted action hidden slots.

    This actor does not invent a new discrete head.  It adapts DreamerVLA world
    model action-hidden slots, applies the OpenVLA LM head to the action-token
    vocabulary tail, samples OpenVLA action token IDs, and decodes those IDs with
    the same bin-center rule used by OpenVLA-OFT's ``ActionTokenizer``.
    """

    def __init__(
        self,
        hidden_dim: int | None = None,
        action_hidden_dim: int = 4096,
        action_dim: int = 7,
        time_horizon: int = 8,
        vocab_size: int = 32000,
        action_token_bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
        adapter_type: str = "residual_mlp",
        adapter_hidden_dim: int = 1024,
        freeze_lm_head: bool = True,
        init_lm_head_ckpt: str | None = None,
        head_type: str = "oft_discrete_token",
        **_: Any,
    ) -> None:
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        self.action_token_bins = int(action_token_bins)
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.adapter_type = str(adapter_type).lower()
        self.head_type = str(head_type).lower()
        if self.head_type != "oft_discrete_token":
            raise ValueError(
                "OpenVLADiscreteTokenActor requires head_type='oft_discrete_token'."
            )
        if self.adapter_type not in {"identity", "mlp", "residual_mlp"}:
            raise ValueError(
                "adapter_type must be one of {'identity', 'mlp', 'residual_mlp'}"
            )
        if self.action_token_bins < 2:
            raise ValueError("action_token_bins must be >= 2")

        self.token_count = self.time_horizon * self.action_dim
        expected_flat = self.token_count * self.action_hidden_dim
        self.hidden_dim = expected_flat if hidden_dim is None else int(hidden_dim)
        if self.hidden_dim != expected_flat:
            raise ValueError(
                "OpenVLADiscreteTokenActor flat hidden dim mismatch: "
                f"hidden_dim={self.hidden_dim}, expected {expected_flat}"
            )

        lm_head_state = (
            self._load_lm_head_state_dict(str(init_lm_head_ckpt))
            if init_lm_head_ckpt
            else {}
        )
        weight = lm_head_state.get("weight")
        if weight is not None:
            vocab_size = int(weight.shape[0])
            if int(weight.shape[1]) != self.action_hidden_dim:
                raise ValueError(
                    "OpenVLA LM head hidden dim mismatch: "
                    f"checkpoint={tuple(weight.shape)}, action_hidden_dim={self.action_hidden_dim}"
                )
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= self.action_token_bins:
            raise ValueError(
                f"vocab_size={self.vocab_size} must exceed action_token_bins={self.action_token_bins}"
            )

        if self.adapter_type == "identity":
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.action_hidden_dim),
                nn.Linear(self.action_hidden_dim, int(adapter_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(adapter_hidden_dim), self.action_hidden_dim),
            )
            if self.adapter_type == "residual_mlp":
                final_linear = self.adapter[-1]
                if isinstance(final_linear, nn.Linear):
                    nn.init.zeros_(final_linear.weight)
                    nn.init.zeros_(final_linear.bias)

        self.lm_head = nn.Linear(self.action_hidden_dim, self.vocab_size, bias=False)
        if lm_head_state:
            missing, unexpected = self.lm_head.load_state_dict(lm_head_state, strict=False)
            logger.info(
                "Loaded OpenVLA LM head from %s; missing=%d unexpected=%d",
                init_lm_head_ckpt,
                len(missing),
                len(unexpected),
            )
        if bool(freeze_lm_head):
            for param in self.lm_head.parameters():
                param.requires_grad = False

        bins = torch.linspace(self.min_action, self.max_action, self.action_token_bins)
        centers = (bins[:-1] + bins[1:]) / 2.0
        valid_token_ids = torch.arange(
            self.vocab_size - self.action_token_bins,
            self.vocab_size,
            dtype=torch.long,
        )
        discretized = self.vocab_size - valid_token_ids
        center_indices = torch.clamp(
            discretized - 1,
            min=0,
            max=int(centers.numel()) - 1,
        )
        self.register_buffer("_bins", bins, persistent=False)
        self.register_buffer("_bin_centers", centers, persistent=False)
        self.register_buffer("_action_token_ids", valid_token_ids, persistent=False)
        self.register_buffer(
            "_action_values_by_class", centers[center_indices], persistent=False
        )

    def _load_lm_head_state_dict(self, ckpt_path: str) -> dict[str, torch.Tensor]:
        prefixes = (
            "lm_head.",
            "vla.lm_head.",
            "model.lm_head.",
            "language_model.lm_head.",
            "base_model.model.lm_head.",
        )
        if is_hf_checkpoint(ckpt_path):
            for prefix in prefixes:
                tensors = load_hf_prefixed_tensors(ckpt_path, prefix)
                if tensors:
                    return tensors
            return {}

        path = Path(ckpt_path).expanduser()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        candidates: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            for key in ("state_dict", "model"):
                value = payload.get(key)
                if isinstance(value, dict):
                    candidates.append(value)
            encoder_sd = payload.get("state_dicts", {}).get("encoder")
            if isinstance(encoder_sd, dict):
                candidates.append(encoder_sd)
            candidates.append(payload)

        strip_prefixes = ("module.", "backbone.", *prefixes)
        return extract_state_dict(candidates, strip_prefixes, {"weight"})

    def _reshape_action_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            if hidden.shape[1:] != (self.token_count, self.action_hidden_dim):
                raise ValueError(
                    "OpenVLADiscreteTokenActor hidden sequence shape mismatch: "
                    f"got {tuple(hidden.shape)}, expected [B,{self.token_count},{self.action_hidden_dim}]"
                )
            action_hidden = hidden
        elif hidden.ndim == 2:
            if int(hidden.shape[-1]) != self.hidden_dim:
                raise ValueError(
                    f"OpenVLADiscreteTokenActor hidden dim mismatch: got {hidden.shape[-1]}, "
                    f"expected {self.hidden_dim}"
                )
            action_hidden = hidden.reshape(
                hidden.shape[0], self.token_count, self.action_hidden_dim
            )
        else:
            raise ValueError(
                f"Unsupported OpenVLADiscreteTokenActor hidden shape: {tuple(hidden.shape)}"
            )
        return action_hidden.to(dtype=self.lm_head.weight.dtype)

    def _action_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        action_hidden = self._reshape_action_hidden(hidden)
        adapted = self.adapter(action_hidden)
        if self.adapter_type == "residual_mlp":
            adapted = action_hidden + adapted
        return adapted

    def _action_token_logits(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_hidden = self._action_hidden(hidden)
        # Compute ONLY the action-token columns by slicing the lm_head weight to
        # the action token rows first, so the full [.., vocab_size] logits tensor
        # is never materialized (the vocab tail is only used to extract these
        # columns). Mathematically + gradient identical to
        # ``self.lm_head(action_hidden).index_select(-1, action_token_ids)`` —
        # lm_head has no bias, and unselected rows get zero gradient either way.
        ids = self._action_token_ids.to(device=action_hidden.device)
        action_weight = self.lm_head.weight.index_select(0, ids)
        action_logits = torch.nn.functional.linear(action_hidden, action_weight)
        return action_logits.float(), action_hidden.float()

    def _classes_to_actions(self, classes: torch.Tensor) -> torch.Tensor:
        values = self._action_values_by_class.to(device=classes.device, dtype=torch.float32)
        flat = values.index_select(0, classes.reshape(-1).long())
        return flat.reshape(classes.shape)

    def _classes_to_token_ids(self, classes: torch.Tensor) -> torch.Tensor:
        token_ids = self._action_token_ids.to(device=classes.device)
        flat = token_ids.index_select(0, classes.reshape(-1).long())
        return flat.reshape(classes.shape)

    def _token_ids_to_classes(self, token_ids: torch.Tensor) -> torch.Tensor:
        start = self.vocab_size - self.action_token_bins
        classes = token_ids.long() - int(start)
        return classes.clamp(min=0, max=self.action_token_bins - 1)

    def _actions_to_classes(self, actions: torch.Tensor) -> torch.Tensor:
        clipped = actions.float().clamp(self.min_action, self.max_action)
        bins = self._bins.to(device=actions.device, dtype=clipped.dtype)
        discretized = torch.bucketize(clipped.reshape(-1), bins)
        discretized = discretized.clamp(min=1, max=self.action_token_bins)
        classes = self.action_token_bins - discretized
        return classes.reshape(actions.shape).long()

    def _chunk_from_classes(self, classes: torch.Tensor) -> torch.Tensor:
        actions = self._classes_to_actions(classes)
        return actions.reshape(classes.shape[0], self.time_horizon, self.action_dim)

    def reference_action_chunk(self, hidden: torch.Tensor) -> torch.Tensor:
        logits, _ = self._action_token_logits(hidden)
        classes = logits.argmax(dim=-1)
        return self._chunk_from_classes(classes)

    def forward(self, batch: dict[str, Any]) -> Any:
        mode = batch.get("mode")
        hidden = batch["hidden"]
        logits, action_hidden = self._action_token_logits(hidden)
        dist = Categorical(logits=logits)
        greedy_classes = logits.argmax(dim=-1)
        greedy_chunk = self._chunk_from_classes(greedy_classes)
        greedy_token_ids = self._classes_to_token_ids(greedy_classes)

        if mode == "sample":
            deterministic = bool(batch.get("deterministic", False))
            classes = greedy_classes if deterministic else dist.sample()
            token_ids = self._classes_to_token_ids(classes)
            action_chunk = self._chunk_from_classes(classes)
            token_log_prob = dist.log_prob(classes)
            extra = {
                "action_chunk": greedy_chunk,
                "action_token_ids": token_ids.reshape(
                    token_ids.shape[0], self.time_horizon, self.action_dim
                ),
                "greedy_action_token_ids": greedy_token_ids.reshape(
                    greedy_token_ids.shape[0], self.time_horizon, self.action_dim
                ),
                "action_token_logits": logits,
                "action_hidden": action_hidden,
                "mean_chunk": greedy_chunk,
                "std_chunk": torch.zeros_like(greedy_chunk),
            }
            if bool(batch.get("return_chunk", False)):
                log_prob = token_log_prob.sum(dim=-1)
                return action_chunk, log_prob, extra
            first_action = action_chunk[:, 0, :]
            first_log_prob = token_log_prob.reshape(
                logits.shape[0], self.time_horizon, self.action_dim
            )[:, 0].sum(dim=-1)
            return first_action, first_log_prob, extra

        if mode == "evaluate":
            action = batch["action"]
            action_token_ids = batch.get("action_token_ids")
            if action_token_ids is not None:
                classes = self._token_ids_to_classes(action_token_ids.to(logits.device))
                if classes.ndim == 3:
                    classes = classes.reshape(classes.shape[0], -1)
                elif classes.ndim != 2:
                    raise ValueError(
                        f"action_token_ids must be [B,N] or [B,T,A], got {tuple(classes.shape)}"
                    )
            else:
                classes = self._actions_to_classes(action.to(logits.device))
                if classes.ndim == 3:
                    classes = classes.reshape(classes.shape[0], -1)
                elif classes.ndim == 2:
                    classes = classes.reshape(classes.shape[0], -1)
                else:
                    raise ValueError(
                        f"action must be [B,A] or [B,T,A], got {tuple(action.shape)}"
                    )

            token_log_prob = dist.log_prob(classes)
            token_entropy = dist.entropy()
            if action.ndim == 3:
                log_prob = token_log_prob.sum(dim=-1)
                entropy = token_entropy.sum(dim=-1)
            elif action.ndim == 2:
                log_prob = token_log_prob[:, : self.action_dim].sum(dim=-1)
                entropy = token_entropy[:, : self.action_dim].sum(dim=-1)
            else:
                raise ValueError(f"action must be [B,A] or [B,T,A], got {tuple(action.shape)}")
            return (
                log_prob,
                entropy,
                {
                    "action_chunk": greedy_chunk,
                    "action_token_ids": greedy_token_ids.reshape(
                        greedy_token_ids.shape[0], self.time_horizon, self.action_dim
                    ),
                    "action_token_logits": logits,
                    "action_hidden": action_hidden,
                    "mean_chunk": greedy_chunk,
                    "std_chunk": torch.zeros_like(greedy_chunk),
                },
            )
        raise ValueError(f"Unknown OpenVLADiscreteTokenActor forward mode: {mode!r}")


__all__ = ["OpenVLADiscreteTokenActor"]
