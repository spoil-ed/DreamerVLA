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


class LatentToOpenVLAHiddenStateActor(BaseActor):
    """Bridge query-before VLA hidden states to OpenVLA discrete action tokens."""

    def __init__(
        self,
        hidden_dim: int | None = None,
        source_token_count: int | None = None,
        source_token_dim: int = 4096,
        hidden_state_dim: int = 4096,
        action_dim: int = 7,
        time_horizon: int = 8,
        bridge_hidden_dim: int = 1024,
        num_bridge_layers: int = 2,
        num_bridge_heads: int = 8,
        bridge_dropout: float = 0.1,
        vocab_size: int = 32000,
        action_token_bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
        adapter_type: str = "residual_mlp",
        adapter_hidden_dim: int = 1024,
        freeze_lm_head: bool = True,
        init_lm_head_ckpt: str | None = None,
        head_type: str = "oft_discrete_token",
        **kwargs: Any,
    ) -> None:
        if "action_hidden_dim" in kwargs:
            raise TypeError(
                "LatentToOpenVLAHiddenStateActor uses hidden_state_dim; "
                "action_hidden_dim belongs only to legacy action-hidden routes."
            )
        super().__init__()
        self.source_token_count = (
            int(source_token_count) if source_token_count is not None else None
        )
        self.source_token_dim = int(source_token_dim)
        self.hidden_state_dim = int(hidden_state_dim)
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        self.action_token_bins = int(action_token_bins)
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.adapter_type = str(adapter_type).lower()
        self.head_type = str(head_type).lower()
        self.bridge_hidden_dim = int(bridge_hidden_dim)
        if self.head_type != "oft_discrete_token":
            raise ValueError(
                "LatentToOpenVLAHiddenStateActor requires head_type='oft_discrete_token'."
            )
        if self.adapter_type not in {"identity", "mlp", "residual_mlp"}:
            raise ValueError(
                "adapter_type must be one of {'identity', 'mlp', 'residual_mlp'}"
            )
        if self.action_token_bins < 2:
            raise ValueError("action_token_bins must be >= 2")
        if self.bridge_hidden_dim % int(num_bridge_heads) != 0:
            raise ValueError(
                "bridge_hidden_dim must be divisible by num_bridge_heads: "
                f"{self.bridge_hidden_dim} % {int(num_bridge_heads)} != 0"
            )

        self.action_token_count = self.time_horizon * self.action_dim
        expected_flat = (
            None
            if self.source_token_count is None
            else self.source_token_count * self.source_token_dim
        )
        self.hidden_dim = (
            int(hidden_dim)
            if hidden_dim is not None
            else int(expected_flat)
            if expected_flat is not None
            else None
        )
        if (
            expected_flat is not None
            and self.hidden_dim is not None
            and self.hidden_dim != expected_flat
        ):
            raise ValueError(
                "LatentToOpenVLAHiddenStateActor flat hidden dim mismatch: "
                f"hidden_dim={self.hidden_dim}, expected source_token_count * "
                f"source_token_dim = {expected_flat}"
            )

        lm_head_state = (
            self._load_lm_head_state_dict(str(init_lm_head_ckpt))
            if init_lm_head_ckpt
            else {}
        )
        weight = lm_head_state.get("weight")
        if weight is not None:
            vocab_size = int(weight.shape[0])
            if int(weight.shape[1]) != self.hidden_state_dim:
                raise ValueError(
                    "OpenVLA LM head hidden-state dim mismatch: "
                    f"checkpoint={tuple(weight.shape)}, hidden_state_dim={self.hidden_state_dim}"
                )
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= self.action_token_bins:
            raise ValueError(
                f"vocab_size={self.vocab_size} must exceed action_token_bins={self.action_token_bins}"
            )

        self.source_proj = (
            nn.Identity()
            if self.source_token_dim == self.bridge_hidden_dim
            else nn.Linear(self.source_token_dim, self.bridge_hidden_dim)
        )
        self.action_queries = nn.Parameter(
            torch.randn(self.action_token_count, self.bridge_hidden_dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.bridge_hidden_dim,
            nhead=int(num_bridge_heads),
            dim_feedforward=self.bridge_hidden_dim * 4,
            dropout=float(bridge_dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.bridge = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(num_bridge_layers),
            norm=nn.LayerNorm(self.bridge_hidden_dim),
        )
        self.hidden_state_proj = (
            nn.Identity()
            if self.bridge_hidden_dim == self.hidden_state_dim
            else nn.Linear(self.bridge_hidden_dim, self.hidden_state_dim)
        )

        if self.adapter_type == "identity":
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.hidden_state_dim),
                nn.Linear(self.hidden_state_dim, int(adapter_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(adapter_hidden_dim), self.hidden_state_dim),
            )
            if self.adapter_type == "residual_mlp":
                final_linear = self.adapter[-1]
                if isinstance(final_linear, nn.Linear):
                    nn.init.zeros_(final_linear.weight)
                    nn.init.zeros_(final_linear.bias)

        self.lm_head = nn.Linear(self.hidden_state_dim, self.vocab_size, bias=False)
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

    def _source_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 2:
            if self.hidden_dim is not None and int(hidden.shape[-1]) != self.hidden_dim:
                raise ValueError(
                    f"flat hidden dim mismatch: got {hidden.shape[-1]}, expected {self.hidden_dim}"
                )
            if self.source_token_count is None:
                if int(hidden.shape[-1]) % self.source_token_dim != 0:
                    raise ValueError(
                        "flat hidden dim must be divisible by source_token_dim when "
                        "source_token_count is omitted"
                    )
                token_count = int(hidden.shape[-1]) // self.source_token_dim
            else:
                token_count = self.source_token_count
            return hidden.reshape(hidden.shape[0], token_count, self.source_token_dim)
        if hidden.ndim == 3:
            if int(hidden.shape[-1]) != self.source_token_dim:
                raise ValueError(
                    f"source token dim mismatch: got {hidden.shape[-1]}, expected {self.source_token_dim}"
                )
            if self.source_token_count is not None and int(hidden.shape[1]) != int(
                self.source_token_count
            ):
                raise ValueError(
                    "source token count mismatch: "
                    f"got {hidden.shape[1]}, expected {self.source_token_count}"
                )
            return hidden
        raise ValueError(
            f"hidden must be flat [B,N*D] or tokenized [B,N,D], got {tuple(hidden.shape)}"
        )

    def _hidden_state_slots(self, hidden: torch.Tensor) -> torch.Tensor:
        source = self._source_tokens(hidden)
        dtype = self.action_queries.dtype
        source = self.source_proj(source.to(dtype=dtype))
        queries = self.action_queries.to(device=source.device, dtype=source.dtype)
        queries = queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        bridged = self.bridge(tgt=queries, memory=source)
        hidden_state = self.hidden_state_proj(bridged)
        adapted = self.adapter(hidden_state)
        if self.adapter_type == "residual_mlp":
            adapted = hidden_state + adapted
        return adapted

    def _action_token_logits(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state = self._hidden_state_slots(hidden)
        ids = self._action_token_ids.to(device=hidden_state.device)
        action_weight = self.lm_head.weight.index_select(0, ids)
        action_logits = torch.nn.functional.linear(
            hidden_state.to(dtype=action_weight.dtype), action_weight
        )
        return action_logits.float(), hidden_state.float()

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
        logits, hidden_state = self._action_token_logits(hidden)
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
                "hidden_state": hidden_state,
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
                    "hidden_state": hidden_state,
                    "mean_chunk": greedy_chunk,
                    "std_chunk": torch.zeros_like(greedy_chunk),
                },
            )
        raise ValueError(f"Unknown LatentToOpenVLAHiddenStateActor forward mode: {mode!r}")


__all__ = ["LatentToOpenVLAHiddenStateActor"]
