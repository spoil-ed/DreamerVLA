from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from dreamervla.algorithms.actor._load import extract_state_dict
from dreamervla.algorithms.actor.base_actor import BaseActor
from dreamervla.utils.hf_checkpoint import is_hf_checkpoint, load_hf_prefixed_tensors

logger = logging.getLogger(__name__)


class LatentToOpenVLAHiddenStateActor(BaseActor):
    """Bridge ``[B,256,4096]`` hidden tokens to internal action slots."""

    # Raw-image preprocessing remains owned by the frozen OpenVLA-OFT
    # checkpoint.  Standalone evaluators use this narrow capability marker to
    # compose that extractor with this restored hidden-token actor.
    requires_external_hidden_extractor = True

    def __init__(
        self,
        hidden_dim: int | None = None,
        source_token_count: int | None = 256,
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
        if "hidden_token_dim" in kwargs:
            raise TypeError(
                "LatentToOpenVLAHiddenStateActor uses hidden_state_dim; "
                "hidden_token_dim belongs to the removed 56x1024 route."
            )
        if hidden_dim is not None:
            raise TypeError(
                "flat hidden_dim observations are removed; pass tokenized hidden_token [B,256,4096]"
            )
        super().__init__()
        self.source_token_count = int(source_token_count or 256)
        self.source_token_dim = int(source_token_dim)
        if self.source_token_count != 256 or self.source_token_dim != 4096:
            raise ValueError(
                "LatentToOpenVLAHiddenStateActor requires source tokens "
                "[256,4096], got "
                f"[{self.source_token_count},{self.source_token_dim}]"
            )
        self.hidden_state_dim = int(hidden_state_dim)
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        if self.hidden_state_dim != 4096:
            raise ValueError(
                f"OpenVLA decoder hidden_state_dim is fixed to 4096, got {self.hidden_state_dim}"
            )
        if self.action_dim != 7 or self.time_horizon != 8:
            raise ValueError(
                "LIBERO mainline action geometry is fixed to [8,7], got "
                f"[{self.time_horizon},{self.action_dim}]"
            )
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
            raise ValueError("adapter_type must be one of {'identity', 'mlp', 'residual_mlp'}")
        if self.action_token_bins < 2:
            raise ValueError("action_token_bins must be >= 2")
        if self.bridge_hidden_dim % int(num_bridge_heads) != 0:
            raise ValueError(
                "bridge_hidden_dim must be divisible by num_bridge_heads: "
                f"{self.bridge_hidden_dim} % {int(num_bridge_heads)} != 0"
            )

        self.action_token_count = self.time_horizon * self.action_dim
        self.hidden_dim = None

        lm_head_state = (
            self._load_lm_head_state_dict(str(init_lm_head_ckpt)) if init_lm_head_ckpt else {}
        )
        weight = lm_head_state.get("weight")
        action_weight = None
        if weight is not None:
            loaded_rows = int(weight.shape[0])
            if loaded_rows > self.action_token_bins:
                vocab_size = loaded_rows
                action_weight = weight[-self.action_token_bins :].contiguous()
            elif loaded_rows == self.action_token_bins:
                action_weight = weight.contiguous()
            else:
                raise ValueError(
                    "OpenVLA LM head must contain at least action_token_bins rows: "
                    f"checkpoint={tuple(weight.shape)}, bins={self.action_token_bins}"
                )
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

        # Only these rows can contribute to the categorical action logits. Keeping
        # the full frozen vocabulary head would inflate FSDP, checkpoints, and
        # Actor->Rollout synchronization without changing any policy output.
        self.lm_head = nn.Linear(
            self.hidden_state_dim,
            self.action_token_bins,
            bias=False,
        )
        if action_weight is not None:
            missing, unexpected = self.lm_head.load_state_dict(
                {"weight": action_weight}, strict=False
            )
            logger.info(
                "Loaded OpenVLA action-token LM-head rows from %s; missing=%d unexpected=%d",
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
        self.register_buffer("_action_values_by_class", centers[center_indices], persistent=False)

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

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Accept legacy policy checkpoints that stored the full vocabulary head."""

        key = f"{prefix}lm_head.weight"
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            rows = int(value.shape[0])
            if rows > self.action_token_bins:
                state_dict[key] = value[-self.action_token_bins :].contiguous()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _source_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
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
            "hidden must be tokenized hidden_token [B,256,4096]; "
            f"flat observations are closed, got {tuple(hidden.shape)}"
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
        action_weight = self.lm_head.weight
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
            logprob_type = str(batch.get("logprob_type", "sequence")).lower()
            if logprob_type not in {"sequence", "token_level"}:
                raise ValueError(
                    f"logprob_type must be 'sequence' or 'token_level', got {logprob_type!r}"
                )
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
                log_prob = (
                    token_log_prob.reshape(
                        token_log_prob.shape[0], self.time_horizon, self.action_dim
                    )
                    if logprob_type == "token_level"
                    else token_log_prob.sum(dim=-1)
                )
                return action_chunk, log_prob, extra
            first_action = action_chunk[:, 0, :]
            first_log_prob = token_log_prob.reshape(
                logits.shape[0], self.time_horizon, self.action_dim
            )[:, 0].sum(dim=-1)
            return first_action, first_log_prob, extra

        if mode == "evaluate":
            action = batch["action"]
            logprob_type = str(batch.get("logprob_type", "sequence")).lower()
            if logprob_type not in {"sequence", "token_level"}:
                raise ValueError(
                    f"logprob_type must be 'sequence' or 'token_level', got {logprob_type!r}"
                )
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
                    raise ValueError(f"action must be [B,A] or [B,T,A], got {tuple(action.shape)}")

            token_log_prob = dist.log_prob(classes)
            token_entropy = dist.entropy()
            if logprob_type == "token_level" and action.ndim == 3:
                log_prob = token_log_prob.reshape(
                    token_log_prob.shape[0], self.time_horizon, self.action_dim
                )
                entropy = token_entropy.reshape(
                    token_entropy.shape[0], self.time_horizon, self.action_dim
                )
            elif logprob_type == "token_level" and action.ndim == 2:
                log_prob = token_log_prob[:, : self.action_dim]
                entropy = token_entropy[:, : self.action_dim]
            elif action.ndim == 3:
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
