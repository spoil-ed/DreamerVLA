from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.actor.base_actor import BaseActor
from src.models.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    L1RegressionActionHead,
    PerTokenRegressionActionHead,
)


class Pi0ActionHiddenActor(BaseActor):
    """Decode predicted pi0 action-query hidden states with VLA final head."""

    def __init__(
        self,
        hidden_dim: int = 5120,
        action_hidden_dim: int = 1024,
        action_dim: int = 7,
        time_horizon: int = 5,
        adapter_type: str = "residual_mlp",
        adapter_hidden_dim: int = 1024,
        freeze_output_projection: bool = True,
        initial_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        freeze_log_std: bool = False,
        init_action_head_ckpt: str | None = None,
        head_type: str = "pi0_query",
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.freeze_log_std = bool(freeze_log_std)
        self.adapter_type = str(adapter_type).lower()
        self.head_type = str(head_type).lower()
        if self.head_type not in {"pi0_query", "legacy"}:
            raise ValueError("head_type must be one of {'pi0_query', 'legacy'}")
        if self.head_type == "pi0_query":
            self.token_count = self.time_horizon
        else:
            self.token_count = self.time_horizon * self.action_dim
        expected_flat = self.token_count * self.action_hidden_dim
        if self.hidden_dim != expected_flat:
            raise ValueError(
                "Pi0ActionHiddenActor flat hidden dim mismatch: "
                f"head_type={self.head_type}, hidden_dim={self.hidden_dim}, "
                f"expected token_count={self.token_count} * action_hidden_dim="
                f"{self.action_hidden_dim} = {expected_flat}"
            )
        if self.adapter_type not in {"identity", "mlp", "residual_mlp"}:
            raise ValueError(
                "adapter_type must be one of {'identity', 'mlp', 'residual_mlp'}"
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

        if self.head_type == "pi0_query":
            self.output_projection = PerTokenRegressionActionHead(
                self.action_hidden_dim,
                self.action_hidden_dim,
                self.action_dim,
            )
        else:
            self.output_projection = L1RegressionActionHead(
                self.action_hidden_dim,
                self.action_hidden_dim,
                self.time_horizon,
                self.action_dim,
            )
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(initial_log_std)),
            requires_grad=not self.freeze_log_std,
        )

        if init_action_head_ckpt:
            ckpt_path = str(init_action_head_ckpt)
            import os as _os

            if _os.path.isdir(ckpt_path):
                self._load_output_projection_from_hf_dir(ckpt_path)
            else:
                self._load_output_projection_from_vla_ckpt(ckpt_path)
        if bool(freeze_output_projection):
            for param in self.output_projection.parameters():
                param.requires_grad = False

    def _load_output_projection_from_hf_dir(self, model_dir: str) -> None:
        """Load output_projection weights from a HuggingFace safetensors directory."""
        import glob as _glob
        import json as _json
        import os as _os

        from safetensors.torch import load_file as _load_safetensors

        prefix = "action_head.output_projection."
        index_path = _os.path.join(model_dir, "model.safetensors.index.json")
        files: list[str]
        if _os.path.isfile(index_path):
            with open(index_path, "r") as fh:
                index = _json.load(fh).get("weight_map", {})
            files = sorted(
                {
                    _os.path.join(model_dir, p)
                    for k, p in index.items()
                    if k.startswith(prefix)
                }
            )
        else:
            files = sorted(_glob.glob(_os.path.join(model_dir, "*.safetensors")))
        output_projection_sd: dict[str, torch.Tensor] = {}
        for path in files:
            tensors = _load_safetensors(path)
            for k, v in tensors.items():
                if k.startswith(prefix):
                    output_projection_sd[k[len(prefix) :]] = v.to(dtype=torch.float32)
        if not output_projection_sd:
            raise RuntimeError(f"No '{prefix}' tensors found in HF dir: {model_dir}")
        missing, unexpected = self.output_projection.load_state_dict(
            output_projection_sd, strict=False
        )
        print(
            f"[Pi0ActionHiddenActor] loaded {len(output_projection_sd)} output_projection tensors "
            f"from HF dir {model_dir}; missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[Pi0ActionHiddenActor] WARN missing (first 5): {missing[:5]}")
        if unexpected:
            print(f"[Pi0ActionHiddenActor] WARN unexpected (first 5): {unexpected[:5]}")

    def _load_output_projection_from_vla_ckpt(self, ckpt_path: str) -> None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        direct_sd = self._extract_direct_action_head_state_dict(payload)
        if direct_sd:
            missing, unexpected = self.output_projection.load_state_dict(
                direct_sd, strict=False
            )
            print(
                f"[Pi0ActionHiddenActor] loaded {len(direct_sd)} output_projection tensors "
                f"from standalone action-head ckpt {ckpt_path}; "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing:
                print(
                    f"[Pi0ActionHiddenActor] WARN missing output_projection tensors (first 5): {missing[:5]}"
                )
            if unexpected:
                print(
                    f"[Pi0ActionHiddenActor] WARN unexpected output_projection tensors (first 5): {unexpected[:5]}"
                )
            return

        encoder_sd = payload.get("state_dicts", {}).get("encoder")
        if encoder_sd is None:
            raise RuntimeError(
                f"Pi0 action-hidden actor checkpoint has no state_dicts.encoder: {ckpt_path}"
            )
        prefix = "backbone.action_head.output_projection."
        output_projection_sd = {
            key[len(prefix) :]: value
            for key, value in encoder_sd.items()
            if key.startswith(prefix)
        }
        if not output_projection_sd:
            raise RuntimeError(
                f"Pi0 action-hidden actor checkpoint has no '{prefix}' output_projection tensors: {ckpt_path}"
            )
        missing, unexpected = self.output_projection.load_state_dict(
            output_projection_sd, strict=False
        )
        print(
            f"[Pi0ActionHiddenActor] loaded {len(output_projection_sd)} output_projection tensors "
            f"from VLA ckpt; missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(
                f"[Pi0ActionHiddenActor] WARN missing output_projection tensors (first 5): {missing[:5]}"
            )
        if unexpected:
            print(
                f"[Pi0ActionHiddenActor] WARN unexpected output_projection tensors (first 5): {unexpected[:5]}"
            )
        del payload

    def _extract_direct_action_head_state_dict(
        self, payload: Any
    ) -> dict[str, torch.Tensor]:
        """Extract a plain action-head state_dict, e.g. OpenVLA-OFT component ckpts."""
        if not isinstance(payload, dict):
            return {}

        candidates: list[dict[str, Any]] = []
        for key in ("state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                candidates.append(value)
        candidates.append(payload)

        expected_keys = set(self.output_projection.state_dict().keys())
        prefixes = ("module.", "output_projection.", "action_head.")
        for candidate in candidates:
            if not candidate or not all(
                isinstance(k, str) and isinstance(v, torch.Tensor)
                for k, v in candidate.items()
            ):
                continue
            normalized: dict[str, torch.Tensor] = {}
            for key, value in candidate.items():
                normalized_key = key
                changed = True
                while changed:
                    changed = False
                    for prefix in prefixes:
                        if normalized_key.startswith(prefix):
                            normalized_key = normalized_key[len(prefix) :]
                            changed = True
                            break
                if normalized_key in expected_keys:
                    normalized[normalized_key] = value.to(dtype=torch.float32)
            if normalized:
                return normalized
        return {}

    def _reshape_action_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            if hidden.shape[1:] != (self.token_count, self.action_hidden_dim):
                raise ValueError(
                    "Pi0ActionHiddenActor hidden sequence shape mismatch: "
                    f"head_type={self.head_type}, got {tuple(hidden.shape)}, "
                    f"expected [B,{self.token_count},{self.action_hidden_dim}]"
                )
            action_hidden = hidden
        elif hidden.ndim == 2:
            if int(hidden.shape[-1]) != self.hidden_dim:
                raise ValueError(
                    f"Pi0ActionHiddenActor hidden dim mismatch: got {hidden.shape[-1]}, expected {self.hidden_dim}"
                )
            action_hidden = hidden.reshape(
                hidden.shape[0], self.token_count, self.action_hidden_dim
            )
        else:
            raise ValueError(
                f"Unsupported Pi0ActionHiddenActor hidden shape: {tuple(hidden.shape)}"
            )

        param_dtype = next(self.output_projection.parameters()).dtype
        return action_hidden.to(dtype=param_dtype)

    def _action_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        action_hidden = self._reshape_action_hidden(hidden)
        adapted = self.adapter(action_hidden)
        if self.adapter_type == "residual_mlp":
            adapted = action_hidden + adapted
        return adapted

    def _action_chunk(self, hidden: torch.Tensor) -> torch.Tensor:
        action_hidden = self._action_hidden(hidden)
        actions = self.output_projection(action_hidden)
        return actions.reshape(
            action_hidden.shape[0], self.time_horizon, self.action_dim
        ).float()

    def reference_action_chunk(self, hidden: torch.Tensor) -> torch.Tensor:
        """Original frozen VLA head output before the trainable actor adapter."""
        action_hidden = self._reshape_action_hidden(hidden)
        actions = self.output_projection(action_hidden)
        return actions.reshape(
            action_hidden.shape[0], self.time_horizon, self.action_dim
        ).float()

    def forward(self, batch: dict[str, Any]) -> Any:
        mode = batch.get("mode")
        hidden = batch["hidden"]
        action_chunk = self._action_chunk(hidden)
        dist, mean, std = self._normal_from_action_chunk(action_chunk)
        chunk_dist, mean_chunk, std_chunk = self._normal_from_full_action_chunk(
            action_chunk
        )
        if mode == "sample":
            deterministic = bool(batch.get("deterministic", False))
            if bool(batch.get("return_chunk", False)):
                action = mean_chunk if deterministic else chunk_dist.rsample()
                log_prob = chunk_dist.log_prob(action).sum(dim=(-1, -2))
                return (
                    action,
                    log_prob,
                    {
                        "mean": mean,
                        "std": std,
                        "mean_chunk": mean_chunk,
                        "std_chunk": std_chunk,
                        "action_chunk": mean_chunk,
                    },
                )
            action = mean if deterministic else dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return (
                action,
                log_prob,
                {"mean": mean, "std": std, "action_chunk": mean_chunk},
            )
        if mode == "evaluate":
            action = batch["action"]
            if action.ndim == 3:
                log_prob = chunk_dist.log_prob(action).sum(dim=(-1, -2))
                entropy = chunk_dist.entropy().sum(dim=(-1, -2))
                return (
                    log_prob,
                    entropy,
                    {
                        "mean": mean,
                        "std": std,
                        "mean_chunk": mean_chunk,
                        "std_chunk": std_chunk,
                    },
                )
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return (
                log_prob,
                entropy,
                {
                    "mean": mean,
                    "std": std,
                    "mean_chunk": mean_chunk,
                    "std_chunk": std_chunk,
                },
            )
        raise ValueError(f"Unknown Pi0ActionHiddenActor forward mode: {mode!r}")


__all__ = ["Pi0ActionHiddenActor"]
