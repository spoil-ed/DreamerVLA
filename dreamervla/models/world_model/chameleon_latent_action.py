from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChameleonLatentActionOutput:
    pred_latent: torch.Tensor
    target_latent: torch.Tensor
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class ChameleonLatentActionWorldModel(nn.Module):
    """Z-only action-conditioned latent dynamics.

    This model intentionally has no Dreamer deterministic state ``h`` and no
    posterior/prior KL.  A frozen Chameleon backbone outside this module
    produces latents:

        z_t, z_{t+k} = encoder(image_t), encoder(image_{t+k})

    The trainable model only learns:

        f(z_t, a_t, ..., a_{t+k-1}) -> z_{t+k}

    ``latent`` can be either [B, C] image-pooled Chameleon hidden states or
    [B, N, C] per-image-token Chameleon hidden states.  The same module handles
    both by applying the latent projection/output heads over the last axis.
    """

    def __init__(
        self,
        latent_dim: int = 4096,
        action_dim: int = 7,
        model_dim: int = 1024,
        action_layers: int = 2,
        action_heads: int = 8,
        predictor_hidden_dim: int = 2048,
        dropout: float = 0.1,
        residual: bool = True,
        mse_coef: float = 1.0,
        cosine_coef: float = 0.1,
        delta_coef: float = 0.0,
        action_pool: str = "last",
        max_action_steps: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.residual = bool(residual)
        self.mse_coef = float(mse_coef)
        self.cosine_coef = float(cosine_coef)
        self.delta_coef = float(delta_coef)
        self.action_pool = str(action_pool).lower()
        if self.action_pool not in {"last", "mean"}:
            raise ValueError("action_pool must be 'last' or 'mean'")

        self.latent_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.model_dim),
            nn.GELU(),
        )
        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.model_dim),
            nn.GELU(),
        )
        self.action_pos = nn.Parameter(
            torch.zeros(1, int(max_action_steps), self.model_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(action_heads),
            dim_feedforward=max(int(predictor_hidden_dim), self.model_dim * 2),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(action_layers),
            enable_nested_tensor=False,
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, int(predictor_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(predictor_hidden_dim), self.latent_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.action_pos, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_actions(self, action_seq: torch.Tensor) -> torch.Tensor:
        """Encode a k-step action sequence into one summary vector [B, D]."""
        if action_seq.ndim != 3:
            raise ValueError(
                f"action_seq must be [B,K,A], got {tuple(action_seq.shape)}"
            )
        if action_seq.shape[-1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {action_seq.shape[-1]}, expected {self.action_dim}"
            )
        K = int(action_seq.shape[1])
        if K < 1:
            raise ValueError("action_seq must contain at least one action")
        if K > self.action_pos.shape[1]:
            raise ValueError(
                f"action horizon K={K} exceeds max_action_steps={self.action_pos.shape[1]}"
            )
        x = self.action_proj(action_seq)
        x = x + self.action_pos[:, :K].to(dtype=x.dtype, device=x.device)
        x = self.action_encoder(x)
        if self.action_pool == "mean":
            return x.mean(dim=1)
        return x[:, -1]

    def predict(self, latent: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent dim mismatch: got {latent.shape[-1]}, expected {self.latent_dim}"
            )
        action_summary = self.encode_actions(action_seq)
        z = self.latent_proj(latent)
        if latent.ndim == 3:
            action_summary = action_summary[:, None, :].expand(-1, latent.shape[1], -1)
        elif latent.ndim != 2:
            raise ValueError(
                f"latent must be [B,C] or [B,N,C], got {tuple(latent.shape)}"
            )
        delta_or_pred = self.fuse(torch.cat([z, action_summary], dim=-1))
        return latent + delta_or_pred if self.residual else delta_or_pred

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        latent = batch["latent"]
        target = batch["target_latent"]
        action_seq = batch["action_seq"]
        pred = self.predict(latent, action_seq)

        mse_loss = F.mse_loss(pred.float(), target.float())
        pred_flat = pred.float().flatten(start_dim=1)
        target_flat = target.float().flatten(start_dim=1)
        cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
        cosine_loss = 1.0 - cosine

        delta_loss = mse_loss.new_zeros(())
        if self.delta_coef > 0:
            delta_pred = pred.float() - latent.float()
            delta_target = target.float() - latent.float()
            delta_loss = F.mse_loss(delta_pred, delta_target)

        loss = (
            self.mse_coef * mse_loss
            + self.cosine_coef * cosine_loss
            + self.delta_coef * delta_loss
        )

        with torch.no_grad():
            target_delta = target.float() - latent.float()
            pred_delta = pred.float() - latent.float()
            metrics = {
                "mse_loss": mse_loss.detach(),
                "cosine_loss": cosine_loss.detach(),
                "delta_loss": delta_loss.detach(),
                "pred_target_cos": cosine.detach(),
                "latent_norm": latent.float().flatten(start_dim=1).norm(dim=-1).mean(),
                "target_norm": target.float().flatten(start_dim=1).norm(dim=-1).mean(),
                "pred_norm": pred.float().flatten(start_dim=1).norm(dim=-1).mean(),
                "target_delta_norm": target_delta.flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "pred_delta_norm": pred_delta.flatten(start_dim=1).norm(dim=-1).mean(),
                "action_norm": action_seq.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
            }
        return {"pred_latent": pred, "target_latent": target, "loss": loss, **metrics}


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(float(max_period), device=t.device))
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(
            self.timestep_embedding(t, self.frequency_embedding_size).to(
                next(self.parameters()).dtype
            )
        )


class ChameleonLatentFlowWorldModel(nn.Module):
    """LaDiWM-style frozen-latent action world model.

    The workspace supplies frozen Chameleon latents ``z_t`` and ``z_{t+k}``.
    This module trains on their residual with the same rectified-flow target
    used in LaDiWM:

        delta = z_{t+k} - z_t
        C = -delta
        x_noisy = delta + t * C + t * eps = (1 - t) * delta + t * eps

    A transformer receives ``x_noisy``, ``z_t``, ``t`` and the action sequence,
    then predicts ``C``.  For logging, ``z_hat = z_t - C_pred`` is compared with
    the target latent.
    """

    def __init__(
        self,
        latent_dim: int = 4096,
        action_dim: int = 7,
        model_dim: int = 1024,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_layers: int = 2,
        max_action_steps: int = 16,
        time_embed_dim: int = 256,
        flow_loss_coef: float = 1.0,
        latent_mse_coef: float = 0.1,
        cosine_coef: float = 0.01,
        variation_coef: float = 0.0,
        eps: float = 1e-3,
        context_frames: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.flow_loss_coef = float(flow_loss_coef)
        self.latent_mse_coef = float(latent_mse_coef)
        self.cosine_coef = float(cosine_coef)
        self.variation_coef = float(variation_coef)
        self.eps = float(eps)
        self.context_frames = int(context_frames)

        self.latent_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.model_dim),
        )
        self.noisy_delta_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.model_dim),
        )
        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.model_dim),
            nn.SiLU(),
        )
        action_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(heads),
            dim_feedforward=int(self.model_dim * mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(
            action_layer,
            num_layers=int(action_layers),
            enable_nested_tensor=False,
        )
        self.action_pos = nn.Parameter(
            torch.zeros(1, int(max_action_steps), self.model_dim)
        )
        self.time_embed = _TimestepEmbedder(
            self.model_dim, frequency_embedding_size=int(time_embed_dim)
        )

        self.cond_type = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.latent_type = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.noisy_type = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        block = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(heads),
            dim_feedforward=int(self.model_dim * mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            block,
            num_layers=int(depth),
            enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(self.model_dim)
        self.out = nn.Linear(self.model_dim, self.latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for param in (
            self.action_pos,
            self.cond_type,
            self.latent_type,
            self.noisy_type,
        ):
            nn.init.normal_(param, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _as_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 2:
            return x[:, None, :], True
        if x.ndim == 3:
            return x, False
        raise ValueError(f"latent must be [B,C] or [B,N,C], got {tuple(x.shape)}")

    def encode_actions(self, action_seq: torch.Tensor) -> torch.Tensor:
        if action_seq.ndim != 3:
            raise ValueError(
                f"action_seq must be [B,K,A], got {tuple(action_seq.shape)}"
            )
        if action_seq.shape[-1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {action_seq.shape[-1]}, expected {self.action_dim}"
            )
        K = int(action_seq.shape[1])
        if not (1 <= K <= self.action_pos.shape[1]):
            raise ValueError(
                f"action horizon K={K} outside [1,{self.action_pos.shape[1]}]"
            )
        x = self.action_proj(action_seq)
        x = x + self.action_pos[:, :K].to(dtype=x.dtype, device=x.device)
        return self.action_encoder(x)[:, -1]

    def encode_action_sequence(self, action_seq: torch.Tensor) -> torch.Tensor:
        if action_seq.ndim != 3:
            raise ValueError(
                f"action_seq must be [B,K,A], got {tuple(action_seq.shape)}"
            )
        if action_seq.shape[-1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {action_seq.shape[-1]}, expected {self.action_dim}"
            )
        K = int(action_seq.shape[1])
        if not (1 <= K <= self.action_pos.shape[1]):
            raise ValueError(
                f"action horizon K={K} outside [1,{self.action_pos.shape[1]}]"
            )
        x = self.action_proj(action_seq)
        x = x + self.action_pos[:, :K].to(dtype=x.dtype, device=x.device)
        return self.action_encoder(x)

    def predict_C(
        self,
        latent: torch.Tensor,
        noisy_delta: torch.Tensor,
        action_seq: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        latent_tokens, squeezed = self._as_tokens(latent)
        noisy_tokens, noisy_squeezed = self._as_tokens(noisy_delta)
        if (
            squeezed != noisy_squeezed
            or latent_tokens.shape[:2] != noisy_tokens.shape[:2]
        ):
            raise ValueError("latent and noisy_delta must have matching token layout")
        action_cond = self.encode_actions(action_seq)
        time_cond = self.time_embed(t.to(device=latent.device))
        cond = (action_cond + time_cond)[:, None, :] + self.cond_type.to(
            dtype=latent.dtype, device=latent.device
        )
        z = self.latent_proj(latent_tokens) + self.latent_type.to(
            dtype=latent.dtype, device=latent.device
        )
        x = self.noisy_delta_proj(noisy_tokens) + self.noisy_type.to(
            dtype=latent.dtype, device=latent.device
        )
        tokens = torch.cat([cond, z, x], dim=1)
        tokens = self.transformer(tokens)
        pred_tokens = tokens[:, 1 + z.shape[1] :]
        pred_C = self.out(self.out_norm(pred_tokens))
        return pred_C[:, 0] if squeezed else pred_C

    def _flatten_latent_sequence(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.ndim == 3:
            B, T, C = x.shape
            if C != self.latent_dim:
                raise ValueError(
                    f"latent dim mismatch: got {C}, expected {self.latent_dim}"
                )
            return x, (B, T)
        if x.ndim == 4:
            B, T, N, C = x.shape
            if C != self.latent_dim:
                raise ValueError(
                    f"latent dim mismatch: got {C}, expected {self.latent_dim}"
                )
            return x.reshape(B, T * N, C), (B, T, N)
        raise ValueError(
            f"latent_seq must be [B,T,C] or [B,T,N,C], got {tuple(x.shape)}"
        )

    def predict_C_sequence(
        self,
        context_latent: torch.Tensor,
        noisy_delta_seq: torch.Tensor,
        action_seq: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        context_tokens, _ = self._flatten_latent_sequence(context_latent)
        noisy_tokens, shape = self._flatten_latent_sequence(noisy_delta_seq)
        B = int(noisy_delta_seq.shape[0])
        K = int(noisy_delta_seq.shape[1])
        if action_seq.shape[:2] != (B, K):
            raise ValueError(
                f"action_seq must align with residual sequence [B,K], got {tuple(action_seq.shape[:2])}, expected {(B, K)}"
            )

        time_cond = self.time_embed(t.to(device=noisy_delta_seq.device))
        action_cond = self.encode_action_sequence(action_seq)
        if noisy_delta_seq.ndim == 3:
            delta_cond = action_cond + time_cond[:, None, :]
        else:
            N = int(noisy_delta_seq.shape[2])
            delta_cond = (
                action_cond[:, :, None, :]
                .expand(B, K, N, self.model_dim)
                .reshape(B, K * N, self.model_dim)
            )
            delta_cond = delta_cond + time_cond[:, None, :]

        context = self.latent_proj(context_tokens)
        context = context + self.latent_type.to(
            dtype=context.dtype, device=context.device
        )
        noisy = self.noisy_delta_proj(noisy_tokens)
        noisy = noisy + delta_cond.to(dtype=noisy.dtype, device=noisy.device)
        noisy = noisy + self.noisy_type.to(dtype=noisy.dtype, device=noisy.device)
        global_cond = time_cond[:, None, :] + self.cond_type.to(
            dtype=context.dtype, device=context.device
        )
        tokens = torch.cat([global_cond, context, noisy], dim=1)
        tokens = self.transformer(tokens)
        pred_tokens = tokens[:, 1 + context.shape[1] :]
        pred_tokens = self.out(self.out_norm(pred_tokens))
        if len(shape) == 2:
            return pred_tokens.reshape(B, K, self.latent_dim)
        return pred_tokens.reshape(B, K, shape[2], self.latent_dim)

    def _forward_sequence(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        latent_seq = batch["latent_seq"]
        action_seq = batch["action_seq"]
        T = int(latent_seq.shape[1])
        context_frames = int(batch.get("context_frames", self.context_frames))
        if not (1 <= context_frames < T):
            raise ValueError(f"context_frames={context_frames} must be in [1,{T - 1}]")

        target_future = latent_seq[:, context_frames:]
        prev_latents = latent_seq[:, context_frames - 1 : -1]
        delta_seq = target_future - prev_latents
        C = -delta_seq
        if action_seq.shape[1] != delta_seq.shape[1]:
            raise ValueError(
                f"action horizon {action_seq.shape[1]} must match residual horizon {delta_seq.shape[1]}"
            )

        B = int(latent_seq.shape[0])
        t = (
            torch.rand(B, device=latent_seq.device, dtype=torch.float32)
            * (1.0 - self.eps)
            + self.eps
        )
        t_view = t.reshape(B, *((1,) * (delta_seq.ndim - 1))).to(dtype=delta_seq.dtype)
        noise = torch.randn_like(delta_seq)
        noisy_delta = delta_seq + C * t_view + noise * t_view
        pred_C = self.predict_C_sequence(
            context_latent=latent_seq[:, :context_frames],
            noisy_delta_seq=noisy_delta,
            action_seq=action_seq,
            t=t,
        )
        pred_delta_seq = -pred_C
        start = latent_seq[:, context_frames - 1 : context_frames]
        pred_future = start + torch.cumsum(pred_delta_seq, dim=1)

        flow_loss = F.mse_loss(pred_C.float(), C.float())
        latent_mse_loss = F.mse_loss(pred_future.float(), target_future.float())
        pred_flat = pred_future.float().flatten(start_dim=1)
        target_flat = target_future.float().flatten(start_dim=1)
        cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
        cosine_loss = 1.0 - cosine
        variation_loss = pred_C.float().abs().mean()
        loss = (
            self.flow_loss_coef * flow_loss
            + self.latent_mse_coef * latent_mse_loss
            + self.cosine_coef * cosine_loss
            + self.variation_coef * variation_loss
        )

        with torch.no_grad():
            endpoint_cos = F.cosine_similarity(
                pred_future[:, -1].float().flatten(start_dim=1),
                target_future[:, -1].float().flatten(start_dim=1),
                dim=-1,
            ).mean()
            metrics = {
                "flow_loss": flow_loss.detach(),
                "latent_mse_loss": latent_mse_loss.detach(),
                "cosine_loss": cosine_loss.detach(),
                "variation_loss": variation_loss.detach(),
                "pred_target_cos": cosine.detach(),
                "endpoint_pred_target_cos": endpoint_cos.detach(),
                "latent_norm": latent_seq[:, :context_frames]
                .float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "target_norm": target_future.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "pred_norm": pred_future.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "target_delta_norm": delta_seq.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "pred_delta_norm": pred_delta_seq.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "action_norm": action_seq.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "t_mean": t.mean(),
            }
        return {
            "pred_latent_seq": pred_future,
            "target_latent_seq": target_future,
            "pred_latent": pred_future[:, -1],
            "target_latent": target_future[:, -1],
            "pred_C": pred_C,
            "target_C": C,
            "loss": loss,
            **metrics,
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "latent_seq" in batch:
            return self._forward_sequence(batch)
        latent = batch["latent"]
        target = batch["target_latent"]
        action_seq = batch["action_seq"]
        if latent.shape != target.shape:
            raise ValueError(
                f"latent and target shape mismatch: {tuple(latent.shape)} vs {tuple(target.shape)}"
            )
        delta = target - latent
        C = -delta
        B = int(latent.shape[0])
        t = (
            torch.rand(B, device=latent.device, dtype=torch.float32) * (1.0 - self.eps)
            + self.eps
        )
        t_view = t.reshape(B, *((1,) * (delta.ndim - 1))).to(dtype=delta.dtype)
        noise = torch.randn_like(delta)
        noisy_delta = delta + C * t_view + noise * t_view
        pred_C = self.predict_C(latent, noisy_delta, action_seq, t)
        pred_latent = latent - pred_C

        flow_loss = F.mse_loss(pred_C.float(), C.float())
        latent_mse_loss = F.mse_loss(pred_latent.float(), target.float())
        pred_flat = pred_latent.float().flatten(start_dim=1)
        target_flat = target.float().flatten(start_dim=1)
        cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
        cosine_loss = 1.0 - cosine
        variation_loss = pred_C.float().abs().mean()
        loss = (
            self.flow_loss_coef * flow_loss
            + self.latent_mse_coef * latent_mse_loss
            + self.cosine_coef * cosine_loss
            + self.variation_coef * variation_loss
        )

        with torch.no_grad():
            metrics = {
                "flow_loss": flow_loss.detach(),
                "latent_mse_loss": latent_mse_loss.detach(),
                "cosine_loss": cosine_loss.detach(),
                "variation_loss": variation_loss.detach(),
                "pred_target_cos": cosine.detach(),
                "latent_norm": latent.float().flatten(start_dim=1).norm(dim=-1).mean(),
                "target_norm": target.float().flatten(start_dim=1).norm(dim=-1).mean(),
                "pred_norm": pred_latent.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "target_delta_norm": delta.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "pred_delta_norm": (-pred_C)
                .float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "action_norm": action_seq.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "t_mean": t.mean(),
            }
        return {
            "pred_latent": pred_latent,
            "target_latent": target,
            "pred_C": pred_C,
            "target_C": C,
            "loss": loss,
            **metrics,
        }


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _modulate_tokens(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


class _CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(x)
        kv = self.kv_norm(context)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return x + out


class _LaDiWMDiTBlock(nn.Module):
    """Two-stream DiT block, alternating LaDiWM-style spatial/temporal mixing."""

    def __init__(
        self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.cross_attn1 = _CrossAttentionBlock(dim, heads, dropout=dropout)
        self.cross_attn2 = _CrossAttentionBlock(dim, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        hidden = int(dim * mlp_ratio)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )
        self.adaLN1 = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.adaLN2 = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        context1: torch.Tensor,
        context2: torch.Tensor,
        time_fea: torch.Tensor,
        action_fea: torch.Tensor,
        shape: tuple[int, int, int],
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, N = shape
        x1 = self.cross_attn1(x1, context1)
        x2 = self.cross_attn2(x2, context2)

        if mode == "spatial":
            x1_m = x1.reshape(B, K, N, -1).reshape(B * K, N, -1)
            x2_m = x2.reshape(B, K, N, -1).reshape(B * K, N, -1)
            cond = (action_fea + time_fea[:, None, :]).reshape(B * K, -1)
            shift1, scale1, gate_attn1, shift_mlp1, scale_mlp1, gate_mlp1 = self.adaLN1(
                cond
            ).chunk(6, dim=-1)
            shift2, scale2, gate_attn2, shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN2(
                cond
            ).chunk(6, dim=-1)
            q1 = _modulate(self.norm1(x1_m), shift1, scale1)
            q2 = _modulate(self.norm2(x2_m), shift2, scale2)
            mixed, _ = self.self_attn(
                torch.cat([q1, q2], dim=1),
                torch.cat([q1, q2], dim=1),
                torch.cat([q1, q2], dim=1),
                need_weights=False,
            )
            x1_m = x1_m + gate_attn1.unsqueeze(1) * mixed[:, :N]
            x2_m = x2_m + gate_attn2.unsqueeze(1) * mixed[:, N:]
            x1_m = x1_m + gate_mlp1.unsqueeze(1) * self.mlp1(
                _modulate(self.norm3(x1_m), shift_mlp1, scale_mlp1)
            )
            x2_m = x2_m + gate_mlp2.unsqueeze(1) * self.mlp2(
                _modulate(self.norm4(x2_m), shift_mlp2, scale_mlp2)
            )
            return x1_m.reshape(B, K * N, -1), x2_m.reshape(B, K * N, -1)

        if mode == "temporal":
            x1_m = x1.reshape(B, K, N, -1).permute(0, 2, 1, 3).reshape(B * N, K, -1)
            x2_m = x2.reshape(B, K, N, -1).permute(0, 2, 1, 3).reshape(B * N, K, -1)
            cond = (
                (action_fea + time_fea[:, None, :])[:, None, :, :]
                .expand(B, N, K, -1)
                .reshape(B * N, K, -1)
            )
            shift1, scale1, gate_attn1, shift_mlp1, scale_mlp1, gate_mlp1 = self.adaLN1(
                cond
            ).chunk(6, dim=-1)
            shift2, scale2, gate_attn2, shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN2(
                cond
            ).chunk(6, dim=-1)
            q1 = _modulate_tokens(self.norm1(x1_m), shift1, scale1)
            q2 = _modulate_tokens(self.norm2(x2_m), shift2, scale2)
            mixed, _ = self.self_attn(
                torch.cat([q1, q2], dim=1),
                torch.cat([q1, q2], dim=1),
                torch.cat([q1, q2], dim=1),
                need_weights=False,
            )
            x1_m = x1_m + gate_attn1 * mixed[:, :K]
            x2_m = x2_m + gate_attn2 * mixed[:, K:]
            x1_m = x1_m + gate_mlp1 * self.mlp1(
                _modulate_tokens(self.norm3(x1_m), shift_mlp1, scale_mlp1)
            )
            x2_m = x2_m + gate_mlp2 * self.mlp2(
                _modulate_tokens(self.norm4(x2_m), shift_mlp2, scale_mlp2)
            )
            x1_m = x1_m.reshape(B, N, K, -1).permute(0, 2, 1, 3).reshape(B, K * N, -1)
            x2_m = x2_m.reshape(B, N, K, -1).permute(0, 2, 1, 3).reshape(B, K * N, -1)
            return x1_m, x2_m

        raise ValueError("mode must be 'spatial' or 'temporal'")


class _LaDiWMFinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, out_dim)
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        time_fea: torch.Tensor,
        action_fea: torch.Tensor,
        shape: tuple[int, int, int],
    ) -> torch.Tensor:
        B, K, N = shape
        x = x.reshape(B, K, N, -1).reshape(B * K, N, -1)
        cond = (action_fea + time_fea[:, None, :]).reshape(B * K, -1)
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x.reshape(B, K, N, -1)


class ChameleonLaDiWMFlowWorldModel(nn.Module):
    """Closer LaDiWM analogue over frozen Chameleon token latents.

    This keeps DreamerVLA's pretokenized dataloader, but follows LaDiWM more
    closely than the endpoint predictor:
      - use latent sequence residuals, not just z_t -> z_{t+k};
      - split Chameleon hidden channels into two interacting streams;
      - encode history latents as cross-attention context;
      - inject per-step action embeddings through AdaLN in alternating
        spatial/temporal transformer blocks.
    """

    def __init__(
        self,
        latent_dim: int = 4096,
        action_dim: int = 7,
        model_dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_action_steps: int = 16,
        max_image_tokens: int = 256,
        time_embed_dim: int = 256,
        flow_loss_coef: float = 1.0,
        latent_mse_coef: float = 0.1,
        cosine_coef: float = 0.01,
        aux_loss_coef: float = 1.0,
        variation_coef: float = 3e-4,
        eps: float = 1e-3,
        context_frames: int = 1,
    ) -> None:
        super().__init__()
        if latent_dim < 2:
            raise ValueError("latent_dim must be at least 2")
        self.latent_dim = int(latent_dim)
        self.stream1_dim = self.latent_dim // 2
        self.stream2_dim = self.latent_dim - self.stream1_dim
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.context_frames = int(context_frames)
        self.flow_loss_coef = float(flow_loss_coef)
        self.latent_mse_coef = float(latent_mse_coef)
        self.cosine_coef = float(cosine_coef)
        self.aux_loss_coef = float(aux_loss_coef)
        self.variation_coef = float(variation_coef)
        self.eps = float(eps)

        self.in1 = nn.Linear(self.stream1_dim, self.model_dim)
        self.in2 = nn.Linear(self.stream2_dim, self.model_dim)
        self.context1 = nn.Linear(self.stream1_dim, self.model_dim)
        self.context2 = nn.Linear(self.stream2_dim, self.model_dim)
        self.time_encoder = _TimestepEmbedder(
            self.model_dim, frequency_embedding_size=int(time_embed_dim)
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, int(max_action_steps), 1, self.model_dim),
            requires_grad=False,
        )
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, 1, int(max_image_tokens), self.model_dim),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                _LaDiWMDiTBlock(
                    self.model_dim,
                    int(heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.final1 = _LaDiWMFinalLayer(self.model_dim, self.stream1_dim)
        self.final2 = _LaDiWMFinalLayer(self.model_dim, self.stream2_dim)
        self.aux1 = _LaDiWMFinalLayer(self.model_dim, self.stream1_dim)
        self.aux2 = _LaDiWMFinalLayer(self.model_dim, self.stream2_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.time_encoder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_encoder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.action_encoder[1].weight, std=0.02)
        nn.init.normal_(self.action_encoder[3].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN1[-1].weight, 0)
            nn.init.constant_(block.adaLN1[-1].bias, 0)
            nn.init.constant_(block.adaLN2[-1].weight, 0)
            nn.init.constant_(block.adaLN2[-1].bias, 0)
        for head in (self.final1, self.final2, self.aux1, self.aux2):
            nn.init.constant_(head.adaLN[-1].weight, 0)
            nn.init.constant_(head.adaLN[-1].bias, 0)
            nn.init.constant_(head.linear.weight, 0)
            nn.init.constant_(head.linear.bias, 0)

    def _ensure_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 3:
            return x[:, :, None, :], True
        if x.ndim == 4:
            return x, False
        raise ValueError(
            f"latent_seq must be [B,T,C] or [B,T,N,C], got {tuple(x.shape)}"
        )

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent dim mismatch: got {x.shape[-1]}, expected {self.latent_dim}"
            )
        return x[..., : self.stream1_dim], x[..., self.stream1_dim :]

    def _action_features(
        self, action_seq: torch.Tensor, K: int, dtype: torch.dtype
    ) -> torch.Tensor:
        if action_seq.ndim != 3 or action_seq.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_seq must be [B,K,{self.action_dim}], got {tuple(action_seq.shape)}"
            )
        if action_seq.shape[1] != K:
            raise ValueError(
                f"action horizon {action_seq.shape[1]} must match residual horizon {K}"
            )
        return self.action_encoder(action_seq).to(dtype=dtype)

    def _add_pos(self, x: torch.Tensor) -> torch.Tensor:
        B, K, N, D = x.shape
        if K > self.temporal_pos.shape[1] or N > self.spatial_pos.shape[2]:
            raise ValueError(
                f"sequence shape K={K}, N={N} exceeds max_action_steps={self.temporal_pos.shape[1]}, "
                f"max_image_tokens={self.spatial_pos.shape[2]}"
            )
        pos = self.temporal_pos[:, :K].to(dtype=x.dtype, device=x.device)
        pos = pos + self.spatial_pos[:, :, :N].to(dtype=x.dtype, device=x.device)
        return x + pos

    def predict_C_sequence(
        self,
        context_latent: torch.Tensor,
        noisy_delta_seq: torch.Tensor,
        action_seq: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_latent, _ = self._ensure_tokens(context_latent)
        noisy_delta_seq, _ = self._ensure_tokens(noisy_delta_seq)
        B, K, N, _ = noisy_delta_seq.shape
        context1_raw, context2_raw = self._split(context_latent)
        noisy1_raw, noisy2_raw = self._split(noisy_delta_seq)
        context1 = self.context1(context1_raw).reshape(B, -1, self.model_dim)
        context2 = self.context2(context2_raw).reshape(B, -1, self.model_dim)
        x1 = self._add_pos(self.in1(noisy1_raw)).reshape(B, K * N, self.model_dim)
        x2 = self._add_pos(self.in2(noisy2_raw)).reshape(B, K * N, self.model_dim)
        time_fea = self.time_encoder(
            t.clamp_min(self.eps).log().to(device=noisy_delta_seq.device)
        )
        action_fea = self._action_features(action_seq, K, dtype=x1.dtype)

        shape = (B, K, N)
        for idx, block in enumerate(self.blocks):
            mode = "spatial" if (idx % 2) == 0 else "temporal"
            x1, x2 = block(
                x1, x2, context1, context2, time_fea, action_fea, shape, mode=mode
            )

        y1 = self.final1(x1, time_fea, action_fea, shape)
        y2 = self.final2(x2, time_fea, action_fea, shape)
        aux = torch.cat(
            [
                self.aux1(x1, time_fea, action_fea, shape),
                self.aux2(x2, time_fea, action_fea, shape),
            ],
            dim=-1,
        )
        return torch.cat([y1, y2], dim=-1), aux

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "latent_seq" not in batch:
            raise ValueError(
                "ChameleonLaDiWMFlowWorldModel expects latent_seq from the reused sequence dataloader"
            )
        latent_seq, squeezed = self._ensure_tokens(batch["latent_seq"])
        action_seq = batch["action_seq"]
        T = int(latent_seq.shape[1])
        context_frames = int(batch.get("context_frames", self.context_frames))
        if not (1 <= context_frames < T):
            raise ValueError(f"context_frames={context_frames} must be in [1,{T - 1}]")

        target_future = latent_seq[:, context_frames:]
        prev_latents = latent_seq[:, context_frames - 1 : -1]
        delta_seq = target_future - prev_latents
        C = -delta_seq
        B = int(latent_seq.shape[0])
        t = (
            torch.rand(B, device=latent_seq.device, dtype=torch.float32)
            * (1.0 - self.eps)
            + self.eps
        )
        t_view = t.reshape(B, *((1,) * (delta_seq.ndim - 1))).to(dtype=delta_seq.dtype)
        noise = torch.randn_like(delta_seq)
        noisy_delta = delta_seq + C * t_view + noise * t_view
        pred_C, aux_C = self.predict_C_sequence(
            context_latent=latent_seq[:, :context_frames],
            noisy_delta_seq=noisy_delta,
            action_seq=action_seq,
            t=t,
        )
        pred_delta_seq = -pred_C
        pred_future = latent_seq[:, context_frames - 1 : context_frames] + torch.cumsum(
            pred_delta_seq, dim=1
        )

        flow_loss = F.mse_loss(pred_C.float(), C.float())
        aux_loss = F.mse_loss(aux_C.float(), C.float())
        latent_mse_loss = F.mse_loss(pred_future.float(), target_future.float())
        pred_flat = pred_future.float().flatten(start_dim=1)
        target_flat = target_future.float().flatten(start_dim=1)
        cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
        cosine_loss = 1.0 - cosine
        variation_loss = pred_C.float().abs().mean()
        loss = (
            self.flow_loss_coef * flow_loss
            + self.aux_loss_coef * aux_loss
            + self.latent_mse_coef * latent_mse_loss
            + self.cosine_coef * cosine_loss
            + self.variation_coef * variation_loss
        )

        if squeezed:
            pred_future_out = pred_future[:, :, 0]
            target_future_out = target_future[:, :, 0]
            pred_C_out = pred_C[:, :, 0]
            C_out = C[:, :, 0]
        else:
            pred_future_out = pred_future
            target_future_out = target_future
            pred_C_out = pred_C
            C_out = C

        with torch.no_grad():
            endpoint_cos = F.cosine_similarity(
                pred_future[:, -1].float().flatten(start_dim=1),
                target_future[:, -1].float().flatten(start_dim=1),
                dim=-1,
            ).mean()
            metrics = {
                "flow_loss": flow_loss.detach(),
                "aux_loss": aux_loss.detach(),
                "latent_mse_loss": latent_mse_loss.detach(),
                "cosine_loss": cosine_loss.detach(),
                "variation_loss": variation_loss.detach(),
                "pred_target_cos": cosine.detach(),
                "endpoint_pred_target_cos": endpoint_cos.detach(),
                "latent_norm": latent_seq[:, :context_frames]
                .float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "target_norm": target_future.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "pred_norm": pred_future.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "target_delta_norm": delta_seq.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "pred_delta_norm": pred_delta_seq.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "action_norm": action_seq.float()
                .flatten(start_dim=1)
                .norm(dim=-1)
                .mean(),
                "t_mean": t.mean(),
            }
        return {
            "pred_latent_seq": pred_future_out,
            "target_latent_seq": target_future_out,
            "pred_latent": pred_future_out[:, -1],
            "target_latent": target_future_out[:, -1],
            "pred_C": pred_C_out,
            "target_C": C_out,
            "loss": loss,
            **metrics,
        }

    @torch.no_grad()
    def action_sensitivity_metrics(
        self, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Compare real / zero / shuffled actions under the same flow noise.

        This is the diagnostic that answers whether action is being used:
        real_action_error should become lower than zero/shuffled-action errors.
        """
        if "latent_seq" not in batch:
            return {}
        was_training = self.training
        self.eval()
        try:
            latent_seq, _ = self._ensure_tokens(batch["latent_seq"])
            action_seq = batch["action_seq"]
            T = int(latent_seq.shape[1])
            context_frames = int(batch.get("context_frames", self.context_frames))
            if not (1 <= context_frames < T):
                return {}
            target_future = latent_seq[:, context_frames:]
            prev_latents = latent_seq[:, context_frames - 1 : -1]
            delta_seq = target_future - prev_latents
            C = -delta_seq
            B, K = int(latent_seq.shape[0]), int(delta_seq.shape[1])
            if action_seq.shape[1] != K:
                return {}

            # Keep t/noise fixed across action variants; otherwise the action
            # comparison is contaminated by flow-noise variance.
            t = torch.full((B,), 0.5, device=latent_seq.device, dtype=torch.float32)
            t_view = t.reshape(B, *((1,) * (delta_seq.ndim - 1))).to(
                dtype=delta_seq.dtype
            )
            noise = torch.randn_like(delta_seq)
            noisy_delta = delta_seq + C * t_view + noise * t_view

            zero_action = torch.zeros_like(action_seq)
            if B > 1:
                perm = torch.roll(torch.arange(B, device=action_seq.device), shifts=1)
                shuffled_action = action_seq.index_select(0, perm)
            elif K > 1:
                shuffled_action = torch.roll(action_seq, shifts=1, dims=1)
            else:
                shuffled_action = zero_action

            start = latent_seq[:, context_frames - 1 : context_frames]

            def predict_future(
                actions: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                pred_C, _aux_C = self.predict_C_sequence(
                    context_latent=latent_seq[:, :context_frames],
                    noisy_delta_seq=noisy_delta,
                    action_seq=actions,
                    t=t,
                )
                return start + torch.cumsum(-pred_C, dim=1), pred_C

            real_future, real_C = predict_future(action_seq)
            zero_future, zero_C = predict_future(zero_action)
            shuffle_future, shuffle_C = predict_future(shuffled_action)

            def l2_error(pred: torch.Tensor) -> torch.Tensor:
                return (
                    (pred.float() - target_future.float())
                    .flatten(start_dim=1)
                    .norm(dim=-1)
                    .mean()
                )

            real_error = l2_error(real_future)
            zero_error = l2_error(zero_future)
            shuffle_error = l2_error(shuffle_future)
            real_flow = F.mse_loss(real_C.float(), C.float())
            zero_flow = F.mse_loss(zero_C.float(), C.float())
            shuffle_flow = F.mse_loss(shuffle_C.float(), C.float())
            return {
                "action_real_error": real_error,
                "action_zero_error": zero_error,
                "action_shuffle_error": shuffle_error,
                "action_margin_zero": zero_error - real_error,
                "action_margin_shuffle": shuffle_error - real_error,
                "action_real_flow_mse": real_flow,
                "action_zero_flow_mse": zero_flow,
                "action_shuffle_flow_mse": shuffle_flow,
            }
        finally:
            if was_training:
                self.train()


__all__ = [
    "ChameleonLatentActionOutput",
    "ChameleonLatentActionWorldModel",
    "ChameleonLatentFlowWorldModel",
    "ChameleonLaDiWMFlowWorldModel",
]
