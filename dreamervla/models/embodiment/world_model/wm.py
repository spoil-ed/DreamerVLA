from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamervla.models.embodiment.world_model.base_world_model import BaseWorldModel


class WorldModel(BaseWorldModel):
    """World model over tokenized OpenVLA-OFT input embeddings.

    The external observation boundary is tokenized ``[B,T,256,4096]`` (or one
    frame ``[B,256,4096]``). Flat action-query/hidden-token observations are not
    accepted. Internally the model appends one action-conditioning token per
    environment timestep and predicts future observation tokens.
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        token_count: int | None = None,
        token_dim: int = 4096,
        time_horizon: int | None = 8,
        model_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        num_hist: int = 3,
        num_pred: int = 1,
        max_seq_len: int = 128,
        hidden_loss_scale: float = 1.0,
        cosine_loss_scale: float = 0.1,
        rollout_loss_scale: float = 0.0,
        rollout_horizon: int = 0,
        rollout_context: int | None = None,
        reward_head_type: str = "none",
        reward_loss_scale: float = 0.0,
        reward_hidden_dim: int = 1024,
        reward_init_logit: float = 0.0,
        reward_pos_weight: float | None = None,
        success_return_head_type: str = "none",
        success_return_loss_scale: float = 0.0,
        success_return_hidden_dim: int | None = None,
        success_return_init_logit: float = 0.0,
        success_return_loss_type: str = "bce",
        return_predictions: bool = False,
        freeze_backbone: bool = False,
        freeze_input_embeddings: bool = False,
        latent_stage: str | None = None,
        latent_source: str = "OpenVLA-OFT hidden_token [256,4096]",
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.token_dim = int(token_dim)
        self.time_horizon = int(time_horizon) if time_horizon is not None else 8
        self.token_count = int(token_count) if token_count is not None else 256
        self.obs_dim = int(obs_dim) if obs_dim is not None else self.token_count * self.token_dim
        if (self.token_count == 56 and self.token_dim == 1024) or self.obs_dim == 56 * 1024:
            raise ValueError(
                "the removed 56x1024 observation interface is closed; use hidden_token [256,4096]"
            )
        self.model_dim = int(model_dim)
        self.num_hist = int(num_hist)
        self.num_pred = int(num_pred)
        self.max_seq_len = int(max_seq_len)
        self.hidden_loss_scale = float(hidden_loss_scale)
        self.cosine_loss_scale = float(cosine_loss_scale)
        self.rollout_loss_scale = float(rollout_loss_scale)
        self.rollout_horizon = int(rollout_horizon)
        self.rollout_context = (
            int(rollout_context) if rollout_context is not None else self.num_hist
        )
        self.reward_head_type = str(reward_head_type).lower()
        self.reward_loss_scale = float(reward_loss_scale)
        self.reward_hidden_dim = int(reward_hidden_dim)
        self.reward_init_logit = float(reward_init_logit)
        self.reward_pos_weight_value = (
            None if reward_pos_weight is None else float(reward_pos_weight)
        )
        self.reward_enabled = self.reward_head_type not in {
            "",
            "none",
            "null",
            "false",
            "0",
        }
        self.success_return_head_type = str(success_return_head_type).lower()
        self.success_return_loss_scale = float(success_return_loss_scale)
        self.success_return_hidden_dim = (
            int(success_return_hidden_dim)
            if success_return_hidden_dim is not None
            else self.reward_hidden_dim
        )
        self.success_return_init_logit = float(success_return_init_logit)
        self.success_return_loss_type = str(success_return_loss_type).lower()
        self.success_return_enabled = self.success_return_head_type not in {
            "",
            "none",
            "null",
            "false",
            "0",
        }
        self.return_predictions = bool(return_predictions)
        self.freeze_backbone_requested = bool(freeze_backbone)
        self.freeze_input_embeddings_requested = bool(freeze_input_embeddings)
        self.latent_stage = None if latent_stage is None else str(latent_stage)
        self.latent_source = str(latent_source)
        self.slots_per_step = self.token_count + 1
        self.decoder = None
        self.emb_criterion = nn.MSELoss()
        if self.token_count * self.token_dim != self.obs_dim:
            raise ValueError(
                "token_count * token_dim must equal obs_dim: "
                f"{self.token_count} * {self.token_dim} != {self.obs_dim}"
            )
        if self.num_pred < 1:
            raise ValueError(f"num_pred must be positive, got {self.num_pred}")
        if self.rollout_horizon < 0:
            raise ValueError(f"rollout_horizon must be non-negative, got {self.rollout_horizon}")
        if self.rollout_context < 1:
            raise ValueError(f"rollout_context must be positive, got {self.rollout_context}")
        if self.reward_loss_scale < 0.0:
            raise ValueError(
                f"reward_loss_scale must be non-negative, got {self.reward_loss_scale}"
            )
        if self.reward_loss_scale > 0.0 and not self.reward_enabled:
            raise ValueError(
                "reward_loss_scale > 0 requires reward_head_type in {'binary', 'scalar'}"
            )
        if self.reward_enabled and self.reward_head_type not in {
            "binary",
            "bernoulli",
            "sigmoid",
            "scalar",
            "mse",
            "regression",
        }:
            raise ValueError(
                f"Unsupported reward_head_type: {reward_head_type!r}; "
                "use 'binary' (BCE classification) / 'scalar' (MSE regression) / 'none'"
            )
        self.reward_is_scalar = self.reward_enabled and self.reward_head_type in {
            "scalar",
            "mse",
            "regression",
        }
        if self.reward_hidden_dim < 1:
            raise ValueError(f"reward_hidden_dim must be positive, got {self.reward_hidden_dim}")
        if self.success_return_loss_scale < 0.0:
            raise ValueError(
                f"success_return_loss_scale must be non-negative, got {self.success_return_loss_scale}"
            )
        if self.success_return_loss_scale > 0.0 and not self.success_return_enabled:
            raise ValueError(
                "success_return_loss_scale > 0 requires success_return_head_type='binary'"
            )
        if self.success_return_enabled and self.success_return_head_type not in {
            "binary",
            "bernoulli",
            "sigmoid",
        }:
            raise ValueError(
                f"Unsupported success_return_head_type: {success_return_head_type!r}; use 'binary' or 'none'"
            )
        if self.success_return_hidden_dim < 1:
            raise ValueError(
                f"success_return_hidden_dim must be positive, got {self.success_return_hidden_dim}"
            )
        if self.success_return_loss_type not in {"bce", "mse"}:
            raise ValueError("success_return_loss_type must be one of {'bce', 'mse'}")

        self.obs_norm = nn.LayerNorm(self.token_dim)
        self.obs_proj = nn.Linear(self.token_dim, self.model_dim)
        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.model_dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_seq_len * self.slots_per_step, self.model_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(heads),
            dim_feedforward=int(mlp_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.out_norm = nn.LayerNorm(self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, self.token_dim)
        if self.reward_enabled:
            self.reward_norm = nn.LayerNorm(self.token_dim)
            self.reward_head = nn.Sequential(
                nn.Linear(self.token_dim, self.reward_hidden_dim),
                nn.GELU(),
                nn.Linear(self.reward_hidden_dim, 1),
            )
            final = self.reward_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.constant_(final.bias, self.reward_init_logit)
            if self.reward_pos_weight_value is not None:
                self.register_buffer(
                    "reward_pos_weight",
                    torch.tensor(float(self.reward_pos_weight_value), dtype=torch.float32),
                )
            else:
                self.reward_pos_weight = None
        if self.success_return_enabled:
            self.success_return_norm = nn.LayerNorm(self.token_dim)
            self.success_return_head = nn.Sequential(
                nn.Linear(self.token_dim, self.success_return_hidden_dim),
                nn.GELU(),
                nn.Linear(self.success_return_hidden_dim, 1),
            )
            final = self.success_return_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.constant_(final.bias, self.success_return_init_logit)
        if self.freeze_backbone_requested:
            self.freeze_backbone()
        if self.freeze_input_embeddings_requested:
            self.freeze_input_embeddings()

    def freeze_input_embeddings(self) -> int:
        """Freeze observation/action projection layers (the input "encoder" of this WM).

        Used for reward-head fine-tuning runs where the obs token embedding and
        action embedding should stay fixed at their pretrained values while the
        dynamics predictor and reward head adapt to a new reward target.
        """
        frozen = 0
        for name in ("obs_norm", "obs_proj", "action_proj"):
            module = getattr(self, name, None)
            if not isinstance(module, nn.Module):
                continue
            module.eval()
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameter.requires_grad = False
                    frozen += parameter.numel()
        if isinstance(self.pos_embedding, nn.Parameter) and self.pos_embedding.requires_grad:
            self.pos_embedding.requires_grad = False
            frozen += self.pos_embedding.numel()
        return frozen

    def freeze_backbone(self) -> int:
        """Freeze only an attached feature backbone/encoder, if this variant has one.

        The precomputed-hidden VLA route does not instantiate the upstream
        feature encoder, so this is intentionally a no-op there.  We keep
        it explicit to avoid accidentally freezing the dynamics predictor.
        """
        frozen = 0
        seen: set[int] = set()
        for name in ("backbone", "encoder", "base_model"):
            module = getattr(self, name, None)
            if not isinstance(module, nn.Module) or module is self or id(module) in seen:
                continue
            seen.add(id(module))
            module.eval()
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameter.requires_grad = False
                    frozen += parameter.numel()
        return frozen

    def _module_dtype(self) -> torch.dtype:
        return self.obs_proj.weight.dtype

    def _module_device(self) -> torch.device:
        return self.obs_proj.weight.device

    def _validate_obs_embedding(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        """Return the tokenized ``[B,T,N,D]`` view used by the WM path."""
        return self.obs_to_tokens(obs_embedding)

    def _validate_actions(self, actions: torch.Tensor, steps: int) -> torch.Tensor:
        if actions.ndim == 2:
            actions = actions[:, None]
        if actions.ndim != 3:
            raise ValueError(
                f"actions must be [B,T,{self.action_dim}] or [B,{self.action_dim}], got {tuple(actions.shape)}"
            )
        if actions.shape[1] < steps:
            raise ValueError(
                f"actions length {actions.shape[1]} is shorter than obs length {steps}"
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {actions.shape[-1]}, expected {self.action_dim}"
            )
        return actions[:, :steps].to(device=self._module_device(), dtype=self._module_dtype())

    def obs_to_tokens(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        """Validate tokenized observations and normalize to ``[B,T,N,D]``."""
        if obs_embedding.ndim == 3:
            if obs_embedding.shape[1:] == (self.token_count, self.token_dim):
                tokens = obs_embedding[:, None]
            else:
                raise ValueError(
                    "obs_embedding must be tokenized [B,N,token_dim] or "
                    "[B,T,N,token_dim]; flat observations are closed; "
                    f"got {tuple(obs_embedding.shape)}"
                )
        elif obs_embedding.ndim == 4:
            if obs_embedding.shape[-2:] != (self.token_count, self.token_dim):
                raise ValueError(
                    f"tokenized obs shape mismatch: got {tuple(obs_embedding.shape)}, "
                    f"expected trailing dims ({self.token_count}, {self.token_dim})"
                )
            tokens = obs_embedding
        else:
            raise ValueError(
                "obs_embedding must be tokenized [B,N,token_dim] or "
                "[B,T,N,token_dim]; flat observations are closed; "
                f"got {tuple(obs_embedding.shape)}"
            )
        if tokens.shape[1] > self.max_seq_len:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds max_seq_len={self.max_seq_len}"
            )
        return tokens.to(device=self._module_device(), dtype=self._module_dtype())

    def _obs_embedding_from_obs(self, obs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        if "obs_embedding" in obs:
            return obs["obs_embedding"]
        raise KeyError("WorldModel expects obs to contain `obs_embedding`.")

    def _block_causal_mask(
        self,
        steps: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        frame_ids = torch.arange(int(steps), device=device).repeat_interleave(self.slots_per_step)
        future = frame_ids[None, :] > frame_ids[:, None]
        mask = torch.zeros(
            int(steps) * self.slots_per_step,
            int(steps) * self.slots_per_step,
            device=device,
            dtype=dtype or torch.float32,
        )
        return mask.masked_fill(future, float("-inf"))

    def encode_obs(self, obs: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        visual = self.obs_to_tokens(self._obs_embedding_from_obs(obs))
        proprio = visual.new_zeros(visual.shape[0], visual.shape[1], 0)
        return {"visual": visual, "proprio": proprio}

    def encode_act(self, act: torch.Tensor) -> torch.Tensor:
        if act.ndim == 2:
            act = act[:, None]
        return self.action_proj(self._validate_actions(act, int(act.shape[1])))

    def encode(
        self, obs: dict[str, torch.Tensor] | torch.Tensor, act: torch.Tensor
    ) -> torch.Tensor:
        obs_tokens = self.obs_to_tokens(self._obs_embedding_from_obs(obs))
        steps = obs_tokens.shape[1]
        act_emb = self.encode_act(act)
        obs_emb = self.obs_proj(self.obs_norm(obs_tokens))
        z = torch.cat([obs_emb, act_emb.unsqueeze(2)], dim=2)
        z = z.reshape(z.shape[0], steps * self.slots_per_step, self.model_dim)
        z = z + self.pos_embedding[:, : z.shape[1]]
        return z.reshape(z.shape[0], steps, self.slots_per_step, self.model_dim)

    def separate_emb(self, z: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        visual_model = z[:, :, : self.token_count]
        act_emb = z[:, :, -1]
        visual = self.out_proj(self.out_norm(visual_model))
        proprio = visual.new_zeros(visual.shape[0], visual.shape[1], 0)
        return {"visual": visual, "proprio": proprio}, act_emb

    def replace_actions_from_z(self, z: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        z = z.clone()
        z[:, :, -1] = self.encode_act(act)
        return z

    def decode_obs(
        self, z_obs: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        zero = z_obs["visual"].new_zeros(())
        return z_obs, zero

    def decode(self, z: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        z_obs, _ = self.separate_emb(z)
        return self.decode_obs(z_obs)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"z must be [B,T,P,D], got {tuple(z.shape)}")
        bsz, steps, slots, dim = z.shape
        if slots != self.slots_per_step or dim != self.model_dim:
            raise ValueError(
                f"z shape mismatch: got slots={slots}, dim={dim}; "
                f"expected slots={self.slots_per_step}, dim={self.model_dim}"
            )
        flat = z.reshape(bsz, steps * slots, dim)
        pred = self.predictor(
            flat, mask=self._block_causal_mask(steps, flat.device, dtype=flat.dtype)
        )
        return self.out_norm(pred).reshape(bsz, steps, slots, dim)

    def predict_next_tokens(
        self, obs_embedding: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        z = self.encode(obs_embedding, actions)
        z_pred = self.predict(z)
        z_obs_pred, _ = self.separate_emb(z_pred)
        return z_obs_pred["visual"]

    def _actions_or_zeros(
        self, actions: torch.Tensor | None, batch: int, steps: int
    ) -> torch.Tensor:
        if actions is None:
            return torch.zeros(
                batch,
                steps,
                self.action_dim,
                device=self._module_device(),
                dtype=self._module_dtype(),
            )
        return self._validate_actions(actions, steps)

    def observe_sequence(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor]]:
        obs_tokens = self.obs_to_tokens(self._obs_embedding_from_obs(batch))
        bsz, steps = obs_tokens.shape[:2]
        actions = self._actions_or_zeros(batch.get("actions"), bsz, steps)

        histories: list[torch.Tensor] = []
        action_histories: list[torch.Tensor] = []
        for step in range(steps):
            indices = torch.arange(
                step - self.num_hist + 1,
                step + 1,
                device=obs_tokens.device,
            ).clamp_min(0)
            histories.append(obs_tokens.index_select(1, indices))
            action_histories.append(actions.index_select(1, indices))

        latent = {
            "hidden": obs_tokens,
            "history": torch.stack(histories, dim=1),
            "actions": torch.stack(action_histories, dim=1),
        }
        return {"latent": latent}

    def encode_latent(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        obs_tokens = self.obs_to_tokens(hidden)
        bsz = int(obs_tokens.shape[0])
        if obs_tokens.shape[1] >= self.num_hist:
            history = obs_tokens[:, -self.num_hist :]
        else:
            pad = obs_tokens[:, :1].expand(-1, self.num_hist - obs_tokens.shape[1], -1, -1)
            history = torch.cat([pad, obs_tokens], dim=1)
        actions = torch.zeros(
            bsz,
            self.num_hist,
            self.action_dim,
            device=history.device,
            dtype=history.dtype,
        )
        return {
            "hidden": history[:, -1],
            "history": history,
            "actions": actions,
        }

    def observe_next(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        is_first: bool | torch.Tensor = False,
    ) -> dict[str, torch.Tensor]:
        if isinstance(is_first, torch.Tensor) and bool(is_first.detach().flatten()[0].item()):
            return self.encode_latent(hidden)
        if isinstance(is_first, bool) and is_first:
            return self.encode_latent(hidden)

        history = self._latent_history(latent)
        bsz = int(history.shape[0])
        next_hidden = self.obs_to_tokens(hidden)[:, -1]
        action_history = self._latent_actions(latent, bsz).clone()
        action = actions
        if action.ndim == 3:
            action = action[:, 0]
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"observe_next action must be [B,{self.action_dim}], got {tuple(actions.shape)}"
            )
        action_history[:, -1] = action.to(device=history.device, dtype=history.dtype)

        next_history = torch.cat([history[:, 1:], next_hidden[:, None]], dim=1)
        next_action_history = torch.cat(
            [
                action_history[:, 1:],
                action_history.new_zeros(bsz, 1, self.action_dim),
            ],
            dim=1,
        )
        return {
            "hidden": next_hidden,
            "history": next_history,
            "actions": next_action_history,
        }

    def _latent_hidden(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(latent, torch.Tensor):
            return latent
        hidden = latent.get("hidden")
        if isinstance(hidden, torch.Tensor):
            return hidden
        history = latent.get("history")
        if isinstance(history, torch.Tensor):
            if history.ndim == 3:
                return history[:, -1]
            if history.ndim == 4:
                if history.shape[-1] == self.obs_dim:
                    return history[:, :, -1]
                return history[:, -1]
            if history.ndim == 5:
                return history[:, :, -1]
        raise KeyError("VLA latent must contain `hidden` or `history`.")

    def _latent_history(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(latent, dict) and isinstance(latent.get("history"), torch.Tensor):
            history = latent["history"]
            if history.ndim == 4 and history.shape[-1] == self.obs_dim:
                history = history[:, -1]
            elif history.ndim == 5:
                history = history[:, -1]
            if history.ndim not in {3, 4}:
                raise ValueError(
                    "latent history must be [B,H,obs_dim], [B,H,N,token_dim], "
                    f"[B,T,H,obs_dim], or [B,T,H,N,token_dim]; got {tuple(history.shape)}"
                )
            return self.obs_to_tokens(history)
        hidden = self._latent_hidden(latent)
        tokens = self.obs_to_tokens(hidden)
        if tokens.shape[1] >= self.num_hist:
            return tokens[:, -self.num_hist :]
        pad = tokens[:, :1].expand(-1, self.num_hist - tokens.shape[1], -1, -1)
        return torch.cat([pad, tokens], dim=1)

    def _latent_actions(
        self, latent: dict[str, torch.Tensor] | torch.Tensor, batch: int
    ) -> torch.Tensor:
        if isinstance(latent, dict) and isinstance(latent.get("actions"), torch.Tensor):
            actions = latent["actions"]
            if actions.ndim == 4:
                actions = actions[:, -1]
            if actions.ndim != 3:
                raise ValueError(
                    f"flat latent action history must be [B,H,A], got {tuple(actions.shape)}"
                )
            if actions.shape[1] == self.num_hist:
                return actions.to(device=self._module_device(), dtype=self._module_dtype())
            if actions.shape[1] > self.num_hist:
                return actions[:, -self.num_hist :].to(
                    device=self._module_device(), dtype=self._module_dtype()
                )
            pad = actions[:, :1].expand(-1, self.num_hist - actions.shape[1], -1)
            return torch.cat([pad, actions], dim=1).to(
                device=self._module_device(), dtype=self._module_dtype()
            )
        return torch.zeros(
            batch,
            self.num_hist,
            self.action_dim,
            device=self._module_device(),
            dtype=self._module_dtype(),
        )

    def predict_next(
        self, latent: dict[str, torch.Tensor] | torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        history = self._latent_history(latent)
        bsz = int(history.shape[0])
        action_history = self._latent_actions(latent, bsz).clone()
        action = actions
        if action.ndim == 3:
            action = action[:, 0]
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Dreamer action must be [B,{self.action_dim}], got {tuple(actions.shape)}"
            )
        action_history[:, -1] = action.to(device=history.device, dtype=history.dtype)

        z = self.encode(history, action_history)
        pred_z = self.predict(z)
        pred_obs, _ = self.separate_emb(pred_z[:, -1:])
        next_hidden = pred_obs["visual"][:, -1]

        if self.num_hist > 1:
            next_history = torch.cat([history[:, 1:], next_hidden[:, None]], dim=1)
            next_action_history = torch.cat(
                [
                    action_history[:, 1:],
                    action_history.new_zeros(bsz, 1, self.action_dim),
                ],
                dim=1,
            )
        else:
            next_history = next_hidden[:, None]
            next_action_history = action_history.new_zeros(bsz, 1, self.action_dim)
        return {
            "hidden": next_hidden,
            "history": next_history,
            "actions": next_action_history,
        }

    def actor_input(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return self._latent_hidden(latent)

    def critic_input(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        hidden = self._latent_hidden(latent)
        tokens = self.obs_to_tokens(hidden)
        pooled = tokens.mean(dim=2)
        if hidden.ndim == 2 or (
            hidden.ndim == 3 and hidden.shape[1:] == (self.token_count, self.token_dim)
        ):
            return pooled[:, 0]
        return pooled

    def reward_from_latent(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        hidden = self._latent_hidden(latent)
        reward = self.predict_reward(hidden)
        if self.obs_to_tokens(hidden).shape[1] == 1:
            return reward[:, 0]
        return reward

    def success_return_from_latent(
        self, latent: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        hidden = self._latent_hidden(latent)
        success_return = self.predict_success_return(hidden)
        if self.obs_to_tokens(hidden).shape[1] == 1:
            return success_return[:, 0]
        return success_return

    def continue_from_latent(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return torch.ones_like(self.reward_from_latent(latent))

    def _hidden_loss_terms(
        self, hidden_pred: torch.Tensor, hidden_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_mse = F.mse_loss(hidden_pred.float(), hidden_target.float())
        pred_norm = F.normalize(hidden_pred.float(), dim=-1)
        target_norm = F.normalize(hidden_target.float(), dim=-1)
        hidden_cosine = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()
        hidden_loss = self.hidden_loss_scale * hidden_mse + self.cosine_loss_scale * hidden_cosine
        return hidden_loss, hidden_mse, hidden_cosine

    def _require_reward_head(self) -> None:
        if not self.reward_enabled:
            raise RuntimeError(
                "reward head is disabled; set reward_head_type='binary' or 'scalar' to enable it"
            )

    def _require_success_return_head(self) -> None:
        if not self.success_return_enabled:
            raise RuntimeError(
                "success-return head is disabled; set success_return_head_type='binary' to enable it"
            )

    def reward_logits(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        self._require_reward_head()
        tokens = self.obs_to_tokens(obs_embedding)
        pooled = self.reward_norm(tokens).mean(dim=2)
        return self.reward_head(pooled).squeeze(-1)

    def predict_reward(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        logits = self.reward_logits(obs_embedding).float()
        if self.reward_is_scalar:
            # Scalar regression head: output is the predicted value directly.
            return logits
        return torch.sigmoid(logits)

    def success_return_logits(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        self._require_success_return_head()
        tokens = self.obs_to_tokens(obs_embedding)
        pooled = self.success_return_norm(tokens).mean(dim=2)
        return self.success_return_head(pooled).squeeze(-1)

    def predict_success_return(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.success_return_logits(obs_embedding).float())

    def _align_reward_target(self, rewards: torch.Tensor, target_steps: int) -> torch.Tensor:
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if rewards.ndim != 2:
            raise ValueError(f"rewards must be [B,T] or [B,T,1], got {tuple(rewards.shape)}")
        if rewards.shape[1] == target_steps + self.num_pred:
            rewards = rewards[:, self.num_pred :]
        elif rewards.shape[1] != target_steps:
            raise ValueError(
                f"reward length mismatch: got {rewards.shape[1]}, expected {target_steps} "
                f"or {target_steps + self.num_pred}"
            )
        return rewards.to(device=self._module_device())

    def _reward_loss_terms(
        self,
        hidden_target: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = self.reward_logits(hidden_target.detach())
        target = self._align_reward_target(rewards, int(logits.shape[1])).to(dtype=logits.dtype)
        if self.reward_is_scalar:
            # MSE regression to a continuous per-step reward target (e.g. progress delta).
            pred = logits.float()
            target_f = target.float()
            reward_loss = F.mse_loss(pred, target_f)
            abs_err = (pred - target_f).abs().mean().detach()
            out = {
                "reward_loss": reward_loss,
                "reward_pred_mean": pred.mean().detach(),
                "reward_target_mean": target_f.mean().detach(),
                "reward_mae": abs_err,
            }
            if self.return_predictions:
                out.update(
                    {
                        "reward_logits": logits,
                        "reward_pred": pred,
                        "reward_target": target,
                    }
                )
            return out
        pos_weight = getattr(self, "reward_pos_weight", None)
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device=logits.device, dtype=torch.float32)
        reward_loss = F.binary_cross_entropy_with_logits(
            logits.float(),
            target.float(),
            pos_weight=pos_weight,
        )
        pred = torch.sigmoid(logits.float())
        acc = ((pred >= 0.5) == (target.float() >= 0.5)).float().mean()
        out = {
            "reward_loss": reward_loss,
            "reward_pred_mean": pred.mean().detach(),
            "reward_target_mean": target.float().mean().detach(),
            "reward_binary_acc": acc.detach(),
        }
        if self.return_predictions:
            out.update(
                {
                    "reward_logits": logits,
                    "reward_pred": pred,
                    "reward_target": target,
                }
            )
        return out

    def _success_return_loss_terms(
        self,
        hidden_target: torch.Tensor,
        success_to_go: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = self.success_return_logits(hidden_target.detach())
        target = self._align_reward_target(success_to_go, int(logits.shape[1])).to(
            dtype=logits.dtype
        )
        target = target.float().clamp(0.0, 1.0)
        pred = torch.sigmoid(logits.float())
        if self.success_return_loss_type == "mse":
            success_return_loss = F.mse_loss(pred, target)
        else:
            success_return_loss = F.binary_cross_entropy_with_logits(logits.float(), target)
        out = {
            "success_return_loss": success_return_loss,
            "success_return_pred_mean": pred.mean().detach(),
            "success_return_target_mean": target.mean().detach(),
            "success_return_mse": F.mse_loss(pred.detach(), target.detach()).detach(),
        }
        if self.return_predictions:
            out.update(
                {
                    "success_return_logits": logits,
                    "success_return_pred": pred,
                    "success_return_target": target,
                }
            )
        return out

    def _open_loop_rollout_loss(
        self, obs_embedding: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        obs_tokens = self.obs_to_tokens(obs_embedding)
        steps = int(obs_tokens.shape[1])
        context_len = min(self.rollout_context, steps - 1)
        horizon = min(self.rollout_horizon, steps - context_len)
        zero = obs_tokens.new_zeros(())
        if horizon < 1:
            return {
                "rollout_loss": zero,
                "rollout_mse": zero,
                "rollout_cosine_loss": zero,
                "rollout_horizon": zero,
            }

        seed = obs_tokens[:, :context_len]
        rollout_actions = actions[:, : context_len + horizon]
        rollout, _ = self._rollout_hidden(seed, rollout_actions)
        hidden_pred = rollout[:, context_len : context_len + horizon]
        hidden_target = obs_tokens[:, context_len : context_len + horizon].detach()
        rollout_loss, rollout_mse, rollout_cosine = self._hidden_loss_terms(
            hidden_pred, hidden_target
        )
        out = {
            "rollout_loss": rollout_loss,
            "rollout_mse": rollout_mse.detach(),
            "rollout_cosine_loss": rollout_cosine.detach(),
            "rollout_horizon": rollout_mse.detach().new_tensor(float(horizon)),
        }
        if self.return_predictions:
            out.update(
                {
                    "rollout_hidden_pred": hidden_pred,
                    "rollout_hidden_target": hidden_target,
                }
            )
        return out

    def _loss(
        self,
        obs_embedding: torch.Tensor,
        actions: torch.Tensor,
        current_actions: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
        success_to_go: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        obs_tokens = self.obs_to_tokens(obs_embedding)
        if obs_tokens.shape[1] <= self.num_pred:
            raise ValueError(
                f"sequence length {obs_tokens.shape[1]} must be greater than num_pred={self.num_pred}"
            )
        transition_actions = current_actions if current_actions is not None else actions
        pred_tokens_all = self.predict_next_tokens(obs_embedding, transition_actions)
        pred_tokens = pred_tokens_all[:, : -self.num_pred]
        target_tokens = obs_tokens[:, self.num_pred :].detach()
        hidden_pred = pred_tokens
        hidden_target = target_tokens

        loss, hidden_mse, hidden_cosine = self._hidden_loss_terms(hidden_pred, hidden_target)
        rollout_out: dict[str, torch.Tensor] = {}
        if self.rollout_loss_scale > 0.0 and self.rollout_horizon > 0:
            rollout_out = self._open_loop_rollout_loss(obs_embedding, transition_actions)
            loss = loss + self.rollout_loss_scale * rollout_out["rollout_loss"]
        reward_out: dict[str, torch.Tensor] = {}
        if self.reward_loss_scale > 0.0:
            if rewards is None:
                raise KeyError("reward_loss_scale > 0 requires batch to contain `rewards`")
            reward_out = self._reward_loss_terms(hidden_target, rewards)
            loss = loss + self.reward_loss_scale * reward_out["reward_loss"]
        success_return_out: dict[str, torch.Tensor] = {}
        if self.success_return_loss_scale > 0.0:
            if success_to_go is None:
                raise KeyError(
                    "success_return_loss_scale > 0 requires batch to contain `success_to_go`"
                )
            success_return_out = self._success_return_loss_terms(hidden_target, success_to_go)
            loss = loss + self.success_return_loss_scale * success_return_out["success_return_loss"]
        zero = loss.new_zeros(())
        out = {
            "_loss": loss,
            "loss": loss.detach(),
            "next_latent_loss": hidden_mse.detach(),
            "next_latent_mse": hidden_mse.detach(),
            "next_latent_cosine_loss": hidden_cosine.detach(),
            "hidden_loss": hidden_mse.detach(),
            "hidden_mse": hidden_mse.detach(),
            "hidden_cosine_loss": hidden_cosine.detach(),
            "hidden_pred_norm": hidden_pred.detach().float().norm(dim=-1).mean(),
            "hidden_target_norm": hidden_target.detach().float().norm(dim=-1).mean(),
            "rec_loss": zero.detach(),
            "dyn_loss": zero.detach(),
            "rep_loss": zero.detach(),
            "image_mse": zero.detach(),
            "image_psnr": zero.detach(),
        }
        if rollout_out:
            out.update(
                {
                    "rollout_loss": rollout_out["rollout_loss"].detach(),
                    "rollout_mse": rollout_out["rollout_mse"],
                    "rollout_cosine_loss": rollout_out["rollout_cosine_loss"],
                    "rollout_horizon": rollout_out["rollout_horizon"],
                }
            )
        if reward_out:
            out.update(
                {
                    "reward_loss": reward_out["reward_loss"].detach(),
                    "reward_pred_mean": reward_out["reward_pred_mean"],
                    "reward_target_mean": reward_out["reward_target_mean"],
                    "reward_binary_acc": reward_out["reward_binary_acc"],
                }
            )
        if success_return_out:
            out.update(
                {
                    "success_return_loss": success_return_out["success_return_loss"].detach(),
                    "success_return_pred_mean": success_return_out["success_return_pred_mean"],
                    "success_return_target_mean": success_return_out["success_return_target_mean"],
                    "success_return_mse": success_return_out["success_return_mse"],
                }
            )
        if self.return_predictions:
            out.update(
                {
                    "hidden_pred": hidden_pred,
                    "hidden_target": hidden_target,
                    "hidden_pred_tokens": pred_tokens,
                }
            )
            if rollout_out:
                out.update(
                    {
                        "rollout_hidden_pred": rollout_out["rollout_hidden_pred"],
                        "rollout_hidden_target": rollout_out["rollout_hidden_target"],
                    }
                )
            if reward_out:
                out.update(
                    {
                        "reward_logits": reward_out["reward_logits"],
                        "reward_pred": reward_out["reward_pred"],
                        "reward_target": reward_out["reward_target"],
                    }
                )
            if success_return_out:
                out.update(
                    {
                        "success_return_logits": success_return_out["success_return_logits"],
                        "success_return_pred": success_return_out["success_return_pred"],
                        "success_return_target": success_return_out["success_return_target"],
                    }
                )
        return out

    def _rollout_hidden(
        self, obs_embedding: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tokens = self.obs_to_tokens(obs_embedding)
        actions = self._validate_actions(actions, int(actions.shape[1]))
        if obs_tokens.shape[0] != actions.shape[0]:
            raise ValueError(
                f"batch mismatch: obs_embedding batch={obs_tokens.shape[0]}, actions batch={actions.shape[0]}"
            )
        rollout = obs_tokens
        z_rollout = self.encode(rollout, actions[:, : rollout.shape[1]])
        while rollout.shape[1] < actions.shape[1]:
            context_len = min(self.num_hist, int(rollout.shape[1]))
            context_start = int(rollout.shape[1]) - context_len
            cur_z = self.encode(
                rollout[:, context_start:],
                actions[:, context_start : int(rollout.shape[1])],
            )
            pred_z = self.predict(cur_z)
            pred_obs, _ = self.separate_emb(pred_z[:, -1:])
            next_hidden = pred_obs["visual"]
            rollout = torch.cat([rollout, next_hidden], dim=1)
            z_new = self.encode(
                rollout[:, -1:], actions[:, rollout.shape[1] - 1 : rollout.shape[1]]
            )
            z_rollout = torch.cat([z_rollout, z_new], dim=1)
        return rollout, z_rollout

    @torch.no_grad()
    def rollout(
        self,
        obs_embedding: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        *,
        obs_0: dict[str, torch.Tensor] | torch.Tensor | None = None,
        act: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
        if obs_0 is not None or act is not None:
            if obs_0 is None or act is None:
                raise ValueError("WM-style rollout requires both obs_0 and act")
            rollout, z_rollout = self._rollout_hidden(self._obs_embedding_from_obs(obs_0), act)
            return self.encode_obs(rollout), z_rollout
        if obs_embedding is None or actions is None:
            raise ValueError(
                "rollout requires either (obs_embedding, actions) or (obs_0=..., act=...)"
            )
        rollout, _ = self._rollout_hidden(obs_embedding, actions)
        out = {
            "obs_embedding": rollout,
            "obs_tokens": rollout,
        }
        if self.reward_enabled:
            out["reward_pred"] = self.predict_reward(rollout)
        if self.success_return_enabled:
            out["success_return_pred"] = self.predict_success_return(rollout)
        return out

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        success_to_go = (
            batch.get("success_to_go")
            if batch.get("success_to_go") is not None
            else batch.get("return_to_go", batch.get("return_targets"))
        )
        return self._loss(
            batch["obs_embedding"],
            batch["actions"],
            batch.get("current_actions"),
            batch.get("rewards"),
            success_to_go,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict) and batch.get("mode") == "observe_sequence":
            return self.observe_sequence(batch)
        if isinstance(batch, dict) and batch.get("mode") == "encode_latent":
            hidden = batch.get("hidden", batch.get("obs_embedding"))
            if hidden is None:
                raise KeyError("encode_latent mode requires `hidden` or `obs_embedding`")
            return self.encode_latent(hidden)
        if isinstance(batch, dict) and batch.get("mode") == "observe_next":
            hidden = batch.get("hidden", batch.get("obs_embedding"))
            if hidden is None:
                raise KeyError("observe_next mode requires `hidden` or `obs_embedding`")
            return self.observe_next(
                batch["latent"],
                hidden,
                batch["actions"],
                batch.get("is_first", False),
            )
        if isinstance(batch, dict) and batch.get("mode") == "predict_next":
            return self.predict_next(batch["latent"], batch["actions"])
        if isinstance(batch, dict) and batch.get("mode") == "actor_input":
            return self.actor_input(batch["latent"])
        if isinstance(batch, dict) and batch.get("mode") == "critic_input":
            return self.critic_input(batch["latent"])
        if isinstance(batch, dict) and batch.get("mode") == "continue":
            return self.continue_from_latent(batch["latent"])
        if isinstance(batch, dict) and batch.get("mode") == "rollout":
            return self.rollout(batch["obs_embedding"], batch["actions"])
        if isinstance(batch, dict) and batch.get("mode") == "reward":
            if "latent" in batch:
                return self.reward_from_latent(batch["latent"])
            obs_embedding = batch.get("obs_embedding", batch.get("hidden"))
            if obs_embedding is None:
                raise KeyError("reward mode requires `obs_embedding` or `hidden`")
            return self.predict_reward(obs_embedding)
        if isinstance(batch, dict) and batch.get("mode") in {
            "success_return",
            "return",
        }:
            if "latent" in batch:
                return self.success_return_from_latent(batch["latent"])
            obs_embedding = batch.get("obs_embedding", batch.get("hidden"))
            if obs_embedding is None:
                raise KeyError("success_return mode requires `obs_embedding` or `hidden`")
            return self.predict_success_return(obs_embedding)
        return self.loss(batch)


__all__ = ["WorldModel"]
