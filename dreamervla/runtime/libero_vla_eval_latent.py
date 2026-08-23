"""OpenVLA hidden-token and Dreamer latent evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image


class EmbodiedEvalLatentMixin:
    def _pixel_obs_for_wm(
        self, frame_history: list[tuple[Image.Image, Image.Image]]
    ) -> torch.Tensor:
        wm = getattr(self, "_unwrapped_world_model", None) or self.world_model
        wm_encoder = getattr(wm, "encoder", None)
        image_size = int(
            getattr(
                wm_encoder,
                "image_size",
                OmegaConf.select(self.cfg, "world_model.image_size", default=64),
            )
        )

        raw_obs = getattr(self, "_libero_current_raw_obs", None)
        if (
            isinstance(raw_obs, dict)
            and "agentview_image" in raw_obs
            and "robot0_eye_in_hand_image" in raw_obs
        ):
            third = np.asarray(raw_obs["agentview_image"], dtype=np.uint8)
            wrist = np.asarray(raw_obs["robot0_eye_in_hand_image"], dtype=np.uint8)
        else:
            # Base LIBERO VLA eval stores 180-degree-rotated PILs. Rotate them
            # back here so pixel DreamerV3 sees the same orientation as the
            # offline pixel HDF5 dataset.
            third_pil, wrist_pil = frame_history[-1]
            third = np.asarray(third_pil, dtype=np.uint8)[::-1, ::-1]
            wrist = np.asarray(wrist_pil, dtype=np.uint8)[::-1, ::-1]

        third = self._resize_hwc_uint8(third, image_size)
        wrist = self._resize_hwc_uint8(wrist, image_size)
        chw = np.concatenate(
            [third.transpose(2, 0, 1), wrist.transpose(2, 0, 1)],
            axis=0,
        ).astype(np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0).to(self.device)

    def _dreamer_obs_embedding_from_eval_inputs(
        self,
        item_processor: Any,
        frame_history: list[tuple[Image.Image, Image.Image]],
        state: np.ndarray,
        task_description: str,
    ) -> tuple[Any, list[int] | None]:
        if self._wm_expects_pixel_images():
            return self._pixel_obs_for_wm(frame_history), None

        oft_extractor = getattr(self, "_dreamer_oft_extractor", None)
        if oft_extractor is not None:
            raw_obs = getattr(self, "_libero_current_raw_obs", None)
            if not isinstance(raw_obs, dict):
                raise RuntimeError("OFT Dreamer eval requires current LIBERO raw obs")
            obs = self._dreamer_oft_obs_from_libero_raw(raw_obs, state)
            result = oft_extractor.step(obs, task_description)
            hidden = getattr(result, "hidden_state", None)
            if hidden is None:
                hidden = result[1]
            obs_tensor = torch.as_tensor(hidden, device=self.device).float()
            if obs_tensor.ndim in {1, 2}:
                obs_tensor = obs_tensor.unsqueeze(0)
            if obs_tensor.ndim < 2:
                raise ValueError(
                    f"OFT obs_embedding must have at least 1 dim, got {tuple(obs_tensor.shape)}"
                )
            out: dict[str, torch.Tensor] = {"obs_embedding": obs_tensor}
            lang_emb = getattr(result, "lang_emb", None)
            if lang_emb is not None:
                lang = torch.as_tensor(lang_emb, device=self.device).float()
                if lang.ndim == 1:
                    lang = lang.unsqueeze(0)
                out["lang_emb"] = lang
            proprio = torch.as_tensor(state, device=self.device).float()
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            out["proprio"] = proprio
            return out, None

        raise RuntimeError(
            "token world-model evaluation requires the OpenVLA-OFT extractor; "
            "alternate observation encoders are closed"
        )

    @staticmethod
    def _dreamer_oft_obs_from_libero_raw(
        raw_obs: dict[str, Any],
        state: np.ndarray,
    ) -> dict[str, Any]:
        if "agentview_rgb" in raw_obs:
            third = raw_obs["agentview_rgb"]
        else:
            third = raw_obs["agentview_image"]
        if "eye_in_hand_rgb" in raw_obs:
            wrist = raw_obs["eye_in_hand_rgb"]
        else:
            wrist = raw_obs["robot0_eye_in_hand_image"]
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        return {
            "agentview_rgb": np.ascontiguousarray(np.asarray(third, dtype=np.uint8)),
            "eye_in_hand_rgb": np.ascontiguousarray(np.asarray(wrist, dtype=np.uint8)),
            "state": state_arr,
            "proprio": state_arr,
        }

    def _dreamer_dummy_sequence_inputs(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        input_ids = torch.zeros(batch, seq_len + 1, dtype=torch.long, device=hidden_states.device)
        input_ids[:, seq_len] = self._action_token_id
        attention_mask = torch.ones(
            batch, seq_len + 1, dtype=torch.bool, device=hidden_states.device
        )
        return input_ids, attention_mask

    def _dreamer_action_chunk_from_latent(
        self,
        latent: Any,
        input_ids: list[int] | None = None,
        action_steps: int = 1,
        live_hidden: Any | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if bool(getattr(self, "_real_relabel_enabled", False)):
            self._last_real_relabel_actor_step = None
        actor_input_mode = str(
            OmegaConf.select(self.cfg, "algorithm.actor_input_mode", default="pooled")
        ).lower()
        if actor_input_mode == "sequence":
            hidden_states = self.world_model(
                {"mode": "actor_input_sequence", "latent": latent}
            ).float()
            if input_ids is not None:
                seq_input_ids = torch.tensor(
                    [input_ids + [self._action_token_id]],
                    dtype=torch.long,
                    device=self.device,
                )
                if seq_input_ids.shape[1] < hidden_states.shape[1] + 1:
                    pad = hidden_states.shape[1] + 1 - seq_input_ids.shape[1]
                    seq_input_ids = F.pad(seq_input_ids, (0, pad), value=0)
                    seq_input_ids[:, hidden_states.shape[1]] = self._action_token_id
                seq_input_ids = seq_input_ids[:, : hidden_states.shape[1] + 1]
                seq_attention_mask = torch.ones_like(seq_input_ids, dtype=torch.bool)
            else:
                seq_input_ids, seq_attention_mask = self._dreamer_dummy_sequence_inputs(
                    hidden_states
                )
            action, _, _ = self.policy(
                {
                    "mode": "sample",
                    "hidden_states": hidden_states,
                    "input_ids": seq_input_ids,
                    "attention_mask": seq_attention_mask,
                    "target_token_id": self._action_token_id,
                    "deterministic": bool(getattr(self, "_dreamer_deterministic", True)),
                    "return_chunk": True,
                }
            )
            action_chunk_np = action.squeeze(0).detach().cpu().float().numpy()
        else:
            feat = self.world_model({"mode": "actor_input", "latent": latent}).float()
            feat = self._maybe_add_hidden_noise(feat)
            action, _, _ = self.policy(
                {
                    "mode": "sample",
                    "hidden": feat,
                    "deterministic": bool(getattr(self, "_dreamer_deterministic", True)),
                    "return_chunk": True,
                }
            )
            action_chunk_np = action.squeeze(0).detach().cpu().float().numpy()

        if action_chunk_np.ndim == 1:
            action_chunk_np = action_chunk_np.reshape(1, -1)
        else:
            action_chunk_np = action_chunk_np.reshape(-1, action_chunk_np.shape[-1])
        max_actions = max(int(action_steps), 1)
        raw_actions = [
            np.asarray(row[:7], dtype=np.float32).copy() for row in action_chunk_np[:max_actions]
        ]
        env_actions = [
            self._dreamer_policy_raw_to_env_action(row).astype(np.float32, copy=False)
            for row in raw_actions
        ]
        latent_actions = [
            self._dreamer_latent_action_from_raw_env(raw, env).astype(np.float32, copy=False)
            for raw, env in zip(raw_actions, env_actions, strict=True)
        ]
        if not env_actions:
            return [], []
        raw_action_np = raw_actions[0]
        action_np = env_actions[0]
        if bool(getattr(self, "_real_relabel_enabled", False)) and "feat" in locals():
            old_log_prob = float("nan")
            try:
                raw_action_t = torch.as_tensor(
                    raw_action_np, dtype=feat.dtype, device=feat.device
                ).reshape(1, -1)
                with torch.no_grad():
                    old_log_prob_t, _entropy_t, _extra_eval = self.policy(
                        {
                            "mode": "evaluate",
                            "hidden": feat.detach().float(),
                            "action": raw_action_t,
                        }
                    )
                old_log_prob = float(old_log_prob_t.detach().float().reshape(-1)[0].cpu())
            except Exception:
                old_log_prob = float("nan")
            self._last_real_relabel_actor_step = {
                "actor_input": feat.detach().float().reshape(feat.shape[0], -1)[0].cpu().tolist(),
                "raw_action": np.asarray(raw_action_np, dtype=np.float32).reshape(-1).tolist(),
                "old_log_prob": old_log_prob,
            }
        live_hidden_tensor = (
            self._hidden_tensor_from_eval_obs(live_hidden) if live_hidden is not None else None
        )
        self._record_hidden_action_compare(
            live_hidden=live_hidden_tensor,
            recon_hidden=feat if "feat" in locals() else None,
            recon_action_raw=raw_action_np,
            executed_action=action_np,
            context=getattr(self, "_libero_current_eval_context", None),
            source="online_latent",
        )
        live_trace_hidden = self._hidden_token_grid_for_trace(live_hidden_tensor)
        recon_trace_hidden = self._hidden_token_grid_for_trace(feat if "feat" in locals() else None)
        self._write_policy_trace(
            source="dreamer",
            state=np.asarray(
                getattr(self, "_libero_current_eval_context_state", []),
                dtype=np.float32,
            ),
            action_chunk_raw=action_chunk_np[:max_actions],
            action_chunk_env=np.stack(env_actions, axis=0),
            live_hidden_token_grid=live_trace_hidden,
            recon_hidden_token_grid=recon_trace_hidden,
            obs_embedding=live_hidden,
            actor_input=feat if "feat" in locals() else None,
            latent=latent,
            input_ids=np.asarray(input_ids, dtype=np.float32) if input_ids is not None else None,
        )
        if bool(OmegaConf.select(self.cfg, "eval.log_action_stats", default=False)):
            count = int(getattr(self, "_dreamer_eval_action_log_count", 0))
            limit = int(OmegaConf.select(self.cfg, "eval.log_action_stats_limit", default=8))
            if count < limit:
                print(
                    "  [Eval][online-action] "
                    f"raw={np.array2string(raw_action_np, precision=4, suppress_small=False)} "
                    f"env={np.array2string(action_np, precision=4, suppress_small=False)} "
                    f"latent={np.array2string(latent_actions[0], precision=4, suppress_small=False)} "
                    f"abs_mean={float(np.mean(np.abs(action_np))):.5f} "
                    f"max_abs={float(np.max(np.abs(action_np))):.5f} "
                    f"chunk={len(env_actions)} action_steps={int(action_steps)}",
                    flush=True,
                )
            self._dreamer_eval_action_log_count = count + 1
        return env_actions, latent_actions

    def _dreamer_online_reset(self) -> None:
        self._dreamer_online_latent = None
        self._dreamer_online_prev_action = None
        oft_extractor = getattr(self, "_dreamer_oft_extractor", None)
        if oft_extractor is not None and hasattr(oft_extractor, "reset"):
            oft_extractor.reset()
        planner = getattr(self, "_tdmpc_mpc_planner", None)
        if planner is not None:
            planner.reset()

    def _dreamer_online_update_latent(self, obs_embedding: Any) -> Any:
        hidden = self._hidden_tensor_from_eval_obs(obs_embedding)
        if getattr(self, "_dreamer_online_latent", None) is None:
            latent = self.world_model({"mode": "encode_latent", "hidden": hidden})
        else:
            prev_action = getattr(self, "_dreamer_online_prev_action", None)
            if not isinstance(prev_action, torch.Tensor):
                raise RuntimeError("online_latent update missing previous executed action")
            latent = self.world_model(
                {
                    "mode": "observe_next",
                    "latent": self._dreamer_online_latent,
                    "hidden": hidden,
                    "actions": prev_action,
                    "is_first": False,
                }
            )
        latent = self._latent_with_eval_sidecars(latent, obs_embedding)
        self._dreamer_online_latent = latent
        return latent
