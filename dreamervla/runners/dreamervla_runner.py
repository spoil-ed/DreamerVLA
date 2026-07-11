"""DreamerVLA closed-loop training runner.

Each training step runs two interleaved phases on the same data batch:

  Phase-1  World-model (WM) pretraining
           WM learns (obs, action) → next_obs transitions and reward prediction.
           Encoder is frozen; policy/critic are not involved.

  Phase-2  Actor-critic update (DreamerV3-style)
           H-step imagination in WM latent space:
             policy samples actions → WM gives rewards → target_critic
             bootstraps λ-returns → percentile-normalised advantages
           Twohot symlog critic; Polyak-averaged target critic.

Both phases can be toggled independently via `training.run_wm_phase` and
`training.run_actor_critic_phase`.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import pathlib
import random
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from dreamervla.algorithms.dreamervla import (
    imagine_actor_critic_step,
    world_model_pretrain_step,
)
from dreamervla.algorithms.registry import get_actor_update_route
from dreamervla.dataset import BaseDataset
from dreamervla.algorithms.critic.twohot_critic import ReturnPercentileTracker
from dreamervla.runners._dreamer_runner_common import save_viz_strip
from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.distributed import NopretokenizeSFTDistributedHelper
from dreamervla.utils.checkpoint_util import TopKCheckpointManager
from dreamervla.utils.ema import EMAHelper
from dreamervla.utils.hf_checkpoint import (
    is_hf_checkpoint,
    resolve_hf_checkpoint_dir,
)
from dreamervla.utils.optim import build_optimizer
from dreamervla.utils.paths import checkpoints_path, data_path
from dreamervla.utils.seed import set_seed
from dreamervla.utils.torch_utils import freeze_module


class DreamerVLARunner(BaseRunner):
    """Closed-loop DreamerVLA: WM pretraining + DreamerV3 actor-critic (twohot + target critic)."""

    runner_name = "joint_dreamervla"
    runner_status = "current"
    runner_family = "actor"
    include_keys = ("global_step", "epoch")
    # encoder is frozen — no need to checkpoint it.
    exclude_keys = ("encoder", "_unwrapped_world_model")

    @property
    def default_vla_init_dir(self) -> str:
        return str(checkpoints_path("VLA_model_256", "libero_goal"))

    @property
    def default_output_dir(self) -> str:
        return str(data_path("outputs", "dreamervla"))

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        if output_dir is None:
            output_dir = str(
                OmegaConf.select(
                    config, "training.out_dir", default=self.default_output_dir
                )
            )
        super().__init__(config, output_dir=output_dir)

        self.distributed = NopretokenizeSFTDistributedHelper.initialize(
            strategy=str(
                OmegaConf.select(config, "training.distributed_strategy", default="ddp")
            ),
            fsdp_mixed_precision=str(
                OmegaConf.select(
                    config, "training.fsdp_mixed_precision", default="bf16"
                )
            ),
            enable_activation_checkpointing=bool(
                OmegaConf.select(
                    config, "training.enable_activation_checkpointing", default=True
                )
            ),
        )
        self.rank = self.distributed.rank
        self.local_rank = self.distributed.local_rank
        self.world_size = self.distributed.world_size
        self.device = self.distributed.resolve_device(str(self.config.trainer.device))
        if self.distributed.is_main_process:
            self.print_config()
        set_seed(int(self.config.seed) + self.rank)

        # ── model placeholders ──────────────────────────────────────────────
        self.encoder = None  # OpenVLA-OFT policy / frozen feature extractor
        self.policy = None  # VLAPolicy         — Dreamer actor (latent space)
        self.ref_policy = None  # Frozen actor snapshot for KL/BC anchoring
        self.critic = None  # TwohotCritic      — twohot symlog value function
        self.target_critic = None  # TwohotCritic      — Polyak-averaged target copy
        self.world_model = None  # DreamerV3 WM      — dynamics + reward

        # ── optimizer placeholders ──────────────────────────────────────────
        self.policy_optimizer = None
        self.critic_optimizer = None
        self.world_model_optimizer = None
        self.world_model_ema: EMAHelper | None = None
        self.return_tracker: ReturnPercentileTracker | None = None
        self.vq_model = None
        self.actor_update_route = None
        self._real_relabel_steps: list[dict[str, Any]] = []
        self._real_relabel_rng = random.Random(
            int(OmegaConf.select(config, "seed", default=0)) + 1701
        )

    # ──────────────────────────────────────────────────────────────────────
    # Embedding helpers
    # ──────────────────────────────────────────────────────────────────────

    def _encode_hidden_from_tokenized(
        self, input_ids_list: list[list[int]]
    ) -> torch.Tensor:
        """Run the frozen encoder on a list of token sequences → pooled float32 tensor."""
        if self.encoder is None:
            raise ValueError(
                "Encoder must be initialised before calling _encode_hidden_from_tokenized."
            )
        if not input_ids_list:
            hidden_dim = int(
                OmegaConf.select(self.cfg, "world_model.hidden_dim", default=1)
            )
            return torch.zeros((0, hidden_dim), device=self.device, dtype=torch.float32)

        labels_list = [[-100] * len(seq) for seq in input_ids_list]
        lengths = [len(seq) for seq in input_ids_list]
        with torch.no_grad():
            _, _, _, hidden_states, _, _, _ = self.encoder.backbone(
                input_ids=input_ids_list,
                labels=labels_list,
                training=True,
                output_hidden_states=True,
                att_mask=False,
            )
        attention_mask = torch.zeros(
            hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
        )
        for idx, length in enumerate(lengths):
            if length > 0:
                attention_mask[idx, :length] = True
        weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(
            1.0
        )
        return pooled.float().detach()

    def _make_obs_dict(
        self, input_ids_list: list[list[int]]
    ) -> dict[str, torch.Tensor]:
        """Encode tokenized obs → {obs_embedding: tensor} for policy/WM passthrough."""
        return {"obs_embedding": self._encode_hidden_from_tokenized(input_ids_list)}

    # ──────────────────────────────────────────────────────────────────────
    # Batch assembly
    # ──────────────────────────────────────────────────────────────────────

    def _wm_io_mode(self) -> str:
        wm = getattr(self, "_unwrapped_world_model", None) or self.world_model
        return str(getattr(wm, "io_mode", "hidden"))

    def _obs_embedding_for_wm(self, input_ids_list: list[list[int]]) -> torch.Tensor:
        """Produce the obs_embedding the WM expects: BPE ids in token mode,
        pooled hiddens otherwise."""
        if self._wm_io_mode() == "token":
            return self._extract_image_bpe_ids(input_ids_list)
        return self._encode_hidden_from_tokenized(input_ids_list)

    def _obs_embedding_sequence_for_wm(
        self, input_ids_seq: list[list[list[int]]]
    ) -> torch.Tensor:
        """Encode a nested [B][T][token] observation sequence for Dreamer starts."""
        if not input_ids_seq:
            hidden_dim = int(
                OmegaConf.select(self.cfg, "world_model.hidden_dim", default=1)
            )
            return torch.zeros(
                (0, 0, hidden_dim), device=self.device, dtype=torch.float32
            )
        batch_size = len(input_ids_seq)
        seq_len = len(input_ids_seq[0])
        if seq_len == 0:
            hidden_dim = int(
                OmegaConf.select(self.cfg, "world_model.hidden_dim", default=1)
            )
            return torch.zeros(
                (batch_size, 0, hidden_dim), device=self.device, dtype=torch.float32
            )
        flat: list[list[int]] = []
        for row in input_ids_seq:
            if len(row) != seq_len:
                raise ValueError(
                    "All wm_obs_input_ids_seq rows must have the same length."
                )
            flat.extend([list(step) for step in row])
        encoded = self._obs_embedding_for_wm(flat)
        return encoded.reshape(batch_size, seq_len, *encoded.shape[1:])

    def _build_wm_pretrain_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        """Assemble the WM pretrain batch for retained DreamerV3-style routes."""
        if (
            isinstance(batch.get("obs_embedding"), torch.Tensor)
            and isinstance(batch.get("actions"), torch.Tensor)
        ):
            return {
                key: value
                for key in (
                    "obs_embedding",
                    "actions",
                    "current_actions",
                    "rewards",
                    "dones",
                    "is_first",
                    "is_terminal",
                    "is_last",
                    "success_to_go",
                    "return_to_go",
                    "return_targets",
                    "task_ids",
                    "proprio",
                    "lang_emb",
                )
                if (value := batch.get(key)) is not None
            }
        if isinstance(batch.get("images"), torch.Tensor) and isinstance(
            batch.get("actions"), torch.Tensor
        ):
            return {
                key: value
                for key in (
                    "images",
                    "actions",
                    "current_actions",
                    "rewards",
                    "dones",
                    "is_first",
                    "is_terminal",
                    "is_last",
                    "success_to_go",
                    "return_to_go",
                    "return_targets",
                    "task_ids",
                )
                if (value := batch.get(key)) is not None
            }
        if isinstance(batch.get("tokens"), torch.Tensor) and isinstance(
            batch.get("actions"), torch.Tensor
        ):
            return {
                key: value
                for key in (
                    "tokens",
                    "actions",
                    "current_actions",
                    "rewards",
                    "dones",
                    "is_first",
                    "is_terminal",
                    "is_last",
                    "success_to_go",
                    "return_to_go",
                    "return_targets",
                )
                if (value := batch.get(key)) is not None
            }

        obs_seq = batch.get("wm_obs_input_ids_seq")
        action_seq = batch.get("action_seq")
        if isinstance(obs_seq, list) and isinstance(action_seq, torch.Tensor):
            hidden_seq = self._obs_embedding_sequence_for_wm(obs_seq)
            if hidden_seq.shape[1] < 2:
                return None
            actions = action_seq.to(self.device)
            bsz, steps = hidden_seq.shape[:2]
            obs_flat = hidden_seq[:, :-1].reshape(
                bsz * (steps - 1), *hidden_seq.shape[2:]
            )
            next_flat = hidden_seq[:, 1:].reshape(
                bsz * (steps - 1), *hidden_seq.shape[2:]
            )
            action_flat = actions[:, 1:].reshape(bsz * (steps - 1), *actions.shape[2:])
            wm_batch = {
                "obs_embedding": obs_flat,
                "next_obs_embedding": next_flat,
                "action": action_flat,
            }
            reward_seq = batch.get("reward_seq")
            if isinstance(reward_seq, torch.Tensor):
                wm_batch["reward"] = reward_seq.to(self.device)[:, 1:].reshape(
                    bsz * (steps - 1)
                )
            done_seq = batch.get("done_seq")
            if isinstance(done_seq, torch.Tensor):
                wm_batch["done"] = done_seq.to(self.device)[:, 1:].reshape(
                    bsz * (steps - 1)
                )
            return wm_batch

        obs_ids = batch.get("wm_obs_input_ids")
        next_obs_ids = batch.get("wm_next_obs_input_ids")
        if not isinstance(obs_ids, list) or not isinstance(next_obs_ids, list):
            return None

        action = batch.get("conditioning_action") or batch.get("action")
        if not isinstance(action, torch.Tensor):
            return None

        wm_batch: dict[str, Any] = {
            "obs_embedding": self._obs_embedding_for_wm(obs_ids),
            "next_obs_embedding": self._obs_embedding_for_wm(next_obs_ids),
            "action": action.to(self.device),
        }
        reward = batch.get("reward")
        if isinstance(reward, torch.Tensor):
            wm_batch["reward"] = reward.to(self.device)
        return wm_batch

    def _build_actor_critic_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        """Assemble {obs: {obs_embedding}} for imagine_actor_critic_step.
        obs_embedding is BPE ids in token mode, pooled hiddens otherwise.
        """

        def _replay_value_fields() -> dict[str, torch.Tensor]:
            fields: dict[str, torch.Tensor] = {}
            for key in ("rewards", "reward", "dones", "is_terminal", "is_last"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor) and value.ndim >= 2:
                    fields[key] = value.to(self.device)
            return fields

        obs_embedding = batch.get("obs_embedding")
        if isinstance(obs_embedding, torch.Tensor):
            if obs_embedding.ndim != 3:
                raise ValueError(
                    "follow-up flat actor batch expects obs_embedding "
                    f"[B,T,D], got {tuple(obs_embedding.shape)}"
                )
            return {
                "obs": {
                    "obs_embedding": obs_embedding.to(self.device),
                    "actions": batch["actions"].to(self.device),
                    "is_first": batch["is_first"].to(self.device),
                    **_replay_value_fields(),
                    **(
                        {
                            "actor_input_ids": batch["actor_input_ids"].to(self.device),
                            "actor_attention_mask": batch["actor_attention_mask"].to(
                                self.device
                            ),
                        }
                        if isinstance(batch.get("actor_input_ids"), torch.Tensor)
                        and isinstance(batch.get("actor_attention_mask"), torch.Tensor)
                        else {}
                    ),
                }
            }

        images = batch.get("images")
        if isinstance(images, torch.Tensor):
            if images.ndim != 5:
                raise ValueError(
                    f"Dreamer pixel actor batch expects images [B,T,C,H,W], got {tuple(images.shape)}"
                )
            return {
                "obs": {
                    "images": images.to(self.device),
                    "actions": batch["actions"].to(self.device),
                    "is_first": batch["is_first"].to(self.device),
                    **_replay_value_fields(),
                }
            }

        tokens = batch.get("tokens")
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim not in {3, 4}:
                raise ValueError(
                    f"Dreamer token actor batch expects tokens [B,T,N] or [B,T,V,N], got {tuple(tokens.shape)}"
                )
            return {
                "obs": {
                    "tokens": tokens.to(self.device),
                    "actions": batch["actions"].to(self.device),
                    "is_first": batch["is_first"].to(self.device),
                    **_replay_value_fields(),
                }
            }

        obs_seq = batch.get("wm_obs_input_ids_seq")
        action_seq = batch.get("action_seq")
        if isinstance(obs_seq, list) and isinstance(action_seq, torch.Tensor):
            hidden_seq = self._obs_embedding_sequence_for_wm(obs_seq)
            bsz, steps = hidden_seq.shape[:2]
            is_first = torch.zeros(bsz, steps, device=self.device, dtype=torch.bool)
            if steps > 0:
                is_first[:, 0] = True
            return {
                "obs": {
                    "hidden_seq": hidden_seq,
                    "actions": action_seq.to(self.device),
                    "is_first": is_first,
                }
            }

        obs_ids = batch.get("wm_obs_input_ids") or batch.get("input_ids")
        if not isinstance(obs_ids, list):
            return None
        hidden = self._obs_embedding_for_wm(obs_ids)
        actions = batch.get("action")
        if not isinstance(actions, torch.Tensor):
            actions = torch.zeros(
                hidden.shape[0],
                1,
                int(OmegaConf.select(self.cfg, "world_model.action_dim", default=7)),
            )
        if actions.ndim == 2:
            actions = actions[:, None]
        is_first = torch.ones(hidden.shape[0], 1, device=self.device, dtype=torch.bool)
        return {
            "obs": {
                "hidden_seq": hidden[:, None],
                "actions": actions.to(self.device),
                "is_first": is_first,
            }
        }

    def _load_real_relabel_steps(self, cfg: DictConfig) -> list[dict[str, Any]]:
        relabel_cfg = OmegaConf.select(
            cfg, "algorithm.real_rollout_relabel", default=None
        )
        if relabel_cfg is None or not bool(
            OmegaConf.select(relabel_cfg, "enabled", default=False)
        ):
            return []
        raw_paths = OmegaConf.select(relabel_cfg, "paths", default=None)
        if raw_paths is None:
            single_path = OmegaConf.select(relabel_cfg, "path", default=None)
            raw_paths = [single_path] if single_path else []
        paths = [pathlib.Path(str(path)).expanduser() for path in list(raw_paths)]
        if not paths:
            raise ValueError(
                "algorithm.real_rollout_relabel.enabled=true requires `path` or `paths`."
            )

        baseline = float(OmegaConf.select(relabel_cfg, "outcome_baseline", default=0.5))
        positive_weight = float(
            OmegaConf.select(relabel_cfg, "positive_weight", default=1.0)
        )
        negative_weight = float(
            OmegaConf.select(relabel_cfg, "negative_weight", default=1.0)
        )
        terminal_only = (
            str(
                OmegaConf.select(relabel_cfg, "reward_placement", default="terminal")
            ).lower()
            == "terminal"
        )
        max_steps_per_traj = int(
            OmegaConf.select(relabel_cfg, "max_steps_per_trajectory", default=0)
        )
        lower = float(
            OmegaConf.select(relabel_cfg, "accuracy_lower_bound", default=0.01)
        )
        upper = float(
            OmegaConf.select(relabel_cfg, "accuracy_upper_bound", default=0.99)
        )
        keep_all_failures = bool(
            OmegaConf.select(
                relabel_cfg, "keep_all_failures_as_negatives", default=True
            )
        )

        records: list[dict[str, Any]] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"real rollout relabel jsonl not found: {path}")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        groups: dict[str, list[dict[str, Any]]] = {}
        for row in records:
            groups.setdefault(str(row.get("prompt_key", "")), []).append(row)
        kept_prompt_keys: set[str] = set()
        for prompt_key, rows in groups.items():
            acc = sum(float(bool(row.get("complete", False))) for row in rows) / max(
                len(rows), 1
            )
            if lower <= acc <= upper or (keep_all_failures and acc <= 0.0):
                kept_prompt_keys.add(prompt_key)

        steps: list[dict[str, Any]] = []
        skipped_missing_trace = 0
        for row in records:
            if str(row.get("prompt_key", "")) not in kept_prompt_keys:
                continue
            actor_inputs = row.get("actor_inputs")
            raw_actions = row.get("raw_actions")
            old_log_probs = row.get("old_log_probs")
            actor_step_indices = row.get("actor_step_indices")
            if not (
                isinstance(actor_inputs, list)
                and isinstance(raw_actions, list)
                and isinstance(old_log_probs, list)
            ):
                skipped_missing_trace += 1
                continue
            n = min(len(actor_inputs), len(raw_actions), len(old_log_probs))
            if isinstance(actor_step_indices, list):
                n = min(n, len(actor_step_indices))
                step_indices = [int(value) for value in actor_step_indices[:n]]
            else:
                step_indices = list(range(n))
            if max_steps_per_traj > 0:
                n = min(n, max_steps_per_traj)
                step_indices = step_indices[:n]
            if n <= 0:
                continue
            complete = bool(row.get("complete", False))
            finish_step = max(1, int(row.get("finish_step", n)))
            finish_env_index = finish_step - 1
            terminal_trace_idx = max(
                (
                    idx
                    for idx, env_idx in enumerate(step_indices)
                    if int(env_idx) <= finish_env_index
                ),
                default=n - 1,
            )
            advantage = float(row.get("acc", float(complete))) - baseline
            sample_weight = positive_weight if advantage > 0.0 else negative_weight
            for idx in range(n):
                if terminal_only and idx != terminal_trace_idx:
                    continue
                old_log_prob = float(old_log_probs[idx])
                if not math.isfinite(old_log_prob):
                    continue
                steps.append(
                    {
                        "hidden": actor_inputs[idx],
                        "action": raw_actions[idx],
                        "old_log_prob": old_log_prob,
                        "advantage": advantage,
                        "weight": sample_weight,
                        "complete": complete,
                    }
                )

        if self.distributed.is_main_process:
            print(
                "[real-relabel] "
                f"records={len(records)} kept_prompts={len(kept_prompt_keys)}/{len(groups)} "
                f"steps={len(steps)} skipped_missing_trace={skipped_missing_trace}",
                flush=True,
            )
        return steps

    def _sample_real_relabel_batch(
        self, algorithm_cfg: DictConfig
    ) -> dict[str, torch.Tensor] | None:
        steps = getattr(self, "_real_relabel_steps", [])
        if not steps:
            return None
        relabel_cfg = OmegaConf.select(
            algorithm_cfg, "real_rollout_relabel", default=None
        )
        if (
            relabel_cfg is None
            or float(OmegaConf.select(relabel_cfg, "loss_scale", default=0.0)) <= 0.0
        ):
            return None
        batch_size = max(
            1, int(OmegaConf.select(relabel_cfg, "batch_size", default=64))
        )
        replace = len(steps) < batch_size
        if replace:
            chosen = [
                steps[self._real_relabel_rng.randrange(len(steps))]
                for _ in range(batch_size)
            ]
        else:
            chosen = self._real_relabel_rng.sample(steps, batch_size)
        hidden = torch.tensor(
            [row["hidden"] for row in chosen], dtype=torch.float32, device=self.device
        )
        action = torch.tensor(
            [row["action"] for row in chosen], dtype=torch.float32, device=self.device
        )
        old_log_prob = torch.tensor(
            [row["old_log_prob"] for row in chosen],
            dtype=torch.float32,
            device=self.device,
        )
        advantage = torch.tensor(
            [row["advantage"] for row in chosen],
            dtype=torch.float32,
            device=self.device,
        )
        weight = torch.tensor(
            [row["weight"] for row in chosen], dtype=torch.float32, device=self.device
        )
        return {
            "hidden": hidden,
            "action": action,
            "old_log_prob": old_log_prob,
            "advantage": advantage,
            "weight": weight,
        }

    # ── Token DreamerV3 WM visualisation ────────────────────────────────────

    def _maybe_build_viz(self) -> None:
        if not self.distributed.is_main_process:
            return
        viz_cfg = OmegaConf.select(self.cfg, "viz", default=None)
        if viz_cfg is None or not bool(
            OmegaConf.select(viz_cfg, "enabled", default=False)
        ):
            return
        cfg_path = OmegaConf.select(
            viz_cfg,
            "vqgan_config_path",
            default=str(
                pathlib.Path(__file__).resolve().parents[2]
                / "data"
                / "checkpoints"
                / "chameleon"
                / "tokenizer"
                / "vqgan.yaml"
            ),
        )
        ckpt_path = OmegaConf.select(
            viz_cfg,
            "vqgan_ckpt_path",
            default=str(
                pathlib.Path(__file__).resolve().parents[2]
                / "data"
                / "checkpoints"
                / "chameleon"
                / "tokenizer"
                / "vqgan.ckpt"
            ),
        )
        try:
            from dreamervla.utils.vq_image_decoder import load_vq_model

            viz_device_cfg = OmegaConf.select(viz_cfg, "device", default=None)
            viz_device = (
                self.device
                if viz_device_cfg is None
                else torch.device(str(viz_device_cfg))
            )
            self.vq_model = load_vq_model(
                cfg_path=cfg_path, ckpt_path=ckpt_path, device=viz_device
            )
            print(f"[dreamer-vla][viz] VQGAN ready on {viz_device}")
        except Exception as exc:
            print(
                f"[dreamer-vla][viz] failed to build VQGAN visualizer, disabling: {exc}"
            )
            self.vq_model = None

    @torch.no_grad()
    def _decode_token_view(self, token_ids: torch.Tensor, h: int, w: int):
        from dreamervla.utils.vq_image_decoder import tensor_to_pil, vq_tokens_to_pixels

        if self.vq_model is None:
            return None
        token_ids = token_ids.reshape(1, h * w).to(
            device=next(self.vq_model.parameters()).device,
            dtype=torch.long,
        )
        pixels = vq_tokens_to_pixels(token_ids, self.vq_model, h_latent=h, w_latent=w)
        return tensor_to_pil(pixels[0])

    _save_viz_strip = staticmethod(save_viz_strip)

    @torch.no_grad()
    def _maybe_save_token_viz(self, batch: dict[str, Any]) -> None:
        if not self.distributed.is_main_process:
            return
        viz_cfg = OmegaConf.select(self.cfg, "viz", default=None)
        if viz_cfg is None or self.vq_model is None:
            return
        every = int(OmegaConf.select(viz_cfg, "every_n_steps", default=500))
        if every <= 0 or self.global_step % every != 0:
            return
        wm = getattr(
            self, "_unwrapped_world_model", None
        ) or self.distributed.unwrap_module(self.world_model)
        if (
            wm is None
            or not hasattr(wm, "encoder")
            or not hasattr(wm, "rssm")
            or not hasattr(wm, "decoder")
        ):
            return

        tokens = batch.get("tokens")
        actions = batch.get("actions")
        is_first = batch.get("is_first")
        if not isinstance(tokens, torch.Tensor) or not isinstance(
            actions, torch.Tensor
        ):
            return
        if (
            not isinstance(is_first, torch.Tensor)
            or tokens.ndim != 4
            or tokens.shape[1] < 2
        ):
            return

        tokens = tokens.to(self.device, non_blocking=True).long()
        actions = actions.to(self.device, non_blocking=True)
        is_first = is_first.to(self.device, non_blocking=True)
        was_training = wm.training
        wm.eval()
        try:
            enc = wm.encoder(tokens)
            seq = wm.rssm.observe(enc, actions.to(dtype=enc.dtype), is_first)
            post_logits = wm.decoder(seq["deter"], seq["stoch"])
            post_pred = post_logits.argmax(dim=-1)

            deter0 = seq["deter"][:, 0]
            stoch0 = seq["stoch"][:, 0]
            action1 = actions[:, 1].to(device=deter0.device, dtype=deter0.dtype)
            prior_deter1 = wm.rssm._core(deter0, stoch0, action1)
            prior_logits1 = wm.rssm._prior(prior_deter1)
            prior_idx1 = prior_logits1.argmax(dim=-1)
            prior_stoch1 = F.one_hot(prior_idx1, wm.rssm.classes).to(
                dtype=prior_logits1.dtype
            )
            prior_dec_logits = wm.decoder(prior_deter1[:, None], prior_stoch1[:, None])
            prior_pred = prior_dec_logits.argmax(dim=-1)[:, 0]
        finally:
            if was_training:
                wm.train()

        b, _t, num_views, tokens_per_view = tokens.shape
        h, w = tuple(int(x) for x in wm.encoder.spatial_grid)
        if h * w != tokens_per_view:
            print(
                f"[dreamer-vla][viz] skip: h*w={h * w} != tokens_per_view={tokens_per_view}"
            )
            return
        view_labels = list(
            OmegaConf.select(viz_cfg, "view_labels", default=["third", "wrist"])
        )
        if len(view_labels) != num_views:
            view_labels = [f"view{idx}" for idx in range(num_views)]

        num_samples = min(int(OmegaConf.select(viz_cfg, "num_samples", default=2)), b)
        cell_size = int(OmegaConf.select(viz_cfg, "cell_size", default=192))
        out_dir = pathlib.Path(self.output_dir) / "viz"
        saved = 0
        for sample_idx in range(num_samples):
            panels: list[tuple[str, Any]] = []
            for view_idx, label in enumerate(view_labels):
                panels.extend(
                    [
                        (
                            f"{label} cur",
                            self._decode_token_view(
                                tokens[sample_idx, 0, view_idx], h, w
                            ),
                        ),
                        (
                            f"{label} recon",
                            self._decode_token_view(
                                post_pred[sample_idx, 0, view_idx], h, w
                            ),
                        ),
                        (
                            f"{label} next",
                            self._decode_token_view(
                                tokens[sample_idx, 1, view_idx], h, w
                            ),
                        ),
                        (
                            f"{label} prior",
                            self._decode_token_view(
                                prior_pred[sample_idx, view_idx], h, w
                            ),
                        ),
                    ]
                )
            path = out_dir / f"step_{self.global_step:07d}_sample{sample_idx:02d}.png"
            self._save_viz_strip(path, panels, cell_size=cell_size)
            saved += 1
        if saved:
            print(
                f"[dreamer-vla][viz] step {self.global_step}: wrote {saved} panel(s) under {out_dir}"
            )

    # ── Token-mode helpers for VLA/image-token visualization ────────────────

    def _get_image_bpe_set(self) -> set[int]:
        cached = getattr(self, "_image_bpe_set_cache", None)
        if cached is not None:
            return cached
        if self.encoder is None:
            raise RuntimeError("encoder must be built before _get_image_bpe_set()")
        vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
        self._image_bpe_set_cache = set(vocab_mapping.bpe2img.keys())
        return self._image_bpe_set_cache

    def _extract_image_bpe_ids(
        self,
        input_ids_list: list[list[int]],
    ) -> torch.Tensor:
        """Pull the WM's target image block's BPE ids out of each tokenized
        sample, returning [B, n_img_tok] long.  Same logic as
        PretokenizeWMRunner._extract_image_bpe_ids.
        """
        from dreamervla.utils.wm_image_viz import extract_image_blocks

        wm_inner = getattr(self, "_unwrapped_world_model", None) or self.world_model
        n_img_tok = int(getattr(wm_inner, "n_image_tokens", 256))
        which_block = int(OmegaConf.select(self.cfg, "viz.which_block", default=-2))
        img_bpe = self._get_image_bpe_set()
        if not input_ids_list:
            return torch.zeros((0, n_img_tok), dtype=torch.long, device=self.device)
        rows: list[list[int]] = []
        for idx, seq in enumerate(input_ids_list):
            blocks = extract_image_blocks(list(seq))
            if not blocks:
                raise ValueError(f"sample {idx}: no image block found in tokens")
            bidx = which_block if which_block >= 0 else len(blocks) + which_block
            if not (0 <= bidx < len(blocks)):
                raise ValueError(
                    f"sample {idx}: which_block={which_block} out of range "
                    f"(have {len(blocks)} blocks)"
                )
            _start, _end, block_ids = blocks[bidx]
            tok_ids = [int(tok) for tok in block_ids if int(tok) in img_bpe]
            if len(tok_ids) != n_img_tok:
                raise ValueError(
                    f"sample {idx}: block has {len(tok_ids)} image tokens, "
                    f"expected {n_img_tok}"
                )
            rows.append(tok_ids)
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    # ── Cross-run init ckpt loaders ──────────────────────────────────────────

    def _load_encoder_init_ckpt(self, ckpt_path: str) -> None:
        """Load `state_dicts.encoder` from a runner-format .ckpt into
        self.encoder (already instantiated). Used to inject a fine-tuned VLA
        encoder ckpt at the start of cotrain.
        """
        path = pathlib.Path(ckpt_path).expanduser().resolve()
        if is_hf_checkpoint(path):
            if self.distributed.is_main_process:
                print(
                    "[init] encoder weights already come from HF checkpoint "
                    f"{resolve_hf_checkpoint_dir(path)}"
                )
            return
        if not path.is_file():
            raise FileNotFoundError(f"init.encoder_state_ckpt not found: {path}")
        if self.distributed.is_main_process:
            print(f"[init] loading encoder weights from {path} ...")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sd = payload.get("state_dicts", {}).get("encoder")
        if sd is None:
            raise RuntimeError(f"{path} has no state_dicts.encoder")
        target_dtype = next(self.encoder.parameters()).dtype
        sd = {
            k: (v.to(dtype=target_dtype) if torch.is_floating_point(v) else v)
            for k, v in sd.items()
        }
        missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
        if self.distributed.is_main_process:
            print(
                f"[init] encoder loaded: {len(sd)} tensors, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        del payload

    def _load_world_model_init_ckpt(self, ckpt_path: str) -> None:
        """Load world-model weights into self.world_model before wrapping.

        Supports both DreamerVLA runner checkpoints
        (`state_dicts.world_model`) and standalone DreamerV3 WM checkpoints
        (`model`).
        """
        path = pathlib.Path(ckpt_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"init.world_model_state_ckpt not found: {path}")
        if self.distributed.is_main_process:
            print(f"[init] loading world_model weights from {path} ...")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sd = payload.get("state_dicts", {}).get("world_model")
        if sd is None:
            sd = payload.get("model")
        if sd is None:
            raise RuntimeError(
                f"{path} has no state_dicts.world_model or model state dict"
            )
        target_dtype = next(self.world_model.parameters()).dtype
        sd = {
            k: (v.to(dtype=target_dtype) if torch.is_floating_point(v) else v)
            for k, v in sd.items()
        }
        model_sd = self.world_model.state_dict()
        reset_reward_head = bool(
            OmegaConf.select(
                self.cfg, "init.reset_world_model_reward_head", default=False
            )
        )
        remapped: dict[str, torch.Tensor] = {}
        skipped_reward_head = 0
        for key, value in sd.items():
            if key.startswith("module."):
                key = key.removeprefix("module.")
            if reset_reward_head and key.startswith("reward_head."):
                skipped_reward_head += 1
                continue
            if key.startswith("reward_head.net.") and not key.startswith(
                "reward_head.net.net."
            ):
                candidate = key.replace("reward_head.net.", "reward_head.net.net.", 1)
                if candidate in model_sd:
                    key = candidate
            remapped[key] = value
        sd = remapped
        mismatched = [
            (k, tuple(v.shape), tuple(model_sd[k].shape))
            for k, v in sd.items()
            if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape)
        ]
        if mismatched:
            sd = {
                k: v
                for k, v in sd.items()
                if k not in model_sd or tuple(v.shape) == tuple(model_sd[k].shape)
            }
        missing, unexpected = self.world_model.load_state_dict(sd, strict=False)
        if self.distributed.is_main_process:
            print(
                f"[init] world_model loaded: {len(sd)} tensors, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if mismatched:
                print(f"[init] skipped shape-mismatched tensors: {len(mismatched)}")
                print(f"[init] shape mismatches (first 5): {mismatched[:5]}")
            if skipped_reward_head:
                print(
                    f"[init] reset reward_head; skipped {skipped_reward_head} checkpoint tensors"
                )
            if missing:
                print(f"[init] missing (first 5): {missing[:5]}")
            if unexpected:
                print(f"[init] unexpected (first 5): {unexpected[:5]}")
        del payload
        self.distributed.barrier()

    def _load_dreamervla_init_ckpt(self, ckpt_path: str) -> None:
        """Selectively initialise from a previous DreamerVLA checkpoint.

        This is intentionally separate from resume: it lets a new actor class
        reuse the old WM/critic while skipping incompatible policy and optimizer
        state.
        """
        path = pathlib.Path(ckpt_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"init.dreamervla_state_ckpt not found: {path}")
        if self.distributed.is_main_process:
            print(f"[init] loading DreamerVLA init state from {path} ...")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dicts = payload.get("state_dicts", {})
        if not isinstance(state_dicts, dict) or not state_dicts:
            raise RuntimeError(f"{path} has no state_dicts")

        load_plan = {
            "world_model": bool(
                OmegaConf.select(
                    self.cfg, "init.load_dreamervla_world_model", default=True
                )
            ),
            "critic": bool(
                OmegaConf.select(self.cfg, "init.load_dreamervla_critic", default=True)
            ),
            "target_critic": bool(
                OmegaConf.select(
                    self.cfg, "init.load_dreamervla_target_critic", default=True
                )
            ),
            "policy": bool(
                OmegaConf.select(self.cfg, "init.load_dreamervla_policy", default=False)
            ),
            "return_tracker": bool(
                OmegaConf.select(
                    self.cfg, "init.load_dreamervla_return_tracker", default=True
                )
            ),
        }
        strict = bool(
            OmegaConf.select(self.cfg, "init.load_dreamervla_strict", default=False)
        )
        for key, enabled in load_plan.items():
            if (
                not enabled
                or key not in state_dicts
                or getattr(self, key, None) is None
            ):
                continue
            if key == "return_tracker":
                self.return_tracker.load_state_dict(state_dicts[key])
                if self.distributed.is_main_process:
                    print("[init] return_tracker loaded from DreamerVLA checkpoint")
                continue
            self._load_compatible_module_state(key, state_dicts[key], strict=strict)
        del payload
        self.distributed.barrier()

    def _load_compatible_module_state(
        self, key: str, state_dict: dict[str, Any], *, strict: bool = False
    ) -> None:
        module = getattr(self, key)
        context_keys = {"policy", "critic", "target_critic", "world_model"}
        context = (
            self.distributed.model_state_dict_context(module, rank0_only=False)
            if key in context_keys
            else contextlib.nullcontext()
        )
        with context:
            target_sd = module.state_dict()
            try:
                target_dtype = next(module.parameters()).dtype
            except StopIteration:
                target_dtype = None
            filtered: dict[str, Any] = {}
            mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
            for raw_key, value in state_dict.items():
                load_key = (
                    raw_key.removeprefix("module.")
                    if isinstance(raw_key, str)
                    else raw_key
                )
                if (
                    key == "world_model"
                    and isinstance(load_key, str)
                    and load_key.startswith("reward_head.net.")
                    and not load_key.startswith("reward_head.net.net.")
                ):
                    candidate = load_key.replace(
                        "reward_head.net.", "reward_head.net.net.", 1
                    )
                    if candidate in target_sd or f"module.{candidate}" in target_sd:
                        load_key = candidate
                candidate_keys = [load_key]
                if isinstance(load_key, str):
                    if not load_key.startswith("module."):
                        candidate_keys.append(f"module.{load_key}")
                    else:
                        candidate_keys.append(load_key.removeprefix("module."))
                target_key = next(
                    (
                        candidate
                        for candidate in candidate_keys
                        if candidate in target_sd
                    ),
                    load_key,
                )
                if target_key in target_sd and isinstance(value, torch.Tensor):
                    if tuple(value.shape) != tuple(target_sd[target_key].shape):
                        mismatched.append(
                            (
                                str(target_key),
                                tuple(value.shape),
                                tuple(target_sd[target_key].shape),
                            )
                        )
                        continue
                    if target_dtype is not None and torch.is_floating_point(value):
                        value = value.to(dtype=target_dtype)
                filtered[str(target_key)] = value
            missing, unexpected = module.load_state_dict(filtered, strict=strict)
        if self.distributed.is_main_process:
            print(
                f"[init] {key} loaded from DreamerVLA checkpoint: tensors={len(filtered)} "
                f"missing={len(missing)} unexpected={len(unexpected)} strict={strict}"
            )
            if mismatched:
                print(
                    f"[init] {key} skipped shape-mismatched tensors: {len(mismatched)}"
                )
                print(f"[init] {key} shape mismatches (first 5): {mismatched[:5]}")
            if missing:
                print(f"[init] {key} missing (first 5): {missing[:5]}")
            if unexpected:
                print(f"[init] {key} unexpected (first 5): {unexpected[:5]}")

    def _attach_image_token_mapping(self) -> None:
        """For spatial_codec / token-mode WMs, attach the encoder's image-vocab
        BPE mapping (and lm_head in hidden mode). Required before WM forward.
        """
        wm = getattr(self, "_unwrapped_world_model", None) or self.world_model
        if not getattr(wm, "spatial_codec", False):
            return
        if self.encoder is None:
            return
        try:
            lm_head = self.encoder.backbone.lm_head
            vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
            image_token_bpe_ids = torch.tensor(
                sorted(vocab_mapping.bpe2img.keys()),
                dtype=torch.long,
            )
            full_vocab_size = int(lm_head.weight.shape[0])
            wm_io_mode = str(getattr(wm, "io_mode", "hidden"))
            wm.attach_lm_head(
                lm_head if wm_io_mode == "hidden" else None,
                image_token_bpe_ids,
                full_vocab_size=full_vocab_size,
            )
            if self.distributed.is_main_process:
                tag = "lm_head" if wm_io_mode == "hidden" else "vocab (token mode)"
                print(
                    f"[wm] attached {tag} for image-vocab CE "
                    f"(image_vocab={image_token_bpe_ids.numel()}, full_vocab={full_vocab_size})"
                )
        except Exception as exc:
            if self.distributed.is_main_process:
                print(f"[wm] attach_lm_head failed — token-mode WM will crash: {exc}")

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────────

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        if key == "policy" and self.policy is not None:
            with self.distributed.model_state_dict_context(self.policy):
                return self.policy.state_dict()
        if (
            key == "policy_optimizer"
            and self.policy_optimizer is not None
            and self.policy is not None
        ):
            return self.distributed.optimizer_state_dict(
                self.policy, self.policy_optimizer
            )
        if key == "critic" and self.critic is not None:
            with self.distributed.model_state_dict_context(self.critic):
                return self.critic.state_dict()
        if (
            key == "critic_optimizer"
            and self.critic_optimizer is not None
            and self.critic is not None
        ):
            return self.distributed.optimizer_state_dict(
                self.critic, self.critic_optimizer
            )
        if key == "target_critic" and self.target_critic is not None:
            with self.distributed.model_state_dict_context(self.target_critic):
                return self.target_critic.state_dict()
        if key == "return_tracker" and self.return_tracker is not None:
            return self.return_tracker.state_dict()
        if key == "world_model" and self.world_model is not None:
            with self.distributed.model_state_dict_context(self.world_model):
                return self.world_model.state_dict()
        if (
            key == "world_model_optimizer"
            and self.world_model_optimizer is not None
            and self.world_model is not None
        ):
            return self.distributed.optimizer_state_dict(
                self.world_model, self.world_model_optimizer
            )
        return value.state_dict()

    def _load_state_dict_from_checkpoint(
        self, key: str, value: Any, state_dict: dict[str, Any], **kwargs: Any
    ) -> None:
        if key == "policy" and self.policy is not None:
            with self.distributed.model_state_dict_context(self.policy):
                value.load_state_dict(state_dict, **kwargs)
            return
        if (
            key == "policy_optimizer"
            and self.policy_optimizer is not None
            and self.policy is not None
        ):
            self.distributed.load_optimizer_state_dict(
                self.policy, self.policy_optimizer, state_dict
            )
            return
        if key == "critic" and self.critic is not None:
            with self.distributed.model_state_dict_context(self.critic):
                value.load_state_dict(state_dict, **kwargs)
            return
        if (
            key == "critic_optimizer"
            and self.critic_optimizer is not None
            and self.critic is not None
        ):
            self.distributed.load_optimizer_state_dict(
                self.critic, self.critic_optimizer, state_dict
            )
            return
        if key == "target_critic" and self.target_critic is not None:
            with self.distributed.model_state_dict_context(self.target_critic):
                value.load_state_dict(state_dict, **kwargs)
            return
        if key == "return_tracker" and self.return_tracker is not None:
            value.load_state_dict(state_dict)
            return
        if key == "world_model" and self.world_model is not None:
            with self.distributed.model_state_dict_context(self.world_model):
                value.load_state_dict(state_dict, **kwargs)
            return
        if (
            key == "world_model_optimizer"
            and self.world_model_optimizer is not None
            and self.world_model is not None
        ):
            self.distributed.load_optimizer_state_dict(
                self.world_model, self.world_model_optimizer, state_dict
            )
            return
        value.load_state_dict(state_dict, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_val_loss(
        self, val_dataloader: DataLoader, split_name: str
    ) -> dict[str, float]:
        if self.world_model is None:
            return {}
        self.world_model.eval()
        val_losses: list[float] = []
        val_transition_losses: list[float] = []
        for batch in val_dataloader:
            wm_batch = self._build_wm_pretrain_batch(batch)
            if wm_batch is None:
                continue
            wm_batch = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in wm_batch.items()
            }
            # Route via __call__ so FSDP gathers params before compute_loss_dict.
            loss_dict = self.world_model(wm_batch)
            loss_value = loss_dict.get("loss", loss_dict.get("_loss"))
            if not isinstance(loss_value, torch.Tensor):
                continue
            val_losses.append(float(loss_value.item()))
            transition_value = loss_dict.get("transition_loss")
            val_transition_losses.append(
                float(transition_value.item())
                if isinstance(transition_value, torch.Tensor)
                else 0.0
            )
        self.world_model.train()
        if not val_losses:
            return {}
        count = max(self.distributed.reduce_sum(len(val_losses)), 1.0)
        metrics = {
            f"val_{split_name}_wm_loss": self.distributed.reduce_sum(sum(val_losses))
            / count,
            f"val_{split_name}_wm_transition_loss": self.distributed.reduce_sum(
                sum(val_transition_losses)
            )
            / count,
        }
        if self.distributed.is_main_process:
            print(
                f"  [Val {split_name}] "
                + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )
        return metrics

    def run(self) -> list[dict[str, float | str | int]]:  # noqa: C901
        history: list[dict[str, float | str | int]] = []
        if self.distributed.is_main_process:
            print("DreamerVLA Runner begin.")
        cfg = copy.deepcopy(self.cfg)

        # ── dataset & dataloader ────────────────────────────────────────
        dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, BaseDataset)

        train_dataloader = self.make_distributed_dataloader(
            dataset,
            cfg.dataloader,
            sanitize_worker_kwargs=True,
        )
        val_dataloaders = self.make_val_dataloaders(cfg, sanitize_worker_kwargs=True)

        # ── encoder (frozen) ───────────────────────────────────────────
        encoder_cfg_root = OmegaConf.select(cfg, "encoder", default=None)
        if encoder_cfg_root is None:
            self.encoder = None
        else:
            encoder_cfg = self._build_frozen_encoder_cfg(cfg)
            encoder_init_ckpt = OmegaConf.select(
                cfg, "init.encoder_state_ckpt", default=None
            )
            if encoder_init_ckpt and is_hf_checkpoint(encoder_init_ckpt):
                with open_dict(encoder_cfg):
                    encoder_cfg.model_path = str(
                        resolve_hf_checkpoint_dir(encoder_init_ckpt)
                    )
            self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
            freeze_module(self.encoder)
            if encoder_init_ckpt:
                self._load_encoder_init_ckpt(str(encoder_init_ckpt))

        # ── world model ────────────────────────────────────────────────
        world_model_cfg = OmegaConf.select(cfg, "world_model")
        if world_model_cfg is None:
            raise ValueError("`world_model` config section is required.")
        wm_hidden_dim = self.infer_hidden_dim_from_dataset(
            dataset
        ) or self.infer_hidden_dim_from_encoder(self.encoder)
        instantiate_kwargs: dict[str, Any] = {}
        if wm_hidden_dim is not None:
            instantiate_kwargs["hidden_dim"] = wm_hidden_dim
        # token-mode WM needs num_image_tokens_vocab; auto-fill from encoder
        # when not pinned in the cfg.
        if (
            str(OmegaConf.select(world_model_cfg, "io_mode", default="hidden"))
            == "token"
            and OmegaConf.select(world_model_cfg, "num_image_tokens_vocab") is None
        ):
            vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
            instantiate_kwargs["num_image_tokens_vocab"] = len(vocab_mapping.bpe2img)
        self.world_model = hydra.utils.instantiate(
            world_model_cfg, **instantiate_kwargs
        ).to(self.device)

        fsdp_precision = str(
            OmegaConf.select(cfg, "training.fsdp_mixed_precision", default="bf16")
        )
        _dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        self.world_model = self.world_model.to(
            dtype=_dtype_map.get(fsdp_precision, torch.bfloat16)
        )

        # Pre-FSDP wiring: attach image-token vocab mapping (required for token
        # mode forward), then load init ckpt while the module is still
        # unwrapped.  Keep an unwrapped reference for batch-builder helpers
        # that need to read attributes like io_mode and n_image_tokens.
        self._unwrapped_world_model = self.world_model
        self._attach_image_token_mapping()
        wm_init_ckpt = OmegaConf.select(
            cfg, "init.world_model_state_ckpt", default=None
        )
        if wm_init_ckpt:
            self._load_world_model_init_ckpt(str(wm_init_ckpt))

        self.world_model = self.distributed.wrap_trainable_module(self.world_model)

        wm_optim_cfg = OmegaConf.select(cfg, "optim.world_model")
        if wm_optim_cfg is None:
            raise ValueError("`optim.world_model` must be configured.")
        self.world_model_optimizer = build_optimizer(self.world_model, wm_optim_cfg)

        # ── policy (Dreamer actor) ─────────────────────────────────────
        policy_cfg = OmegaConf.select(cfg, "policy")
        if policy_cfg is None:
            raise ValueError("`policy` config section is required.")
        policy_target = str(OmegaConf.select(policy_cfg, "_target_", default=""))
        require_encoder_ckpt = bool(
            OmegaConf.select(cfg, "init.require_encoder_state_ckpt", default=False)
        )
        encoder_state_ckpt = OmegaConf.select(
            cfg, "init.encoder_state_ckpt", default=None
        )
        if (
            require_encoder_ckpt
            and "VLAActionHeadActor" in policy_target
            and not encoder_state_ckpt
        ):
            raise ValueError(
                "This DreamerVLA config reuses the VLA action head, so it needs a "
                "matching goal VLA checkpoint. Pass one with "
                "`init.encoder_state_ckpt=data/outputs/vla/<run>/checkpoints/latest.ckpt` or "
                "`VLA_STATE_CKPT=data/outputs/vla/<run>/checkpoints/latest.ckpt bash scripts/train_dreamervla.sh`."
            )
        policy_module = hydra.utils.instantiate(policy_cfg).to(self.device)
        algorithm_cfg_for_ref = OmegaConf.select(cfg, "algorithm", default={})
        use_ref_policy = (
            float(OmegaConf.select(algorithm_cfg_for_ref, "kl_coef", default=0.0)) > 0.0
            or float(
                OmegaConf.select(
                    algorithm_cfg_for_ref, "actor_bc_to_ref_scale", default=0.0
                )
            )
            > 0.0
        )
        if use_ref_policy:
            self.ref_policy = copy.deepcopy(policy_module).to(self.device)
            freeze_module(self.ref_policy)
            self.ref_policy.eval()
        self.policy = policy_module
        # Policy stays in float32 — its forward casts inputs via .float()
        # internally, so feature dtype is normalised at the call site.
        self.policy = self.distributed.wrap_trainable_module(self.policy)

        policy_optim_cfg = OmegaConf.select(cfg, "optim.policy")
        if policy_optim_cfg is None:
            raise ValueError("`optim.policy` must be configured.")
        self.policy_optimizer = build_optimizer(self.policy, policy_optim_cfg)

        # ── twohot critic (online + target copy) ───────────────────────
        critic_cfg = OmegaConf.select(cfg, "critic")
        if critic_cfg is None:
            raise ValueError("`critic` config section is required.")
        tdmpc_value_mode = str(
            OmegaConf.select(cfg, "algorithm.tdmpc_ac.value_mode", default="state")
        ).lower()
        if tdmpc_value_mode in {"state_action", "q", "q_za", "q(z,a)"}:
            critic_cfg = OmegaConf.create(
                OmegaConf.to_container(critic_cfg, resolve=True)
            )
            critic_action_dim = int(
                OmegaConf.select(
                    cfg,
                    "algorithm.tdmpc_ac.action_dim",
                    default=OmegaConf.select(cfg, "policy.action_dim", default=7),
                )
            )
            critic_cfg.hidden_dim = int(critic_cfg.hidden_dim) + critic_action_dim
            print(
                "[tdmpc_ac] using state_action critic: "
                f"critic.hidden_dim={critic_cfg.hidden_dim} (+ action_dim={critic_action_dim})"
            )

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.target_critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        # Critic / target_critic stay in float32 — feature dtype normalised
        # at call site (imagine_actor_critic_step).
        freeze_module(
            self.target_critic
        )  # updated by Polyak averaging, never by optimiser
        self.critic = self.distributed.wrap_trainable_module(self.critic)
        # target_critic must be FSDP-wrapped with the SAME flattening as
        # critic, otherwise soft_update sees one side as 1-D shards and the
        # other as 2-D layers and dim mismatches. wrap_trainable_module is
        # idempotent on already-frozen modules — it'll FSDP-wrap them under
        # FSDP strategy and DDP-skip them otherwise.
        self.target_critic = self.distributed.wrap_trainable_module(self.target_critic)

        critic_optim_cfg = OmegaConf.select(cfg, "optim.critic")
        if critic_optim_cfg is None:
            raise ValueError("`optim.critic` must be configured.")
        self.critic_optimizer = build_optimizer(self.critic, critic_optim_cfg)

        # ── classifier (only needed by the lumos route) ─────────
        # Loaded frozen — provides the outcome reward (LatentSuccessClassifier
        # over the imagined latent video). Skipped when update_type != LUMOS.
        self.classifier = None
        self.classifier_threshold = 0.5
        actor_update_kind = str(
            OmegaConf.select(cfg, "algorithm.update_type", default="dreamer")
        ).lower()
        self.actor_update_route = (
            None
            if actor_update_kind == "dreamer"
            else get_actor_update_route(actor_update_kind)
        )
        if (
            self.actor_update_route is not None
            and self.actor_update_route.requires_classifier
        ):
            from dreamervla.algorithms.critic import build_classifier

            classifier_ckpt_path = OmegaConf.select(
                cfg, "init.classifier_state_ckpt", default=None
            )
            if not classifier_ckpt_path:
                raise ValueError(
                    f"update_type={actor_update_kind} requires init.classifier_state_ckpt — "
                    "path to a LatentSuccessClassifier .ckpt (model+threshold+config)."
                )
            cls_payload = torch.load(
                str(classifier_ckpt_path), map_location="cpu", weights_only=False
            )
            cls_cfg_blob = cls_payload.get("config", {}).get("classifier")
            if cls_cfg_blob is None:
                raise RuntimeError(
                    f"classifier ckpt {classifier_ckpt_path} has no config.classifier blob"
                )
            self.classifier = build_classifier(cls_cfg_blob).to(self.device).eval()
            self.classifier.load_state_dict(cls_payload["model"])
            freeze_module(self.classifier)
            override_thresh = OmegaConf.select(
                cfg, "algorithm.lumos.classifier_threshold", default=None
            )
            self.classifier_threshold = float(
                override_thresh
                if override_thresh is not None
                else cls_payload.get("threshold", 0.5)
            )
            if self.distributed.is_main_process:
                print(
                    f"[init] classifier loaded from {classifier_ckpt_path}; "
                    f"threshold={self.classifier_threshold:.4f}; "
                    f"ckpt F1={cls_payload.get('f1', float('nan')):.4f}",
                    flush=True,
                )

        # ── return percentile tracker ──────────────────────────────────
        self.return_tracker = ReturnPercentileTracker(
            decay=float(
                OmegaConf.select(cfg, "algorithm.return_tracker.decay", default=0.99)
            ),
            low=float(
                OmegaConf.select(cfg, "algorithm.return_tracker.low", default=0.05)
            ),
            high=float(
                OmegaConf.select(cfg, "algorithm.return_tracker.high", default=0.95)
            ),
        )
        self._maybe_build_viz()

        dreamervla_init_ckpt = OmegaConf.select(
            cfg, "init.dreamervla_state_ckpt", default=None
        )
        if dreamervla_init_ckpt:
            self._load_dreamervla_init_ckpt(str(dreamervla_init_ckpt))

        if (
            bool(OmegaConf.select(cfg, "training.use_ema", default=False))
            and self.world_model_ema is None
        ):
            self.world_model_ema = EMAHelper(
                self.world_model,
                decay=float(OmegaConf.select(cfg, "ema.decay", default=0.9999)),
                update_after_step=int(
                    OmegaConf.select(cfg, "ema.update_after_step", default=0)
                ),
            )

        self.resume(cfg)
        if bool(OmegaConf.select(cfg, "training.resume", default=False)):
            steps_per_epoch = len(train_dataloader)
            if (
                steps_per_epoch > 0
                and self.global_step > 0
                and (self.global_step + 1) % steps_per_epoch == 0
            ):
                if self.distributed.is_main_process:
                    print(
                        "[resume] checkpoint was saved at an epoch boundary; "
                        f"advancing epoch {self.epoch} -> {self.epoch + 1} "
                        f"and global_step {self.global_step} -> {self.global_step + 1}"
                    )
                self.epoch += 1
                self.global_step += 1

        lr_scheduler_name = str(
            OmegaConf.select(cfg, "training.lr_scheduler", default="constant")
        )
        lr_warmup_steps = int(
            OmegaConf.select(cfg, "training.lr_warmup_steps", default=0)
        )
        num_epochs_cfg = OmegaConf.select(cfg, "training.num_epochs", default=20)
        num_epochs = 20 if num_epochs_cfg is None else int(num_epochs_cfg)
        total_training_steps = (len(train_dataloader) * num_epochs) // int(
            cfg.training.gradient_accumulate_every
        )
        wm_lr_scheduler = get_scheduler(
            lr_scheduler_name,
            optimizer=self.world_model_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=total_training_steps,
            last_epoch=self.global_step - 1,
        )
        policy_lr_scheduler = get_scheduler(
            lr_scheduler_name,
            optimizer=self.policy_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=total_training_steps,
            last_epoch=self.global_step - 1,
        )
        critic_lr_scheduler = get_scheduler(
            lr_scheduler_name,
            optimizer=self.critic_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=total_training_steps,
            last_epoch=self.global_step - 1,
        )

        run_wm_phase = bool(
            OmegaConf.select(cfg, "training.run_wm_phase", default=True)
        )
        run_ac_phase = bool(
            OmegaConf.select(cfg, "training.run_actor_critic_phase", default=True)
        )
        algorithm_cfg = OmegaConf.select(cfg, "algorithm")
        if algorithm_cfg is None:
            raise ValueError("`algorithm` config section is required.")
        actor_update_kind = str(
            OmegaConf.select(algorithm_cfg, "update_type", default="dreamer")
        ).lower()
        actor_update_route = getattr(self, "actor_update_route", None)
        if actor_update_kind != "dreamer" and actor_update_route is None:
            actor_update_route = get_actor_update_route(actor_update_kind)
        optim_cfg = OmegaConf.select(cfg, "optim")
        self._real_relabel_steps = self._load_real_relabel_steps(cfg)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        if cfg.training.debug:
            num_epochs = 3
            cfg.training.num_epochs = num_epochs
            cfg.training.max_train_steps = 2
            cfg.training.checkpoint_every = 1

        if self.distributed.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
        self.distributed.barrier()

        train_log_path = os.path.join(self.output_dir, "dreamervla_logs.json.txt")
        train_logger_cm = self.distributed.logger_context(train_log_path)
        progress_total = max(1, num_epochs * len(train_dataloader))

        try:
            with train_logger_cm as train_json_logger:
                reached_max_steps = False
                self.console_banner("TRAINING", subtitle=f"{num_epochs} epochs")
                while self.epoch < num_epochs:
                    self.set_dataloader_epoch(train_dataloader, self.epoch)

                    step_log: dict[str, float | str | int] = {}
                    epoch_wm_losses: list[float] = []
                    epoch_actor_losses: list[float] = []
                    epoch_critic_losses: list[float] = []
                    epoch_returns: list[float] = []
                    epoch_rewards: list[float] = []
                    epoch_scales: list[float] = []

                    for batch_idx, batch in enumerate(train_dataloader):
                        local_metrics: dict[str, float] = {}
                        step_had_update = False

                        # Phase 1 — world-model pretraining
                        if run_wm_phase:
                            wm_batch = self._build_wm_pretrain_batch(batch)
                            if wm_batch is not None:
                                self.world_model.train()
                                self.policy.eval()
                                self.critic.eval()
                                wm_metrics = world_model_pretrain_step(
                                    policy=self.policy,
                                    world_model=self.world_model,
                                    optimizer=self.world_model_optimizer,
                                    batch=wm_batch,
                                    device=self.device,
                                    optim_cfg=optim_cfg,
                                )
                                wm_lr_scheduler.step()
                                if self.world_model_ema is not None:
                                    self.world_model_ema.step(self.world_model)
                                epoch_wm_losses.append(wm_metrics["loss"])
                                local_metrics["train_wm_loss"] = wm_metrics["loss"]
                                local_metrics["train_wm_transition_loss"] = (
                                    wm_metrics["transition_loss"]
                                )
                                local_metrics["train_wm_reward_loss"] = wm_metrics[
                                    "reward_loss"
                                ]
                                local_metrics["train_wm_grad_norm"] = wm_metrics[
                                    "grad_norm"
                                ]
                                for name, value in wm_metrics.items():
                                    if name not in {
                                        "loss",
                                        "transition_loss",
                                        "reward_loss",
                                        "grad_norm",
                                    }:
                                        local_metrics[f"train_wm_{name}"] = value
                                local_metrics["wm_lr"] = float(
                                    wm_lr_scheduler.get_last_lr()[0]
                                )
                                step_had_update = True

                        # Phase 2 — DreamerV3 actor-critic imagination
                        if run_ac_phase:
                            ac_batch = self._build_actor_critic_batch(batch)
                            if ac_batch is not None:
                                self.world_model.eval()
                                if actor_update_kind == "dreamer":
                                    ac_metrics = imagine_actor_critic_step(
                                        policy=self.policy,
                                        world_model=self.world_model,
                                        critic=self.critic,
                                        target_critic=self.target_critic,
                                        actor_optimizer=self.policy_optimizer,
                                        critic_optimizer=self.critic_optimizer,
                                        return_tracker=self.return_tracker,
                                        obs=ac_batch["obs"],
                                        device=self.device,
                                        algorithm_cfg=algorithm_cfg,
                                        optim_cfg=optim_cfg,
                                        ref_policy=self.ref_policy,
                                    )
                                else:
                                    if actor_update_route is None:
                                        raise RuntimeError(
                                            "Actor update route was not resolved."
                                        )
                                    actor_kwargs = {
                                        "policy": self.policy,
                                        actor_update_route.world_model_arg: self.world_model,
                                        "actor_optimizer": self.policy_optimizer,
                                        "obs": ac_batch["obs"],
                                        "device": self.device,
                                        "algorithm_cfg": algorithm_cfg,
                                        "optim_cfg": optim_cfg,
                                        "ref_policy": self.ref_policy,
                                    }
                                    if actor_update_route.requires_classifier:
                                        actor_kwargs.update(
                                            {
                                                "classifier": self.classifier,
                                                "classifier_threshold": self.classifier_threshold,
                                            }
                                        )
                                    if actor_update_route.uses_real_relabel:
                                        actor_kwargs["real_relabel_batch"] = (
                                            self._sample_real_relabel_batch(
                                                algorithm_cfg
                                            )
                                        )
                                    if actor_update_route.uses_critic:
                                        actor_kwargs.update(
                                            {
                                                "critic": self.critic,
                                                "target_critic": self.target_critic,
                                                "critic_optimizer": self.critic_optimizer,
                                            }
                                        )
                                    ac_metrics = actor_update_route.step_fn(
                                        **actor_kwargs
                                    )
                                epoch_actor_losses.append(ac_metrics["actor_loss"])
                                epoch_critic_losses.append(
                                    ac_metrics["critic_loss"]
                                )
                                epoch_returns.append(ac_metrics["returns_mean"])
                                epoch_rewards.append(ac_metrics["reward_mean"])
                                epoch_scales.append(ac_metrics["return_scale"])
                                local_metrics.update(
                                    {
                                        "train_actor_loss": ac_metrics[
                                            "actor_loss"
                                        ],
                                        "train_actor_bc_loss": ac_metrics.get(
                                            "actor_bc_loss", 0.0
                                        ),
                                        "train_actor_bc_scale": ac_metrics.get(
                                            "actor_bc_scale", 0.0
                                        ),
                                        "train_actor_vla_drift_raw_mse": ac_metrics.get(
                                            "actor_vla_drift_raw_mse", 0.0
                                        ),
                                        "train_actor_vla_drift_env_mse": ac_metrics.get(
                                            "actor_vla_drift_env_mse", 0.0
                                        ),
                                        "train_actor_vla_drift_env_mse_clipped": ac_metrics.get(
                                            "actor_vla_drift_env_mse_clipped", 0.0
                                        ),
                                        "train_actor_vla_drift_env_mae": ac_metrics.get(
                                            "actor_vla_drift_env_mae", 0.0
                                        ),
                                        "train_critic_loss": ac_metrics[
                                            "critic_loss"
                                        ],
                                        "train_returns_mean": ac_metrics[
                                            "returns_mean"
                                        ],
                                        "train_returns_std": ac_metrics[
                                            "returns_std"
                                        ],
                                        "train_raw_returns_mean": ac_metrics.get(
                                            "raw_returns_mean",
                                            ac_metrics["returns_mean"],
                                        ),
                                        "train_raw_returns_std": ac_metrics.get(
                                            "raw_returns_std",
                                            ac_metrics["returns_std"],
                                        ),
                                        "train_advantage_mean": ac_metrics.get(
                                            "advantage_mean", 0.0
                                        ),
                                        "train_advantage_std": ac_metrics.get(
                                            "advantage_std", 0.0
                                        ),
                                        "train_advantage_mag": ac_metrics.get(
                                            "advantage_mag", 0.0
                                        ),
                                        "train_return_norm_enabled": ac_metrics.get(
                                            "return_norm_enabled", 0.0
                                        ),
                                        "train_return_norm_low": ac_metrics.get(
                                            "return_norm_low", 0.0
                                        ),
                                        "train_return_norm_high": ac_metrics.get(
                                            "return_norm_high", 0.0
                                        ),
                                        "train_return_norm_scale": ac_metrics.get(
                                            "return_norm_scale", 1.0
                                        ),
                                        "train_return_norm_batch_scale": ac_metrics.get(
                                            "return_norm_batch_scale", 1.0
                                        ),
                                        "train_ret_normed_min": ac_metrics.get(
                                            "ret_normed_min", 0.0
                                        ),
                                        "train_ret_normed_max": ac_metrics.get(
                                            "ret_normed_max", 0.0
                                        ),
                                        "train_ret_normed_rate": ac_metrics.get(
                                            "ret_normed_rate", 0.0
                                        ),
                                        "train_return_scale": ac_metrics[
                                            "return_scale"
                                        ],
                                        "train_reward_mean": ac_metrics[
                                            "reward_mean"
                                        ],
                                        "train_success_return_shaping_scale": ac_metrics.get(
                                            "success_return_shaping_scale", 0.0
                                        ),
                                        "train_success_return_mean": ac_metrics.get(
                                            "success_return_mean", 0.0
                                        ),
                                        "train_success_return_delta_mean": ac_metrics.get(
                                            "success_return_delta_mean", 0.0
                                        ),
                                        "train_success_return_delta_std": ac_metrics.get(
                                            "success_return_delta_std", 0.0
                                        ),
                                        "train_continue_mean": ac_metrics.get(
                                            "continue_mean", 1.0
                                        ),
                                        "train_value_mean": ac_metrics[
                                            "value_mean"
                                        ],
                                        "train_critic_target_mean": ac_metrics.get(
                                            "critic_target_mean", 0.0
                                        ),
                                        "train_repval_loss": ac_metrics.get(
                                            "repval_loss", 0.0
                                        ),
                                        "train_repval_applied": ac_metrics.get(
                                            "repval_applied", 0.0
                                        ),
                                        "train_repval_weight_mean": ac_metrics.get(
                                            "repval_weight_mean", 0.0
                                        ),
                                        "train_imagine_weight_mean": ac_metrics.get(
                                            "imagine_weight_mean", 1.0
                                        ),
                                        "train_actor_grad_norm": ac_metrics[
                                            "actor_grad_norm"
                                        ],
                                        "train_critic_grad_norm": ac_metrics[
                                            "critic_grad_norm"
                                        ],
                                        "train_ppo_update_epochs": ac_metrics.get(
                                            "ppo_update_epochs", 1.0
                                        ),
                                        "train_ppo_ratio_mean": ac_metrics.get(
                                            "ppo_ratio_mean", 1.0
                                        ),
                                        "train_ppo_ratio_min": ac_metrics.get(
                                            "ppo_ratio_min", 1.0
                                        ),
                                        "train_ppo_ratio_max": ac_metrics.get(
                                            "ppo_ratio_max", 1.0
                                        ),
                                        "train_ppo_clipfrac": ac_metrics.get(
                                            "ppo_clipfrac", 0.0
                                        ),
                                        "train_real_relabel_applied": ac_metrics.get(
                                            "real_relabel_applied", 0.0
                                        ),
                                        "train_real_relabel_loss": ac_metrics.get(
                                            "real_relabel_loss", 0.0
                                        ),
                                        "train_real_relabel_term": ac_metrics.get(
                                            "real_relabel_term", 0.0
                                        ),
                                        "train_real_relabel_ratio_mean": ac_metrics.get(
                                            "real_relabel_ratio_mean", 1.0
                                        ),
                                        "train_real_relabel_clipfrac": ac_metrics.get(
                                            "real_relabel_clipfrac", 0.0
                                        ),
                                        "train_real_relabel_advantage_mean": ac_metrics.get(
                                            "real_relabel_advantage_mean", 0.0
                                        ),
                                    }
                                )
                                for name, value in ac_metrics.items():
                                    if (
                                        name.startswith("actor_grad_norm_")
                                        or name.startswith("tdmpc_")
                                    ) and isinstance(value, (int, float)):
                                        local_metrics[f"train_{name}"] = value
                                policy_lr_scheduler.step()
                                critic_lr_scheduler.step()
                                local_metrics["policy_lr"] = float(
                                    policy_lr_scheduler.get_last_lr()[0]
                                )
                                local_metrics["critic_lr"] = float(
                                    critic_lr_scheduler.get_last_lr()[0]
                                )
                                step_had_update = True

                        if not step_had_update:
                            continue

                        reduced = self.distributed.reduce_mean_dict(local_metrics)
                        step_log = {
                            **reduced,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                        }

                        self.console_progress(
                            self.global_step, progress_total, "train"
                        )

                        self._maybe_save_token_viz(batch)

                        is_last_batch = batch_idx == len(train_dataloader) - 1
                        if not is_last_batch:
                            train_json_logger.log(step_log)
                            self.log_metrics(step_log, step=self.global_step)
                            self.global_step += 1

                        if (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= cfg.training.max_train_steps - 1
                        ):
                            reached_max_steps = True
                            break

                    if not epoch_wm_losses and not epoch_actor_losses:
                        self.global_step += 1
                        self.epoch += 1
                        continue

                    if epoch_wm_losses:
                        wm_n = max(
                            self.distributed.reduce_sum(len(epoch_wm_losses)), 1.0
                        )
                        step_log["epoch_wm_loss"] = (
                            self.distributed.reduce_sum(sum(epoch_wm_losses)) / wm_n
                        )
                    if epoch_actor_losses:
                        ac_n = max(
                            self.distributed.reduce_sum(len(epoch_actor_losses)), 1.0
                        )
                        step_log["epoch_actor_loss"] = (
                            self.distributed.reduce_sum(sum(epoch_actor_losses)) / ac_n
                        )
                        step_log["epoch_critic_loss"] = (
                            self.distributed.reduce_sum(sum(epoch_critic_losses)) / ac_n
                        )
                        step_log["epoch_returns_mean"] = (
                            self.distributed.reduce_sum(sum(epoch_returns)) / ac_n
                        )
                        step_log["epoch_reward_mean"] = (
                            self.distributed.reduce_sum(sum(epoch_rewards)) / ac_n
                        )
                        step_log["epoch_return_scale"] = (
                            self.distributed.reduce_sum(sum(epoch_scales)) / ac_n
                        )

                    step_log.setdefault("epoch_wm_loss", float("inf"))
                    step_log.setdefault("epoch_actor_loss", float("inf"))
                    step_log.setdefault("epoch_critic_loss", float("inf"))

                    _epoch_train_keys = {
                        "epoch_wm_loss",
                        "epoch_actor_loss",
                        "epoch_critic_loss",
                        "epoch_returns_mean",
                        "epoch_reward_mean",
                        "epoch_return_scale",
                    }
                    _train_console_metrics = {
                        self._normalize_metric_name(k): v
                        for k, v in step_log.items()
                        if k in _epoch_train_keys
                    }
                    if _train_console_metrics:
                        self.console_metrics(
                            f"train · epoch {self.epoch}", _train_console_metrics
                        )

                    eval_every = int(
                        OmegaConf.select(cfg, "eval.eval_every", default=1)
                    )
                    _eval_console_metrics: dict[str, float] = {}
                    if val_dataloaders and (self.epoch % eval_every) == 0:
                        for split_name, val_dl in val_dataloaders.items():
                            _val_metrics = self.evaluate_val_loss(val_dl, split_name)
                            step_log.update(_val_metrics)
                            _eval_console_metrics.update(
                                {
                                    self._normalize_metric_name(k): v
                                    for k, v in _val_metrics.items()
                                }
                            )
                    if _eval_console_metrics:
                        self.console_metrics(
                            f"eval · epoch {self.epoch}", _eval_console_metrics
                        )

                    train_json_logger.log(step_log)
                    self.log_metrics(step_log, step=self.global_step)
                    history.append(dict(step_log))

                    if (self.epoch % cfg.training.checkpoint_every) == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        metric_dict = {
                            k.replace("/", "_"): v for k, v in step_log.items()
                        }
                        topk_ckpt_path = None
                        if self.distributed.is_main_process:
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        topk_ckpt_path = self.distributed.broadcast_object(
                            topk_ckpt_path
                        )
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                    self.global_step += 1
                    self.epoch += 1
                    if reached_max_steps:
                        break
                self.console_banner("TRAINING", done=True)
        finally:
            self.distributed.barrier()
            self.distributed.cleanup()

        return history


__all__ = ["DreamerVLARunner"]
