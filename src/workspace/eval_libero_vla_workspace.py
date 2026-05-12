"""Eval-only workspace: load a VLA/Dreamer checkpoint and run LIBERO rollouts.

No training, no optimizer, no dataset. Reuses the rollout logic that already
lives on ``PretokenizeVLAWorkspace.evaluate_libero`` so there is exactly one
code path for LIBERO success-rate measurement.

Typical use:

  bash scripts/eval_libero_vla.sh \\
    eval.ckpt_path=/path/to/pretokenize_vla/checkpoints/epoch=013-train_vla_loss=1.984.ckpt \\
    eval.task_suite_name=libero_goal \\
    eval.num_episodes_per_task=10

LIBERO rollout is strictly single-process; the script enforces a single GPU
and this workspace forces ``distributed_strategy=ddp`` so the encoder is not
sharded (FSDP sharding would block single-rank inference).
"""
from __future__ import annotations

import copy
import gc
import json
import os
import pathlib
import time
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image

from src.utils.torch_utils import freeze_module
from src.workspace.pretokenize_vla_workspace import PretokenizeVLAWorkspace


class EvalLiberoVLAWorkspace(PretokenizeVLAWorkspace):
    """Load a VLA or Dreamer ckpt -> run LIBERO rollout -> dump JSON metrics."""

    default_output_dir = "/home/user01/liops/workspace/DreamerVLA/data/outputs/eval/eval_libero_vla"

    def run(self) -> list[dict[str, Any]]:
        if self.distributed.is_main_process:
            print("EvalLiberoVLA Workspace begin.")
        cfg = copy.deepcopy(self.cfg)

        if self.world_size != 1:
            raise RuntimeError(
                f"EvalLiberoVLAWorkspace must run on a single process (got world_size={self.world_size}). "
                "Rollout evaluation does not support multi-process inference."
            )
        if self.distributed.uses_fsdp:
            raise RuntimeError(
                "EvalLiberoVLAWorkspace requires DDP (not FSDP). "
                "Pass `training.distributed_strategy=ddp`."
            )

        ckpt_path = OmegaConf.select(cfg, "eval.ckpt_path", default=None)
        ckpt_path = str(pathlib.Path(str(ckpt_path)).expanduser().resolve()) if ckpt_path else None
        payload = None
        ckpt_kind = str(OmegaConf.select(cfg, "eval.ckpt_kind", default="auto")).lower()
        if ckpt_kind not in {"auto", "vla", "dreamer"}:
            raise ValueError("eval.ckpt_kind must be one of: auto, vla, dreamer")
        if ckpt_path and ckpt_kind in {"auto", "dreamer"}:
            payload = self._load_checkpoint_payload(ckpt_path)
            state_keys = set(payload.get("state_dicts", {}).keys())
            is_dreamer = {"world_model", "policy"}.issubset(state_keys)
            if ckpt_kind == "dreamer" and not is_dreamer:
                raise RuntimeError(f"{ckpt_path} does not look like a Dreamer checkpoint: {sorted(state_keys)}")
            if is_dreamer:
                return self._run_dreamer_eval(cfg, ckpt_path, payload)

        # ── encoder (inference only; no optimiser, no distributed wrapping) ──
        encoder_cfg = self._build_trainable_encoder_cfg(cfg)
        with open_dict(encoder_cfg):
            encoder_cfg.freeze_backbone = True
        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        self.encoder.eval()

        # ── optional: load VLA checkpoint (produced by PretokenizeVLAWorkspace) ─
        if ckpt_path:
            if self.distributed.is_main_process:
                print(f"  [Eval] loading VLA checkpoint: {ckpt_path}")
            # Only restore the encoder; skip optimiser / EMA / step counters.
            # (The ckpt was produced by PretokenizeVLAWorkspace which writes
            # vla_optimizer too, but that attribute is None here.)
            if payload is None:
                payload = self._load_checkpoint_payload(ckpt_path)
            self.load_payload(
                payload,
                exclude_keys=("vla_optimizer", "vla_ema"),
                include_keys=(),  # don't restore global_step / epoch
            )
        else:
            if self.distributed.is_main_process:
                print("  [Eval] no eval.ckpt_path set → evaluating init VLA weights "
                      f"({OmegaConf.select(cfg, 'init.vla_ckpt_path')})")

        # ── rollout ──────────────────────────────────────────────────────────
        os.makedirs(self.output_dir, exist_ok=True)
        metrics = self.evaluate_libero(epoch=-1)

        # ── dump metrics ─────────────────────────────────────────────────────
        if self.distributed.is_main_process:
            metrics_out = {
                "ckpt_path": ckpt_path,
                "task_suite": str(OmegaConf.select(cfg, "eval.task_suite_name", default="libero_goal")),
                "num_episodes_per_task": int(OmegaConf.select(cfg, "eval.num_episodes_per_task", default=10)),
                "action_steps": int(OmegaConf.select(cfg, "eval.action_steps", default=10)),
                **metrics,
            }
            out_path = os.path.join(self.output_dir, "eval_libero_metrics.json")
            with open(out_path, "w") as f:
                json.dump(metrics_out, f, indent=2)
            print(f"  [Eval] wrote metrics → {out_path}")

        return [metrics]

    def _load_checkpoint_payload(self, ckpt_path: str) -> dict[str, Any]:
        if self.distributed.is_main_process:
            print(f"  [Eval] reading checkpoint: {ckpt_path}")
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def _run_dreamer_eval(
        self,
        eval_cfg_root: DictConfig,
        ckpt_path: str,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.distributed.is_main_process:
            print("  [Eval] detected Dreamer checkpoint; using world_model + policy rollout.")

        train_cfg = copy.deepcopy(payload.get("cfg"))
        if train_cfg is None:
            raise RuntimeError(f"{ckpt_path} has no saved cfg; cannot rebuild Dreamer modules.")
        with open_dict(train_cfg):
            train_cfg.eval = copy.deepcopy(eval_cfg_root.eval)
            if OmegaConf.select(train_cfg, "encoder", default=None) is None:
                train_cfg.encoder = copy.deepcopy(eval_cfg_root.encoder)
            # Dreamer checkpoints may carry a stale init/encoder path when the
            # training launch overrode it from the shell.  Let eval-time
            # overrides rebuild the frozen VLA backbone/action-head correctly.
            eval_vla_path = OmegaConf.select(eval_cfg_root, "init.vla_ckpt_path", default=None)
            if eval_vla_path is not None:
                train_cfg.init.vla_ckpt_path = eval_vla_path
                if OmegaConf.select(train_cfg, "encoder", default=None) is not None:
                    train_cfg.encoder.model_path = eval_vla_path
            eval_encoder_ckpt = OmegaConf.select(eval_cfg_root, "init.encoder_state_ckpt", default=None)
            if eval_encoder_ckpt is not None:
                train_cfg.init.encoder_state_ckpt = eval_encoder_ckpt
            eval_horizon = OmegaConf.select(eval_cfg_root, "encoder.time_horizon", default=None)
            if eval_horizon is not None and OmegaConf.select(train_cfg, "encoder", default=None) is not None:
                train_cfg.encoder.time_horizon = eval_horizon
            train_cfg.training.out_dir = self.output_dir
            train_cfg.training.distributed_strategy = "ddp"
            train_cfg.training.enable_activation_checkpointing = False
            train_cfg.trainer.device = str(eval_cfg_root.trainer.device)
        self.cfg = train_cfg
        self.config = train_cfg

        self._dreamer_eval = True
        self._dreamer_deterministic = bool(OmegaConf.select(train_cfg, "eval.dreamer_deterministic", default=True))
        self._dreamer_action_repeat = max(1, int(OmegaConf.select(train_cfg, "eval.dreamer_action_repeat", default=1)))
        self._dreamer_clip_actions = bool(OmegaConf.select(train_cfg, "eval.dreamer_clip_actions", default=True))
        self._dreamer_rollout_mode = str(OmegaConf.select(train_cfg, "eval.dreamer_rollout_mode", default="stateless")).lower()
        if self._dreamer_rollout_mode not in {"stateless", "online_rssm"}:
            raise ValueError("eval.dreamer_rollout_mode must be one of: stateless, online_rssm")
        self._dreamer_actor_input_source = str(
            OmegaConf.select(train_cfg, "eval.dreamer_actor_input_source", default="rssm")
        ).lower()
        if self._dreamer_actor_input_source not in {"rssm", "encoder", "encoder_sequence"}:
            raise ValueError("eval.dreamer_actor_input_source must be one of: rssm, encoder, encoder_sequence")
        self._dreamer_policy_source = str(
            OmegaConf.select(train_cfg, "eval.dreamer_policy_source", default="ckpt")
        ).lower()
        if self._dreamer_policy_source not in {"ckpt", "init"}:
            raise ValueError("eval.dreamer_policy_source must be one of: ckpt, init")
        self._hidden_noise_std = float(OmegaConf.select(train_cfg, "eval.hidden_noise_std", default=0.0))
        self._hidden_noise_seed = int(OmegaConf.select(train_cfg, "eval.hidden_noise_seed", default=0))
        self._hidden_noise_generator = torch.Generator(device=self.device)
        self._hidden_noise_generator.manual_seed(self._hidden_noise_seed)
        self._hidden_noise_mse_sum = 0.0
        self._hidden_noise_cosine_sum = 0.0
        self._hidden_noise_count = 0

        self._build_dreamer_modules(train_cfg, payload)
        os.makedirs(self.output_dir, exist_ok=True)
        metrics = self.evaluate_libero(epoch=-1)
        if self._hidden_noise_count > 0:
            metrics = dict(metrics)
            metrics["hidden_noise_std"] = float(self._hidden_noise_std)
            metrics["hidden_noise_seed"] = int(self._hidden_noise_seed)
            metrics["hidden_noise_mean_mse"] = float(self._hidden_noise_mse_sum / self._hidden_noise_count)
            metrics["hidden_noise_mean_cosine_loss"] = float(self._hidden_noise_cosine_sum / self._hidden_noise_count)
            metrics["hidden_noise_count"] = int(self._hidden_noise_count)

        if self.distributed.is_main_process:
            metrics_out = {
                "ckpt_path": ckpt_path,
                "ckpt_kind": "dreamer",
                "task_suite": str(OmegaConf.select(train_cfg, "eval.task_suite_name", default="libero_goal")),
                "num_episodes_per_task": int(OmegaConf.select(train_cfg, "eval.num_episodes_per_task", default=10)),
                "action_steps": int(OmegaConf.select(train_cfg, "eval.action_steps", default=10)),
                "dreamer_action_repeat": int(self._dreamer_action_repeat),
                "dreamer_deterministic": bool(self._dreamer_deterministic),
                "dreamer_clip_actions": bool(self._dreamer_clip_actions),
                "dreamer_rollout_mode": str(self._dreamer_rollout_mode),
                "dreamer_actor_input_source": str(self._dreamer_actor_input_source),
                "dreamer_policy_source": str(self._dreamer_policy_source),
                "hidden_noise_std": float(self._hidden_noise_std),
                "hidden_noise_seed": int(self._hidden_noise_seed),
                **metrics,
            }
            out_path = os.path.join(self.output_dir, "eval_libero_metrics.json")
            with open(out_path, "w") as f:
                json.dump(metrics_out, f, indent=2)
            print(f"  [Eval] wrote metrics -> {out_path}")
        return [metrics]

    def evaluate_libero(self, epoch: int) -> dict[str, float]:
        if (
            getattr(self, "_dreamer_eval", False)
            and getattr(self, "_dreamer_rollout_mode", "stateless") == "online_rssm"
        ):
            return self._evaluate_libero_online_rssm(epoch)
        return super().evaluate_libero(epoch)

    def _maybe_add_hidden_noise(self, hidden: torch.Tensor) -> torch.Tensor:
        noise_std = float(getattr(self, "_hidden_noise_std", 0.0))
        if noise_std <= 0.0:
            return hidden
        noise = torch.randn(
            hidden.shape,
            generator=getattr(self, "_hidden_noise_generator", None),
            device=hidden.device,
            dtype=hidden.dtype,
        ) * noise_std
        perturbed = hidden + noise
        with torch.no_grad():
            mse = (perturbed.float() - hidden.float()).square().mean()
            pred = F.normalize(perturbed.float(), dim=-1)
            target = F.normalize(hidden.float(), dim=-1)
            cosine = 1.0 - (pred * target).sum(dim=-1).mean()
            self._hidden_noise_mse_sum += float(mse.detach().cpu())
            self._hidden_noise_cosine_sum += float(cosine.detach().cpu())
            self._hidden_noise_count += 1
        return perturbed

    def _build_dreamer_modules(self, cfg: DictConfig, payload: dict[str, Any]) -> None:
        state_dicts = payload.get("state_dicts", {})

        encoder_cfg = self._build_frozen_encoder_cfg(cfg)
        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        freeze_module(self.encoder)
        if "encoder" in state_dicts:
            self._load_module_state(self.encoder, state_dicts["encoder"], "encoder")
        else:
            encoder_init_ckpt = OmegaConf.select(cfg, "init.encoder_state_ckpt", default=None)
            if encoder_init_ckpt:
                encoder_payload = self._load_checkpoint_payload(str(encoder_init_ckpt))
                encoder_sd = encoder_payload.get("state_dicts", {}).get("encoder")
                if encoder_sd is None:
                    raise RuntimeError(f"{encoder_init_ckpt} has no state_dicts.encoder")
                self._load_module_state(self.encoder, encoder_sd, "encoder")
                del encoder_payload
        self.encoder.eval()

        world_model_cfg = OmegaConf.select(cfg, "world_model")
        if world_model_cfg is None:
            raise ValueError("Dreamer eval requires `world_model` in the saved cfg.")
        instantiate_kwargs: dict[str, Any] = {}
        if (
            str(OmegaConf.select(world_model_cfg, "io_mode", default="hidden")) == "token"
            and OmegaConf.select(world_model_cfg, "num_image_tokens_vocab") is None
        ):
            vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
            instantiate_kwargs["num_image_tokens_vocab"] = len(vocab_mapping.bpe2img)
        self.world_model = hydra.utils.instantiate(world_model_cfg, **instantiate_kwargs).to(self.device)
        fsdp_precision = str(OmegaConf.select(cfg, "training.fsdp_mixed_precision", default="bf16"))
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.world_model = self.world_model.to(dtype=dtype_map.get(fsdp_precision, torch.bfloat16))
        self._unwrapped_world_model = self.world_model
        self._attach_image_token_mapping()
        self._load_module_state(self.world_model, state_dicts["world_model"], "world_model")
        self.world_model.eval()

        policy_cfg = OmegaConf.select(cfg, "policy")
        if policy_cfg is None:
            raise ValueError("Dreamer eval requires `policy` in the saved cfg.")
        if getattr(self, "_dreamer_policy_source", "ckpt") == "ckpt":
            # The Dreamer checkpoint below fully restores the policy.  Avoid
            # re-reading the 40GB VLA training checkpoint just to warm-start
            # action_head during construction.
            policy_cfg = copy.deepcopy(policy_cfg)
            with open_dict(policy_cfg):
                policy_cfg.init_action_head_ckpt = None
            if self.distributed.is_main_process:
                print("  [Eval] policy source=ckpt; skipped action_head warm-start during policy init.")
        self.policy = hydra.utils.instantiate(policy_cfg).to(self.device)
        if getattr(self, "_dreamer_policy_source", "ckpt") == "ckpt":
            self._load_module_state(self.policy, state_dicts["policy"], "policy")
        elif self.distributed.is_main_process:
            print("  [Eval] using init policy/action_head; skipped Dreamer checkpoint policy state.")
        self.policy.eval()

        # Drop optimizer/critic tensors as soon as possible; Dreamer eval only needs actor + WM.
        for key in ("policy_optimizer", "critic_optimizer", "world_model_optimizer", "critic", "target_critic"):
            state_dicts.pop(key, None)
        gc.collect()

    def _build_frozen_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = self.build_encoder_cfg(cfg)
        with open_dict(encoder_cfg):
            encoder_cfg.model_path = self._resolve_vla_init_path()
            encoder_cfg.freeze_backbone = True
        return encoder_cfg

    def _load_module_state(self, module: Any, state_dict: dict[str, Any], name: str) -> None:
        target_dtype = next(module.parameters()).dtype
        converted = {
            self._strip_wrapping_prefix(key): (
                value.to(dtype=target_dtype)
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
                else value
            )
            for key, value in state_dict.items()
        }
        if name == "world_model":
            model_sd = module.state_dict()
            remapped = {}
            for key, value in converted.items():
                if key.startswith("reward_head.net.") and not key.startswith("reward_head.net.net."):
                    candidate = key.replace("reward_head.net.", "reward_head.net.net.", 1)
                    if candidate in model_sd:
                        key = candidate
                remapped[key] = value
            converted = remapped
        missing, unexpected = module.load_state_dict(converted, strict=False)
        if self.distributed.is_main_process:
            print(
                f"  [Eval] loaded {name}: tensors={len(converted)} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing:
                print(f"  [Eval]   missing first 5: {missing[:5]}")
            if unexpected:
                print(f"  [Eval]   unexpected first 5: {unexpected[:5]}")

    @staticmethod
    def _strip_wrapping_prefix(key: str) -> str:
        for prefix in ("_fsdp_wrapped_module.", "module."):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    def _attach_image_token_mapping(self) -> None:
        wm = getattr(self, "_unwrapped_world_model", None) or self.world_model
        if wm is None or not getattr(wm, "spatial_codec", False) or self.encoder is None:
            return
        lm_head = self.encoder.backbone.lm_head
        vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
        image_token_bpe_ids = torch.tensor(sorted(vocab_mapping.bpe2img.keys()), dtype=torch.long)
        full_vocab_size = int(lm_head.weight.shape[0])
        wm_io_mode = str(getattr(wm, "io_mode", "hidden"))
        wm.attach_lm_head(
            lm_head if wm_io_mode == "hidden" else None,
            image_token_bpe_ids,
            full_vocab_size=full_vocab_size,
        )
        if self.distributed.is_main_process:
            tag = "lm_head" if wm_io_mode == "hidden" else "vocab (token mode)"
            print(f"  [Eval] attached {tag} for image-token mapping.")

    def _wm_io_mode(self) -> str:
        wm = getattr(self, "_unwrapped_world_model", None) or getattr(self, "world_model", None)
        if wm is None:
            return "hidden"
        explicit = getattr(wm, "io_mode", None)
        if explicit is not None:
            return str(explicit)
        encoder = getattr(wm, "encoder", None)
        if encoder is not None and encoder.__class__.__name__ == "DreamerV3TokenEncoder":
            return "token"
        return "hidden"

    def _wm_expects_image_vocab_tokens(self) -> bool:
        wm = getattr(self, "_unwrapped_world_model", None) or getattr(self, "world_model", None)
        encoder = getattr(wm, "encoder", None)
        return encoder is not None and encoder.__class__.__name__ == "DreamerV3TokenEncoder"

    def _wm_expects_pixel_images(self) -> bool:
        wm = getattr(self, "_unwrapped_world_model", None) or getattr(self, "world_model", None)
        encoder = getattr(wm, "encoder", None)
        return encoder is not None and encoder.__class__.__name__ == "DreamerV3PixelEncoder"

    def _get_image_bpe_set(self) -> set[int]:
        cached = getattr(self, "_image_bpe_set_cache", None)
        if cached is not None:
            return cached
        vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
        self._image_bpe_set_cache = set(vocab_mapping.bpe2img.keys())
        return self._image_bpe_set_cache

    def _extract_image_bpe_ids(self, input_ids_list: list[list[int]]) -> torch.Tensor:
        from src.utils.wm_image_viz import extract_image_blocks

        wm = getattr(self, "_unwrapped_world_model", None) or self.world_model
        wm_encoder = getattr(wm, "encoder", None)
        n_img_tok = int(getattr(wm, "n_image_tokens", getattr(wm_encoder, "n_image_tokens", 256)))
        which_blocks_cfg = OmegaConf.select(self.cfg, "eval.dreamer_which_image_blocks", default=None)
        if which_blocks_cfg is None:
            which_blocks = [int(OmegaConf.select(self.cfg, "eval.dreamer_which_image_block", default=-2))]
        else:
            which_blocks = [int(item) for item in which_blocks_cfg]
        img_bpe = self._get_image_bpe_set()
        bpe2img = None
        if self._wm_expects_image_vocab_tokens():
            bpe2img = self.encoder.backbone.model.vocabulary_mapping.bpe2img
        rows: list[list[int]] = []
        for idx, seq in enumerate(input_ids_list):
            blocks = extract_image_blocks(list(seq))
            if not blocks:
                raise ValueError(f"rollout sample {idx}: no image block found in tokens")
            tok_ids: list[int] = []
            for which_block in which_blocks:
                bidx = which_block if which_block >= 0 else len(blocks) + which_block
                if not (0 <= bidx < len(blocks)):
                    raise ValueError(f"rollout sample {idx}: image block {which_block} out of range")
                _start, _end, block_ids = blocks[bidx]
                tok_ids.extend(int(tok) for tok in block_ids if int(tok) in img_bpe)
            if len(tok_ids) != n_img_tok:
                raise ValueError(
                    f"rollout sample {idx}: image blocks {which_blocks} have {len(tok_ids)} image tokens, expected {n_img_tok}"
                )
            if bpe2img is not None:
                tok_ids = [int(bpe2img[int(tok)]) for tok in tok_ids]
            rows.append(tok_ids)
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def _encode_hidden_from_tokenized(self, input_ids_list: list[list[int]]) -> torch.Tensor:
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
        attention_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        for idx, length in enumerate(lengths):
            if length > 0:
                attention_mask[idx, :length] = True
        weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        return ((hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)).float().detach()

    def _encode_hidden_sequence_from_tokenized(
        self,
        input_ids_list: list[list[int]],
        target_token_id: int = 10004,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        max_len = int(hidden_states.shape[1])
        input_rows = []
        mask_rows = []
        for seq, length in zip(input_ids_list, lengths):
            # Append the action trigger as a marker.  The actor consumes all
            # hidden states before this marker, matching native ActionHead.
            row = [int(tok) for tok in seq[:max_len]] + [int(target_token_id)]
            mask = [1] * min(int(length), max_len) + [1]
            target_len = max_len + 1
            if len(row) < target_len:
                row.extend([0] * (target_len - len(row)))
                mask.extend([0] * (target_len - len(mask)))
            input_rows.append(row[:target_len])
            mask_rows.append(mask[:target_len])
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(mask_rows, dtype=torch.bool, device=self.device)
        return hidden_states.float().detach(), input_ids, attention_mask

    def _obs_embedding_for_wm(self, input_ids_list: list[list[int]]) -> torch.Tensor:
        if self._wm_io_mode() == "token":
            return self._extract_image_bpe_ids(input_ids_list)
        return self._encode_hidden_from_tokenized(input_ids_list)

    @staticmethod
    def _resize_hwc_uint8(image: np.ndarray, size: int) -> np.ndarray:
        if image.shape[0] == size and image.shape[1] == size:
            return np.ascontiguousarray(image)
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        return np.asarray(Image.fromarray(image).resize((size, size), resample=resample), dtype=np.uint8)

    def _pixel_obs_for_wm(self, frame_history: list[tuple[Image.Image, Image.Image]]) -> torch.Tensor:
        wm = getattr(self, "_unwrapped_world_model", None) or self.world_model
        wm_encoder = getattr(wm, "encoder", None)
        image_size = int(getattr(wm_encoder, "image_size", OmegaConf.select(self.cfg, "world_model.image_size", default=64)))

        raw_obs = getattr(self, "_libero_current_raw_obs", None)
        if isinstance(raw_obs, dict) and "agentview_image" in raw_obs and "robot0_eye_in_hand_image" in raw_obs:
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
    ) -> tuple[torch.Tensor, list[int] | None]:
        if self._wm_expects_pixel_images():
            return self._pixel_obs_for_wm(frame_history), None

        img_c: list[Image.Image] = []
        for third_pil, wrist_pil in frame_history:
            img_c.extend([third_pil, wrist_pil])
        human_val = f"Finish the task: {task_description}." + "<|state|>" + "<|image|>" * len(img_c)
        conv = {
            "conversations": [{"from": "human", "value": human_val}],
            "image": img_c,
            "action": [],
            "state": [state],
        }
        tokens = item_processor.process_item(conv, training_mode=False)
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        input_ids = [int(tok) for tok in tokens]
        return self._obs_embedding_for_wm([input_ids]), input_ids

    def _dreamer_dummy_sequence_inputs(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        input_ids = torch.zeros(batch, seq_len + 1, dtype=torch.long, device=hidden_states.device)
        input_ids[:, seq_len] = 10004
        attention_mask = torch.ones(batch, seq_len + 1, dtype=torch.bool, device=hidden_states.device)
        return input_ids, attention_mask

    def _dreamer_action_from_latent(
        self,
        latent: Any,
        input_ids: list[int] | None = None,
        action_steps: int = 1,
    ) -> np.ndarray:
        actor_input_mode = str(OmegaConf.select(self.cfg, "algorithm.actor_input_mode", default="pooled")).lower()
        if actor_input_mode == "sequence":
            hidden_states = self.world_model({"mode": "actor_input_sequence", "latent": latent}).float()
            if input_ids is not None:
                seq_input_ids = torch.tensor([input_ids + [10004]], dtype=torch.long, device=self.device)
                if seq_input_ids.shape[1] < hidden_states.shape[1] + 1:
                    pad = hidden_states.shape[1] + 1 - seq_input_ids.shape[1]
                    seq_input_ids = F.pad(seq_input_ids, (0, pad), value=0)
                    seq_input_ids[:, hidden_states.shape[1]] = 10004
                seq_input_ids = seq_input_ids[:, : hidden_states.shape[1] + 1]
                seq_attention_mask = torch.ones_like(seq_input_ids, dtype=torch.bool)
            else:
                seq_input_ids, seq_attention_mask = self._dreamer_dummy_sequence_inputs(hidden_states)
            action, _, _ = self.policy({
                "mode": "sample",
                "hidden_states": hidden_states,
                "input_ids": seq_input_ids,
                "attention_mask": seq_attention_mask,
                "target_token_id": 10004,
                "deterministic": bool(getattr(self, "_dreamer_deterministic", True)),
                "return_chunk": True,
            })
            action_np = action.squeeze(0).detach().cpu().float().numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]
            else:
                action_np = action_np.reshape(-1, action_np.shape[-1])[0]
        else:
            feat = self.world_model({"mode": "actor_input", "latent": latent}).float()
            feat = self._maybe_add_hidden_noise(feat)
            action, _, _ = self.policy({
                "mode": "sample",
                "hidden": feat,
                "deterministic": bool(getattr(self, "_dreamer_deterministic", True)),
            })
            action_np = action.squeeze(0).detach().cpu().float().numpy()
            if action_np.ndim > 1:
                action_np = action_np.reshape(-1, action_np.shape[-1])[0]

        action_np = np.asarray(action_np[:7], dtype=np.float32)
        raw_action_np = action_np.copy()
        if bool(getattr(self, "_dreamer_clip_actions", True)):
            min_values = np.array([-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0])
            max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
            action_np = np.clip(action_np, min_values, max_values)
        if bool(OmegaConf.select(self.cfg, "eval.log_action_stats", default=False)):
            count = int(getattr(self, "_dreamer_eval_action_log_count", 0))
            limit = int(OmegaConf.select(self.cfg, "eval.log_action_stats_limit", default=8))
            if count < limit:
                print(
                    "  [Eval][online-action] "
                    f"raw={np.array2string(raw_action_np, precision=4, suppress_small=False)} "
                    f"clipped={np.array2string(action_np, precision=4, suppress_small=False)} "
                    f"abs_mean={float(np.mean(np.abs(action_np))):.5f} "
                    f"max_abs={float(np.max(np.abs(action_np))):.5f} "
                    f"action_steps={int(action_steps)}",
                    flush=True,
                )
            self._dreamer_eval_action_log_count = count + 1
        return action_np.astype(np.float32, copy=False)

    def _dreamer_online_reset(self) -> None:
        self._dreamer_online_latent = None
        self._dreamer_online_prev_action = None

    def _dreamer_online_update_latent(self, obs_embedding: torch.Tensor) -> Any:
        if getattr(self, "_dreamer_online_latent", None) is None:
            latent = self.world_model({"mode": "encode_latent", "hidden": obs_embedding})
        else:
            prev_action = getattr(self, "_dreamer_online_prev_action", None)
            if not isinstance(prev_action, torch.Tensor):
                raise RuntimeError("online_rssm latent update missing previous executed action")
            latent = self.world_model({
                "mode": "observe_next",
                "latent": self._dreamer_online_latent,
                "hidden": obs_embedding,
                "actions": prev_action,
                "is_first": False,
            })
        self._dreamer_online_latent = latent
        return latent

    def _evaluate_libero_online_rssm(self, epoch: int) -> dict[str, float]:
        if not self.distributed.is_main_process:
            return {}
        if self.distributed.uses_fsdp:
            print("  [Eval] Skipping online_rssm eval under FSDP. Use scripts/eval_libero_vla.sh.")
            return {}

        from libero.libero import benchmark as libero_benchmark
        from src.env import (
            TASK_MAX_STEPS,
            get_libero_dummy_action,
            get_libero_env,
            get_libero_image,
            quat2axisangle,
            save_rollout_video,
        )

        eval_cfg = OmegaConf.select(self.cfg, "eval", default=None)
        task_suite_name = str(OmegaConf.select(eval_cfg, "task_suite_name", default="libero_goal"))
        num_episodes = int(OmegaConf.select(eval_cfg, "num_episodes_per_task", default=10))
        action_steps = int(OmegaConf.select(eval_cfg, "action_steps", default=5))
        resolution = int(OmegaConf.select(self.cfg, "encoder.resolution", default=256))
        history_length = int(OmegaConf.select(eval_cfg, "history_length", default=2))
        save_video = bool(OmegaConf.select(eval_cfg, "save_video", default=False))
        video_max_episodes = int(OmegaConf.select(eval_cfg, "video_max_episodes", default=1))
        video_dir = os.path.join(self.output_dir, "videos")

        item_processor = self.encoder._build_processor(self.device)
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        total_tasks = int(task_suite.n_tasks)
        task_ids_cfg = OmegaConf.select(eval_cfg, "task_ids", default=None)
        if task_ids_cfg is not None:
            task_ids = [int(task_id) for task_id in task_ids_cfg]
        else:
            task_start = int(OmegaConf.select(eval_cfg, "task_start", default=0))
            max_tasks = OmegaConf.select(eval_cfg, "max_tasks", default=None)
            task_stop = total_tasks if max_tasks is None else min(total_tasks, task_start + int(max_tasks))
            task_ids = list(range(task_start, task_stop))
        if not task_ids:
            raise ValueError("LIBERO eval selected no tasks; check eval.task_ids/task_start/max_tasks.")
        max_steps_cfg = OmegaConf.select(eval_cfg, "max_steps", default=None)
        max_steps = int(max_steps_cfg if max_steps_cfg is not None else TASK_MAX_STEPS.get(task_suite_name, 300))
        print(
            f"  [Eval][online_rssm] suite='{task_suite_name}' tasks={task_ids} "
            f"episodes_per_task={num_episodes} max_steps={max_steps} history_length={history_length}",
            flush=True,
        )

        self.encoder.eval()
        total_episodes, total_successes = 0, 0
        run_t0 = time.time()
        for task_index, task_id in enumerate(task_ids):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=resolution)
            n_eps = min(num_episodes, len(initial_states))
            print(
                f"  [Eval][online_rssm] >>> Task {task_id} ({task_index + 1}/{len(task_ids)}): "
                f"\"{task_description}\" episodes={n_eps}",
                flush=True,
            )
            task_successes = 0
            task_t0 = time.time()
            for episode_idx in range(n_eps):
                self._dreamer_online_reset()
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                done = False
                for _ in range(10):
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    if done:
                        break
                ep_t0 = time.time()
                frame_history: list[tuple[Image.Image, Image.Image]] = []
                should_record = save_video and total_episodes < video_max_episodes
                rollout_images: list[np.ndarray] = []
                steps_taken = 0

                for step_idx in range(max_steps):
                    img = get_libero_image(obs, resolution)
                    wrist_img = get_libero_image(obs, resolution, "robot0_eye_in_hand_image")
                    if should_record:
                        rollout_images.append(img)
                    state = np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    )
                    third_pil = Image.fromarray(img)
                    wrist_pil = Image.fromarray(wrist_img)
                    frame_history.append((third_pil, wrist_pil))
                    if len(frame_history) > history_length:
                        frame_history = frame_history[-history_length:]
                    padded = [frame_history[0]] * (history_length - len(frame_history)) + frame_history

                    self._libero_current_raw_obs = obs
                    obs_embedding, input_ids = self._dreamer_obs_embedding_from_eval_inputs(
                        item_processor,
                        padded,
                        state,
                        task_description,
                    )
                    with torch.no_grad():
                        latent = self._dreamer_online_update_latent(obs_embedding)
                        action = self._dreamer_action_from_latent(latent, input_ids=input_ids, action_steps=action_steps)
                    obs, _, done, _ = env.step(action.tolist())
                    self._dreamer_online_prev_action = torch.from_numpy(action).to(self.device).reshape(1, -1)
                    steps_taken = step_idx + 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                video_path = None
                if should_record and rollout_images:
                    video_path = save_rollout_video(video_dir, rollout_images, total_episodes, bool(done), task_description)
                total_episodes += 1
                ep_dt = time.time() - ep_t0
                tag = "OK " if done else "FAIL"
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"  [Eval][online_rssm]   ep {episode_idx + 1}/{n_eps} {tag} "
                    f"steps={steps_taken} time={ep_dt:5.1f}s "
                    f"task_succ={task_successes}/{episode_idx + 1} "
                    f"total_succ={total_successes}/{total_episodes}"
                    f"{' video=' + video_path if video_path else ''}",
                    flush=True,
                )
            env.close()
            rate = task_successes / max(n_eps, 1)
            print(
                f"  [Eval][online_rssm] <<< Task {task_id} done: success={rate:.1%} "
                f"({task_successes}/{n_eps}) time={time.time() - task_t0:.1f}s",
                flush=True,
            )

        avg_success = total_successes / max(total_episodes, 1)
        print(
            f"  [Eval][online_rssm] Epoch {epoch} overall success rate: {avg_success:.1%} "
            f"({total_successes}/{total_episodes}) total_time={time.time() - run_t0:.1f}s",
            flush=True,
        )
        return {
            "eval_success_rate": avg_success,
            "eval_total_episodes": float(total_episodes),
            "eval_total_successes": float(total_successes),
            "eval_dreamer_rollout_mode_online_rssm": 1.0,
        }

    def _generate_actions(
        self,
        backbone: Any,
        item_processor: Any,
        frame_history: list[tuple[Image.Image, Image.Image]],
        state: np.ndarray,
        task_description: str,
        action_steps: int,
    ) -> list[np.ndarray]:
        if not getattr(self, "_dreamer_eval", False):
            return super()._generate_actions(backbone, item_processor, frame_history, state, task_description, action_steps)

        with torch.no_grad():
            if self._wm_expects_pixel_images():
                obs_embedding = self._pixel_obs_for_wm(frame_history)
            else:
                img_c: list[Image.Image] = []
                for third_pil, wrist_pil in frame_history:
                    img_c.extend([third_pil, wrist_pil])
                human_val = f"Finish the task: {task_description}." + "<|state|>" + "<|image|>" * len(img_c)
                conv = {
                    "conversations": [{"from": "human", "value": human_val}],
                    "image": img_c,
                    "action": [],
                    "state": [state],
                }
                tokens = item_processor.process_item(conv, training_mode=False)
                if isinstance(tokens, tuple):
                    tokens = tokens[0]
                input_ids = [int(tok) for tok in tokens]
                obs_embedding = self._obs_embedding_for_wm([input_ids])
            actor_input_source = getattr(self, "_dreamer_actor_input_source", "rssm")
            if actor_input_source == "encoder_sequence":
                if self._wm_expects_pixel_images():
                    raise RuntimeError("eval.dreamer_actor_input_source=encoder_sequence requires tokenized VLA inputs")
                hidden_states, seq_input_ids, seq_attention_mask = self._encode_hidden_sequence_from_tokenized([input_ids])
                hidden_states = self._maybe_add_hidden_noise(hidden_states)
                action, _, _ = self.policy({
                    "mode": "sample",
                    "hidden_states": hidden_states,
                    "input_ids": seq_input_ids,
                    "attention_mask": seq_attention_mask,
                    "target_token_id": 10004,
                    "deterministic": bool(getattr(self, "_dreamer_deterministic", True)),
                    "return_chunk": True,
                })
                action_chunk = action.squeeze(0).detach().cpu().float().numpy()
                actions = self._unnorm_actions(action_chunk)
                if actions.ndim == 1:
                    actions = actions[None]
                if bool(OmegaConf.select(self.cfg, "eval.log_action_stats", default=False)):
                    print(
                        "  [Eval][action-seq] "
                        f"chunk_shape={tuple(actions.shape)} "
                        f"first={np.array2string(actions[0], precision=4, suppress_small=False)}",
                        flush=True,
                    )
                return [actions[i].astype(np.float32) for i in range(min(len(actions), int(action_steps)))]

            if actor_input_source == "encoder":
                if not hasattr(self.world_model, "encoder"):
                    raise RuntimeError("eval.dreamer_actor_input_source=encoder requires world_model.encoder")
                feat = self.world_model.encoder(obs_embedding)
                if feat.ndim == 3:
                    if feat.shape[1] != 1:
                        raise RuntimeError(
                            "eval.dreamer_actor_input_source=encoder expected a single observation embedding; "
                            f"got encoder output shape {tuple(feat.shape)}"
                        )
                    feat = feat[:, 0]
                feat = feat.float()
                feat = self._maybe_add_hidden_noise(feat)
            else:
                latent = self.world_model({"mode": "encode_latent", "hidden": obs_embedding})
                if hasattr(self.world_model, "actor_input"):
                    feat = self.world_model.actor_input(latent).float()
                else:
                    feat = latent.feature().float()
                feat = self._maybe_add_hidden_noise(feat)
            action, _, _ = self.policy({
                "mode": "sample",
                "hidden": feat,
                "deterministic": bool(getattr(self, "_dreamer_deterministic", True)),
            })
        action_np = action.squeeze(0).detach().cpu().float().numpy()
        if action_np.ndim > 1:
            action_np = action_np.reshape(-1, action_np.shape[-1])[0]
        action_np = action_np[:7]
        raw_action_np = action_np.copy()
        if bool(getattr(self, "_dreamer_clip_actions", True)):
            min_values = np.array([-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0])
            max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
            action_np = np.clip(action_np, min_values, max_values)
        if bool(OmegaConf.select(self.cfg, "eval.log_action_stats", default=False)):
            count = int(getattr(self, "_dreamer_eval_action_log_count", 0))
            limit = int(OmegaConf.select(self.cfg, "eval.log_action_stats_limit", default=8))
            if count < limit:
                print(
                    "  [Eval][action] "
                    f"raw={np.array2string(raw_action_np, precision=4, suppress_small=False)} "
                    f"clipped={np.array2string(action_np, precision=4, suppress_small=False)} "
                    f"abs_mean={float(np.mean(np.abs(action_np))):.5f} "
                    f"max_abs={float(np.max(np.abs(action_np))):.5f}",
                    flush=True,
                )
            self._dreamer_eval_action_log_count = count + 1
        repeat = min(int(getattr(self, "_dreamer_action_repeat", 1)), max(int(action_steps), 1))
        return [action_np.astype(np.float32) for _ in range(repeat)]


__all__ = ["EvalLiberoVLAWorkspace"]
