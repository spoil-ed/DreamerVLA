"""Eval-only runner: load a VLA/Dreamer checkpoint and run LIBERO rollouts.

No training, no optimizer, no dataset. Reuses the rollout logic that already
lives on ``PretokenizeVLARunner.evaluate_libero`` so there is exactly one
code path for LIBERO success-rate measurement.

Typical use:

  bash scripts/eval_libero_vla.sh \\
    eval.ckpt_path=data/outputs/vla/<run>/checkpoints/latest.ckpt \\
    eval.task_suite_name=libero_goal \\
    eval.num_episodes_per_task=10

LIBERO rollout is strictly single-process; the script enforces a single GPU
and this runner forces ``distributed_strategy=ddp`` so the encoder is not
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
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from transformers import GenerationConfig

from dreamervla.constants import DEFAULT_ACTION_TOKEN_ID
from dreamervla.runners import _embodied_eval_helpers as _eh
from dreamervla.runners._embodied_eval_action_mixin import EmbodiedEvalActionMixin
from dreamervla.runners._embodied_eval_export_mixin import EmbodiedEvalExportMixin
from dreamervla.runners._embodied_eval_image_token_mixin import EmbodiedEvalImageTokenMixin
from dreamervla.runners._embodied_eval_latent_mixin import EmbodiedEvalLatentMixin
from dreamervla.runners.pretokenize_vla_runner import PretokenizeVLARunner
from dreamervla.utils.hf_checkpoint import (
    is_hf_checkpoint,
    load_runner_payload,
    resolve_hf_checkpoint_dir,
)
from dreamervla.utils.paths import data_path
from dreamervla.utils.torch_utils import freeze_module


class _OFTBaseEvalAdapter:
    """Adapter that lets the existing LIBERO eval loop drive an OFT extractor."""

    def __init__(self, extractor: Any) -> None:
        self.extractor = extractor
        self.backbone = self

    def _build_processor(self, _device: torch.device) -> None:
        return None

    def eval(self) -> _OFTBaseEvalAdapter:
        return self


class EmbodiedEvalRunner(
    EmbodiedEvalExportMixin,
    EmbodiedEvalImageTokenMixin,
    EmbodiedEvalActionMixin,
    EmbodiedEvalLatentMixin,
    PretokenizeVLARunner,
):
    """Load a VLA or Dreamer ckpt -> run LIBERO rollout -> dump JSON metrics."""

    runner_name = "libero_eval"
    runner_status = "current"
    runner_family = "eval"

    @property
    def default_output_dir(self) -> str:
        return str(data_path("outputs", "eval", "eval_libero_vla"))

    @staticmethod
    def _oft_base_policy_cfg(cfg: DictConfig, ckpt_path: str) -> dict[str, Any]:
        return {
            "model_path": str(ckpt_path),
            "num_images_in_input": int(
                OmegaConf.select(cfg, "task.openvla_oft.num_images_in_input", default=1)
            ),
            "policy_mode": "auto",
            "unnorm_key": str(
                OmegaConf.select(
                    cfg,
                    "task.openvla_oft.dataset_statistics_key",
                    default="libero_goal_no_noops",
                )
            ),
            "expected_action_head_type": OmegaConf.select(
                cfg, "task.openvla_oft.expected_action_head_type", default=None
            ),
            "expected_include_state": OmegaConf.select(
                cfg, "task.openvla_oft.expected_include_state", default=None
            ),
            "_rank": 0,
        }

    @staticmethod
    def _use_oft_base_eval(
        cfg: DictConfig,
        *,
        ckpt_kind: str,
        ckpt_is_hf_vla: bool,
    ) -> bool:
        target = str(
            OmegaConf.select(cfg, "task.openvla_oft.sft_policy_target", default="")
        )
        return (
            str(ckpt_kind).lower() == "vla"
            and bool(ckpt_is_hf_vla)
            and target.endswith("OpenVLAOFTPolicy")
        )

    @staticmethod
    def _oft_base_eval_obs_from_libero_raw(
        raw_obs: dict[str, Any],
        state: np.ndarray,
    ) -> dict[str, Any]:
        third = raw_obs.get("agentview_rgb", raw_obs.get("agentview_image"))
        wrist = raw_obs.get(
            "eye_in_hand_rgb", raw_obs.get("robot0_eye_in_hand_image")
        )
        if third is None or wrist is None:
            raise KeyError("OFT base eval requires LIBERO agentview and wrist images")
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        return {
            "agentview_rgb": np.ascontiguousarray(np.asarray(third, dtype=np.uint8)),
            "eye_in_hand_rgb": np.ascontiguousarray(np.asarray(wrist, dtype=np.uint8)),
            "state": state_arr,
            "proprio": state_arr,
        }

    def _build_oft_base_eval_adapter(
        self,
        cfg: DictConfig,
        ckpt_path: str,
    ) -> _OFTBaseEvalAdapter:
        from dreamervla.workers.inference.oft_rollout import OFTRolloutBundle

        policy_cfg = self._oft_base_policy_cfg(cfg, ckpt_path)
        image_keys = list(
            OmegaConf.select(
                cfg,
                "task.image_keys",
                default=["agentview_rgb", "eye_in_hand_rgb"],
            )
        )
        bundle = OFTRolloutBundle(
            policy_cfg=policy_cfg,
            unnorm_key=str(policy_cfg["unnorm_key"]),
            image_keys=image_keys,
            history=int(
                OmegaConf.select(
                    cfg, "task.openvla_oft.input_tokens.expected_history", default=1
                )
            ),
            rotate_images_180=bool(
                OmegaConf.select(
                    cfg,
                    "task.openvla_oft.input_tokens.expected_rotate_images_180",
                    default=True,
                )
            ),
            center_crop=bool(
                OmegaConf.select(cfg, "task.openvla_oft.center_crop", default=True)
            ),
            obs_hidden_source=str(
                OmegaConf.select(
                    cfg,
                    "task.openvla_oft.input_tokens.expected_obs_hidden_source",
                    default="action_query",
                )
            ),
            expected_action_head_type=policy_cfg.get("expected_action_head_type"),
            expected_include_state=policy_cfg.get("expected_include_state"),
            device=str(self.device),
        )
        extractor = bundle.make_extractor()
        self._base_oft_extractor = extractor
        return _OFTBaseEvalAdapter(extractor)


    @property
    def _action_token_id(self) -> int:
        """Action-token id used for all token insertions (X-03; adjustable)."""
        return int(
            OmegaConf.select(
                self.cfg, "eval.target_token_id", default=DEFAULT_ACTION_TOKEN_ID
            )
        )

    def run(self) -> list[dict[str, Any]]:
        if self.distributed.is_main_process:
            print("EvalLiberoVLA Runner begin.")
        cfg = copy.deepcopy(self.cfg)

        if self.world_size != 1:
            raise RuntimeError(
                f"EmbodiedEvalRunner must run on a single process (got world_size={self.world_size}). "
                "Rollout evaluation does not support multi-process inference."
            )
        if self.distributed.uses_fsdp:
            raise RuntimeError(
                "EmbodiedEvalRunner requires DDP (not FSDP). "
                "Pass `training.distributed_strategy=ddp`."
            )

        ckpt_path = OmegaConf.select(cfg, "eval.ckpt_path", default=None)
        ckpt_path = (
            str(pathlib.Path(str(ckpt_path)).expanduser().resolve())
            if ckpt_path
            else None
        )
        payload = None
        ckpt_kind = str(OmegaConf.select(cfg, "eval.ckpt_kind", default="auto")).lower()
        if ckpt_kind not in {"auto", "vla", "dreamer"}:
            raise ValueError("eval.ckpt_kind must be one of: auto, vla, dreamer")
        ckpt_is_hf_vla = bool(ckpt_path and is_hf_checkpoint(ckpt_path))
        if ckpt_is_hf_vla and ckpt_kind == "dreamer":
            raise RuntimeError(
                f"{ckpt_path} is a Hugging Face VLA checkpoint, not a Dreamer checkpoint."
            )
        if ckpt_path and not ckpt_is_hf_vla and ckpt_kind in {"auto", "dreamer"}:
            payload = self._load_checkpoint_payload(ckpt_path)
            state_keys = set(payload.get("state_dicts", {}).keys())
            is_dreamer = {"world_model", "policy"}.issubset(state_keys)
            if ckpt_kind == "dreamer" and not is_dreamer:
                raise RuntimeError(
                    f"{ckpt_path} does not look like a Dreamer checkpoint: {sorted(state_keys)}"
                )
            if is_dreamer:
                return self._run_dreamer_eval(cfg, ckpt_path, payload)

        # ── encoder/policy (inference only; no optimiser, no distributed wrapping) ──
        if self._use_oft_base_eval(
            cfg,
            ckpt_kind=ckpt_kind,
            ckpt_is_hf_vla=ckpt_is_hf_vla,
        ):
            self.encoder = self._build_oft_base_eval_adapter(cfg, str(ckpt_path))
        else:
            encoder_cfg = self._build_trainable_encoder_cfg(cfg)
            if ckpt_is_hf_vla:
                with open_dict(encoder_cfg):
                    encoder_cfg.model_path = ckpt_path
            with open_dict(encoder_cfg):
                encoder_cfg.freeze_backbone = True
            self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
            self.encoder.eval()

        # ── optional: load VLA checkpoint (produced by PretokenizeVLARunner) ─
        if ckpt_path and not ckpt_is_hf_vla:
            if self.distributed.is_main_process:
                print(f"  [Eval] loading VLA checkpoint: {ckpt_path}")
            # Only restore the encoder; skip optimiser / EMA / step counters.
            # (The ckpt was produced by PretokenizeVLARunner which writes
            # vla_optimizer too, but that attribute is None here.)
            if payload is None:
                payload = self._load_checkpoint_payload(ckpt_path)
            self._normalize_vla_encoder_state_for_single_process_eval(payload)
            self.load_payload(
                payload,
                exclude_keys=("vla_optimizer", "vla_ema"),
                include_keys=(),  # don't restore global_step / epoch
            )
        elif ckpt_is_hf_vla:
            if self.distributed.is_main_process:
                print(f"  [Eval] loaded HF VLA checkpoint: {ckpt_path}")
        else:
            if self.distributed.is_main_process:
                print(
                    "  [Eval] no eval.ckpt_path set → evaluating init VLA weights "
                    f"({OmegaConf.select(cfg, 'init.vla_ckpt_path')})"
                )

        # ── rollout ──────────────────────────────────────────────────────────
        os.makedirs(self.output_dir, exist_ok=True)
        self._init_policy_trace(cfg)
        task_suite_name = str(
            OmegaConf.select(cfg, "eval.task_suite_name", default="libero_goal")
        )
        self.console_banner("EVALUATION", subtitle=f"suite={task_suite_name}")
        metrics = self.evaluate_libero(epoch=-1)
        eval_rate = float(metrics.get("eval_success_rate", 0.0))
        self.console_metrics(
            "eval",
            {
                "eval/success_rate": eval_rate,
                "eval/episodes": float(metrics.get("eval_total_episodes", 0.0)),
                "eval/successes": float(metrics.get("eval_total_successes", 0.0)),
            },
            force=True,
        )
        self.console_banner(
            "EVALUATION", done=True, subtitle=f"succ {eval_rate:.3f}"
        )

        # ── dump metrics ─────────────────────────────────────────────────────
        if self.distributed.is_main_process:
            metrics_out = {
                "ckpt_path": ckpt_path,
                "task_suite": task_suite_name,
                "num_episodes_per_task": int(
                    OmegaConf.select(cfg, "eval.num_episodes_per_task", default=50)
                ),
                "seed": int(
                    OmegaConf.select(
                        cfg,
                        "eval.seed",
                        default=OmegaConf.select(cfg, "seed", default=0),
                    )
                ),
                "num_steps_wait": int(
                    OmegaConf.select(cfg, "eval.num_steps_wait", default=10)
                ),
                "action_steps": int(
                    OmegaConf.select(cfg, "eval.action_steps", default=10)
                ),
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
            return load_runner_payload(ckpt_path)
        except TypeError:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    _normalize_vla_encoder_state_for_single_process_eval = staticmethod(
        _eh.normalize_vla_encoder_state_for_single_process_eval
    )
    _checkpoint_cfg_from_payload = staticmethod(_eh.checkpoint_cfg_from_payload)

    def _run_dreamer_eval(
        self,
        eval_cfg_root: DictConfig,
        ckpt_path: str,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.distributed.is_main_process:
            print(
                "  [Eval] detected Dreamer checkpoint; using world_model + policy rollout."
            )

        try:
            train_cfg = self._checkpoint_cfg_from_payload(payload)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{ckpt_path} has no saved cfg; cannot rebuild Dreamer modules."
            ) from exc
        with open_dict(train_cfg):
            train_cfg.eval = copy.deepcopy(eval_cfg_root.eval)
            if OmegaConf.select(train_cfg, "encoder", default=None) is None:
                train_cfg.encoder = copy.deepcopy(eval_cfg_root.encoder)
            # Dreamer checkpoints may carry a stale init/encoder path when the
            # training launch overrode it from the shell.  Let eval-time
            # overrides rebuild the frozen VLA backbone/action-head correctly.
            eval_vla_path = OmegaConf.select(
                eval_cfg_root, "init.vla_ckpt_path", default=None
            )
            if eval_vla_path is not None:
                train_cfg.init.vla_ckpt_path = eval_vla_path
                if OmegaConf.select(train_cfg, "encoder", default=None) is not None:
                    train_cfg.encoder.model_path = eval_vla_path
            eval_encoder_ckpt = OmegaConf.select(
                eval_cfg_root, "init.encoder_state_ckpt", default=None
            )
            if eval_encoder_ckpt is not None:
                train_cfg.init.encoder_state_ckpt = eval_encoder_ckpt
            eval_horizon = OmegaConf.select(
                eval_cfg_root, "encoder.time_horizon", default=None
            )
            if (
                eval_horizon is not None
                and OmegaConf.select(train_cfg, "encoder", default=None) is not None
            ):
                train_cfg.encoder.time_horizon = eval_horizon
            train_cfg.training.out_dir = self.output_dir
            train_cfg.training.distributed_strategy = "ddp"
            train_cfg.training.enable_activation_checkpointing = False
            train_cfg.trainer.device = str(eval_cfg_root.trainer.device)
        self.cfg = train_cfg
        self.config = train_cfg

        self._dreamer_eval = True
        self._dreamer_deterministic = bool(
            OmegaConf.select(train_cfg, "eval.dreamer_deterministic", default=True)
        )
        self._dreamer_action_repeat = max(
            1, int(OmegaConf.select(train_cfg, "eval.dreamer_action_repeat", default=1))
        )
        self._dreamer_clip_actions = bool(
            OmegaConf.select(train_cfg, "eval.dreamer_clip_actions", default=True)
        )
        self._dreamer_rollout_mode = str(
            OmegaConf.select(
                train_cfg, "eval.dreamer_rollout_mode", default="stateless"
            )
        ).lower()
        if self._dreamer_rollout_mode not in {"stateless", "online_rssm"}:
            raise ValueError(
                "eval.dreamer_rollout_mode must be one of: stateless, online_rssm"
            )
        self._dreamer_actor_input_source = str(
            OmegaConf.select(
                train_cfg, "eval.dreamer_actor_input_source", default="rssm"
            )
        ).lower()
        if self._dreamer_actor_input_source not in {
            "rssm",
            "encoder",
            "encoder_sequence",
        }:
            raise ValueError(
                "eval.dreamer_actor_input_source must be one of: rssm, encoder, encoder_sequence"
            )
        self._dreamer_policy_source = str(
            OmegaConf.select(train_cfg, "eval.dreamer_policy_source", default="ckpt")
        ).lower()
        if self._dreamer_policy_source not in {"ckpt", "init"}:
            raise ValueError("eval.dreamer_policy_source must be one of: ckpt, init")
        self._tdmpc_mpc_enabled = bool(
            OmegaConf.select(train_cfg, "eval.tdmpc_mpc.enabled", default=False)
        )
        self._tdmpc_mpc_use_target_critic = bool(
            OmegaConf.select(
                train_cfg, "eval.tdmpc_mpc.use_target_critic", default=True
            )
        )
        self._tdmpc_mpc_planner = (
            self._build_tdmpc_mpc_planner(train_cfg)
            if self._tdmpc_mpc_enabled
            else None
        )
        self._hidden_noise_std = float(
            OmegaConf.select(train_cfg, "eval.hidden_noise_std", default=0.0)
        )
        self._hidden_noise_seed = int(
            OmegaConf.select(train_cfg, "eval.hidden_noise_seed", default=0)
        )
        self._hidden_noise_generator = torch.Generator(device=self.device)
        self._hidden_noise_generator.manual_seed(self._hidden_noise_seed)
        self._hidden_noise_mse_sum = 0.0
        self._hidden_noise_cosine_sum = 0.0
        self._hidden_noise_count = 0
        self._hidden_action_compare_enabled = bool(
            OmegaConf.select(train_cfg, "eval.log_hidden_action_compare", default=False)
        )
        self._hidden_action_compare_limit = int(
            OmegaConf.select(train_cfg, "eval.hidden_action_compare_limit", default=300)
        )
        self._hidden_action_compare_unnorm = bool(
            OmegaConf.select(
                train_cfg,
                "eval.hidden_action_compare_unnorm_policy_outputs",
                default=True,
            )
        )
        self._hidden_action_compare_count = 0
        self._hidden_action_compare_sums: dict[str, float] = {}
        self._hidden_action_compare_path = os.path.join(
            self.output_dir, "hidden_action_compare.jsonl"
        )
        self._hidden_action_compare_summary_path = os.path.join(
            self.output_dir, "hidden_action_compare_summary.json"
        )
        if self._hidden_action_compare_enabled and self.distributed.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self._hidden_action_compare_path, "w"):
                pass
        self._init_policy_trace(train_cfg)
        self._init_real_relabel_export(train_cfg)

        self._build_dreamer_modules(train_cfg, payload)
        os.makedirs(self.output_dir, exist_ok=True)
        dreamer_suite_name = str(
            OmegaConf.select(train_cfg, "eval.task_suite_name", default="libero_goal")
        )
        self.console_banner(
            "EVALUATION",
            subtitle=f"dreamer suite={dreamer_suite_name}",
        )
        metrics = self.evaluate_libero(epoch=-1)
        dreamer_eval_rate = float(metrics.get("eval_success_rate", 0.0))
        self.console_metrics(
            "eval",
            {
                "eval/success_rate": dreamer_eval_rate,
                "eval/episodes": float(metrics.get("eval_total_episodes", 0.0)),
                "eval/successes": float(metrics.get("eval_total_successes", 0.0)),
            },
            force=True,
        )
        self.console_banner(
            "EVALUATION", done=True, subtitle=f"succ {dreamer_eval_rate:.3f}"
        )
        if bool(getattr(self, "_real_relabel_enabled", False)):
            self._write_real_relabel_summary()
            metrics.update(
                {
                    "real_relabel_num_records": float(
                        len(getattr(self, "_real_relabel_records", []))
                    ),
                    "real_relabel_success_rate": float(
                        getattr(self, "_real_relabel_success_rate", 0.0)
                    ),
                }
            )
        if self._hidden_noise_count > 0:
            metrics = dict(metrics)
            metrics["hidden_noise_std"] = float(self._hidden_noise_std)
            metrics["hidden_noise_seed"] = int(self._hidden_noise_seed)
            metrics["hidden_noise_mean_mse"] = float(
                self._hidden_noise_mse_sum / self._hidden_noise_count
            )
            metrics["hidden_noise_mean_cosine_loss"] = float(
                self._hidden_noise_cosine_sum / self._hidden_noise_count
            )
            metrics["hidden_noise_count"] = int(self._hidden_noise_count)
        if int(getattr(self, "_hidden_action_compare_count", 0)) > 0:
            compare_summary = self._hidden_action_compare_summary()
            metrics = dict(metrics)
            metrics.update(
                {
                    f"hidden_action_compare_{key}": value
                    for key, value in compare_summary.items()
                }
            )
            if self.distributed.is_main_process:
                with open(self._hidden_action_compare_summary_path, "w") as f:
                    json.dump(compare_summary, f, indent=2)
                print(
                    f"  [Eval] wrote hidden/action compare summary -> {self._hidden_action_compare_summary_path}"
                )

        if self.distributed.is_main_process:
            metrics_out = {
                "ckpt_path": ckpt_path,
                "ckpt_kind": "dreamer",
                "task_suite": str(
                    OmegaConf.select(
                        train_cfg, "eval.task_suite_name", default="libero_goal"
                    )
                ),
                "num_episodes_per_task": int(
                    OmegaConf.select(
                        train_cfg, "eval.num_episodes_per_task", default=50
                    )
                ),
                "seed": int(
                    OmegaConf.select(
                        train_cfg,
                        "eval.seed",
                        default=OmegaConf.select(train_cfg, "seed", default=0),
                    )
                ),
                "num_steps_wait": int(
                    OmegaConf.select(train_cfg, "eval.num_steps_wait", default=10)
                ),
                "action_steps": int(
                    OmegaConf.select(train_cfg, "eval.action_steps", default=10)
                ),
                "dreamer_action_repeat": int(self._dreamer_action_repeat),
                "dreamer_deterministic": bool(self._dreamer_deterministic),
                "dreamer_clip_actions": bool(self._dreamer_clip_actions),
                "dreamer_unnorm_actions": bool(self._dreamer_should_unnorm_actions()),
                "dreamer_rssm_action_source": str(
                    OmegaConf.select(
                        train_cfg, "eval.dreamer_rssm_action_source", default="env"
                    )
                ),
                "dreamer_rollout_mode": str(self._dreamer_rollout_mode),
                "dreamer_actor_input_source": str(self._dreamer_actor_input_source),
                "dreamer_policy_source": str(self._dreamer_policy_source),
                "tdmpc_mpc_enabled": bool(getattr(self, "_tdmpc_mpc_enabled", False)),
                "dreamer_wm_history_length": int(
                    OmegaConf.select(
                        train_cfg, "eval.dreamer_wm_history_length", default=1
                    )
                ),
                "dreamer_wm_rotate_images": bool(
                    OmegaConf.select(
                        train_cfg, "eval.dreamer_wm_rotate_images", default=False
                    )
                ),
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






















    def _build_dreamer_modules(self, cfg: DictConfig, payload: dict[str, Any]) -> None:
        state_dicts = payload.get("state_dicts", {})

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
        if "encoder" in state_dicts:
            self._load_module_state(self.encoder, state_dicts["encoder"], "encoder")
        else:
            if encoder_init_ckpt and not is_hf_checkpoint(encoder_init_ckpt):
                encoder_payload = self._load_checkpoint_payload(str(encoder_init_ckpt))
                encoder_sd = encoder_payload.get("state_dicts", {}).get("encoder")
                if encoder_sd is None:
                    raise RuntimeError(
                        f"{encoder_init_ckpt} has no state_dicts.encoder"
                    )
                self._load_module_state(self.encoder, encoder_sd, "encoder")
                del encoder_payload
        self.encoder.eval()

        world_model_cfg = OmegaConf.select(cfg, "world_model")
        if world_model_cfg is None:
            raise ValueError("Dreamer eval requires `world_model` in the saved cfg.")
        instantiate_kwargs: dict[str, Any] = {}
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
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        self.world_model = self.world_model.to(
            dtype=dtype_map.get(fsdp_precision, torch.bfloat16)
        )
        self._unwrapped_world_model = self.world_model
        self._attach_image_token_mapping()
        self._load_module_state(
            self.world_model, state_dicts["world_model"], "world_model"
        )
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
                print(
                    "  [Eval] policy source=ckpt; skipped action_head warm-start during policy init."
                )
        self.policy = hydra.utils.instantiate(policy_cfg).to(self.device)
        if getattr(self, "_dreamer_policy_source", "ckpt") == "ckpt":
            self._load_module_state(self.policy, state_dicts["policy"], "policy")
        elif self.distributed.is_main_process:
            print(
                "  [Eval] using init policy/action_head; skipped Dreamer checkpoint policy state."
            )
        self.policy.eval()

        self.target_critic = None
        if bool(getattr(self, "_tdmpc_mpc_enabled", False)) and bool(
            getattr(self, "_tdmpc_mpc_use_target_critic", True)
        ):
            critic_state = state_dicts.get("target_critic") or state_dicts.get("critic")
            critic_cfg = OmegaConf.select(cfg, "critic")
            if critic_cfg is None or critic_state is None:
                if self.distributed.is_main_process:
                    print(
                        "  [Eval][tdmpc-mpc] target critic unavailable; using reward-only MPC."
                    )
            else:
                planner_value_mode = str(
                    OmegaConf.select(cfg, "eval.tdmpc_mpc.value_mode", default="state")
                ).lower()
                if planner_value_mode in {"state_action", "q", "q_za", "q(z,a)"}:
                    critic_cfg = OmegaConf.create(
                        OmegaConf.to_container(critic_cfg, resolve=True)
                    )
                    critic_action_dim = int(
                        OmegaConf.select(
                            cfg,
                            "eval.tdmpc_mpc.action_dim",
                            default=OmegaConf.select(
                                cfg, "algorithm.tdmpc_ac.action_dim", default=7
                            ),
                        )
                    )
                    critic_cfg.hidden_dim = (
                        int(critic_cfg.hidden_dim) + critic_action_dim
                    )
                self.target_critic = hydra.utils.instantiate(critic_cfg).to(self.device)
                self._load_module_state(
                    self.target_critic, critic_state, "target_critic"
                )
                freeze_module(self.target_critic)
                self.target_critic.eval()

        # Drop optimizer/critic tensors as soon as possible after optional MPC critic load.
        for key in (
            "policy_optimizer",
            "critic_optimizer",
            "world_model_optimizer",
            "critic",
            "target_critic",
        ):
            state_dicts.pop(key, None)
        gc.collect()

    def _load_module_state(
        self, module: Any, state_dict: dict[str, Any], name: str
    ) -> None:
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
                if key.startswith("reward_head.net.") and not key.startswith(
                    "reward_head.net.net."
                ):
                    candidate = key.replace(
                        "reward_head.net.", "reward_head.net.net.", 1
                    )
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

    _strip_wrapping_prefix = staticmethod(_eh.strip_wrapping_prefix)













    _resize_hwc_uint8 = staticmethod(_eh.resize_hwc_uint8)








    def _evaluate_libero_online_rssm(self, epoch: int) -> dict[str, float]:
        if not self.distributed.is_main_process:
            return {}
        if self.distributed.uses_fsdp:
            print(
                "  [Eval] Skipping online_rssm eval under FSDP. Use scripts/eval_libero_vla.sh."
            )
            return {}

        from libero.libero import benchmark as libero_benchmark

        from dreamervla.envs import (
            TASK_MAX_STEPS,
            get_libero_dummy_action,
            get_libero_env,
            get_libero_image,
            quat2axisangle,
            resolve_libero_eval_protocol,
            save_rollout_video,
        )

        eval_cfg = OmegaConf.select(self.cfg, "eval", default=None)
        protocol = resolve_libero_eval_protocol(self.cfg, eval_cfg)
        seed = int(protocol["seed"])
        num_steps_wait = int(protocol["num_steps_wait"])
        np.random.seed(seed)
        task_suite_name = str(
            OmegaConf.select(eval_cfg, "task_suite_name", default="libero_goal")
        )
        num_episodes = int(
            OmegaConf.select(eval_cfg, "num_episodes_per_task", default=50)
        )
        action_steps = int(OmegaConf.select(eval_cfg, "action_steps", default=5))
        resolution = int(OmegaConf.select(self.cfg, "encoder.resolution", default=256))
        history_length = int(OmegaConf.select(eval_cfg, "history_length", default=2))
        save_video = bool(OmegaConf.select(eval_cfg, "save_video", default=False))
        video_max_episodes = int(
            OmegaConf.select(eval_cfg, "video_max_episodes", default=1)
        )
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
            task_stop = (
                total_tasks
                if max_tasks is None
                else min(total_tasks, task_start + int(max_tasks))
            )
            task_ids = list(range(task_start, task_stop))
        if not task_ids:
            raise ValueError(
                "LIBERO eval selected no tasks; check eval.task_ids/task_start/max_tasks."
            )
        max_steps_cfg = OmegaConf.select(eval_cfg, "max_steps", default=None)
        max_steps = int(
            max_steps_cfg
            if max_steps_cfg is not None
            else TASK_MAX_STEPS.get(task_suite_name, 300)
        )
        print(
            f"  [Eval][online_rssm] suite='{task_suite_name}' tasks={task_ids} "
            f"episodes_per_task={num_episodes} max_steps={max_steps} history_length={history_length} "
            f"seed={seed} num_steps_wait={num_steps_wait}",
            flush=True,
        )

        self.encoder.eval()
        total_episodes, total_successes = 0, 0
        run_t0 = time.time()
        for task_index, task_id in enumerate(task_ids):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(
                task, resolution=resolution, seed=seed
            )
            n_eps = num_episodes
            print(
                f"  [Eval][online_rssm] >>> Task {task_id} ({task_index + 1}/{len(task_ids)}): "
                f'"{task_description}" episodes={n_eps}',
                flush=True,
            )
            task_successes = 0
            task_t0 = time.time()
            for episode_idx in range(n_eps):
                self._dreamer_online_reset()
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                done = False
                for _ in range(num_steps_wait):
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                ep_t0 = time.time()
                frame_history: list[tuple[Image.Image, Image.Image]] = []
                env_actions_buffer: list[np.ndarray] = []
                rssm_actions_buffer: list[np.ndarray] = []
                should_record = save_video and total_episodes < video_max_episodes
                rollout_images: list[np.ndarray] = []
                steps_taken = 0
                wm_reward_trace: list[float] = []
                action_norm_trace: list[float] = []
                actor_input_trace: list[list[float]] = []
                raw_action_trace: list[list[float]] = []
                old_log_prob_trace: list[float] = []
                actor_step_index_trace: list[int] = []

                for step_idx in range(max_steps):
                    img = get_libero_image(obs, resolution)
                    wrist_img = get_libero_image(
                        obs, resolution, "robot0_eye_in_hand_image"
                    )
                    if should_record:
                        rollout_images.append(img)
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    third_pil = Image.fromarray(img)
                    wrist_pil = Image.fromarray(wrist_img)
                    frame_history.append((third_pil, wrist_pil))
                    if len(frame_history) > history_length:
                        frame_history = frame_history[-history_length:]
                    padded = [frame_history[0]] * (
                        history_length - len(frame_history)
                    ) + frame_history

                    self._libero_current_raw_obs = obs
                    obs_embedding, input_ids = (
                        self._dreamer_obs_embedding_from_eval_inputs(
                            item_processor,
                            padded,
                            state,
                            task_description,
                        )
                    )
                    with torch.no_grad():
                        latent = self._dreamer_online_update_latent(obs_embedding)
                        if bool(getattr(self, "_real_relabel_enabled", False)):
                            try:
                                reward_pred = self.world_model(
                                    {"mode": "reward", "latent": latent}
                                )
                                wm_reward_trace.append(
                                    float(
                                        reward_pred.detach()
                                        .float()
                                        .reshape(-1)[0]
                                        .cpu()
                                    )
                                )
                            except Exception:
                                wm_reward_trace.append(float("nan"))
                        self._libero_current_eval_context = {
                            "task_id": int(task_id),
                            "task_index": int(task_index),
                            "episode_idx": int(episode_idx),
                            "env_step": int(step_idx),
                            "rollout_t": int(step_idx),
                            "task_description": str(task_description),
                        }
                        self._libero_current_eval_context_state = state
                        if not env_actions_buffer:
                            if bool(getattr(self, "_tdmpc_mpc_enabled", False)):
                                env_actions_buffer, rssm_actions_buffer = (
                                    self._tdmpc_mpc_action_chunk_from_latent(
                                        latent,
                                        action_steps=action_steps,
                                    )
                                )
                            else:
                                env_actions_buffer, rssm_actions_buffer = (
                                    self._dreamer_action_chunk_from_latent(
                                        latent,
                                        input_ids=input_ids,
                                        action_steps=action_steps,
                                        live_hidden=obs_embedding,
                                    )
                                )
                            if bool(getattr(self, "_real_relabel_enabled", False)):
                                trace_item = getattr(
                                    self, "_last_real_relabel_actor_step", None
                                )
                                if isinstance(trace_item, dict):
                                    actor_input = trace_item.get("actor_input")
                                    raw_action = trace_item.get("raw_action")
                                    old_log_prob = trace_item.get("old_log_prob")
                                    if isinstance(actor_input, list) and isinstance(
                                        raw_action, list
                                    ):
                                        actor_input_trace.append(actor_input)
                                        raw_action_trace.append(raw_action)
                                        old_log_prob_trace.append(float(old_log_prob))
                                        actor_step_index_trace.append(int(step_idx))
                    if not env_actions_buffer:
                        break
                    action = env_actions_buffer.pop(0)
                    rssm_action = (
                        rssm_actions_buffer.pop(0) if rssm_actions_buffer else action
                    )
                    if bool(
                        OmegaConf.select(
                            self.cfg, "eval.empty_cuda_cache_each_step", default=False
                        )
                    ):
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    obs, _, done, _ = env.step(action.tolist())
                    if bool(getattr(self, "_real_relabel_enabled", False)):
                        action_norm_trace.append(
                            float(np.linalg.norm(np.asarray(action, dtype=np.float32)))
                        )
                    self._dreamer_online_prev_action = (
                        torch.from_numpy(rssm_action).to(self.device).reshape(1, -1)
                    )
                    steps_taken = step_idx + 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                video_path = None
                if should_record and rollout_images:
                    video_path = save_rollout_video(
                        video_dir,
                        rollout_images,
                        total_episodes,
                        bool(done),
                        task_description,
                    )
                total_episodes += 1
                self.console_record_success(bool(done))
                self.console_progress(
                    total_episodes, len(task_ids) * num_episodes, "eval"
                )
                if bool(getattr(self, "_real_relabel_enabled", False)):
                    finite_rewards = [
                        float(x) for x in wm_reward_trace if np.isfinite(float(x))
                    ]
                    policy_mode = (
                        "deterministic"
                        if bool(getattr(self, "_dreamer_deterministic", True))
                        else "sample"
                    )
                    prompt_key = (
                        f"task{int(task_id):02d}_ep{int(episode_idx):03d}_{policy_mode}"
                    )
                    trajectory_id = f"{prompt_key}_sample000"
                    first_ge_08 = next(
                        (
                            idx
                            for idx, value in enumerate(wm_reward_trace)
                            if np.isfinite(float(value)) and float(value) >= 0.8
                        ),
                        -1,
                    )
                    relabel_record = {
                        "trajectory_id": trajectory_id,
                        "prompt_key": prompt_key,
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "sample_idx": 0,
                        "policy_mode": policy_mode,
                        "complete": bool(done),
                        "acc": float(bool(done)),
                        "finish_step": int(steps_taken),
                        "max_steps": int(max_steps),
                        "valid_action_tokens": int(steps_taken * 7),
                        "real_sparse_rewards": self._real_relabel_sparse_rewards(
                            bool(done), int(steps_taken), int(max_steps)
                        ),
                        "reward_relabel": {
                            "type": "terminal_outcome",
                            "positive_step": int(steps_taken - 1) if bool(done) else -1,
                            "target_return": float(bool(done)),
                        },
                        "wm_reward_pred": {
                            "mean": float(np.mean(finite_rewards))
                            if finite_rewards
                            else float("nan"),
                            "max": float(np.max(finite_rewards))
                            if finite_rewards
                            else float("nan"),
                            "last": float(finite_rewards[-1])
                            if finite_rewards
                            else float("nan"),
                            "first_ge_0p8_step": int(first_ge_08),
                            "trace": wm_reward_trace,
                        },
                        "action_norm_mean": float(np.mean(action_norm_trace))
                        if action_norm_trace
                        else float("nan"),
                        "actor_inputs": actor_input_trace,
                        "raw_actions": raw_action_trace,
                        "old_log_probs": old_log_prob_trace,
                        "actor_step_indices": actor_step_index_trace,
                    }
                    self._append_real_relabel_record(relabel_record)
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
            "results/total_success_rate": avg_success,
            "results/total_episodes": float(total_episodes),
            "results/total_successes": float(total_successes),
            "eval_dreamer_rollout_mode_online_rssm": 1.0,
        }

    def _generate_vla_actions_with_trace(
        self,
        backbone: Any,
        item_processor: Any,
        frame_history: list[tuple[Image.Image, Image.Image]],
        state: np.ndarray,
        task_description: str,
        action_steps: int,
    ) -> list[np.ndarray]:
        img_c: list[Image.Image] = []
        for third_pil, wrist_pil in frame_history:
            img_c.extend([third_pil, wrist_pil])
        human_val = (
            f"Finish the task: {task_description}."
            + "<|state|>"
            + "<|image|>" * len(img_c)
        )
        conv = {
            "conversations": [{"from": "human", "value": human_val}],
            "image": img_c,
            "action": [],
            "state": [state],
        }
        tokens = item_processor.process_item(conv, training_mode=False)
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        tokens = [int(tok) for tok in tokens]
        input_ids = torch.tensor(
            tokens, dtype=torch.int64, device=self.device
        ).unsqueeze(0)

        generation_config = GenerationConfig(
            max_new_tokens=1,
            max_length=backbone.config.max_position_embeddings,
            temperature=1,
            top_k=None,
            do_sample=False,
            eos_token_id=[8710],
        )
        if not hasattr(backbone, "generate_action_head"):
            return super()._generate_actions(
                backbone,
                item_processor,
                frame_history,
                state,
                task_description,
                action_steps,
            )

        try:
            predicted = backbone.generate_action_head(input_ids, generation_config)
            action_chunk_raw = predicted.detach().cpu().float().numpy()
            if action_chunk_raw.ndim == 1:
                action_chunk_raw = action_chunk_raw.reshape(1, -1)
            else:
                action_chunk_raw = action_chunk_raw.reshape(
                    -1, action_chunk_raw.shape[-1]
                )
            action_chunk_env = self._unnorm_actions(action_chunk_raw)

            action_hidden = None
            wm_style_action_hidden = None
            if self._use_action_query_obs_hidden():
                hidden_states, seq_input_ids, seq_attention_mask = (
                    self._encode_hidden_sequence_from_tokenized([tokens])
                )
                action_hidden = self.encoder.extract_action_hidden(
                    hidden_states=hidden_states,
                    input_ids=seq_input_ids,
                    attention_mask=seq_attention_mask,
                    target_token_id=self._action_token_id,
                    eval=True,
                )
                try:
                    wm_frame_history = self._dreamer_wm_frame_history(frame_history)
                    wm_tokens = self._dreamer_wm_observation_input_ids(
                        item_processor=item_processor,
                        frame_history=wm_frame_history,
                        state=state,
                        task_description=task_description,
                    )
                    wm_hidden = self._obs_embedding_for_wm([wm_tokens])
                    wm_style_action_hidden = self._action_hidden_tokens_for_trace(
                        wm_hidden
                    )
                except Exception as exc:
                    if bool(
                        OmegaConf.select(
                            self.cfg, "eval.trace_policy_debug_verbose", default=False
                        )
                    ):
                        print(
                            f"  [Eval][trace] failed to compute wm_style_action_hidden: {exc}",
                            flush=True,
                        )

            self._write_policy_trace(
                source="vla",
                state=state,
                action_chunk_raw=action_chunk_raw,
                action_chunk_env=action_chunk_env,
                action_hidden=action_hidden,
                wm_style_action_hidden=wm_style_action_hidden,
                obs_embedding=wm_style_action_hidden,
                input_ids=np.asarray(tokens, dtype=np.float32),
            )
            return [
                action_chunk_env[i].astype(np.float32)
                for i in range(min(len(action_chunk_env), int(action_steps)))
            ]
        except Exception as exc:
            print(f"  [Eval] generate_action_head failed: {exc}", flush=True)
            return super()._generate_actions(
                backbone,
                item_processor,
                frame_history,
                state,
                task_description,
                action_steps,
            )

    def _generate_actions(
        self,
        backbone: Any,
        item_processor: Any,
        frame_history: list[tuple[Image.Image, Image.Image]],
        state: np.ndarray,
        task_description: str,
        action_steps: int,
    ) -> list[np.ndarray]:
        oft_extractor = getattr(self, "_base_oft_extractor", None)
        if oft_extractor is not None:
            raw_obs = getattr(self, "_libero_current_raw_obs", None)
            if not isinstance(raw_obs, dict):
                raise RuntimeError("OFT base eval requires current LIBERO raw obs")
            context = getattr(self, "_libero_current_eval_context", {}) or {}
            if int(context.get("env_step", 0)) == 0 and hasattr(
                oft_extractor, "reset"
            ):
                oft_extractor.reset()
            obs = self._oft_base_eval_obs_from_libero_raw(raw_obs, state)
            decoded = oft_extractor.step(obs, task_description)
            action_chunk = (
                decoded.action_chunk if hasattr(decoded, "action_chunk") else decoded[0]
            )
            from dreamervla.runners.oft_collect_common import process_action

            return [
                process_action(action).astype(np.float32, copy=False)
                for action in list(action_chunk)[: int(action_steps)]
            ]

        if not getattr(self, "_dreamer_eval", False):
            if bool(getattr(self, "_policy_trace_enabled", False)):
                return self._generate_vla_actions_with_trace(
                    backbone,
                    item_processor,
                    frame_history,
                    state,
                    task_description,
                    action_steps,
                )
            return super()._generate_actions(
                backbone,
                item_processor,
                frame_history,
                state,
                task_description,
                action_steps,
            )

        with torch.no_grad():
            if self._wm_expects_pixel_images():
                obs_embedding = self._pixel_obs_for_wm(frame_history)
            else:
                wm_frame_history = self._dreamer_wm_frame_history(frame_history)
                input_ids = self._dreamer_wm_observation_input_ids(
                    item_processor=item_processor,
                    frame_history=wm_frame_history,
                    state=state,
                    task_description=task_description,
                )
                obs_embedding = self._obs_embedding_for_wm([input_ids])
            actor_input_source = getattr(self, "_dreamer_actor_input_source", "rssm")
            if actor_input_source == "encoder_sequence":
                if self._wm_expects_pixel_images():
                    raise RuntimeError(
                        "eval.dreamer_actor_input_source=encoder_sequence requires tokenized VLA inputs"
                    )
                hidden_states, seq_input_ids, seq_attention_mask = (
                    self._encode_hidden_sequence_from_tokenized([input_ids])
                )
                hidden_states = self._maybe_add_hidden_noise(hidden_states)
                action, _, _ = self.policy(
                    {
                        "mode": "sample",
                        "hidden_states": hidden_states,
                        "input_ids": seq_input_ids,
                        "attention_mask": seq_attention_mask,
                        "target_token_id": self._action_token_id,
                        "deterministic": bool(
                            getattr(self, "_dreamer_deterministic", True)
                        ),
                        "return_chunk": True,
                    }
                )
                action_chunk = action.squeeze(0).detach().cpu().float().numpy()
                actions = self._unnorm_actions(action_chunk)
                if actions.ndim == 1:
                    actions = actions[None]
                if bool(
                    OmegaConf.select(self.cfg, "eval.log_action_stats", default=False)
                ):
                    print(
                        "  [Eval][action-seq] "
                        f"chunk_shape={tuple(actions.shape)} "
                        f"first={np.array2string(actions[0], precision=4, suppress_small=False)}",
                        flush=True,
                    )
                return [
                    actions[i].astype(np.float32)
                    for i in range(min(len(actions), int(action_steps)))
                ]

            if actor_input_source == "encoder":
                if not hasattr(self.world_model, "encoder"):
                    raise RuntimeError(
                        "eval.dreamer_actor_input_source=encoder requires world_model.encoder"
                    )
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
                latent = self.world_model(
                    {"mode": "encode_latent", "hidden": obs_embedding}
                )
                if bool(getattr(self, "_tdmpc_mpc_enabled", False)):
                    env_actions, _rssm_actions = (
                        self._tdmpc_mpc_action_chunk_from_latent(
                            latent,
                            action_steps=action_steps,
                        )
                    )
                    return env_actions
                if hasattr(self.world_model, "actor_input"):
                    feat = self.world_model.actor_input(latent).float()
                else:
                    feat = latent.feature().float()
                feat = self._maybe_add_hidden_noise(feat)
            action, _, _ = self.policy(
                {
                    "mode": "sample",
                    "hidden": feat,
                    "deterministic": bool(
                        getattr(self, "_dreamer_deterministic", True)
                    ),
                    "return_chunk": True,
                }
            )
        action_chunk_np = action.squeeze(0).detach().cpu().float().numpy()
        if action_chunk_np.ndim == 1:
            action_chunk_np = action_chunk_np.reshape(1, -1)
        else:
            action_chunk_np = action_chunk_np.reshape(-1, action_chunk_np.shape[-1])
        raw_action_np = np.asarray(action_chunk_np[0, :7], dtype=np.float32).copy()
        action_np = self._dreamer_policy_raw_to_env_action(raw_action_np)
        self._record_hidden_action_compare(
            live_hidden=obs_embedding if actor_input_source == "rssm" else None,
            recon_hidden=feat if actor_input_source == "rssm" else None,
            recon_action_raw=raw_action_np,
            executed_action=action_np,
            context=getattr(self, "_libero_current_eval_context", None),
            source="stateless",
        )
        if bool(OmegaConf.select(self.cfg, "eval.log_action_stats", default=False)):
            count = int(getattr(self, "_dreamer_eval_action_log_count", 0))
            limit = int(
                OmegaConf.select(self.cfg, "eval.log_action_stats_limit", default=8)
            )
            if count < limit:
                print(
                    "  [Eval][action] "
                    f"raw={np.array2string(raw_action_np, precision=4, suppress_small=False)} "
                    f"env={np.array2string(action_np, precision=4, suppress_small=False)} "
                    f"abs_mean={float(np.mean(np.abs(action_np))):.5f} "
                    f"max_abs={float(np.max(np.abs(action_np))):.5f}",
                    flush=True,
                )
            self._dreamer_eval_action_log_count = count + 1
        env_actions = [
            self._dreamer_policy_raw_to_env_action(
                np.asarray(row[:7], dtype=np.float32)
            ).astype(np.float32)
            for row in action_chunk_np[: max(int(action_steps), 1)]
        ]
        if not env_actions:
            return []
        live_hidden = None
        recon_hidden = None
        if actor_input_source == "rssm":
            live_hidden = self._action_hidden_tokens_for_trace(obs_embedding)
            recon_hidden = self._action_hidden_tokens_for_trace(feat)
        self._write_policy_trace(
            source="dreamer",
            state=state,
            action_chunk_raw=action_chunk_np,
            action_chunk_env=np.stack(env_actions, axis=0),
            live_action_hidden=live_hidden,
            recon_action_hidden=recon_hidden,
            obs_embedding=obs_embedding,
            actor_input=feat,
            rssm_latent=latent if "latent" in locals() else None,
            input_ids=np.asarray(input_ids, dtype=np.float32)
            if "input_ids" in locals() and input_ids is not None
            else None,
        )
        return env_actions


__all__ = ["EmbodiedEvalRunner"]
