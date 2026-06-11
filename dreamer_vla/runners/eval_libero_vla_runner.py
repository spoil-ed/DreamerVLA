"""Eval-only runner: load a VLA/Dreamer checkpoint and run LIBERO rollouts.

No training, no optimizer, no dataset. Reuses the rollout logic that already
lives on ``PretokenizeVLARunner.evaluate_libero`` so there is exactly one
code path for LIBERO success-rate measurement.

Typical use:

  bash scripts/eval_libero_vla.sh \\
    eval.ckpt_path=/path/to/pretokenize_vla/checkpoints/epoch=013-train_vla_loss=1.984.ckpt \\
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
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from transformers import GenerationConfig

from dreamer_vla.algorithms.tdmpc_mpc import TDMPCMPCConfig, TDMPCMPCPlanner
from dreamer_vla.runners.pretokenize_vla_runner import PretokenizeVLARunner
from dreamer_vla.utils.torch_utils import freeze_module


class EvalLiberoVLARunner(PretokenizeVLARunner):
    """Load a VLA or Dreamer ckpt -> run LIBERO rollout -> dump JSON metrics."""

    runner_name = "libero_eval"
    runner_status = "current"
    runner_family = "eval"
    default_output_dir = str(
        pathlib.Path(__file__).resolve().parents[2]
        / "data"
        / "outputs"
        / "eval"
        / "eval_libero_vla"
    )

    def run(self) -> list[dict[str, Any]]:
        if self.distributed.is_main_process:
            print("EvalLiberoVLA Runner begin.")
        cfg = copy.deepcopy(self.cfg)

        if self.world_size != 1:
            raise RuntimeError(
                f"EvalLiberoVLARunner must run on a single process (got world_size={self.world_size}). "
                "Rollout evaluation does not support multi-process inference."
            )
        if self.distributed.uses_fsdp:
            raise RuntimeError(
                "EvalLiberoVLARunner requires DDP (not FSDP). "
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
        if ckpt_path and ckpt_kind in {"auto", "dreamer"}:
            payload = self._load_checkpoint_payload(ckpt_path)
            state_keys = set(payload.get("state_dicts", {}).keys())
            is_dreamer = {"world_model", "policy"}.issubset(state_keys)
            if ckpt_kind == "dreamer" and not is_dreamer:
                raise RuntimeError(
                    f"{ckpt_path} does not look like a Dreamer checkpoint: {sorted(state_keys)}"
                )
            if is_dreamer:
                return self._run_dreamer_eval(cfg, ckpt_path, payload)

        # ── encoder (inference only; no optimiser, no distributed wrapping) ──
        encoder_cfg = self._build_trainable_encoder_cfg(cfg)
        with open_dict(encoder_cfg):
            encoder_cfg.freeze_backbone = True
        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        self.encoder.eval()

        # ── optional: load VLA checkpoint (produced by PretokenizeVLARunner) ─
        if ckpt_path:
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
        else:
            if self.distributed.is_main_process:
                print(
                    "  [Eval] no eval.ckpt_path set → evaluating init VLA weights "
                    f"({OmegaConf.select(cfg, 'init.vla_ckpt_path')})"
                )

        # ── rollout ──────────────────────────────────────────────────────────
        os.makedirs(self.output_dir, exist_ok=True)
        self._init_policy_trace(cfg)
        metrics = self.evaluate_libero(epoch=-1)

        # ── dump metrics ─────────────────────────────────────────────────────
        if self.distributed.is_main_process:
            metrics_out = {
                "ckpt_path": ckpt_path,
                "task_suite": str(
                    OmegaConf.select(cfg, "eval.task_suite_name", default="libero_goal")
                ),
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
            return torch.load(
                ckpt_path, map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    @staticmethod
    def _normalize_vla_encoder_state_for_single_process_eval(
        payload: dict[str, Any],
    ) -> None:
        """Make DDP-saved VLA encoder checkpoints load into single-process eval.

        VLA SFT checkpoints saved under DDP can contain backbone keys like
        ``backbone.module.model...``. Eval constructs the unwrapped encoder, so
        those keys need to become ``backbone.model...`` before strict loading.
        """
        encoder_state = payload.get("state_dicts", {}).get("encoder")
        if not isinstance(encoder_state, dict):
            return
        if not any(str(key).startswith("backbone.module.") for key in encoder_state):
            return
        payload["state_dicts"]["encoder"] = {
            (
                str(key).replace("backbone.module.", "backbone.", 1)
                if str(key).startswith("backbone.module.")
                else key
            ): value
            for key, value in encoder_state.items()
        }

    @staticmethod
    def _checkpoint_cfg_from_payload(payload: dict[str, Any]) -> DictConfig:
        cfg = payload.get("cfg")
        if cfg is None:
            raise RuntimeError(
                "Dreamer checkpoint has no saved cfg; cannot rebuild Dreamer modules."
            )
        if isinstance(cfg, DictConfig):
            return copy.deepcopy(cfg)
        if isinstance(cfg, dict):
            return OmegaConf.create(copy.deepcopy(cfg))
        raise TypeError(
            f"Dreamer checkpoint cfg must be DictConfig or dict, got {type(cfg).__name__}"
        )

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
        metrics = self.evaluate_libero(epoch=-1)
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

    def _build_tdmpc_mpc_planner(self, cfg: DictConfig) -> TDMPCMPCPlanner:
        planner_cfg = OmegaConf.select(cfg, "eval.tdmpc_mpc", default={}) or {}
        action_steps = int(OmegaConf.select(cfg, "eval.action_steps", default=1))
        config = TDMPCMPCConfig(
            horizon=int(planner_cfg.get("horizon", 3)),
            iterations=int(planner_cfg.get("iterations", 6)),
            num_samples=int(planner_cfg.get("num_samples", 512)),
            num_elites=int(planner_cfg.get("num_elites", 64)),
            num_pi_trajs=int(planner_cfg.get("num_pi_trajs", 24)),
            action_dim=int(planner_cfg.get("action_dim", 7)),
            min_std=float(planner_cfg.get("min_std", 0.05)),
            max_std=float(planner_cfg.get("max_std", 2.0)),
            temperature=float(planner_cfg.get("temperature", 0.5)),
            gamma=float(planner_cfg.get("gamma", 0.995)),
            terminal_value_scale=float(planner_cfg.get("terminal_value_scale", 1.0)),
            reward_scale=float(planner_cfg.get("reward_scale", 1.0)),
            value_mode=str(planner_cfg.get("value_mode", "state")),
            execute_steps=int(planner_cfg.get("execute_steps", action_steps)),
            eval_mode=bool(planner_cfg.get("eval_mode", True)),
            warm_start=bool(planner_cfg.get("warm_start", True)),
            seed=int(planner_cfg.get("seed", OmegaConf.select(cfg, "seed", default=0))),
        )
        if self.distributed.is_main_process:
            print(
                "  [Eval][tdmpc-mpc] enabled "
                f"horizon={config.horizon} samples={config.num_samples} elites={config.num_elites} "
                f"pi_trajs={config.num_pi_trajs} iterations={config.iterations} "
                f"execute_steps={config.execute_steps}",
                flush=True,
            )
        return TDMPCMPCPlanner(config)

    def _init_real_relabel_export(self, cfg: DictConfig) -> None:
        self._real_relabel_enabled = bool(
            OmegaConf.select(cfg, "eval.export_real_relabel", default=False)
        )
        self._real_relabel_records: list[dict[str, Any]] = []
        self._real_relabel_success_rate = 0.0
        relabel_dir = OmegaConf.select(cfg, "eval.real_relabel_dir", default=None)
        if relabel_dir is None:
            relabel_dir = os.path.join(self.output_dir, "real_relabel")
        self._real_relabel_dir = str(relabel_dir)
        self._real_relabel_jsonl_path = os.path.join(
            self._real_relabel_dir, "real_rollout_relabel_records.jsonl"
        )
        self._real_relabel_summary_path = os.path.join(
            self._real_relabel_dir, "real_rollout_relabel_summary.json"
        )
        if self._real_relabel_enabled and self.distributed.is_main_process:
            os.makedirs(self._real_relabel_dir, exist_ok=True)
            with open(self._real_relabel_jsonl_path, "w"):
                pass

    @staticmethod
    def _real_relabel_sparse_rewards(
        success: bool, finish_step: int, max_steps: int
    ) -> list[float]:
        length = max(1, min(int(finish_step), int(max_steps)))
        rewards = [0.0] * length
        if success:
            rewards[length - 1] = 1.0
        return rewards

    def _append_real_relabel_record(self, record: dict[str, Any]) -> None:
        if not bool(getattr(self, "_real_relabel_enabled", False)):
            return
        self._real_relabel_records.append(record)
        with open(self._real_relabel_jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _write_real_relabel_summary(self) -> None:
        records = list(getattr(self, "_real_relabel_records", []))
        successes = int(sum(int(bool(row.get("complete", False))) for row in records))
        success_rate = successes / max(len(records), 1)
        self._real_relabel_success_rate = float(success_rate)
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in records:
            groups.setdefault(str(row.get("prompt_key", "")), []).append(row)
        group_rows = []
        for prompt_key, rows in sorted(groups.items()):
            acc = (
                float(np.mean([float(row.get("acc", 0.0)) for row in rows]))
                if rows
                else 0.0
            )
            group_rows.append(
                {
                    "prompt_key": prompt_key,
                    "num_samples": len(rows),
                    "successes": int(
                        sum(int(bool(row.get("complete", False))) for row in rows)
                    ),
                    "acc_mean": acc,
                    "keep_by_accuracy_band": bool(0.01 <= acc <= 0.99),
                }
            )
        summary = {
            "num_records": len(records),
            "successes": successes,
            "success_rate": float(success_rate),
            "records_jsonl": str(getattr(self, "_real_relabel_jsonl_path", "")),
            "wmpo_style_filter": {
                "accuracy_lower_bound": 0.01,
                "accuracy_upper_bound": 0.99,
                "num_prompt_groups": len(group_rows),
                "num_kept_prompt_groups": int(
                    sum(int(row["keep_by_accuracy_band"]) for row in group_rows)
                ),
                "num_records": len(records),
                "num_kept_records": int(
                    sum(
                        len(groups[row["prompt_key"]])
                        for row in group_rows
                        if row["keep_by_accuracy_band"]
                    )
                ),
                "groups": group_rows,
            },
        }
        os.makedirs(self._real_relabel_dir, exist_ok=True)
        with open(self._real_relabel_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"  [Eval] wrote real relabel summary -> {self._real_relabel_summary_path}",
            flush=True,
        )

    def _maybe_add_hidden_noise(self, hidden: torch.Tensor) -> torch.Tensor:
        noise_std = float(getattr(self, "_hidden_noise_std", 0.0))
        if noise_std <= 0.0:
            return hidden
        noise = (
            torch.randn(
                hidden.shape,
                generator=getattr(self, "_hidden_noise_generator", None),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            * noise_std
        )
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

    def _init_policy_trace(self, cfg: DictConfig) -> None:
        self._policy_trace_enabled = bool(
            OmegaConf.select(cfg, "eval.trace_policy_debug", default=False)
        )
        self._policy_trace_limit = int(
            OmegaConf.select(cfg, "eval.trace_policy_debug_limit", default=64)
        )
        self._policy_trace_count = 0
        self._policy_trace_dir = os.path.join(self.output_dir, "policy_trace_arrays")
        self._policy_trace_path = os.path.join(self.output_dir, "policy_trace.jsonl")
        if self._policy_trace_enabled and self.distributed.is_main_process:
            os.makedirs(self._policy_trace_dir, exist_ok=True)
            with open(self._policy_trace_path, "w"):
                pass

    @staticmethod
    def _to_numpy_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _array_summary(value: np.ndarray | None) -> dict[str, Any] | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        return {
            "shape": list(arr.shape),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "l2": float(np.linalg.norm(arr.reshape(-1))),
        }

    def _write_policy_trace(
        self,
        *,
        source: str,
        state: np.ndarray,
        action_chunk_raw: np.ndarray,
        action_chunk_env: np.ndarray,
        action_hidden: Any | None = None,
        wm_style_action_hidden: Any | None = None,
        live_action_hidden: Any | None = None,
        recon_action_hidden: Any | None = None,
        obs_embedding: Any | None = None,
        actor_input: Any | None = None,
        rssm_latent: Any | None = None,
        input_ids: Any | None = None,
    ) -> None:
        if not bool(getattr(self, "_policy_trace_enabled", False)):
            return
        index = int(getattr(self, "_policy_trace_count", 0))
        if index >= int(getattr(self, "_policy_trace_limit", 64)):
            return

        arrays: dict[str, np.ndarray] = {
            "state": np.asarray(state, dtype=np.float32).reshape(-1),
            "action_chunk_raw": np.asarray(action_chunk_raw, dtype=np.float32),
            "action_chunk_env": np.asarray(action_chunk_env, dtype=np.float32),
        }
        optional_arrays = {
            "action_hidden": self._to_numpy_array(action_hidden),
            "wm_style_action_hidden": self._to_numpy_array(wm_style_action_hidden),
            "live_action_hidden": self._to_numpy_array(live_action_hidden),
            "recon_action_hidden": self._to_numpy_array(recon_action_hidden),
            "obs_embedding": self._to_numpy_array(obs_embedding),
            "actor_input": self._to_numpy_array(actor_input),
            "input_ids": self._to_numpy_array(input_ids),
        }
        if rssm_latent is not None:
            for attr in ("deter", "stoch", "logits", "mean", "std", "h"):
                if hasattr(rssm_latent, attr):
                    optional_arrays[f"rssm_{attr}"] = self._to_numpy_array(
                        getattr(rssm_latent, attr)
                    )
        for key, value in optional_arrays.items():
            if value is not None:
                arrays[key] = np.asarray(value, dtype=np.float32)

        array_path = os.path.join(
            self._policy_trace_dir, f"step_{index:06d}_{source}.npz"
        )
        np.savez_compressed(array_path, **arrays)
        context = dict(getattr(self, "_libero_current_eval_context", {}) or {})
        raw_chunk = arrays["action_chunk_raw"].reshape(
            -1, arrays["action_chunk_raw"].shape[-1]
        )
        env_chunk = arrays["action_chunk_env"].reshape(
            -1, arrays["action_chunk_env"].shape[-1]
        )
        record = {
            "index": index,
            "source": str(source),
            "context": context,
            "array_path": array_path,
            "state": arrays["state"].tolist(),
            "first_action_raw": raw_chunk[0].tolist(),
            "first_action_env": env_chunk[0].tolist(),
            "summaries": {
                key: self._array_summary(value) for key, value in arrays.items()
            },
        }
        if self.distributed.is_main_process:
            with open(self._policy_trace_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        self._policy_trace_count = index + 1

    @staticmethod
    def _action_clip_bounds() -> tuple[np.ndarray, np.ndarray]:
        min_values = np.array(
            [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0]
        )
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
        return min_values, max_values

    def _policy_first_action_raw_from_hidden(self, hidden: torch.Tensor) -> np.ndarray:
        action, _, extra = self.policy(
            {
                "mode": "sample",
                "hidden": hidden.detach(),
                "deterministic": True,
                "return_chunk": True,
            }
        )
        action_tensor = extra.get("action_chunk", action)
        action_np = action_tensor.squeeze(0).detach().cpu().float().numpy()
        if action_np.ndim > 1:
            action_np = action_np.reshape(-1, action_np.shape[-1])[0]
        return np.asarray(action_np[:7], dtype=np.float32)

    def _policy_raw_to_env_action_for_compare(
        self, action_raw: np.ndarray
    ) -> np.ndarray:
        action_raw = np.asarray(action_raw[:7], dtype=np.float32)
        if bool(getattr(self, "_hidden_action_compare_unnorm", True)):
            return np.asarray(
                self._unnorm_actions(action_raw.reshape(1, -1))[0], dtype=np.float32
            )
        min_values, max_values = self._action_clip_bounds()
        return np.clip(action_raw, min_values, max_values).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _action_stats(
        prefix: str, left: np.ndarray, right: np.ndarray
    ) -> dict[str, float]:
        diff = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
        return {
            f"{prefix}_mse": float(np.mean(np.square(diff))),
            f"{prefix}_mae": float(np.mean(np.abs(diff))),
            f"{prefix}_max_abs": float(np.max(np.abs(diff))),
        }

    def _dreamer_policy_raw_to_env_action(self, action_raw: np.ndarray) -> np.ndarray:
        action = np.asarray(action_raw[:7], dtype=np.float32)
        if self._dreamer_should_unnorm_actions():
            action = np.asarray(
                self._unnorm_actions(action.reshape(1, -1))[0], dtype=np.float32
            )
        if bool(getattr(self, "_dreamer_clip_actions", True)):
            min_values, max_values = self._action_clip_bounds()
            action = np.clip(action, min_values, max_values)
        return action.astype(np.float32, copy=False)

    def _dreamer_should_unnorm_actions(self) -> bool:
        setting = OmegaConf.select(
            self.cfg, "eval.dreamer_unnorm_actions", default="auto"
        )
        if isinstance(setting, str):
            normalized = setting.lower()
            if normalized in {"auto", ""}:
                policy_name = (
                    self.policy.__class__.__name__ if hasattr(self, "policy") else ""
                )
                policy_target = str(
                    OmegaConf.select(self.cfg, "policy._target_", default="")
                )
                return (
                    "RynnVLAActionHiddenActor" in policy_name
                    or "VLAActionHeadActor" in policy_name
                    or (
                        "RynnVLAActionHiddenActor" in policy_target
                        or "VLAActionHeadActor" in policy_target
                    )
                )
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False
        return bool(setting)

    def _dreamer_rssm_action_from_raw_env(
        self, raw_action: np.ndarray, env_action: np.ndarray
    ) -> np.ndarray:
        source = str(
            OmegaConf.select(self.cfg, "eval.dreamer_rssm_action_source", default="env")
        ).lower()
        if source not in {"env", "raw"}:
            raise ValueError("eval.dreamer_rssm_action_source must be one of: env, raw")
        if source == "raw":
            return np.asarray(raw_action[:7], dtype=np.float32)
        # WM training uses HDF5 LIBERO actions, i.e. the executed/env scale.
        return np.asarray(env_action[:7], dtype=np.float32)

    def _tdmpc_mpc_raw_to_rssm_tensor(self, raw_action: torch.Tensor) -> torch.Tensor:
        raw_action = raw_action[..., :7].float()
        source = str(
            OmegaConf.select(self.cfg, "eval.dreamer_rssm_action_source", default="env")
        ).lower()
        if source not in {"env", "raw"}:
            raise ValueError("eval.dreamer_rssm_action_source must be one of: env, raw")
        if source == "raw":
            return raw_action
        if self._dreamer_should_unnorm_actions():
            low_np, high_np = self._action_clip_bounds()
            low = torch.as_tensor(
                low_np, device=raw_action.device, dtype=raw_action.dtype
            )
            high = torch.as_tensor(
                high_np, device=raw_action.device, dtype=raw_action.dtype
            )
            action = (raw_action + 1.0) * 0.5 * (high - low + 1.0e-8) + low
        else:
            action = raw_action
        if bool(getattr(self, "_dreamer_clip_actions", True)):
            low_np, high_np = self._action_clip_bounds()
            low = torch.as_tensor(
                low_np, device=raw_action.device, dtype=raw_action.dtype
            )
            high = torch.as_tensor(
                high_np, device=raw_action.device, dtype=raw_action.dtype
            )
            action = torch.maximum(torch.minimum(action, high), low)
        return action

    def _tdmpc_mpc_action_chunk_from_latent(
        self,
        latent: Any,
        action_steps: int = 1,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        planner = getattr(self, "_tdmpc_mpc_planner", None)
        if planner is None:
            raise RuntimeError("TD-MPC MPC planner requested but was not initialized.")
        result = planner.plan(
            policy=self.policy,
            world_model=self.world_model,
            latent=latent,
            device=self.device,
            target_critic=getattr(self, "target_critic", None),
            action_transform=self._tdmpc_mpc_raw_to_rssm_tensor,
        )
        raw_chunk_np = result.raw_actions.detach().cpu().float().numpy()
        if raw_chunk_np.ndim == 1:
            raw_chunk_np = raw_chunk_np.reshape(1, -1)
        raw_actions = [
            np.asarray(row[:7], dtype=np.float32).copy()
            for row in raw_chunk_np[: max(int(action_steps), 1)]
        ]
        env_actions = [
            self._dreamer_policy_raw_to_env_action(row).astype(np.float32, copy=False)
            for row in raw_actions
        ]
        rssm_actions = [
            self._dreamer_rssm_action_from_raw_env(raw, env).astype(
                np.float32, copy=False
            )
            for raw, env in zip(raw_actions, env_actions, strict=True)
        ]
        if bool(OmegaConf.select(self.cfg, "eval.log_action_stats", default=False)):
            count = int(getattr(self, "_dreamer_eval_action_log_count", 0))
            limit = int(
                OmegaConf.select(self.cfg, "eval.log_action_stats_limit", default=8)
            )
            if count < limit and env_actions:
                print(
                    "  [Eval][tdmpc-mpc-action] "
                    f"value={float(result.best_value.reshape(-1)[0].cpu()):.5f} "
                    f"elite_mean={float(result.elite_value_mean.reshape(-1)[0].cpu()):.5f} "
                    f"raw={np.array2string(raw_actions[0], precision=4, suppress_small=False)} "
                    f"env={np.array2string(env_actions[0], precision=4, suppress_small=False)} "
                    f"chunk={len(env_actions)}",
                    flush=True,
                )
            self._dreamer_eval_action_log_count = count + 1
        return env_actions, rssm_actions

    def _record_hidden_action_compare(
        self,
        *,
        live_hidden: torch.Tensor | None,
        recon_hidden: torch.Tensor | None,
        recon_action_raw: np.ndarray | None,
        executed_action: np.ndarray | None,
        context: dict[str, Any] | None = None,
        source: str,
    ) -> None:
        if not bool(getattr(self, "_hidden_action_compare_enabled", False)):
            return
        if int(getattr(self, "_hidden_action_compare_count", 0)) >= int(
            getattr(self, "_hidden_action_compare_limit", 300)
        ):
            return
        if live_hidden is None or recon_hidden is None:
            return

        with torch.no_grad():
            live = live_hidden.detach().float().reshape(live_hidden.shape[0], -1)
            recon = recon_hidden.detach().float().reshape(recon_hidden.shape[0], -1)
            if live.shape != recon.shape:
                return
            hidden_diff = recon - live
            hidden_mse = float(hidden_diff.square().mean().detach().cpu())
            hidden_mae = float(hidden_diff.abs().mean().detach().cpu())
            hidden_max_abs = float(hidden_diff.abs().max().detach().cpu())
            live_norm = float(live.norm(dim=-1).mean().detach().cpu())
            recon_norm = float(recon.norm(dim=-1).mean().detach().cpu())
            hidden_cosine_loss = float(
                (1.0 - F.cosine_similarity(recon, live, dim=-1).mean()).detach().cpu()
            )
            live_action_raw = self._policy_first_action_raw_from_hidden(live_hidden)
            if recon_action_raw is None:
                recon_action_raw = self._policy_first_action_raw_from_hidden(
                    recon_hidden
                )

        recon_action_raw = np.asarray(recon_action_raw[:7], dtype=np.float32)
        live_action_env = self._policy_raw_to_env_action_for_compare(live_action_raw)
        recon_action_env = self._policy_raw_to_env_action_for_compare(recon_action_raw)
        record: dict[str, Any] = {
            "index": int(getattr(self, "_hidden_action_compare_count", 0)),
            "source": str(source),
            "context": dict(context or {}),
            "hidden_mse": hidden_mse,
            "hidden_mae": hidden_mae,
            "hidden_max_abs": hidden_max_abs,
            "hidden_cosine_loss": hidden_cosine_loss,
            "live_hidden_norm": live_norm,
            "recon_hidden_norm": recon_norm,
            "recon_to_live_norm_ratio": float(recon_norm / max(live_norm, 1.0e-8)),
            "live_action_raw": live_action_raw.tolist(),
            "recon_action_raw": recon_action_raw.tolist(),
            "live_action_env": live_action_env.tolist(),
            "recon_action_env": recon_action_env.tolist(),
            **self._action_stats(
                "recon_vs_live_raw_action", recon_action_raw, live_action_raw
            ),
            **self._action_stats(
                "recon_vs_live_env_action", recon_action_env, live_action_env
            ),
        }
        if executed_action is not None:
            executed = np.asarray(executed_action[:7], dtype=np.float32)
            record["executed_action"] = executed.tolist()
            record.update(
                self._action_stats(
                    "executed_vs_live_env_action", executed, live_action_env
                )
            )
            record.update(
                self._action_stats(
                    "executed_vs_recon_env_action", executed, recon_action_env
                )
            )
            record.update(
                self._action_stats(
                    "executed_vs_recon_raw_action", executed, recon_action_raw
                )
            )

        sums = getattr(self, "_hidden_action_compare_sums", {})
        for key, value in record.items():
            if isinstance(value, float):
                sums[key] = float(sums.get(key, 0.0) + value)
        self._hidden_action_compare_sums = sums
        self._hidden_action_compare_count = (
            int(getattr(self, "_hidden_action_compare_count", 0)) + 1
        )

        if self.distributed.is_main_process:
            with open(self._hidden_action_compare_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            if record["index"] < int(
                OmegaConf.select(self.cfg, "eval.log_action_stats_limit", default=8)
            ):
                print(
                    "  [Eval][hidden-action-compare] "
                    f"idx={record['index']} hidden_mse={hidden_mse:.6g} "
                    f"hidden_cos={hidden_cosine_loss:.6g} "
                    f"env_action_mse={record['recon_vs_live_env_action_mse']:.6g} "
                    f"exec_vs_live_env={record.get('executed_vs_live_env_action_mse', float('nan')):.6g}",
                    flush=True,
                )

    def _action_hidden_tokens_for_trace(
        self, hidden: torch.Tensor | None
    ) -> torch.Tensor | None:
        if hidden is None:
            return None
        if hidden.ndim == 3:
            return hidden
        if hidden.ndim != 2:
            return None
        token_dim = int(
            OmegaConf.select(self.cfg, "policy.action_hidden_dim", default=1024)
        )
        time_horizon = int(
            OmegaConf.select(self.cfg, "policy.time_horizon", default=5)
        )
        action_dim = int(OmegaConf.select(self.cfg, "policy.action_dim", default=7))
        token_count = time_horizon * action_dim
        expected = token_count * token_dim
        if int(hidden.shape[-1]) != expected:
            return None
        return hidden.reshape(hidden.shape[0], token_count, token_dim)

    def _hidden_action_compare_summary(self) -> dict[str, float | str | int | bool]:
        count = int(getattr(self, "_hidden_action_compare_count", 0))
        sums = getattr(self, "_hidden_action_compare_sums", {})
        summary: dict[str, float | str | int | bool] = {
            "count": count,
            "jsonl_path": str(getattr(self, "_hidden_action_compare_path", "")),
            "policy_outputs_unnormed_for_compare": bool(
                getattr(self, "_hidden_action_compare_unnorm", True)
            ),
        }
        if count <= 0:
            return summary
        for key, value in sorted(sums.items()):
            summary[f"mean_{key}"] = float(value / count)
        return summary

    def _build_dreamer_modules(self, cfg: DictConfig, payload: dict[str, Any]) -> None:
        state_dicts = payload.get("state_dicts", {})

        encoder_cfg = self._build_frozen_encoder_cfg(cfg)
        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        freeze_module(self.encoder)
        if "encoder" in state_dicts:
            self._load_module_state(self.encoder, state_dicts["encoder"], "encoder")
        else:
            encoder_init_ckpt = OmegaConf.select(
                cfg, "init.encoder_state_ckpt", default=None
            )
            if encoder_init_ckpt:
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

    @staticmethod
    def _strip_wrapping_prefix(key: str) -> str:
        for prefix in ("_fsdp_wrapped_module.", "module."):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    def _attach_image_token_mapping(self) -> None:
        wm = getattr(self, "_unwrapped_world_model", None) or self.world_model
        if (
            wm is None
            or not getattr(wm, "spatial_codec", False)
            or self.encoder is None
        ):
            return
        lm_head = self.encoder.backbone.lm_head
        vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
        image_token_bpe_ids = torch.tensor(
            sorted(vocab_mapping.bpe2img.keys()), dtype=torch.long
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
            print(f"  [Eval] attached {tag} for image-token mapping.")

    def _wm_io_mode(self) -> str:
        wm = getattr(self, "_unwrapped_world_model", None) or getattr(
            self, "world_model", None
        )
        if wm is None:
            return "hidden"
        explicit = getattr(wm, "io_mode", None)
        if explicit is not None:
            return str(explicit)
        encoder = getattr(wm, "encoder", None)
        if (
            encoder is not None
            and encoder.__class__.__name__ == "DreamerV3TokenEncoder"
        ):
            return "token"
        return "hidden"

    def _wm_expects_image_vocab_tokens(self) -> bool:
        wm = getattr(self, "_unwrapped_world_model", None) or getattr(
            self, "world_model", None
        )
        encoder = getattr(wm, "encoder", None)
        return (
            encoder is not None
            and encoder.__class__.__name__ == "DreamerV3TokenEncoder"
        )

    def _wm_expects_pixel_images(self) -> bool:
        wm = getattr(self, "_unwrapped_world_model", None) or getattr(
            self, "world_model", None
        )
        encoder = getattr(wm, "encoder", None)
        return (
            encoder is not None
            and encoder.__class__.__name__ == "DreamerV3PixelEncoder"
        )

    def _get_image_bpe_set(self) -> set[int]:
        cached = getattr(self, "_image_bpe_set_cache", None)
        if cached is not None:
            return cached
        vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
        self._image_bpe_set_cache = set(vocab_mapping.bpe2img.keys())
        return self._image_bpe_set_cache

    def _extract_image_bpe_ids(self, input_ids_list: list[list[int]]) -> torch.Tensor:
        from dreamer_vla.utils.wm_image_viz import extract_image_blocks

        wm = getattr(self, "_unwrapped_world_model", None) or self.world_model
        wm_encoder = getattr(wm, "encoder", None)
        n_img_tok = int(
            getattr(wm, "n_image_tokens", getattr(wm_encoder, "n_image_tokens", 256))
        )
        which_blocks_cfg = OmegaConf.select(
            self.cfg, "eval.dreamer_which_image_blocks", default=None
        )
        if which_blocks_cfg is None:
            which_blocks = [
                int(
                    OmegaConf.select(
                        self.cfg, "eval.dreamer_which_image_block", default=-2
                    )
                )
            ]
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
                raise ValueError(
                    f"rollout sample {idx}: no image block found in tokens"
                )
            tok_ids: list[int] = []
            for which_block in which_blocks:
                bidx = which_block if which_block >= 0 else len(blocks) + which_block
                if not (0 <= bidx < len(blocks)):
                    raise ValueError(
                        f"rollout sample {idx}: image block {which_block} out of range"
                    )
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

    def _encode_hidden_from_tokenized(
        self, input_ids_list: list[list[int]]
    ) -> torch.Tensor:
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
        return (
            ((hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))
            .float()
            .detach()
        )

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
        for seq, length in zip(input_ids_list, lengths, strict=True):
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
        if self._use_action_query_obs_hidden():
            hidden_states, input_ids, attention_mask = (
                self._encode_hidden_sequence_from_tokenized(input_ids_list)
            )
            action_hidden = self.encoder.extract_action_hidden(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_token_id=int(
                    OmegaConf.select(self.cfg, "eval.target_token_id", default=10004)
                ),
                eval=True,
            )
            return action_hidden.float().detach()
        return self._encode_hidden_from_tokenized(input_ids_list)

    def _use_action_query_obs_hidden(self) -> bool:
        source = str(
            OmegaConf.select(self.cfg, "eval.obs_hidden_source", default="auto")
        ).lower()
        if source not in {"auto", "pooled", "action_query"}:
            raise ValueError(
                "eval.obs_hidden_source must be one of: auto, pooled, action_query"
            )
        if source == "action_query":
            return True
        if source == "pooled":
            return False
        return str(
            OmegaConf.select(self.cfg, "encoder.action_head_type", default="legacy")
        ).lower() == "legacy"

    def _dreamer_wm_observation_input_ids(
        self,
        item_processor: Any,
        frame_history: list[tuple[Image.Image, Image.Image]],
        state: np.ndarray,
        task_description: str,
    ) -> list[int]:
        img_c: list[Image.Image] = []
        for third_pil, wrist_pil in frame_history:
            img_c.extend([third_pil, wrist_pil])
        prompt_style = str(
            OmegaConf.select(
                self.cfg, "eval.dreamer_wm_prompt_style", default="vla_policy"
            )
        ).lower()
        if prompt_style != "vla_policy":
            raise ValueError("eval.dreamer_wm_prompt_style must be 'vla_policy'")
        if not bool(
            OmegaConf.select(self.cfg, "eval.dreamer_wm_include_state", default=True)
        ):
            raise ValueError("eval.dreamer_wm_include_state must be true")
        if (
            int(OmegaConf.select(self.cfg, "eval.dreamer_wm_history_length", default=2))
            != 2
        ):
            raise ValueError(
                "eval.dreamer_wm_history_length must be 2 to match the existing sidecar"
            )

        human_val = (
            f"Finish the task: {task_description}."
            + "<|state|>"
            + "<|image|>" * len(img_c)
        )
        conv = {
            "conversations": [{"from": "human", "value": human_val}],
            "image": img_c,
            "state": [state],
            "action": [],
        }
        tokens = item_processor.process_item(conv, training_mode=False)
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        return [int(tok) for tok in tokens]

    def _dreamer_wm_frame_history(
        self,
        frame_history: list[tuple[Image.Image, Image.Image]],
    ) -> list[tuple[Image.Image, Image.Image]]:
        """Return the image history used for Dreamer WM encoding.

        Pure VLA rollout uses rotated history frames because its SFT data was
        saved as rotated PNGs.  New action-hidden WM sidecars can now use the
        same rotated two-step policy history; older sidecars can still request
        raw single-frame inputs through eval.dreamer_wm_* overrides.
        """
        if not frame_history:
            return frame_history

        history_cfg = OmegaConf.select(
            self.cfg, "eval.dreamer_wm_history_length", default=None
        )
        if history_cfg is None:
            history_len = len(frame_history)
        else:
            history_len = max(1, int(history_cfg))
        selected = list(frame_history[-history_len:])

        rotate = bool(
            OmegaConf.select(self.cfg, "eval.dreamer_wm_rotate_images", default=False)
        )
        if rotate:
            return selected

        raw_obs = getattr(self, "_libero_current_raw_obs", None)
        if history_len == 1 and isinstance(raw_obs, dict):
            if "agentview_image" in raw_obs and "robot0_eye_in_hand_image" in raw_obs:
                third = np.asarray(raw_obs["agentview_image"], dtype=np.uint8)
                wrist = np.asarray(raw_obs["robot0_eye_in_hand_image"], dtype=np.uint8)
                return [(Image.fromarray(third), Image.fromarray(wrist))]

        # `frame_history` entries were produced by get_libero_image(), which
        # rotates env RGB by 180 degrees. Rotate them back to match HDF5 sidecar
        # preprocessing when raw simulator observations are unavailable.
        restored: list[tuple[Image.Image, Image.Image]] = []
        for third_pil, wrist_pil in selected:
            third = np.asarray(third_pil, dtype=np.uint8)[::-1, ::-1].copy()
            wrist = np.asarray(wrist_pil, dtype=np.uint8)[::-1, ::-1].copy()
            restored.append((Image.fromarray(third), Image.fromarray(wrist)))
        return restored

    @staticmethod
    def _resize_hwc_uint8(image: np.ndarray, size: int) -> np.ndarray:
        if image.shape[0] == size and image.shape[1] == size:
            return np.ascontiguousarray(image)
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        return np.asarray(
            Image.fromarray(image).resize((size, size), resample=resample),
            dtype=np.uint8,
        )

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
    ) -> tuple[torch.Tensor, list[int] | None]:
        if self._wm_expects_pixel_images():
            return self._pixel_obs_for_wm(frame_history), None

        wm_frame_history = self._dreamer_wm_frame_history(frame_history)
        input_ids = self._dreamer_wm_observation_input_ids(
            item_processor=item_processor,
            frame_history=wm_frame_history,
            state=state,
            task_description=task_description,
        )
        return self._obs_embedding_for_wm([input_ids]), input_ids

    def _dreamer_dummy_sequence_inputs(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        input_ids = torch.zeros(
            batch, seq_len + 1, dtype=torch.long, device=hidden_states.device
        )
        input_ids[:, seq_len] = 10004
        attention_mask = torch.ones(
            batch, seq_len + 1, dtype=torch.bool, device=hidden_states.device
        )
        return input_ids, attention_mask

    def _dreamer_action_from_latent(
        self,
        latent: Any,
        input_ids: list[int] | None = None,
        action_steps: int = 1,
        live_hidden: torch.Tensor | None = None,
    ) -> np.ndarray:
        env_actions, _rssm_actions = self._dreamer_action_chunk_from_latent(
            latent=latent,
            input_ids=input_ids,
            action_steps=action_steps,
            live_hidden=live_hidden,
        )
        if not env_actions:
            raise RuntimeError("Dreamer policy produced an empty action chunk")
        return env_actions[0]

    def _dreamer_action_chunk_from_latent(
        self,
        latent: Any,
        input_ids: list[int] | None = None,
        action_steps: int = 1,
        live_hidden: torch.Tensor | None = None,
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
                    [input_ids + [10004]], dtype=torch.long, device=self.device
                )
                if seq_input_ids.shape[1] < hidden_states.shape[1] + 1:
                    pad = hidden_states.shape[1] + 1 - seq_input_ids.shape[1]
                    seq_input_ids = F.pad(seq_input_ids, (0, pad), value=0)
                    seq_input_ids[:, hidden_states.shape[1]] = 10004
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
                    "target_token_id": 10004,
                    "deterministic": bool(
                        getattr(self, "_dreamer_deterministic", True)
                    ),
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
        max_actions = max(int(action_steps), 1)
        raw_actions = [
            np.asarray(row[:7], dtype=np.float32).copy()
            for row in action_chunk_np[:max_actions]
        ]
        env_actions = [
            self._dreamer_policy_raw_to_env_action(row).astype(np.float32, copy=False)
            for row in raw_actions
        ]
        rssm_actions = [
            self._dreamer_rssm_action_from_raw_env(raw, env).astype(
                np.float32, copy=False
            )
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
                old_log_prob = float(
                    old_log_prob_t.detach().float().reshape(-1)[0].cpu()
                )
            except Exception:
                old_log_prob = float("nan")
            self._last_real_relabel_actor_step = {
                "actor_input": feat.detach()
                .float()
                .reshape(feat.shape[0], -1)[0]
                .cpu()
                .tolist(),
                "raw_action": np.asarray(raw_action_np, dtype=np.float32)
                .reshape(-1)
                .tolist(),
                "old_log_prob": old_log_prob,
            }
        self._record_hidden_action_compare(
            live_hidden=live_hidden,
            recon_hidden=feat if "feat" in locals() else None,
            recon_action_raw=raw_action_np,
            executed_action=action_np,
            context=getattr(self, "_libero_current_eval_context", None),
            source="online_rssm",
        )
        live_trace_hidden = self._action_hidden_tokens_for_trace(live_hidden)
        recon_trace_hidden = self._action_hidden_tokens_for_trace(
            feat if "feat" in locals() else None
        )
        self._write_policy_trace(
            source="dreamer",
            state=np.asarray(
                getattr(self, "_libero_current_eval_context_state", []),
                dtype=np.float32,
            ),
            action_chunk_raw=action_chunk_np[:max_actions],
            action_chunk_env=np.stack(env_actions, axis=0),
            live_action_hidden=live_trace_hidden,
            recon_action_hidden=recon_trace_hidden,
            obs_embedding=live_hidden,
            actor_input=feat if "feat" in locals() else None,
            rssm_latent=latent,
            input_ids=np.asarray(input_ids, dtype=np.float32)
            if input_ids is not None
            else None,
        )
        if bool(OmegaConf.select(self.cfg, "eval.log_action_stats", default=False)):
            count = int(getattr(self, "_dreamer_eval_action_log_count", 0))
            limit = int(
                OmegaConf.select(self.cfg, "eval.log_action_stats_limit", default=8)
            )
            if count < limit:
                print(
                    "  [Eval][online-action] "
                    f"raw={np.array2string(raw_action_np, precision=4, suppress_small=False)} "
                    f"env={np.array2string(action_np, precision=4, suppress_small=False)} "
                    f"rssm={np.array2string(rssm_actions[0], precision=4, suppress_small=False)} "
                    f"abs_mean={float(np.mean(np.abs(action_np))):.5f} "
                    f"max_abs={float(np.max(np.abs(action_np))):.5f} "
                    f"chunk={len(env_actions)} action_steps={int(action_steps)}",
                    flush=True,
                )
            self._dreamer_eval_action_log_count = count + 1
        return env_actions, rssm_actions

    def _dreamer_online_reset(self) -> None:
        self._dreamer_online_latent = None
        self._dreamer_online_prev_action = None
        planner = getattr(self, "_tdmpc_mpc_planner", None)
        if planner is not None:
            planner.reset()

    def _dreamer_online_update_latent(self, obs_embedding: torch.Tensor) -> Any:
        if getattr(self, "_dreamer_online_latent", None) is None:
            latent = self.world_model(
                {"mode": "encode_latent", "hidden": obs_embedding}
            )
        else:
            prev_action = getattr(self, "_dreamer_online_prev_action", None)
            if not isinstance(prev_action, torch.Tensor):
                raise RuntimeError(
                    "online_rssm latent update missing previous executed action"
                )
            latent = self.world_model(
                {
                    "mode": "observe_next",
                    "latent": self._dreamer_online_latent,
                    "hidden": obs_embedding,
                    "actions": prev_action,
                    "is_first": False,
                }
            )
        self._dreamer_online_latent = latent
        return latent

    def _evaluate_libero_online_rssm(self, epoch: int) -> dict[str, float]:
        if not self.distributed.is_main_process:
            return {}
        if self.distributed.uses_fsdp:
            print(
                "  [Eval] Skipping online_rssm eval under FSDP. Use scripts/eval_libero_vla.sh."
            )
            return {}

        from libero.libero import benchmark as libero_benchmark

        from dreamer_vla.envs import (
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
                    target_token_id=int(
                        OmegaConf.select(
                            self.cfg, "eval.target_token_id", default=10004
                        )
                    ),
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
                        "target_token_id": 10004,
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


__all__ = ["EvalLiberoVLARunner"]
