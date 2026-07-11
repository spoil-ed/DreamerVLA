"""Action decode/unnorm + TDMPC-MPC + hidden-vs-recon action comparison helpers.

Closed cohesive group extracted from embodied_eval_runner.py (P3 god-file split,
mixin route): inherited by the runner, MRO resolves every self-call unchanged.
Behaviour-preserving.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from dreamervla.algorithms.tdmpc_mpc import TDMPCMPCConfig, TDMPCMPCPlanner
from dreamervla.runners import _embodied_eval_helpers as _eh


class EmbodiedEvalActionMixin:
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

    _action_clip_bounds = staticmethod(_eh.action_clip_bounds)

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

    _action_stats = staticmethod(_eh.action_stats)

    def _dreamer_policy_raw_to_env_action(self, action_raw: np.ndarray) -> np.ndarray:
        action = np.asarray(action_raw[:7], dtype=np.float32)
        if self._dreamer_should_unnorm_actions():
            action = np.asarray(
                self._unnorm_actions(action.reshape(1, -1))[0], dtype=np.float32
            )
        if bool(getattr(self, "_dreamer_clip_actions", True)):
            min_values, max_values = self._action_clip_bounds()
            action = np.clip(action, min_values, max_values)
        action_postprocess = str(
            OmegaConf.select(self.cfg, "eval.action_postprocess", default="none")
        ).strip().lower()
        if action_postprocess in {"openvla_oft", "oft"}:
            from dreamervla.runners.oft_collect_common import process_action

            action = process_action(action)
        elif action_postprocess not in {"", "none", "false"}:
            raise ValueError(
                f"unknown eval.action_postprocess: {action_postprocess!r}"
            )
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
                return "LatentToOpenVLAHiddenStateActor" in {
                    policy_name,
                    policy_target.rsplit(".", 1)[-1],
                }
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False
        return bool(setting)

    def _dreamer_latent_action_source(self) -> str:
        source = OmegaConf.select(
            self.cfg, "eval.dreamer_latent_action_source", default=None
        )
        if source is None:
            source = OmegaConf.select(
                self.cfg, "eval.dreamer_rssm_action_source", default="env"
            )
        source = str(source).strip().lower()
        if source not in {"env", "raw"}:
            raise ValueError(
                "eval.dreamer_latent_action_source must be one of: env, raw"
            )
        return source

    def _dreamer_latent_action_from_raw_env(
        self, raw_action: np.ndarray, env_action: np.ndarray
    ) -> np.ndarray:
        if self._dreamer_latent_action_source() == "raw":
            return np.asarray(raw_action[:7], dtype=np.float32)
        # WM training uses HDF5 LIBERO actions, i.e. the executed/env scale.
        return np.asarray(env_action[:7], dtype=np.float32)

    def _tdmpc_mpc_raw_to_latent_tensor(self, raw_action: torch.Tensor) -> torch.Tensor:
        raw_action = raw_action[..., :7].float()
        if self._dreamer_latent_action_source() == "raw":
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
            action_transform=self._tdmpc_mpc_raw_to_latent_tensor,
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
        latent_actions = [
            self._dreamer_latent_action_from_raw_env(raw, env).astype(
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
        return env_actions, latent_actions

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

    @staticmethod
    def _hidden_token_grid_for_trace(
        hidden: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if hidden is None:
            return None
        if hidden.ndim == 3 and tuple(hidden.shape[-2:]) == (256, 4096):
            return hidden
        return None

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
