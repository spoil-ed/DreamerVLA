#!/usr/bin/env python3
# ruff: noqa: E402
"""Diagnose whether DINO-WM imagined PPO routes match real LIBERO rollout.

This is intentionally a diagnostic script, not a training/eval path.  For a
fixed LIBERO task/init state it:

1. Builds the saved DreamerVLA world model + policy.
2. Encodes the real observation as legacy action-hidden.
3. Samples K first actions from the PPO policy.
4. Rolls each candidate forward inside the WM and ranks by imagined return.
5. Resets the real env for each candidate, executes that first action, then
   continues deterministic closed-loop rollout.
6. Logs action deltas between closed-loop Dreamer actions and the original SFT
   VLA actions at the same real states.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from transformers import GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from libero.libero import benchmark as libero_benchmark
from robosuite.utils.transform_utils import quat2axisangle

from dreamervla.algorithms.dreamervla import (
    _actor_action_for_world_model,
    _detach_latent,
    _world_model_actor_input,
    _world_model_state_reward,
)
from dreamervla.envs import get_libero_dummy_action, get_libero_env, get_libero_image
from dreamervla.runners.eval_libero_vla_runner import EvalLiberoVLARunner


def _array_stats(prefix: str, left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    diff = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(left.reshape(-1)) * np.linalg.norm(right.reshape(-1)))
    cos = (
        float(np.dot(left.reshape(-1), right.reshape(-1)) / denom)
        if denom > 1.0e-12
        else math.nan
    )
    return {
        f"{prefix}_mse": float(np.mean(np.square(diff))),
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_max_abs": float(np.max(np.abs(diff))),
        f"{prefix}_cos": cos,
    }


def _merge_dreamer_eval_cfg(
    eval_cfg_root: DictConfig, payload: dict[str, Any]
) -> DictConfig:
    cfg = payload.get("cfg")
    if cfg is None:
        raise RuntimeError("Dreamer checkpoint has no cfg")
    train_cfg = copy_cfg(cfg)
    with open_dict(train_cfg):
        train_cfg.eval = eval_cfg_root.eval
        if OmegaConf.select(train_cfg, "encoder", default=None) is None:
            train_cfg.encoder = eval_cfg_root.encoder
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
        train_cfg.init.encoder_state_ckpt = eval_encoder_ckpt
        train_cfg.training.out_dir = eval_cfg_root.training.out_dir
        train_cfg.training.distributed_strategy = "ddp"
        train_cfg.training.enable_activation_checkpointing = False
        train_cfg.trainer.device = str(eval_cfg_root.trainer.device)
    return train_cfg


def copy_cfg(cfg: Any) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if isinstance(cfg, dict):
        return OmegaConf.create(cfg)
    raise TypeError(f"Unsupported cfg type: {type(cfg).__name__}")


def _build_runner(args: argparse.Namespace) -> EvalLiberoVLARunner:
    overrides = [
        f"training.out_dir={args.out_dir}",
        f"eval.ckpt_path={args.ckpt}",
        "eval.ckpt_kind=dreamer",
        "init.encoder_state_ckpt=null",
        "eval.obs_hidden_source=action_query",
        f"eval.task_suite_name={args.task_suite}",
        f"eval.action_steps={args.action_steps}",
        f"eval.dreamer_rollout_mode={args.rollout_mode}",
        "eval.dreamer_deterministic=true",
        "trainer.device=cuda:0",
    ]
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"), version_base=None
    ):
        eval_cfg = compose(
            config_name="train", overrides=["experiment=eval_libero_vla", *overrides]
        )
    bootstrap = EvalLiberoVLARunner(eval_cfg)
    payload = bootstrap._load_checkpoint_payload(str(args.ckpt))
    cfg = _merge_dreamer_eval_cfg(eval_cfg, payload)
    ws = EvalLiberoVLARunner(cfg)
    ws._dreamer_eval = True
    ws._dreamer_deterministic = True
    ws._dreamer_action_repeat = 1
    ws._dreamer_clip_actions = True
    ws._dreamer_rollout_mode = str(args.rollout_mode)
    ws._dreamer_actor_input_source = "rssm"
    ws._dreamer_policy_source = "ckpt"
    ws._hidden_noise_std = 0.0
    ws._hidden_noise_seed = 0
    ws._hidden_noise_generator = torch.Generator(device=ws.device)
    ws._hidden_noise_generator.manual_seed(0)
    ws._hidden_noise_mse_sum = 0.0
    ws._hidden_noise_cosine_sum = 0.0
    ws._hidden_noise_count = 0
    ws._hidden_action_compare_enabled = False
    ws._policy_trace_enabled = False
    ws._build_dreamer_modules(cfg, payload)
    return ws


def _state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def _padded_history(
    frame_history: list[tuple[Image.Image, Image.Image]], history_length: int
) -> list[tuple[Image.Image, Image.Image]]:
    return [frame_history[0]] * (history_length - len(frame_history)) + frame_history


def _observe(
    ws: EvalLiberoVLARunner,
    item_processor: Any,
    obs: dict[str, Any],
    frame_history: list[tuple[Image.Image, Image.Image]],
    task_description: str,
    resolution: int,
    history_length: int,
) -> tuple[torch.Tensor, list[int], np.ndarray, list[tuple[Image.Image, Image.Image]]]:
    img = get_libero_image(obs, resolution)
    wrist = get_libero_image(obs, resolution, "robot0_eye_in_hand_image")
    frame_history.append((Image.fromarray(img), Image.fromarray(wrist)))
    if len(frame_history) > history_length:
        frame_history = frame_history[-history_length:]
    state = _state_from_obs(obs)
    ws._libero_current_raw_obs = obs
    obs_embedding, input_ids = ws._dreamer_obs_embedding_from_eval_inputs(
        item_processor,
        _padded_history(frame_history, history_length),
        state,
        task_description,
    )
    if input_ids is None:
        raise RuntimeError("This diagnostic expects token/VLA observation inputs")
    return obs_embedding, input_ids, state, frame_history


def _sft_action(
    ws: EvalLiberoVLARunner, input_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    backbone = ws.encoder.backbone
    generation_config = GenerationConfig(
        max_new_tokens=1,
        max_length=backbone.config.max_position_embeddings,
        temperature=1,
        top_k=None,
        do_sample=False,
        eos_token_id=[8710],
    )
    input_tensor = torch.tensor(
        input_ids, dtype=torch.long, device=ws.device
    ).unsqueeze(0)
    with torch.no_grad():
        raw = (
            backbone.generate_action_head(input_tensor, generation_config)
            .detach()
            .cpu()
            .float()
            .numpy()
        )
    raw = raw.reshape(-1, raw.shape[-1])
    env = np.asarray(ws._unnorm_actions(raw)[0], dtype=np.float32)
    return raw[0, :7].astype(np.float32), env[:7].astype(np.float32)


def _dreamer_action_from_latent(
    ws: EvalLiberoVLARunner,
    latent: Any,
    deterministic: bool,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, torch.Tensor]:
    feat = _world_model_actor_input(ws.world_model, latent).detach().float()
    with torch.no_grad():
        action, _, extra = ws.policy(
            {
                "mode": "sample",
                "hidden": feat,
                "deterministic": deterministic,
                "return_chunk": False,
            }
        )
    raw = action.detach().reshape(-1, 7)[0].cpu().float().numpy().astype(np.float32)
    env = ws._dreamer_policy_raw_to_env_action(raw).astype(np.float32)
    wm_action = _actor_action_for_world_model(action.detach(), ws.cfg.algorithm)
    return action.detach(), raw, env, wm_action.detach()


def _imagine_candidates(
    ws: EvalLiberoVLARunner,
    latent: Any,
    num_candidates: int,
    horizon: int,
) -> list[dict[str, Any]]:
    rows = []
    for candidate in range(num_candidates):
        cur = _detach_latent(latent)
        rewards: list[float] = []
        raw_actions: list[list[float]] = []
        env_actions: list[list[float]] = []
        first_action_tensor = None
        first_wm_action = None
        for step in range(horizon):
            action_tensor, raw, env, wm_action = _dreamer_action_from_latent(
                ws, cur, deterministic=False
            )
            if step == 0:
                first_action_tensor = action_tensor
                first_wm_action = wm_action
            raw_actions.append(raw.tolist())
            env_actions.append(env.tolist())
            with torch.no_grad():
                cur = _detach_latent(
                    ws.world_model(
                        {"mode": "predict_next", "latent": cur, "actions": wm_action}
                    )
                )
                reward = _world_model_state_reward(ws.world_model, cur).detach().float()
            rewards.append(float(reward.reshape(-1)[0].cpu()))
        rows.append(
            {
                "candidate": candidate,
                "imagined_return": float(sum(rewards)),
                "imagined_rewards": rewards,
                "first_raw_action": raw_actions[0],
                "first_env_action": env_actions[0],
                "_first_action_tensor": first_action_tensor,
                "_first_wm_action": first_wm_action,
            }
        )
    rows.sort(key=lambda row: row["imagined_return"], reverse=True)
    return rows


def _rollout_with_forced_first_action(
    ws: EvalLiberoVLARunner,
    env: Any,
    initial_state: Any,
    task_description: str,
    item_processor: Any,
    forced_env_action: np.ndarray,
    forced_wm_action: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    resolution = int(OmegaConf.select(ws.cfg, "encoder.resolution", default=256))
    history_length = int(args.history_length)
    max_steps = int(args.max_steps)
    sft_compare_steps = int(args.sft_compare_steps)
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    for _ in range(10):
        obs, _, done, _ = env.step(get_libero_dummy_action())
        if done:
            break
    ws._dreamer_online_reset()
    frame_history: list[tuple[Image.Image, Image.Image]] = []
    comparisons = []
    steps = 0
    forced_used = False
    for step_idx in range(max_steps):
        obs_embedding, input_ids, state, frame_history = _observe(
            ws,
            item_processor,
            obs,
            frame_history,
            task_description,
            resolution,
            history_length,
        )
        with torch.no_grad():
            if args.rollout_mode == "online_rssm":
                latent = ws._dreamer_online_update_latent(obs_embedding)
            else:
                latent = ws.world_model(
                    {"mode": "encode_latent", "hidden": obs_embedding}
                )
        if not forced_used:
            dreamer_env = forced_env_action.astype(np.float32)
            wm_action = forced_wm_action.reshape(1, -1).to(ws.device)
            forced_used = True
        else:
            _action_tensor, _raw, dreamer_env, wm_action = _dreamer_action_from_latent(
                ws, latent, deterministic=True
            )
        row = {
            "step": step_idx,
            "dreamer_env_action": dreamer_env.tolist(),
        }
        if sft_compare_steps < 0 or step_idx < sft_compare_steps:
            _sft_raw, sft_env = _sft_action(ws, input_ids)
            row["sft_env_action"] = sft_env.tolist()
            row.update(_array_stats("dreamer_vs_sft_env_action", dreamer_env, sft_env))
            comparisons.append(row)
        obs, _, done, _ = env.step(dreamer_env.tolist())
        if args.rollout_mode == "online_rssm":
            ws._dreamer_online_prev_action = wm_action.reshape(1, -1).to(ws.device)
        steps = step_idx + 1
        if done:
            break
    means = {}
    for key in (
        "dreamer_vs_sft_env_action_mse",
        "dreamer_vs_sft_env_action_mae",
        "dreamer_vs_sft_env_action_max_abs",
        "dreamer_vs_sft_env_action_cos",
    ):
        vals = [
            float(row[key]) for row in comparisons if not math.isnan(float(row[key]))
        ]
        means[f"mean_{key}"] = float(np.mean(vals)) if vals else math.nan
    return {
        "success": bool(done),
        "steps": int(steps),
        "action_compare": means,
        "action_compare_rows": comparisons[: args.trace_steps],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=7)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--imagination-horizon", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--action-steps", type=int, default=5)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument(
        "--rollout-mode", choices=["stateless", "online_rssm"], default="stateless"
    )
    parser.add_argument("--trace-steps", type=int, default=20)
    parser.add_argument("--sft-compare-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ws = _build_runner(args)
    item_processor = ws.encoder._build_processor(ws.device)
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(int(args.task_id))
    initial_states = task_suite.get_task_init_states(int(args.task_id))
    initial_state = initial_states[int(args.episode_idx)]
    env, task_description = get_libero_env(
        task,
        resolution=int(OmegaConf.select(ws.cfg, "encoder.resolution", default=256)),
    )

    # Build the shared real initial observation for imagined candidates.
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    for _ in range(10):
        obs, _, done, _ = env.step(get_libero_dummy_action())
        if done:
            break
    frame_history: list[tuple[Image.Image, Image.Image]] = []
    obs_embedding, input_ids, _state, frame_history = _observe(
        ws,
        item_processor,
        obs,
        frame_history,
        task_description,
        int(OmegaConf.select(ws.cfg, "encoder.resolution", default=256)),
        int(args.history_length),
    )
    with torch.no_grad():
        latent = ws.world_model({"mode": "encode_latent", "hidden": obs_embedding})
    sft_raw, sft_env = _sft_action(ws, input_ids)
    candidates = _imagine_candidates(
        ws, latent, int(args.num_candidates), int(args.imagination_horizon)
    )

    real_rows = []
    partial_json = args.out_dir / "imagine_vs_real_routes.partial.json"
    for rank, candidate in enumerate(candidates):
        print(
            f"[route] rank={rank} candidate={candidate['candidate']} imagined_return={candidate['imagined_return']:.6f}",
            flush=True,
        )
        real = _rollout_with_forced_first_action(
            ws=ws,
            env=env,
            initial_state=initial_state,
            task_description=task_description,
            item_processor=item_processor,
            forced_env_action=np.asarray(
                candidate["first_env_action"], dtype=np.float32
            ),
            forced_wm_action=candidate["_first_wm_action"],
            args=args,
        )
        public_candidate = {k: v for k, v in candidate.items() if not k.startswith("_")}
        public_candidate["imagined_rank"] = int(rank)
        public_candidate["first_action_vs_sft"] = _array_stats(
            "first_dreamer_vs_sft_env_action",
            np.asarray(candidate["first_env_action"], dtype=np.float32),
            sft_env,
        )
        real_rows.append({**public_candidate, "real_rollout": real})
        partial_json.write_text(json.dumps({"routes": real_rows}, indent=2))
        print(
            f"[route] done rank={rank} success={real['success']} steps={real['steps']} "
            f"mean_sft_mse={real['action_compare'].get('mean_dreamer_vs_sft_env_action_mse')}",
            flush=True,
        )

    summary = {
        "ckpt": str(args.ckpt),
        "task_suite": args.task_suite,
        "task_id": int(args.task_id),
        "task_description": task_description,
        "episode_idx": int(args.episode_idx),
        "rollout_mode": args.rollout_mode,
        "num_candidates": int(args.num_candidates),
        "imagination_horizon": int(args.imagination_horizon),
        "sft_initial_raw_action": sft_raw.tolist(),
        "sft_initial_env_action": sft_env.tolist(),
        "routes": real_rows,
    }
    out_json = args.out_dir / "imagine_vs_real_routes.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[save] {out_json}")


if __name__ == "__main__":
    main()
