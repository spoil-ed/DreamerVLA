#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from dreamervla.algorithms.dreamervla import (
    imagine_actor_critic_step,
    world_model_pretrain_step,
)
from dreamervla.algorithms.registry import get_actor_update_route
from dreamervla.models.reward import LatentSuccessClassifier, LatentSuccessClassifierConfig


def _init_distributed() -> tuple[int, int, int, bool]:
    """Init NCCL process group from torchrun env vars; no-op for single-process.

    Returns: (rank, world_size, local_rank, is_dist).
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0, False
    local_rank = int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank, True


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Return underlying module from a DDP wrapper, or pass through."""
    return module.module if isinstance(module, DDP) else module


from dreamervla.dataset.online_rollout_dumper import RolloutDumper
from dreamervla.models.critic.twohot_critic import ReturnPercentileTracker
from dreamervla.runners.online_replay import (
    OnlineReplay,
    get_replay_task_stats_global,
)
from dreamervla.runners.online_utils import (
    build_encoder,
    load_world_model_state,
    obs_to_action_hidden,
)
from dreamervla.utils.fixed_step_video import FixedStepVideoRecorder
from dreamervla.utils.optim import build_optimizer
from dreamervla.utils.paths import checkpoints_path
from dreamervla.utils.seed import set_seed
from dreamervla.utils.torch_utils import freeze_module


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online DreamerVLA training with RynnVLA action-hidden inputs."
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT / "configs/dreamervla/online_wmpo_outcome_libero_goal.yaml"
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--world-model-ckpt", required=True)
    parser.add_argument("--resume-ckpt", default=None)
    parser.add_argument(
        "--vla-ckpt-path",
        default=str(checkpoints_path("VLA_model_256", "libero_goal")),
    )
    parser.add_argument("--encoder-state-ckpt", default="")
    parser.add_argument(
        "--action-head-type",
        default="legacy",
        choices=["legacy"],
        help="Action-hidden head used by the online encoder.",
    )
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--total-env-steps",
        "--max-env-steps",
        dest="total_env_steps",
        type=int,
        default=200000,
        help="Training budget in real environment steps; separate from the per-episode horizon.",
    )
    parser.add_argument(
        "--max-train-updates",
        type=int,
        default=None,
        help="Optional training budget in optimizer updates, similar to WMPO total_training_steps.",
    )
    parser.add_argument(
        "--episode-horizon",
        "--max-episode-steps",
        dest="episode_horizon",
        type=int,
        default=200,
        help="Maximum steps in one online episode before timeout; success still terminates early.",
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument(
        "--replay-capacity-mode",
        default="per_task",
        choices=["per_task", "total_sharded"],
        help="'per_task' gives every task --replay-size transitions. 'total_sharded' splits "
        "--replay-size roughly evenly across requested tasks.",
    )
    parser.add_argument("--min-replay", type=int, default=64)
    parser.add_argument(
        "--failure-prefix-steps",
        type=int,
        default=40,
        help="For failed real-env episodes, only sample PPO/WM burn-in windows from the first N steps. "
        "Set <=0 to allow the full failed episode.",
    )
    parser.add_argument(
        "--failure-prefix-ratio",
        type=float,
        default=0.2,
        help="For failed episodes, also cap sampled windows to this fraction of episode length; "
        "the stricter positive cap is used.",
    )
    parser.add_argument(
        "--task-balanced-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample replay windows by cycling available task ids before choosing an episode/window.",
    )
    parser.add_argument(
        "--min-episodes-per-task",
        type=int,
        default=1,
        help="Do not start WM/PPO updates until every requested task has this many valid replay episodes "
        "on the local rank. Set 0 to use only --min-replay.",
    )
    parser.add_argument(
        "--global-coverage-train-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Under DDP, start training once the union of all ranks covers every requested task, "
        "while each rank only needs a non-empty local replay. This avoids waiting for every "
        "rank to complete a full task cycle.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=32.0,
        help="DreamerV3-style replay batch steps trained per real environment step.",
    )
    parser.add_argument(
        "--train-every",
        type=int,
        default=None,
        help="Legacy fixed update cadence; when unset, --train-ratio controls updates DreamerV3-style.",
    )
    parser.add_argument(
        "--updates-per-train",
        type=int,
        default=1,
        help="Updates at each --train-every tick when legacy cadence is enabled.",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--video-every-env-steps", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-max-frames", type=int, default=0)
    parser.add_argument("--video-frame-key", default="third_image")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--deterministic-collect", action="store_true")
    parser.add_argument(
        "--stochastic-collect",
        action="store_true",
        help="Sample noisy full action chunks during real-env collection. By default collection stays "
        "deterministic and only imagined PPO rollouts are stochastic.",
    )
    parser.add_argument(
        "--collect-chunk-steps",
        type=int,
        default=0,
        help="How many actions from each sampled policy chunk to execute in the real env. "
        "0 means use the full returned chunk, which is the normal 5-step chunk setting.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-token-id", type=int, default=10004)
    parser.add_argument("--rssm-action-scale", default="env", choices=["policy", "env"])
    parser.add_argument(
        "--run-wm-phase", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--run-actor-critic-phase", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--wm-refresh-updates-before-ppo",
        type=int,
        default=0,
        help="After replay is ready, run this many WM-only updates before the first PPO update. "
        "This implements collect -> replay -> WM refresh -> PPO instead of immediate interleaving.",
    )
    parser.add_argument(
        "--freeze-wm-after-refresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --wm-refresh-updates-before-ppo > 0, stop supervised WM updates after the refresh phase.",
    )
    parser.add_argument(
        "--freeze-wm-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the WM observation encoder/projection fixed during supervised WM refresh while "
        "still updating transition/prediction/reward layers.",
    )
    parser.add_argument(
        "--actor-update-kind",
        default="dreamer",
        choices=["dreamer", "dense_chunk", "outcome"],
        help="Actor update style. 'dreamer' = Dreamer-style lambda-return imagine_actor_critic_step "
        "(default, back-compat). 'dense_chunk' = chunk-WM PPO with dense per-step state-reward "
        "(NOT recommended — WM reward head is per-window indicator, signal collapses). "
        "'outcome' = full WMPO/verl PPO: chunk-WM rollout + LatentSuccessClassifier outcome reward "
        "+ eos_mask + zero-variance group filter. Requires --classifier-ckpt and ChunkAware WM.",
    )
    parser.add_argument(
        "--classifier-ckpt",
        default=None,
        help="Path to LatentSuccessClassifier .ckpt (with 'model'+'threshold'+'config' keys). "
        "Required when --actor-update-kind=outcome.",
    )
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=None,
        help="Override classifier success threshold. If unset, uses the threshold stored in the ckpt.",
    )
    parser.add_argument(
        "--update-classifier-online",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fine-tune the LatentSuccessClassifier on real online replay windows during training.",
    )
    parser.add_argument("--classifier-lr", type=float, default=1.0e-5)
    parser.add_argument("--classifier-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--classifier-batch-size", type=int, default=16)
    parser.add_argument("--classifier-updates-per-train", type=int, default=1)
    parser.add_argument("--classifier-grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--classifier-early-neg-stride",
        type=int,
        default=8,
        help="Stride for choosing the WMPO-style earlier negative window from an online episode.",
    )
    parser.add_argument(
        "--freeze-log-std",
        action="store_true",
        help="Override config: lock the Gaussian policy std at exp(initial_log_std).",
    )
    parser.add_argument(
        "--bc-to-ref",
        type=float,
        default=None,
        help="Override cfg.algorithm.actor_bc_to_ref_scale (head-level BC anchor against frozen ref_policy).",
    )
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=None,
        help="Override cfg.optim.policy.lr.",
    )
    parser.add_argument(
        "--allow-tiny-trainable",
        action="store_true",
        help="Bypass the silent-freeze guard. By default the script aborts if the policy has "
        "<=1k trainable params (almost certainly the identity+freeze_output_projection bug).",
    )
    # ── piggy-back online rollouts to disk for the WMPO classifier corpus ──
    parser.add_argument(
        "--dump-rollouts-raw-dir",
        default=None,
        help="If set, every completed episode is also written to a sharded HDF5 file under this "
        "dir using the WMReplayClassifierDataset schema (actions/dones/rewards). Pair with "
        "--dump-rollouts-hidden-dir; both must be set to enable dumping.",
    )
    parser.add_argument(
        "--dump-rollouts-hidden-dir",
        default=None,
        help="Sidecar dir for obs_embedding sharded HDF5 files. See --dump-rollouts-raw-dir.",
    )
    parser.add_argument(
        "--dump-rollouts-episodes-per-shard",
        type=int,
        default=25,
        help="Episodes per HDF5 shard pair. Smaller shards = more files; larger = bigger files.",
    )
    parser.add_argument(
        "--dump-rollouts-manifest",
        default=None,
        help="Optional JSONL manifest path (one line per dumped episode).",
    )
    return parser.parse_args()


def online_classifier_update_step(
    *,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: OnlineReplay,
    device: torch.device,
    batch_size: int,
    early_neg_stride: int,
    grad_clip: float,
) -> dict[str, Any]:
    module = _unwrap(classifier)
    cfg = module.cfg
    cls_batch = replay.sample_classifier_windows(
        int(batch_size),
        window=int(cfg.window),
        chunk_size=int(getattr(cfg, "chunk_size", 1)),
        chunk_pool=str(getattr(cfg, "chunk_pool", "last")),
        early_neg_stride=int(early_neg_stride),
    )
    windows = cls_batch["windows"].to(device, non_blocking=True)
    labels = cls_batch["labels"].to(device, non_blocking=True)
    classifier.train()
    logits = classifier(windows)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        classifier.parameters(), max_norm=float(grad_clip)
    )
    optimizer.step()
    classifier.eval()
    module.eval()
    with torch.no_grad():
        probs = torch.softmax(logits.detach(), dim=-1)[:, 1]
        preds = logits.detach().argmax(dim=-1)
        acc = (preds == labels).float().mean()
        pred_pos = preds == 1
        true_pos_label = labels == 1
        tp = (pred_pos & true_pos_label).sum().float()
        fp = (pred_pos & (~true_pos_label)).sum().float()
        fn = ((~pred_pos) & true_pos_label).sum().float()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = (2.0 * precision * recall) / (precision + recall).clamp_min(1.0e-12)
    return {
        "loss": float(loss.detach().cpu().item()),
        "acc": float(acc.detach().cpu().item()),
        "precision": float(precision.detach().cpu().item()),
        "recall": float(recall.detach().cpu().item()),
        "f1": float(f1.detach().cpu().item()),
        "tp": int(tp.detach().cpu().item()),
        "fp": int(fp.detach().cpu().item()),
        "fn": int(fn.detach().cpu().item()),
        "pos_frac": float(labels.float().mean().detach().cpu().item()),
        "prob_mean": float(probs.float().mean().detach().cpu().item()),
        "grad_norm": float(
            grad_norm.detach().cpu().item()
            if isinstance(grad_norm, torch.Tensor)
            else grad_norm
        ),
        "batch": _json_safe(
            {key: value for key, value in cls_batch.items() if key != "windows"}
        ),
    }


def freeze_world_model_encoder_params(world_model: torch.nn.Module) -> list[str]:
    module = _unwrap(world_model)
    prefixes = (
        "pos_embedding",
        "mask_obs_token",
        "obs_norm.",
        "obs_proj.",
    )
    frozen: list[str] = []
    for name, param in module.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad_(False)
            frozen.append(name)
    return frozen


def save_checkpoint(
    out_dir: Path,
    *,
    world_model: torch.nn.Module,
    policy: torch.nn.Module,
    critic: torch.nn.Module,
    target_critic: torch.nn.Module,
    wm_optimizer: torch.optim.Optimizer,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    return_tracker: ReturnPercentileTracker,
    cfg: Any,
    env_step: int,
    update_step: int,
    classifier: torch.nn.Module | None = None,
    classifier_optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step={env_step:07d}-updates={update_step:07d}.ckpt"
    payload = {
        "env_step": int(env_step),
        "update_step": int(update_step),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "state_dicts": {
            "world_model": world_model.state_dict(),
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
            "target_critic": target_critic.state_dict(),
            "world_model_optimizer": wm_optimizer.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "return_tracker": return_tracker.state_dict(),
        },
    }
    if classifier is not None:
        payload["state_dicts"]["classifier"] = classifier.state_dict()
    if classifier_optimizer is not None:
        payload["state_dicts"]["classifier_optimizer"] = (
            classifier_optimizer.state_dict()
        )
    torch.save(payload, path)
    latest = ckpt_dir / "latest.ckpt"
    torch.save(payload, latest)
    print(f"[ckpt] saved {path}", flush=True)
    return path


def load_training_checkpoint(
    ckpt_path: str | Path,
    *,
    world_model: torch.nn.Module,
    policy: torch.nn.Module,
    critic: torch.nn.Module,
    target_critic: torch.nn.Module,
    wm_optimizer: torch.optim.Optimizer,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    return_tracker: ReturnPercentileTracker,
    classifier: torch.nn.Module | None = None,
    classifier_optimizer: torch.optim.Optimizer | None = None,
    policy_strict: bool = True,
    load_policy_optimizer: bool = True,
) -> tuple[int, int]:
    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resume ckpt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dicts = payload.get("state_dicts", {})
    modules = {
        "world_model": world_model,
        "policy": policy,
        "critic": critic,
        "target_critic": target_critic,
    }
    if classifier is not None:
        modules["classifier"] = classifier
    optimizers = {
        "world_model_optimizer": wm_optimizer,
        "policy_optimizer": policy_optimizer,
        "critic_optimizer": critic_optimizer,
    }
    if classifier_optimizer is not None:
        optimizers["classifier_optimizer"] = classifier_optimizer
    for key, module in modules.items():
        if key in state_dicts:
            use_strict = True if key != "policy" else bool(policy_strict)
            missing, unexpected = _unwrap(module).load_state_dict(
                state_dicts[key], strict=use_strict
            )
            if not use_strict and (missing or unexpected):
                print(
                    f"[resume] {key} loaded non-strict: "
                    f"missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}",
                    flush=True,
                )
    for key, optimizer in optimizers.items():
        if key in state_dicts:
            if key == "policy_optimizer" and not bool(load_policy_optimizer):
                print(
                    "[resume] skipping policy_optimizer state (fresh moments for new params)",
                    flush=True,
                )
                continue
            optimizer.load_state_dict(state_dicts[key])
    if "return_tracker" in state_dicts:
        return_tracker.load_state_dict(state_dicts["return_tracker"])
    env_step = int(payload.get("env_step", 0))
    update_step = int(payload.get("update_step", 0))
    print(
        f"[resume] loaded {path} env_step={env_step} update_step={update_step}",
        flush=True,
    )
    return env_step, update_step


def main() -> None:
    args = parse_args()
    actor_update_route = (
        None
        if args.actor_update_kind == "dreamer"
        else get_actor_update_route(args.actor_update_kind)
    )
    rank, world_size, local_rank, is_dist = _init_distributed()
    is_rank0 = rank == 0
    # Per-rank seed offset → each rank's env explores independently.
    set_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if is_dist:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    if is_rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    if is_dist:
        dist.barrier()

    # ── structured traces ─────────────────────────────────────────────────
    # Per-rank files avoid concurrent appends corrupting JSONL under DDP. Rank 0
    # also mirrors PPO traces to the legacy ppo_groups.jsonl path.
    episode_log_f = open(
        out_dir / "logs" / f"episodes_rank{rank}.jsonl", "a", encoding="utf-8"
    )
    train_update_log_f = open(
        out_dir / "logs" / f"train_updates_rank{rank}.jsonl", "a", encoding="utf-8"
    )
    ppo_log_f = open(
        out_dir / "logs" / f"ppo_groups_rank{rank}.jsonl", "a", encoding="utf-8"
    )
    ppo_log_rank0_compat_f = (
        open(out_dir / "logs" / "ppo_groups.jsonl", "a", encoding="utf-8")
        if is_rank0
        else None
    )

    cfg = OmegaConf.load(args.config)
    expected_obs_dim = (
        int(OmegaConf.select(cfg, "policy.time_horizon"))
        * int(OmegaConf.select(cfg, "policy.action_dim"))
        * int(OmegaConf.select(cfg, "policy.action_hidden_dim"))
    )
    cfg_obs_dim = OmegaConf.select(cfg, "world_model.obs_dim")
    if cfg_obs_dim is not None and int(cfg_obs_dim) != expected_obs_dim:
        raise ValueError(
            f"--action-head-type={args.action_head_type} produces obs dim {expected_obs_dim}, "
            f"but config world_model.obs_dim={cfg_obs_dim}."
        )
    cfg.init.vla_ckpt_path = args.vla_ckpt_path
    cfg.init.encoder_state_ckpt = args.encoder_state_ckpt
    cfg.init.world_model_state_ckpt = args.world_model_ckpt
    cfg.training.out_dir = str(out_dir)
    cfg.training.distributed_strategy = "single"
    cfg.algorithm.rssm_action_scale = args.rssm_action_scale
    if bool(args.freeze_log_std):
        cfg.policy.freeze_log_std = True
    if args.bc_to_ref is not None:
        cfg.algorithm.actor_bc_to_ref_scale = float(args.bc_to_ref)
    if args.policy_lr is not None:
        cfg.optim.policy.lr = float(args.policy_lr)

    if is_rank0:
        print(
            f"[online-rynnvla] DDP world_size={world_size} rank={rank} local_rank={local_rank} "
            f"out_dir={out_dir}",
            flush=True,
        )
        print(
            "[online-rynnvla] algo_params "
            f"policy_lr={OmegaConf.select(cfg, 'optim.policy.lr', default='?')} "
            f"wm_lr={OmegaConf.select(cfg, 'optim.world_model.lr', default='?')} "
            f"grad_clip_norm={OmegaConf.select(cfg, 'optim.grad_clip_norm', default='?')} "
            f"initial_log_std={OmegaConf.select(cfg, 'policy.initial_log_std', default='?')} "
            f"min/max_log_std=({OmegaConf.select(cfg, 'policy.min_log_std', default='?')},"
            f"{OmegaConf.select(cfg, 'policy.max_log_std', default='?')}) "
            f"freeze_log_std={OmegaConf.select(cfg, 'policy.freeze_log_std', default=False)} "
            f"actent={OmegaConf.select(cfg, 'algorithm.actent', default='?')} "
            f"kl_coef={OmegaConf.select(cfg, 'algorithm.kl_coef', default=0.0)} "
            f"bc_to_ref_scale={OmegaConf.select(cfg, 'algorithm.actor_bc_to_ref_scale', default=0.0)} "
            f"imag_h={OmegaConf.select(cfg, 'algorithm.imagination_horizon', default='?')}",
            flush=True,
        )
        print(
            f"[online-rynnvla] device={device} task_suite={args.task_suite} task_ids={args.task_ids}",
            flush=True,
        )
        print(f"[online-rynnvla] wm_ckpt={args.world_model_ckpt}", flush=True)
        print(
            f"[online-rynnvla] episode_horizon={args.episode_horizon} "
            f"total_env_steps={args.total_env_steps} "
            f"max_train_updates={args.max_train_updates} "
            f"train_ratio={args.train_ratio} "
            f"collect_chunk_steps={args.collect_chunk_steps}",
            flush=True,
        )
        print(
            "[online-rynnvla] input=vla_policy history=2 state rotate180 action_query",
            flush=True,
        )
        if int(args.video_every_env_steps) > 0:
            print(
                f"[online-rynnvla] video_every_env_steps={args.video_every_env_steps} "
                f"video_fps={args.video_fps} video_frame_key={args.video_frame_key}",
                flush=True,
            )
        OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    encoder = build_encoder(args, device)
    processor = encoder._build_processor(device)
    world_model = hydra.utils.instantiate(cfg.world_model).to(
        device=device, dtype=torch.bfloat16
    )
    load_world_model_state(
        world_model,
        args.world_model_ckpt,
        reset_reward_head=bool(
            OmegaConf.select(cfg, "init.reset_world_model_reward_head", default=False)
        ),
    )
    if bool(args.freeze_wm_encoder):
        frozen_wm_encoder = freeze_world_model_encoder_params(world_model)
        if is_rank0:
            print(
                f"[init] froze WM encoder/projection params: n={len(frozen_wm_encoder)} "
                f"first={frozen_wm_encoder[:8]}",
                flush=True,
            )
    policy = hydra.utils.instantiate(cfg.policy).to(device)

    # WMPO-style frozen reference policy snapshot. Built BEFORE any resume load,
    # so the ref always reflects the SFT init (init_action_head_ckpt), not a
    # resumed RL state. Only kept in memory; not saved with the checkpoint.
    import copy as _copy

    ref_policy = _copy.deepcopy(policy)
    for _p in ref_policy.parameters():
        _p.requires_grad = False
    ref_policy.eval()

    # --- silent-freeze guard (same trap as train_frozen_wm_actor_critic.py) ----
    policy_trainable = [p for p in policy.parameters() if p.requires_grad]
    n_policy_trainable = sum(p.numel() for p in policy_trainable)
    n_policy_total = sum(p.numel() for p in policy.parameters())
    trainable_names = [n for n, p in policy.named_parameters() if p.requires_grad]
    print(
        f"[policy] {type(policy).__name__} trainable={n_policy_trainable:,} "
        f"/ total={n_policy_total:,} ({len(trainable_names)} tensors). "
        f"first trainable names: {trainable_names[:6]}",
        flush=True,
    )
    if n_policy_trainable <= 1000 and not bool(args.allow_tiny_trainable):
        adapter_type = OmegaConf.select(cfg, "policy.adapter_type", default=None)
        frozen_op = OmegaConf.select(
            cfg, "policy.freeze_output_projection", default=None
        )
        raise RuntimeError(
            f"Refusing to start: policy has only {n_policy_trainable} trainable parameters "
            f"({trainable_names}). This is almost certainly the silent-freeze trap: "
            f"adapter_type={adapter_type!r} + freeze_output_projection={frozen_op!r} leaves "
            f"only log_std trainable. Either set freeze_output_projection=false, use a "
            f"non-identity adapter, or pass --allow-tiny-trainable if this is intentional."
        )
    # --------------------------------------------------------------------------

    critic = hydra.utils.instantiate(cfg.critic).to(device)
    target_critic = hydra.utils.instantiate(cfg.critic).to(device)
    target_critic.load_state_dict(critic.state_dict())
    freeze_module(target_critic)

    # DDP-wrap the three trainable modules. Encoder, ref_policy, and
    # target_critic stay un-wrapped. The classifier is wrapped later only when
    # online classifier updates are enabled. find_unused_parameters=True because outcome.py's
    # _temporarily_freeze flips requires_grad on world_model during PPO step.
    if is_dist:
        world_model = DDP(
            world_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        critic = DDP(
            critic,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    wm_optimizer = build_optimizer(world_model, cfg.optim.world_model)
    policy_optimizer = build_optimizer(policy, cfg.optim.policy)
    critic_optimizer = build_optimizer(critic, cfg.optim.critic)

    classifier: torch.nn.Module | None = None
    classifier_optimizer: torch.optim.Optimizer | None = None
    classifier_threshold: float = 0.5
    if actor_update_route is not None and actor_update_route.requires_classifier:
        if not args.classifier_ckpt:
            raise ValueError(
                f"--actor-update-kind={args.actor_update_kind} requires --classifier-ckpt pointing to a "
                "LatentSuccessClassifier .ckpt (model+threshold+config)."
            )
        cls_payload = torch.load(
            args.classifier_ckpt, map_location="cpu", weights_only=False
        )
        cls_config_blob = cls_payload.get("config", {}).get("classifier")
        if cls_config_blob is None:
            raise RuntimeError(
                f"classifier ckpt {args.classifier_ckpt} has no config.classifier blob"
            )
        cls_cfg = LatentSuccessClassifierConfig(**cls_config_blob)
        classifier = LatentSuccessClassifier(cls_cfg).to(device).eval()
        classifier.load_state_dict(cls_payload["model"])
        if bool(args.update_classifier_online):
            for param in classifier.parameters():
                param.requires_grad_(True)
            if is_dist:
                classifier = DDP(
                    classifier,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False,
                )
            classifier_optimizer = torch.optim.AdamW(
                classifier.parameters(),
                lr=float(args.classifier_lr),
                weight_decay=float(args.classifier_weight_decay),
            )
        else:
            freeze_module(classifier)
        classifier_threshold = float(
            args.classifier_threshold
            if args.classifier_threshold is not None
            else cls_payload.get("threshold", 0.5)
        )
        print(
            f"[init] classifier loaded from {args.classifier_ckpt}; "
            f"threshold={classifier_threshold:.4f}; "
            f"ckpt F1={cls_payload.get('f1', float('nan')):.4f}; "
            f"online_update={bool(args.update_classifier_online)}",
            flush=True,
        )
    return_tracker = ReturnPercentileTracker(
        decay=float(
            OmegaConf.select(cfg, "algorithm.return_tracker.decay", default=0.99)
        ),
        low=float(OmegaConf.select(cfg, "algorithm.return_tracker.low", default=0.05)),
        high=float(
            OmegaConf.select(cfg, "algorithm.return_tracker.high", default=0.95)
        ),
    )
    resume_env_step = 0
    resume_update_step = 0
    if args.resume_ckpt:
        resume_env_step, resume_update_step = load_training_checkpoint(
            args.resume_ckpt,
            world_model=world_model,
            policy=policy,
            critic=critic,
            target_critic=target_critic,
            wm_optimizer=wm_optimizer,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            return_tracker=return_tracker,
            classifier=classifier,
            classifier_optimizer=classifier_optimizer,
        )

    task_ids = tuple(
        int(item) for item in str(args.task_ids).split(",") if item.strip()
    )
    rollout_dumper: RolloutDumper | None = None
    # Only rank 0 dumps shards; otherwise multiple ranks would write into the
    # same manifest and corrupt it. Each rank still trains on its own replay.
    if is_rank0 and args.dump_rollouts_raw_dir and args.dump_rollouts_hidden_dir:
        rollout_dumper = RolloutDumper(
            raw_dir=args.dump_rollouts_raw_dir,
            hidden_dir=args.dump_rollouts_hidden_dir,
            episodes_per_shard=int(args.dump_rollouts_episodes_per_shard),
            manifest_path=args.dump_rollouts_manifest,
        )
        print(
            f"[dump-rollouts] writing shards to raw={rollout_dumper.raw_dir} "
            f"hidden={rollout_dumper.hidden_dir} eps_per_shard={rollout_dumper.episodes_per_shard}",
            flush=True,
        )
    elif is_rank0 and (args.dump_rollouts_raw_dir or args.dump_rollouts_hidden_dir):
        raise ValueError(
            "--dump-rollouts-raw-dir and --dump-rollouts-hidden-dir must both be set or both be omitted"
        )

    # Per-rank env seed: different rollouts on each rank gives 4× online data
    # diversity for the shared (DDP-synced) policy gradient updates.
    from dreamervla.envs.train_env import DreamerVLAOnlineTrainEnv

    env_seed = int(args.seed) + rank * 1000
    env = DreamerVLAOnlineTrainEnv(
        task_suite_name=args.task_suite,
        task_id=task_ids[0],
        task_ids=task_ids,
        seed=env_seed,
        max_steps=args.episode_horizon,
        action_input="normalized",
        task_sampling="sequential",
        init_state_sampling="sequential",
        history_length=2,
        include_state=True,
        vla_rotate_180=True,
        obs_hidden_source="action_query",
        action_head_type=args.action_head_type,
    )
    replay = OnlineReplay(
        capacity=args.replay_size,
        sequence_length=args.sequence_length,
        task_ids=task_ids,
        capacity_mode=args.replay_capacity_mode,
        failure_prefix_steps=args.failure_prefix_steps,
        failure_prefix_ratio=args.failure_prefix_ratio,
        task_balanced=args.task_balanced_replay,
        rank=rank,
    )
    obs, _info = env.reset(seed=env_seed)
    episode: list[dict[str, Any]] = []
    episode_return = 0.0
    episode_len = 0
    latent = None
    prev_wm_action: torch.Tensor | None = None
    pending_policy_actions: deque[np.ndarray] = deque()
    pending_chunk_id = 0
    pending_chunk_index = 0
    pending_chunk_len = 0
    update_step = int(resume_update_step)
    last_saved_update = (
        int(resume_update_step)
        if int(resume_update_step) % int(args.save_every) == 0
        else -1
    )
    train_accum = 0.0
    batch_steps = max(int(args.batch_size) * int(args.sequence_length), 1)
    wm_refresh_target = max(0, int(args.wm_refresh_updates_before_ppo))
    wm_refresh_updates = 0
    final_env_step = int(resume_env_step)
    stop_training = False
    if args.max_train_updates is not None and update_step >= int(
        args.max_train_updates
    ):
        stop_training = True
    log_path = out_dir / "online_logs.json.txt"
    start_time = time.time()
    last_metrics: dict[str, float] = {}
    last_phase = "collect"
    last_global_replay_task_stats: dict[str, dict[str, int]] = {}
    last_global_coverage_ready = False
    last_all_ranks_train_ready = False
    # Only rank 0 records videos to avoid 4× duplicated ffmpeg work; the others
    # use a NoOp-like instance configured with every_steps=0.
    video_recorder = FixedStepVideoRecorder(
        every_steps=int(args.video_every_env_steps) if is_rank0 else 0,
        output_dir=Path(args.video_dir).expanduser().resolve()
        if args.video_dir
        else out_dir / "videos",
        fps=int(args.video_fps),
        max_frames=int(args.video_max_frames),
        frame_key=str(args.video_frame_key),
    )

    try:
        for env_step in range(int(resume_env_step) + 1, int(args.total_env_steps) + 1):
            if stop_training:
                break
            final_env_step = int(env_step)
            obs_embedding = obs_to_action_hidden(
                encoder, processor, obs, device, args.target_token_id
            )
            is_first = bool(obs.get("is_first", False)) or latent is None
            with torch.no_grad():
                if is_first:
                    latent = world_model(
                        {"mode": "encode_latent", "hidden": obs_embedding}
                    )
                else:
                    assert prev_wm_action is not None
                    latent = world_model(
                        {
                            "mode": "observe_next",
                            "latent": latent,
                            "hidden": obs_embedding,
                            "actions": prev_wm_action,
                            "is_first": False,
                        }
                    )

                if not pending_policy_actions:
                    feat = world_model(
                        {"mode": "actor_input", "latent": latent}
                    ).float()
                    action_chunk, _log_prob, _extra = policy(
                        {
                            "mode": "sample",
                            "hidden": feat,
                            "deterministic": bool(args.deterministic_collect)
                            or not bool(args.stochastic_collect),
                            "return_chunk": True,
                        }
                    )
                    chunk_np = (
                        action_chunk.reshape(-1, action_chunk.shape[-1])
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                    )
                    collect_chunk_steps = int(args.collect_chunk_steps)
                    if collect_chunk_steps <= 0:
                        collect_chunk_steps = int(chunk_np.shape[0])
                    collect_chunk_steps = max(
                        1, min(collect_chunk_steps, int(chunk_np.shape[0]))
                    )
                    pending_policy_actions.extend(
                        np.asarray(action[:7], dtype=np.float32).copy()
                        for action in chunk_np[:collect_chunk_steps]
                    )
                    pending_chunk_id += 1
                    pending_chunk_index = 0
                    pending_chunk_len = int(len(pending_policy_actions))

                policy_action = pending_policy_actions.popleft()
                current_chunk_id = int(pending_chunk_id)
                current_chunk_index = int(pending_chunk_index)
                current_chunk_len = int(pending_chunk_len)
                pending_chunk_index += 1

            next_obs, reward, terminated, truncated, info = env.step(policy_action)
            done = bool(terminated or truncated)
            video_recorder.capture(next_obs)
            saved_video = video_recorder.maybe_save(
                env_step=env_step,
                episode_len=episode_len + 1,
                episode_return=episode_return + float(reward),
                task_id=int(info.get("task_id", obs.get("task_id", 0))),
                success=bool(terminated),
            )
            if saved_video:
                print(f"[video] saved {saved_video}", flush=True)
            wm_action_np = np.asarray(info["wm_action"], dtype=np.float32).reshape(-1)[
                :7
            ]
            episode.append(
                {
                    "image": np.asarray(obs["image"], dtype=np.uint8),
                    "obs_embedding": obs_embedding.squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    "policy_action": policy_action.astype(np.float32),
                    "wm_action": wm_action_np.astype(np.float32),
                    "reward": np.float32(reward),
                    "done": np.float32(done),
                    "is_first": bool(is_first),
                    "is_terminal": np.float32(terminated),
                    "is_last": np.float32(done),
                    "task_id": int(info.get("task_id", obs.get("task_id", -1))),
                    "collect_chunk_id": current_chunk_id,
                    "collect_chunk_index": current_chunk_index,
                    "collect_chunk_len": current_chunk_len,
                }
            )
            episode_return += float(reward)
            episode_len += 1
            prev_wm_action = (
                torch.from_numpy(wm_action_np)
                .to(device=device, dtype=obs_embedding.dtype)
                .unsqueeze(0)
            )

            local_train_ready = replay.ready_for_training(
                min_transitions=args.min_replay,
                task_ids=task_ids,
                min_episodes_per_task=args.min_episodes_per_task,
            )
            (
                last_global_replay_task_stats,
                last_global_coverage_ready,
                last_all_ranks_train_ready,
            ) = get_replay_task_stats_global(
                replay,
                task_ids=task_ids,
                min_transitions=args.min_replay,
                min_episodes_per_task=args.min_episodes_per_task,
                device=device,
                is_dist=is_dist,
                world_size=world_size,
            )
            if bool(args.global_coverage_train_start):
                local_basic_ready = replay.ready_for_training(
                    min_transitions=args.min_replay,
                    task_ids=task_ids,
                    min_episodes_per_task=0,
                )
                all_ranks_basic_ready = bool(local_basic_ready)
                if is_dist:
                    ready_t = torch.tensor(
                        [int(local_basic_ready)], device=device, dtype=torch.long
                    )
                    dist.all_reduce(ready_t, op=dist.ReduceOp.MIN)
                    all_ranks_basic_ready = bool(int(ready_t.item()))
                all_ranks_train_ready = bool(
                    last_global_coverage_ready and all_ranks_basic_ready
                )
            else:
                all_ranks_train_ready = bool(local_train_ready)
                if is_dist:
                    ready_t = torch.tensor(
                        [int(local_train_ready)], device=device, dtype=torch.long
                    )
                    dist.all_reduce(ready_t, op=dist.ReduceOp.MIN)
                    all_ranks_train_ready = bool(int(ready_t.item()))
                last_all_ranks_train_ready = bool(all_ranks_train_ready)
            num_updates = 0
            if all_ranks_train_ready:
                if args.train_every is not None:
                    if env_step % int(args.train_every) == 0:
                        num_updates = int(args.updates_per_train)
                else:
                    train_accum += float(args.train_ratio) / float(batch_steps)
                    num_updates = int(train_accum)
                    train_accum -= float(num_updates)
            if is_dist and args.train_every is None and all_ranks_train_ready:
                local_num_updates = int(num_updates)
                updates_t = torch.tensor(
                    [local_num_updates], device=device, dtype=torch.long
                )
                dist.all_reduce(updates_t, op=dist.ReduceOp.MIN)
                num_updates = int(updates_t.item())
                if local_num_updates > num_updates:
                    train_accum += float(local_num_updates - num_updates)

            if num_updates > 0:
                for _ in range(num_updates):
                    if args.max_train_updates is not None and update_step >= int(
                        args.max_train_updates
                    ):
                        stop_training = True
                        break
                    batch = replay.sample(args.batch_size)
                    in_wm_refresh = (
                        wm_refresh_target > 0 and wm_refresh_updates < wm_refresh_target
                    )
                    do_wm_phase = bool(args.run_wm_phase) and (
                        wm_refresh_target == 0
                        or in_wm_refresh
                        or not bool(args.freeze_wm_after_refresh)
                    )
                    do_actor_phase = bool(args.run_actor_critic_phase) and (
                        wm_refresh_target == 0
                        or wm_refresh_updates >= wm_refresh_target
                    )
                    if do_wm_phase:
                        wm_metrics = world_model_pretrain_step(
                            policy=policy,
                            world_model=world_model,
                            optimizer=wm_optimizer,
                            batch=batch,
                            device=device,
                            optim_cfg=cfg.optim,
                        )
                        if in_wm_refresh:
                            wm_refresh_updates += 1
                    else:
                        wm_metrics = {"loss": 0.0}
                    classifier_metrics: dict[str, Any] = {
                        "loss": 0.0,
                        "acc": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "pos_frac": 0.0,
                        "prob_mean": 0.0,
                    }
                    if do_actor_phase:
                        last_phase = "ppo"
                        obs_for_update = {
                            "obs_embedding": batch["obs_embedding"],
                            "actions": batch["actions"],
                            "rewards": batch["rewards"],
                            "dones": batch["dones"],
                            "is_first": batch["is_first"],
                            "is_terminal": batch["is_terminal"],
                            "is_last": batch["is_last"],
                        }
                        if args.actor_update_kind == "outcome":
                            # WMPO/verl-style PPO: chunk-WM rollout + LatentSuccessClassifier
                            # outcome reward + eos_mask + zero-variance group filter.

                            # ── Snapshot policy params for drift measurement ─────────
                            policy_module = _unwrap(policy)
                            if ppo_log_f is not None:
                                _prev_params_flat = torch.cat(
                                    [
                                        p.detach().float().flatten()
                                        for p in policy_module.parameters()
                                        if p.requires_grad
                                    ]
                                )

                            if actor_update_route is None:
                                raise RuntimeError(
                                    "Actor update route was not resolved."
                                )
                            ac_metrics = actor_update_route.step_fn(
                                policy=policy,
                                **{actor_update_route.world_model_arg: world_model},
                                classifier=classifier,
                                classifier_threshold=classifier_threshold,
                                actor_optimizer=policy_optimizer,
                                obs=obs_for_update,
                                device=device,
                                algorithm_cfg=cfg.algorithm,
                                optim_cfg=cfg.optim,
                                ref_policy=ref_policy,
                            )

                            # ── Compute drift + write ppo_groups.jsonl ────────────────
                            if ppo_log_f is not None:
                                _curr_params_flat = torch.cat(
                                    [
                                        p.detach().float().flatten()
                                        for p in policy_module.parameters()
                                        if p.requires_grad
                                    ]
                                )
                                _delta = _curr_params_flat - _prev_params_flat
                                _prev_norm = float(_prev_params_flat.norm().item())
                                _drift_l2 = float(_delta.norm().item())
                                _drift_max = (
                                    float(_delta.abs().max().item())
                                    if _delta.numel() > 0
                                    else 0.0
                                )
                                _drift_rel = _drift_l2 / max(_prev_norm, 1e-12)
                                _start_points_per_window = int(
                                    ac_metrics.get(
                                        "wmpo/start_points_per_window",
                                        int(batch["obs_embedding"].shape[1]),
                                    )
                                )
                                _num_groups = int(ac_metrics.get("wmpo/num_groups", 0))
                                _batch_episode_ids = (
                                    batch["episode_ids"].detach().cpu().tolist()
                                )
                                _batch_collection_indices = (
                                    batch["collection_indices"].detach().cpu().tolist()
                                )
                                _batch_task_episode_indices = (
                                    batch["task_episode_indices"]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                )
                                _batch_episode_lengths = (
                                    batch["episode_lengths"].detach().cpu().tolist()
                                )
                                _batch_sample_limits = (
                                    batch["sample_limits"].detach().cpu().tolist()
                                )
                                _batch_source_ranks = (
                                    batch["source_ranks"].detach().cpu().tolist()
                                )
                                _batch_task_ids = (
                                    batch["task_ids"].detach().cpu().tolist()
                                )
                                _batch_start_indices = (
                                    batch["start_indices"].detach().cpu().tolist()
                                )
                                _batch_episode_success = (
                                    batch["episode_success"].detach().cpu().tolist()
                                )
                                _group_source_episode_ids: list[int] = []
                                _group_source_collection_indices: list[int] = []
                                _group_source_task_episode_indices: list[int] = []
                                _group_source_ranks: list[int] = []
                                _group_source_task_ids: list[int] = []
                                _group_source_episode_success: list[bool] = []
                                _group_source_episode_lengths: list[int] = []
                                _group_source_sample_limits: list[int] = []
                                _group_source_window_starts: list[int] = []
                                _group_source_time_offsets: list[int] = []
                                _group_source_absolute_start_indices: list[int] = []
                                for _group_idx in range(_num_groups):
                                    _batch_idx = _group_idx // max(
                                        1, _start_points_per_window
                                    )
                                    _time_offset = _group_idx % max(
                                        1, _start_points_per_window
                                    )
                                    if _batch_idx >= len(_batch_task_ids):
                                        break
                                    _window_start = int(
                                        _batch_start_indices[_batch_idx]
                                    )
                                    _group_source_episode_ids.append(
                                        int(_batch_episode_ids[_batch_idx])
                                    )
                                    _group_source_collection_indices.append(
                                        int(_batch_collection_indices[_batch_idx])
                                    )
                                    _group_source_task_episode_indices.append(
                                        int(_batch_task_episode_indices[_batch_idx])
                                    )
                                    _group_source_ranks.append(
                                        int(_batch_source_ranks[_batch_idx])
                                    )
                                    _group_source_task_ids.append(
                                        int(_batch_task_ids[_batch_idx])
                                    )
                                    _group_source_episode_success.append(
                                        bool(_batch_episode_success[_batch_idx])
                                    )
                                    _group_source_episode_lengths.append(
                                        int(_batch_episode_lengths[_batch_idx])
                                    )
                                    _group_source_sample_limits.append(
                                        int(_batch_sample_limits[_batch_idx])
                                    )
                                    _group_source_window_starts.append(_window_start)
                                    _group_source_time_offsets.append(int(_time_offset))
                                    _group_source_absolute_start_indices.append(
                                        _window_start + int(_time_offset)
                                    )
                                _entry = {
                                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "ts_unix": time.time(),
                                    "rank": int(rank),
                                    "world_size": int(world_size),
                                    "env_step": int(env_step),
                                    "update_step": int(
                                        update_step + 1
                                    ),  # this update hasn't bumped yet
                                    "success_rate_all_rollouts": float(
                                        ac_metrics.get("wmpo/success_rate", 0.0)
                                    ),
                                    "num_groups": _num_groups,
                                    "group_size": int(
                                        ac_metrics.get("wmpo/group_size", 0)
                                    ),
                                    "start_points_per_window": _start_points_per_window,
                                    "num_all_success_groups": int(
                                        ac_metrics.get("wmpo/num_all_success_groups", 0)
                                    ),
                                    "num_all_fail_groups": int(
                                        ac_metrics.get("wmpo/num_all_fail_groups", 0)
                                    ),
                                    "num_mixed_groups": int(
                                        ac_metrics.get("wmpo/num_mixed_groups", 0)
                                    ),
                                    "group_success_rates": list(
                                        ac_metrics.get("wmpo/group_success_rates", [])
                                    ),
                                    "group_success_counts": list(
                                        ac_metrics.get("wmpo/group_success_counts", [])
                                    ),
                                    "group_rollout_successes": list(
                                        ac_metrics.get(
                                            "wmpo/group_rollout_successes", []
                                        )
                                    ),
                                    "group_finish_steps": list(
                                        ac_metrics.get("wmpo/group_finish_steps", [])
                                    ),
                                    "group_has_variance": list(
                                        ac_metrics.get("wmpo/group_has_variance", [])
                                    ),
                                    "sample_episode_ids": _batch_episode_ids,
                                    "sample_collection_indices": _batch_collection_indices,
                                    "sample_task_episode_indices": _batch_task_episode_indices,
                                    "sample_source_ranks": _batch_source_ranks,
                                    "sample_task_ids": _batch_task_ids,
                                    "sample_window_start_indices": _batch_start_indices,
                                    "sample_episode_success": _batch_episode_success,
                                    "sample_episode_lengths": _batch_episode_lengths,
                                    "sample_limits": _batch_sample_limits,
                                    "group_source_episode_ids": _group_source_episode_ids,
                                    "group_source_collection_indices": _group_source_collection_indices,
                                    "group_source_task_episode_indices": _group_source_task_episode_indices,
                                    "group_source_ranks": _group_source_ranks,
                                    "group_source_task_ids": _group_source_task_ids,
                                    "group_source_episode_success": _group_source_episode_success,
                                    "group_source_episode_lengths": _group_source_episode_lengths,
                                    "group_source_sample_limits": _group_source_sample_limits,
                                    "group_source_window_start_indices": _group_source_window_starts,
                                    "group_source_time_offsets": _group_source_time_offsets,
                                    "group_source_absolute_start_indices": _group_source_absolute_start_indices,
                                    # Legacy names kept for older parsers; these are batch-window
                                    # fields, not one value per imagine group.
                                    "start_task_ids": _batch_task_ids,
                                    "start_indices": _batch_start_indices,
                                    "start_episode_success": _batch_episode_success,
                                    "replay_task_stats": replay.task_stats(task_ids),
                                    "global_replay_task_stats": last_global_replay_task_stats,
                                    "phase": last_phase,
                                    "wm_refresh_updates": int(wm_refresh_updates),
                                    "param_drift_l2": _drift_l2,
                                    "param_drift_max": _drift_max,
                                    "param_drift_relative": _drift_rel,
                                    "policy_param_norm": _prev_norm,
                                    "actor_loss": float(
                                        ac_metrics.get("actor_loss", 0.0)
                                    ),
                                    "actor_grad_norm": float(
                                        ac_metrics.get("actor_grad_norm", 0.0)
                                    ),
                                }
                                line = json.dumps(_entry) + "\n"
                                ppo_log_f.write(line)
                                ppo_log_f.flush()
                                if ppo_log_rank0_compat_f is not None:
                                    ppo_log_rank0_compat_f.write(line)
                                    ppo_log_rank0_compat_f.flush()
                                del _prev_params_flat, _curr_params_flat, _delta
                        elif args.actor_update_kind == "dense_chunk":
                            # WMPO chunk-WM PPO with dense per-step state-reward.
                            # No critic — TD-MPC/relabel side losses not yet wired.
                            if actor_update_route is None:
                                raise RuntimeError(
                                    "Actor update route was not resolved."
                                )
                            ac_metrics = actor_update_route.step_fn(
                                policy=policy,
                                **{actor_update_route.world_model_arg: world_model},
                                actor_optimizer=policy_optimizer,
                                obs=obs_for_update,
                                device=device,
                                algorithm_cfg=cfg.algorithm,
                                optim_cfg=cfg.optim,
                                ref_policy=ref_policy,
                            )
                        else:
                            ac_metrics = imagine_actor_critic_step(
                                policy=policy,
                                world_model=world_model,
                                critic=critic,
                                target_critic=target_critic,
                                actor_optimizer=policy_optimizer,
                                critic_optimizer=critic_optimizer,
                                return_tracker=return_tracker,
                                obs=obs_for_update,
                                device=device,
                                algorithm_cfg=cfg.algorithm,
                                optim_cfg=cfg.optim,
                                ref_policy=ref_policy,
                            )
                    else:
                        last_phase = "wm_refresh" if in_wm_refresh else "collect"
                        ac_metrics = {
                            "actor_loss": 0.0,
                            "critic_loss": 0.0,
                            "returns_mean": 0.0,
                            "reward_mean": 0.0,
                        }
                    if (
                        bool(args.update_classifier_online)
                        and classifier is not None
                        and classifier_optimizer is not None
                    ):
                        cls_module = _unwrap(classifier)
                        cls_ready_local = (
                            replay.classifier_window_count(
                                window=int(cls_module.cfg.window),
                                chunk_size=int(
                                    getattr(cls_module.cfg, "chunk_size", 1)
                                ),
                            )
                            > 0
                        )
                        cls_ready = bool(cls_ready_local)
                        if is_dist:
                            cls_ready_t = torch.tensor(
                                [int(cls_ready_local)], device=device, dtype=torch.long
                            )
                            dist.all_reduce(cls_ready_t, op=dist.ReduceOp.MIN)
                            cls_ready = bool(int(cls_ready_t.item()))
                        if cls_ready:
                            cls_metrics_list = []
                            for _cls_update in range(
                                max(1, int(args.classifier_updates_per_train))
                            ):
                                cls_metrics_list.append(
                                    online_classifier_update_step(
                                        classifier=classifier,
                                        optimizer=classifier_optimizer,
                                        replay=replay,
                                        device=device,
                                        batch_size=int(args.classifier_batch_size),
                                        early_neg_stride=int(
                                            args.classifier_early_neg_stride
                                        ),
                                        grad_clip=float(args.classifier_grad_clip),
                                    )
                                )
                            classifier_metrics = {
                                "loss": float(
                                    np.mean([item["loss"] for item in cls_metrics_list])
                                ),
                                "acc": float(
                                    np.mean([item["acc"] for item in cls_metrics_list])
                                ),
                                "precision": float(
                                    np.mean(
                                        [item["precision"] for item in cls_metrics_list]
                                    )
                                ),
                                "recall": float(
                                    np.mean(
                                        [item["recall"] for item in cls_metrics_list]
                                    )
                                ),
                                "f1": float(
                                    np.mean([item["f1"] for item in cls_metrics_list])
                                ),
                                "pos_frac": float(
                                    np.mean(
                                        [item["pos_frac"] for item in cls_metrics_list]
                                    )
                                ),
                                "prob_mean": float(
                                    np.mean(
                                        [item["prob_mean"] for item in cls_metrics_list]
                                    )
                                ),
                                "grad_norm": float(
                                    np.mean(
                                        [item["grad_norm"] for item in cls_metrics_list]
                                    )
                                ),
                                "last_batch": cls_metrics_list[-1].get("batch", {}),
                            }
                    update_step += 1
                    if args.max_train_updates is not None and update_step >= int(
                        args.max_train_updates
                    ):
                        stop_training = True
                    last_metrics = {
                        "wm": float(wm_metrics["loss"]),
                        "dyn": float(
                            wm_metrics.get("dyn_kl", wm_metrics.get("dyn_loss", 0.0))
                        ),
                        "rec": float(
                            wm_metrics.get(
                                "image_decoder_loss",
                                wm_metrics.get("image_recon_mse_loss", 0.0),
                            )
                        ),
                        "rew": float(wm_metrics.get("reward_loss", 0.0)),
                        "actor": float(ac_metrics["actor_loss"]),
                        "critic": float(ac_metrics["critic_loss"]),
                        "G": float(ac_metrics["returns_mean"]),
                        "raw_G": float(
                            ac_metrics.get(
                                "raw_returns_mean", ac_metrics["returns_mean"]
                            )
                        ),
                        "adv": float(ac_metrics.get("advantage_mean", 0.0)),
                        "kl": float(ac_metrics.get("ref_kl_mean", 0.0)),
                        "kl_coef": float(ac_metrics.get("kl_coef", 0.0)),
                        "reward_pred": float(ac_metrics["reward_mean"]),
                        "bc_ref": float(ac_metrics.get("actor_bc_ref_loss", 0.0)),
                        "bc_ref_scale": float(
                            ac_metrics.get("actor_bc_ref_scale", 0.0)
                        ),
                        "repval": float(ac_metrics.get("repval_loss", 0.0)),
                        "cls": float(classifier_metrics.get("loss", 0.0)),
                        "cls_acc": float(classifier_metrics.get("acc", 0.0)),
                        "cls_f1": float(classifier_metrics.get("f1", 0.0)),
                        "cls_prec": float(classifier_metrics.get("precision", 0.0)),
                        "cls_rec": float(classifier_metrics.get("recall", 0.0)),
                        "cls_pos": float(classifier_metrics.get("pos_frac", 0.0)),
                    }
                    train_update_entry = {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ts_unix": time.time(),
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "env_step": int(env_step),
                        "update_step": int(update_step),
                        "phase": str(last_phase),
                        "wm_refresh_updates": int(wm_refresh_updates),
                        "do_wm_phase": bool(do_wm_phase),
                        "do_actor_phase": bool(do_actor_phase),
                        "train_accum": float(train_accum),
                        "batch_episode_ids": batch["episode_ids"]
                        .detach()
                        .cpu()
                        .tolist(),
                        "batch_collection_indices": batch["collection_indices"]
                        .detach()
                        .cpu()
                        .tolist(),
                        "batch_task_episode_indices": batch["task_episode_indices"]
                        .detach()
                        .cpu()
                        .tolist(),
                        "batch_source_ranks": batch["source_ranks"]
                        .detach()
                        .cpu()
                        .tolist(),
                        "batch_task_ids": batch["task_ids"].detach().cpu().tolist(),
                        "batch_window_start_indices": batch["start_indices"]
                        .detach()
                        .cpu()
                        .tolist(),
                        "batch_episode_success": batch["episode_success"]
                        .detach()
                        .cpu()
                        .tolist(),
                        "batch_episode_lengths": batch["episode_lengths"]
                        .detach()
                        .cpu()
                        .tolist(),
                        "batch_sample_limits": batch["sample_limits"]
                        .detach()
                        .cpu()
                        .tolist(),
                        "batch_reward_sums": batch["rewards"]
                        .detach()
                        .cpu()
                        .sum(dim=1)
                        .tolist(),
                        "batch_terminal_counts": batch["is_terminal"]
                        .detach()
                        .cpu()
                        .sum(dim=1)
                        .tolist(),
                        "batch_done_counts": batch["dones"]
                        .detach()
                        .cpu()
                        .sum(dim=1)
                        .tolist(),
                        "local_replay_task_stats": replay.task_stats(task_ids),
                        "global_replay_task_stats": last_global_replay_task_stats,
                        "last_metrics": _json_safe(last_metrics),
                        "wm_metrics": _json_safe(wm_metrics),
                        "classifier_metrics": _json_safe(classifier_metrics),
                        "ac_metrics": _json_safe(ac_metrics),
                    }
                    train_update_log_f.write(json.dumps(train_update_entry) + "\n")
                    train_update_log_f.flush()
                    if stop_training:
                        break

            if is_rank0 and env_step % args.log_every == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                row = {
                    "env_step": env_step,
                    "update_step": update_step,
                    "phase": last_phase,
                    "wm_refresh_updates": wm_refresh_updates,
                    "replay": replay.num_transitions,
                    "local_train_ready": local_train_ready,
                    "global_coverage_ready": last_global_coverage_ready,
                    "all_ranks_train_ready": last_all_ranks_train_ready,
                    "replay_task_stats": replay.task_stats(task_ids),
                    "global_replay_task_stats": last_global_replay_task_stats,
                    "episode_len": episode_len,
                    "episode_return": episode_return,
                    "fps": env_step / elapsed,
                    **last_metrics,
                }
                print(
                    "[online-rynnvla] "
                    + " ".join(
                        f"{key}={value:.4g}"
                        if isinstance(value, float)
                        else f"{key}={value}"
                        for key, value in row.items()
                    ),
                    flush=True,
                )
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (
                is_rank0
                and update_step > 0
                and update_step % args.save_every == 0
                and update_step != last_saved_update
            ):
                save_checkpoint(
                    out_dir,
                    world_model=_unwrap(world_model),
                    policy=_unwrap(policy),
                    critic=_unwrap(critic),
                    target_critic=target_critic,
                    wm_optimizer=wm_optimizer,
                    policy_optimizer=policy_optimizer,
                    critic_optimizer=critic_optimizer,
                    return_tracker=return_tracker,
                    cfg=cfg,
                    env_step=env_step,
                    update_step=update_step,
                    classifier=_unwrap(classifier) if classifier is not None else None,
                    classifier_optimizer=classifier_optimizer,
                )
                last_saved_update = update_step

            if done:
                replay_record = replay.add_episode(episode)
                if rollout_dumper is not None:
                    rollout_dumper.add_episode(
                        episode,
                        task_id=int(info.get("task_id", obs.get("task_id", -1))),
                        success=bool(terminated),
                    )
                if episode_log_f is not None:
                    first_success_step = None
                    for idx, step in enumerate(episode):
                        if (
                            bool(step.get("success", False))
                            or float(step.get("is_terminal", 0.0)) > 0.5
                            or float(step.get("reward", 0.0)) > 0.0
                        ):
                            first_success_step = int(idx)
                            break
                    collect_chunk_ids = [
                        int(step.get("collect_chunk_id", -1))
                        for step in episode
                        if int(step.get("collect_chunk_id", -1)) >= 0
                    ]
                    collect_chunk_indices = [
                        int(step.get("collect_chunk_index", -1))
                        for step in episode
                        if int(step.get("collect_chunk_index", -1)) >= 0
                    ]
                    collect_chunk_lens = [
                        int(step.get("collect_chunk_len", -1))
                        for step in episode
                        if int(step.get("collect_chunk_len", -1)) > 0
                    ]
                    episode_entry = {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ts_unix": time.time(),
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "env_step": int(env_step),
                        "task_id": int(info.get("task_id", obs.get("task_id", -1))),
                        "episode_id": None
                        if replay_record is None
                        else int(replay_record["episode_id"]),
                        "collection_index": None
                        if replay_record is None
                        else int(replay_record["collection_index"]),
                        "task_episode_index": None
                        if replay_record is None
                        else int(replay_record["task_episode_index"]),
                        "episode_len": int(episode_len),
                        "episode_return": float(episode_return),
                        "success": bool(terminated),
                        "truncated": bool(truncated),
                        "first_success_step": first_success_step,
                        "stored_in_replay": replay_record is not None,
                        "replay_transitions": int(replay.num_transitions),
                        "num_collect_chunks": int(len(set(collect_chunk_ids))),
                        "collect_chunk_index_counts": dict(
                            Counter(collect_chunk_indices)
                        ),
                        "collect_chunk_len_counts": dict(Counter(collect_chunk_lens)),
                        "last_collect_chunk_index": (
                            None
                            if not collect_chunk_indices
                            else int(collect_chunk_indices[-1])
                        ),
                    }
                    episode_log_f.write(json.dumps(episode_entry) + "\n")
                    episode_log_f.flush()
                print(
                    f"[episode] rank={rank} env_step={env_step} task={info.get('task_id')} "
                    f"len={episode_len} return={episode_return:.3f} success={bool(terminated)} "
                    f"replay={replay.num_transitions}",
                    flush=True,
                )
                obs, _info = env.reset()
                episode = []
                episode_return = 0.0
                episode_len = 0
                latent = None
                prev_wm_action = None
                pending_policy_actions.clear()
                pending_chunk_index = 0
                pending_chunk_len = 0
            else:
                obs = next_obs

            if stop_training:
                print(
                    f"[online-rynnvla] reached max_train_updates={args.max_train_updates} "
                    f"at env_step={env_step}",
                    flush=True,
                )
                break

        if is_rank0:
            save_checkpoint(
                out_dir,
                world_model=_unwrap(world_model),
                policy=_unwrap(policy),
                critic=_unwrap(critic),
                target_critic=target_critic,
                wm_optimizer=wm_optimizer,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                return_tracker=return_tracker,
                cfg=cfg,
                env_step=final_env_step,
                update_step=update_step,
                classifier=_unwrap(classifier) if classifier is not None else None,
                classifier_optimizer=classifier_optimizer,
            )
    finally:
        env.close()
        if rollout_dumper is not None:
            rollout_dumper.close()
            if is_rank0:
                print(
                    f"[dump-rollouts] flushed {rollout_dumper.total_episodes} episodes "
                    f"({rollout_dumper.total_success} successes) across "
                    f"{rollout_dumper.shards_written} shard(s)",
                    flush=True,
                )
        episode_log_f.close()
        train_update_log_f.close()
        ppo_log_f.close()
        if ppo_log_rank0_compat_f is not None:
            ppo_log_rank0_compat_f.close()
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
