#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_online_pi0_action_hidden_dreamervla import (  # noqa: E402
    OnlineReplay,
    build_encoder,
    load_training_checkpoint,
    load_world_model_state,
    obs_to_action_hidden,
    save_checkpoint,
)
from src.algorithms.dreamer_vla import (  # noqa: E402
    imagine_actor_critic_step,
    world_model_pretrain_step,
)
from src.env.train_env import DreamerVLAOnlineTrainEnv  # noqa: E402
from src.models.critic.twohot_critic import ReturnPercentileTracker  # noqa: E402
from src.utils.fixed_step_video import FixedStepVideoRecorder  # noqa: E402
from src.utils.optim import build_optimizer  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.torch_utils import freeze_module  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-WM DreamerVLA actor/critic training with online or offline replay starts."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/dreamer_vla_libero_goal_pi0_action_hidden_head_actor.yaml"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--world-model-ckpt", required=True)
    parser.add_argument("--resume-ckpt", default=None)
    parser.add_argument("--vla-ckpt-path", default=str(PROJECT_ROOT / "data/ckpts/VLA_model_256/libero_goal"))
    parser.add_argument("--encoder-state-ckpt", default="",
                        help="Optional separate encoder state ckpt (legacy uses HF model_path only, leave empty).")
    parser.add_argument("--action-head-type", default="legacy", choices=["legacy", "pi0_query"],
                        help="Selects extract_action_hidden mode: legacy=35840-dim (v2 WM), pi0_query=5120-dim.")
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-source", choices=["online", "offline", "mixed"], default="online")
    parser.add_argument(
        "--offline-ratio",
        type=float,
        default=0.8,
        help="Probability of drawing actor/critic starts from offline replay when --start-source=mixed.",
    )
    parser.add_argument("--total-env-steps", type=int, default=60000)
    parser.add_argument("--max-train-updates", type=int, default=15000)
    parser.add_argument("--episode-horizon", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--min-replay", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=32.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-offline-windows", type=int, default=None)
    parser.add_argument("--imagination-horizon", type=int, default=10)
    parser.add_argument("--bc-to-vla", type=float, default=0.0)
    parser.add_argument(
        "--bc-to-ref",
        type=float,
        default=None,
        help="Head-level BC anchor against frozen ref_policy (MSE on action_chunk). "
             "Overrides cfg.algorithm.actor_bc_to_ref_scale. Leave unset to use config default.",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=None,
        help="KL(π_now ‖ π_ref) penalty coef (reward shaping). Overrides cfg.algorithm.kl_coef.",
    )
    parser.add_argument(
        "--prev-kl-coef",
        type=float,
        default=None,
        help="WMPO-style KL(π_now ‖ π_prev) penalty coef (reward shaping). "
             "π_prev is an EMA-tracked snapshot of the policy. Overrides cfg.algorithm.prev_kl_coef.",
    )
    parser.add_argument(
        "--prev-policy-tau",
        type=float,
        default=None,
        help="EMA rate for prev_policy tracking. Overrides cfg.algorithm.prev_policy_tau (default 0.02).",
    )
    parser.add_argument(
        "--policy-adapter-type",
        choices=["identity", "mlp", "residual_mlp"],
        default=None,
        help="Override cfg.policy.adapter_type. Use identity for no-drift VLA-head rollouts.",
    )
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=None,
        help="Override cfg.optim.policy.lr. Use 0.0 to keep the policy fixed.",
    )
    parser.add_argument("--target-token-id", type=int, default=10004)
    parser.add_argument("--rssm-action-scale", default="env", choices=["policy", "env"])
    parser.add_argument("--video-every-env-steps", type=int, default=500)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-max-frames", type=int, default=200)
    parser.add_argument("--video-frame-key", default="third_image")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--deterministic-collect", action="store_true")
    parser.add_argument("--freeze-world-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--joint-train-wm",
        action="store_true",
        help="Enable joint world-model training: run world_model_pretrain_step at every "
             "update_step using the same batch as imagine_actor_critic_step. Overrides "
             "--freeze-world-model when set.",
    )
    parser.add_argument(
        "--freeze-log-std",
        action="store_true",
        help="Override config: lock the Gaussian policy std at exp(initial_log_std). "
             "When set, log_std is registered with requires_grad=False and excluded from "
             "the actor optimizer state.",
    )
    parser.add_argument(
        "--resume-policy-non-strict",
        action="store_true",
        help="Load policy state with strict=False (use when resuming across adapter_type changes).",
    )
    parser.add_argument(
        "--reset-policy-optimizer",
        action="store_true",
        help="Skip restoring policy optimizer state (fresh Adam moments for new params).",
    )
    parser.add_argument(
        "--allow-tiny-trainable",
        action="store_true",
        help="Bypass the silent-freeze guard. By default the script aborts if the policy has "
             "<=1k trainable params (almost certainly the identity+freeze_output_projection bug).",
    )
    # ── WMPO-style reward labelling for the WM reward head ─────────────────
    parser.add_argument(
        "--reward-target-mode",
        choices=("raw", "per_window", "diffusion"),
        default="raw",
        help=("raw: use rewards as-is from the batch (legacy). "
              "per_window: WMPO-style binary label, BCE on the LAST latent only. "
              "diffusion: rewrite rewards to gamma^(W-1-t) on positive windows, 0 elsewhere."),
    )
    parser.add_argument("--diffusion-gamma", type=float, default=0.95)
    parser.add_argument("--swap-binary-head", action="store_true",
                        help="Swap WM reward_head with a fresh BinaryRewardHead (random init).")
    parser.add_argument("--binary-init-logit", type=float, default=0.0)
    parser.add_argument("--binary-pos-weight", type=float, default=1.0)
    parser.add_argument("--use-balanced-sampler", action="store_true",
                        help="Use LIBEROBalancedTerminalDataset + BalancedTerminalSampler for offline (50/50 pos/neg).")
    parser.add_argument("--freeze-non-reward-head", action="store_true",
                        help="When --joint-train-wm is on, only update reward_head (freeze encoder/RSSM/decoder/continue_head).")
    parser.add_argument("--wm-lr", type=float, default=None,
                        help="Override cfg.optim.world_model.lr (WMPO uses 1e-4 for reward model).")
    parser.add_argument("--wm-weight-decay", type=float, default=None,
                        help="Override cfg.optim.world_model.weight_decay.")
    return parser.parse_args()


def infinite_batches(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def build_offline_loader(cfg: Any, args: argparse.Namespace) -> tuple[DataLoader, Iterator[dict[str, Any]]]:
    cfg.dataset.sequence_length = int(args.sequence_length)
    cfg.dataloader.batch_size = int(args.batch_size)
    cfg.dataloader.num_workers = int(args.num_workers)
    cfg.dataloader.drop_last = True
    cfg.dataloader.shuffle = True
    if args.max_offline_windows is not None:
        cfg.dataset.max_windows = int(args.max_offline_windows)
    # If WMPO-style balanced sampler requested, swap dataset class to the
    # terminal-aware variant that exposes positive_indices / negative_indices.
    if bool(getattr(args, "use_balanced_sampler", False)):
        cfg.dataset._target_ = "src.dataloader.libero_balanced_terminal_dataset.LIBEROBalancedTerminalDataset"
    dataset = hydra.utils.instantiate(cfg.dataset)
    use_balanced = bool(getattr(args, "use_balanced_sampler", False))
    if use_balanced:
        from src.dataloader.libero_balanced_terminal_dataset import BalancedTerminalSampler
        sampler = BalancedTerminalSampler(
            dataset, num_samples=10**8, positive_ratio=0.5, seed=int(getattr(args, "seed", 0)),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            sampler=sampler,
            drop_last=True,
            num_workers=int(args.num_workers),
            pin_memory=False,
            persistent_workers=bool(int(args.num_workers) > 0),
            prefetch_factor=1 if int(args.num_workers) > 0 else None,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(args.num_workers),
            pin_memory=False,
            persistent_workers=bool(int(args.num_workers) > 0),
            prefetch_factor=1 if int(args.num_workers) > 0 else None,
        )
    return loader, infinite_batches(loader)


def actor_critic_obs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    obs_embedding = batch["obs_embedding"]
    actions = batch["actions"]
    rewards = batch.get("rewards")
    dones = batch.get("dones")
    if rewards is None:
        rewards = torch.zeros(actions.shape[:2], dtype=torch.float32)
    if dones is None:
        dones = torch.zeros(actions.shape[:2], dtype=torch.float32)
    is_first = batch.get("is_first")
    if is_first is None:
        is_first = torch.zeros(actions.shape[:2], dtype=torch.bool)
        is_first[:, 0] = True
    is_terminal = batch.get("is_terminal")
    if is_terminal is None:
        is_terminal = dones
    is_last = batch.get("is_last")
    if is_last is None:
        is_last = dones
    return {
        "obs_embedding": obs_embedding,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "is_first": is_first,
        "is_terminal": is_terminal,
        "is_last": is_last,
    }


def choose_batch(
    *,
    args: argparse.Namespace,
    online_replay: OnlineReplay | None,
    offline_iter: Iterator[dict[str, Any]] | None,
) -> tuple[str, dict[str, Any] | None]:
    can_online = online_replay is not None and online_replay.num_transitions >= int(args.min_replay) and bool(online_replay.episodes)
    can_offline = offline_iter is not None
    if args.start_source == "offline":
        return ("offline", next(offline_iter)) if can_offline else ("none", None)
    if args.start_source == "online":
        return ("online", online_replay.sample(args.batch_size)) if can_online and online_replay is not None else ("none", None)
    if can_offline and (not can_online or random.random() < float(args.offline_ratio)):
        return "offline", next(offline_iter)
    if can_online and online_replay is not None:
        return "online", online_replay.sample(args.batch_size)
    return ("offline", next(offline_iter)) if can_offline else ("none", None)


def log_row(log_path: Path, prefix: str, row: dict[str, Any]) -> None:
    print(
        f"[{prefix}] "
        + " ".join(
            f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in row.items()
        ),
        flush=True,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.init.vla_ckpt_path = args.vla_ckpt_path
    cfg.init.encoder_state_ckpt = args.encoder_state_ckpt if args.encoder_state_ckpt else None
    cfg.init.world_model_state_ckpt = args.world_model_ckpt
    cfg.training.out_dir = str(out_dir)
    cfg.training.distributed_strategy = "single"
    cfg.algorithm.imagination_horizon = int(args.imagination_horizon)
    cfg.algorithm.rssm_action_scale = str(args.rssm_action_scale)
    cfg.algorithm.repval_loss = False
    cfg.algorithm.repval_scale = 0.0
    cfg.algorithm.actor_bc_to_vla_scale = float(args.bc_to_vla)
    if args.bc_to_ref is not None:
        cfg.algorithm.actor_bc_to_ref_scale = float(args.bc_to_ref)
    if args.kl_coef is not None:
        cfg.algorithm.kl_coef = float(args.kl_coef)
    if args.prev_kl_coef is not None:
        cfg.algorithm.prev_kl_coef = float(args.prev_kl_coef)
    if args.prev_policy_tau is not None:
        cfg.algorithm.prev_policy_tau = float(args.prev_policy_tau)
    if args.policy_adapter_type is not None:
        cfg.policy.adapter_type = str(args.policy_adapter_type)
    if args.policy_lr is not None:
        cfg.optim.policy.lr = float(args.policy_lr)
    if bool(args.freeze_log_std):
        cfg.policy.freeze_log_std = True
    # WM optimizer overrides (WMPO-style reward-head fine-tune wants lr=1e-4)
    if args.wm_lr is not None:
        cfg.optim.world_model.lr = float(args.wm_lr)
    if args.wm_weight_decay is not None:
        cfg.optim.world_model.weight_decay = float(args.wm_weight_decay)
    # If swapping to a binary reward head, override the WM cfg before instantiation
    # so the freshly-built head matches the BinaryRewardHead architecture.
    if bool(args.swap_binary_head):
        cfg.world_model.reward_head_type = "binary"
        cfg.world_model.reward_init_logit = float(args.binary_init_logit)
        cfg.world_model.reward_pos_weight = float(args.binary_pos_weight)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    print(f"[frozen-wm] out_dir={out_dir}", flush=True)
    print(
        f"[frozen-wm] device={device} start_source={args.start_source} "
        f"offline_ratio={args.offline_ratio} h={args.imagination_horizon} bc={args.bc_to_vla} "
        f"policy_adapter={OmegaConf.select(cfg, 'policy.adapter_type', default='?')} "
        f"policy_lr={OmegaConf.select(cfg, 'optim.policy.lr', default='?')}",
        flush=True,
    )
    print(
        "[frozen-wm] algo_params "
        f"initial_log_std={OmegaConf.select(cfg, 'policy.initial_log_std', default='?')} "
        f"min_log_std={OmegaConf.select(cfg, 'policy.min_log_std', default='?')} "
        f"max_log_std={OmegaConf.select(cfg, 'policy.max_log_std', default='?')} "
        f"freeze_log_std={OmegaConf.select(cfg, 'policy.freeze_log_std', default=False)} "
        f"actent={OmegaConf.select(cfg, 'algorithm.actent', default='?')} "
        f"kl_coef={OmegaConf.select(cfg, 'algorithm.kl_coef', default=0.0)} "
        f"prev_kl_coef={OmegaConf.select(cfg, 'algorithm.prev_kl_coef', default=0.0)} "
        f"prev_policy_tau={OmegaConf.select(cfg, 'algorithm.prev_policy_tau', default=0.0)} "
        f"kl_penalty_kind={OmegaConf.select(cfg, 'algorithm.kl_penalty_kind', default='kl')} "
        f"bc_to_vla_scale={OmegaConf.select(cfg, 'algorithm.actor_bc_to_vla_scale', default=0.0)} "
        f"bc_to_ref_scale={OmegaConf.select(cfg, 'algorithm.actor_bc_to_ref_scale', default=0.0)} "
        f"grad_clip_norm={OmegaConf.select(cfg, 'optim.grad_clip_norm', default='?')} "
        f"wm_lr={OmegaConf.select(cfg, 'optim.world_model.lr', default='?')} "
        f"weight_decay={OmegaConf.select(cfg, 'optim.policy.weight_decay', default='?')}",
        flush=True,
    )
    print(f"[frozen-wm] wm_ckpt={args.world_model_ckpt}", flush=True)
    print("[frozen-wm] world_model_phase=disabled repval_loss=disabled", flush=True)

    world_model = hydra.utils.instantiate(cfg.world_model).to(device=device, dtype=torch.bfloat16)
    # If we swapped the head architecture, load WM weights but SKIP loading the old
    # reward_head (binary head has different shape than original twohot head).
    load_world_model_state(
        world_model,
        args.world_model_ckpt,
        reset_reward_head=(
            bool(args.swap_binary_head)
            or bool(OmegaConf.select(cfg, "init.reset_world_model_reward_head", default=False))
        ),
    )
    # WMPO-style: only update reward_head, freeze everything else of the WM.
    if bool(args.freeze_non_reward_head):
        n_train = 0
        for name, p in world_model.named_parameters():
            if "reward_head" in name:
                p.requires_grad = True
                n_train += p.numel()
            else:
                p.requires_grad = False
        print(f"[frozen-wm] freeze_non_reward_head=ON  WM trainable (reward_head only) = {n_train:,}",
              flush=True)

    wm_optimizer = build_optimizer(world_model, cfg.optim.world_model)
    # `--joint-train-wm` overrides `--freeze-world-model`: WM stays trainable and
    # in `.train()` mode so that world_model_pretrain_step can backprop through
    # encoder/RSSM/heads. Otherwise we replicate the legacy frozen-WM behavior.
    joint_train_wm = bool(args.joint_train_wm)
    if joint_train_wm:
        world_model.train()
        # If we freeze everything except reward_head, force non-reward modules to eval
        # to disable any potential dropout/BN drift.
        if bool(args.freeze_non_reward_head):
            for name, m in world_model.named_modules():
                if "reward_head" not in name and not any("reward_head" in n for n, _ in m.named_parameters(recurse=False)):
                    m.eval()
        print(
            f"[frozen-wm] joint_train_wm=ON  WM lr={cfg.optim.world_model.lr} "
            f"trainable={sum(p.numel() for p in world_model.parameters() if p.requires_grad):,}",
            flush=True,
        )
    else:
        if bool(args.freeze_world_model):
            freeze_module(world_model)
        world_model.eval()
    print(f"[frozen-wm] reward_target_mode={args.reward_target_mode}  swap_binary_head={bool(args.swap_binary_head)}  use_balanced_sampler={bool(args.use_balanced_sampler)}", flush=True)

    policy = hydra.utils.instantiate(cfg.policy).to(device)

    # WMPO-style frozen reference policy snapshot. Built BEFORE any resume load,
    # so the ref always reflects the SFT init (init_action_head_ckpt), not a
    # resumed RL state. Only kept in memory; not saved with the checkpoint.
    import copy as _copy
    ref_policy = _copy.deepcopy(policy)
    for _p in ref_policy.parameters():
        _p.requires_grad = False
    ref_policy.eval()

    # WMPO-style "previous policy" snapshot for the second KL term:
    # KL(π_now ‖ π_prev) where π_prev is the policy BEFORE the most recent
    # gradient update. Refreshed after each imagine_actor_critic_step.
    prev_policy = _copy.deepcopy(policy)
    for _p in prev_policy.parameters():
        _p.requires_grad = False
    prev_policy.eval()

    # --- silent-freeze guard ---------------------------------------------
    # Multiple earlier experiments accidentally trained nothing in the action
    # path because of (adapter_type=identity + freeze_output_projection=True),
    # leaving only the 7-dim log_std trainable. Catch that configuration here
    # before we waste GPU hours on a "training" run that can't learn.
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
        frozen_op = OmegaConf.select(cfg, "policy.freeze_output_projection", default=None)
        raise RuntimeError(
            f"Refusing to start: policy has only {n_policy_trainable} trainable parameters "
            f"({trainable_names}). This is almost certainly the silent-freeze trap: "
            f"adapter_type={adapter_type!r} + freeze_output_projection={frozen_op!r} leaves "
            f"only log_std trainable. Either set freeze_output_projection=false, use a "
            f"non-identity adapter, or pass --allow-tiny-trainable if this is intentional."
        )
    # ----------------------------------------------------------------------

    critic = hydra.utils.instantiate(cfg.critic).to(device)
    target_critic = hydra.utils.instantiate(cfg.critic).to(device)
    target_critic.load_state_dict(critic.state_dict())
    freeze_module(target_critic)
    policy_optimizer = build_optimizer(policy, cfg.optim.policy)
    critic_optimizer = build_optimizer(critic, cfg.optim.critic)
    return_tracker = ReturnPercentileTracker(
        decay=float(OmegaConf.select(cfg, "algorithm.return_tracker.decay", default=0.99)),
        low=float(OmegaConf.select(cfg, "algorithm.return_tracker.low", default=0.05)),
        high=float(OmegaConf.select(cfg, "algorithm.return_tracker.high", default=0.95)),
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
            policy_strict=not bool(args.resume_policy_non_strict),
            load_policy_optimizer=not bool(args.reset_policy_optimizer),
        )
        if bool(args.freeze_world_model):
            freeze_module(world_model)
        world_model.eval()

    offline_loader = None
    offline_iter = None
    if args.start_source in {"offline", "mixed"}:
        offline_loader, offline_iter = build_offline_loader(cfg, args)
        print(f"[frozen-wm] offline_windows={len(offline_loader.dataset)}", flush=True)

    collect_online = args.start_source in {"online", "mixed"}
    encoder = None
    processor = None
    env = None
    replay = None
    video_recorder = None
    obs = None
    episode: list[dict[str, Any]] = []
    episode_return = 0.0
    episode_len = 0
    latent = None
    prev_wm_action: torch.Tensor | None = None
    task_ids = tuple(int(item) for item in str(args.task_ids).split(",") if item.strip())
    if collect_online:
        encoder = build_encoder(args, device)
        processor = encoder._build_processor(device)
        env = DreamerVLAOnlineTrainEnv(
            task_suite_name=args.task_suite,
            task_id=task_ids[0],
            task_ids=task_ids,
            seed=args.seed,
            max_steps=args.episode_horizon,
            action_input="normalized",
            task_sampling="sequential",
            init_state_sampling="sequential",
            history_length=2,
            include_state=True,
            vla_rotate_180=True,
            obs_hidden_source="action_query",
            action_head_type=str(args.action_head_type),
        )
        replay = OnlineReplay(capacity=args.replay_size, sequence_length=args.sequence_length)
        obs, _info = env.reset(seed=args.seed)
        video_recorder = FixedStepVideoRecorder(
            every_steps=int(args.video_every_env_steps),
            output_dir=Path(args.video_dir).expanduser().resolve() if args.video_dir else out_dir / "videos",
            fps=int(args.video_fps),
            max_frames=int(args.video_max_frames),
            frame_key=str(args.video_frame_key),
        )

    update_step = int(resume_update_step)
    final_env_step = int(resume_env_step)
    last_saved_update = int(resume_update_step) if int(resume_update_step) % int(args.save_every) == 0 else -1
    train_accum = 0.0
    batch_steps = max(int(args.batch_size) * int(args.sequence_length), 1)
    log_path = out_dir / "frozen_wm_logs.json.txt"
    start_time = time.time()
    source_counts = {"online": 0, "offline": 0}

    def train_updates(num_updates: int, env_step: int) -> dict[str, float]:
        nonlocal update_step, last_saved_update
        last_metrics: dict[str, float] = {}
        for _ in range(int(num_updates)):
            if args.max_train_updates is not None and update_step >= int(args.max_train_updates):
                break
            source, batch = choose_batch(args=args, online_replay=replay, offline_iter=offline_iter)
            if batch is None:
                break
            source_counts[source] = source_counts.get(source, 0) + 1
            # Mark each window as "positive" (last step is success terminal) before any
            # reward-target transformation. Online batches use env-derived sparse rewards
            # (terminal +1); offline batches read either dense progress or sparse fields.
            rewards_t = batch["rewards"]
            if isinstance(rewards_t, torch.Tensor):
                last_rew = rewards_t[..., -1]
            else:
                last_rew = torch.as_tensor(rewards_t)[..., -1]
            is_positive_window = (last_rew > 0.5).to(dtype=torch.float32)  # [B]

            # Apply WMPO-style reward-target transformation BEFORE WM training.
            if args.reward_target_mode == "diffusion":
                W = int(args.sequence_length)
                gamma_t = float(args.diffusion_gamma)
                decay = torch.tensor(
                    [gamma_t ** (W - 1 - t) for t in range(W)],
                    dtype=torch.float32, device=is_positive_window.device,
                )
                # rewards[b, t] = decay[t]  if positive window else 0
                new_rewards = decay.unsqueeze(0) * is_positive_window.unsqueeze(-1)
                batch["rewards"] = new_rewards
            elif args.reward_target_mode == "per_window":
                # Build a sparse target: zeros everywhere except last step,
                # which gets the per-window {0, 1} label. WM-step still computes
                # per-step BCE but the dominant gradient comes from the last step
                # because the head is freshly trained from a near-zero init logit.
                W = int(args.sequence_length)
                new_rewards = torch.zeros(
                    (is_positive_window.shape[0], W),
                    dtype=torch.float32, device=is_positive_window.device,
                )
                new_rewards[:, -1] = is_positive_window
                batch["rewards"] = new_rewards
            # else: raw mode → keep batch["rewards"] as-is

            wm_metrics: dict[str, float] = {"loss": 0.0}
            if joint_train_wm:
                # WM loss expects raw batch keys (incl. `images`), not the
                # stripped actor_critic_obs view. Pass the full batch through.
                wm_metrics = world_model_pretrain_step(
                    policy=policy,
                    world_model=world_model,
                    optimizer=wm_optimizer,
                    batch=batch,
                    device=device,
                    optim_cfg=cfg.optim,
                )
                # imagine step expects world_model in eval() for inference-only
                # rollouts; switch back to train() after each AC step below.
                world_model.eval()
            ac_metrics = imagine_actor_critic_step(
                policy=policy,
                world_model=world_model,
                critic=critic,
                target_critic=target_critic,
                actor_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                return_tracker=return_tracker,
                obs=actor_critic_obs(batch),
                device=device,
                algorithm_cfg=cfg.algorithm,
                optim_cfg=cfg.optim,
                ref_policy=ref_policy,
                prev_policy=prev_policy,
            )
            # EMA-track prev_policy → current policy (similar to target_critic_tau).
            # KL(π_now ‖ π_prev) penalises rapid changes between consecutive updates.
            with torch.no_grad():
                _tau = float(cfg.algorithm.get("prev_policy_tau", 0.02))
                if _tau > 0.0:
                    for _pn, _pp in zip(policy.parameters(), prev_policy.parameters()):
                        _pp.data.lerp_(_pn.data, _tau)
            update_step += 1
            if joint_train_wm:
                # Restore train mode for the NEXT WM pretrain step.
                world_model.train()
            last_metrics = {
                "wm": float(wm_metrics.get("loss", 0.0)),
                "dyn": float(wm_metrics.get("dyn_kl", wm_metrics.get("dyn_loss", 0.0))),
                "rew": float(wm_metrics.get("reward_loss", 0.0)),
                "actor": float(ac_metrics["actor_loss"]),
                "critic": float(ac_metrics["critic_loss"]),
                "G": float(ac_metrics["returns_mean"]),
                "raw_G": float(ac_metrics.get("raw_returns_mean", ac_metrics["returns_mean"])),
                "adv": float(ac_metrics.get("advantage_mean", 0.0)),
                "adv_mag": float(ac_metrics.get("advantage_mag", 0.0)),
                "kl": float(ac_metrics.get("ref_kl_mean", 0.0)),
                "kl_coef": float(ac_metrics.get("kl_coef", 0.0)),
                "reward_pred": float(ac_metrics["reward_mean"]),
                "bc": float(ac_metrics.get("actor_bc_loss", 0.0)),
                "bc_ref": float(ac_metrics.get("actor_bc_ref_loss", 0.0)),
                "bc_ref_scale": float(ac_metrics.get("actor_bc_ref_scale", 0.0)),
                "rraw_p10": float(ac_metrics.get("reward_raw_p10", 0.0)),
                "rraw_p50": float(ac_metrics.get("reward_raw_p50", 0.0)),
                "rraw_p90": float(ac_metrics.get("reward_raw_p90", 0.0)),
                "rraw_min": float(ac_metrics.get("reward_raw_min", 0.0)),
                "rraw_max": float(ac_metrics.get("reward_raw_max", 0.0)),
                "kl_p50": float(ac_metrics.get("kl_p50", 0.0)),
                "kl_p90": float(ac_metrics.get("kl_p90", 0.0)),
                "g_pg": float(ac_metrics.get("actor_grad_norm_pg", 0.0)),
                "g_bcref": float(ac_metrics.get("actor_grad_norm_bc_ref", 0.0)),
                "g_ent": float(ac_metrics.get("actor_grad_norm_entropy", 0.0)),
                "g_out": float(ac_metrics.get("actor_grad_norm_output_projection", 0.0)),
                "cos_pg_bc": float(ac_metrics.get("actor_grad_cos_pg_bcref", 0.0)),
                "logp_mean": float(ac_metrics.get("log_prob_mean", 0.0)),
                "adv_pos": float(ac_metrics.get("advantage_pos_frac", 0.0)),
                "drift_raw": float(ac_metrics.get("actor_vla_drift_raw_mse", 0.0)),
                "drift_env": float(ac_metrics.get("actor_vla_drift_env_mse", 0.0)),
                "drift_env_clip": float(ac_metrics.get("actor_vla_drift_env_mse_clipped", 0.0)),
                "drift_env_mae": float(ac_metrics.get("actor_vla_drift_env_mae", 0.0)),
                "repval": float(ac_metrics.get("repval_loss", 0.0)),
                "src_online": float(source_counts.get("online", 0)),
                "src_offline": float(source_counts.get("offline", 0)),
            }
            if update_step > 0 and update_step % int(args.save_every) == 0 and update_step != last_saved_update:
                save_checkpoint(
                    out_dir,
                    world_model=world_model,
                    policy=policy,
                    critic=critic,
                    target_critic=target_critic,
                    wm_optimizer=wm_optimizer,
                    policy_optimizer=policy_optimizer,
                    critic_optimizer=critic_optimizer,
                    return_tracker=return_tracker,
                    cfg=cfg,
                    env_step=env_step,
                    update_step=update_step,
                )
                last_saved_update = update_step
        return last_metrics

    try:
        if not collect_online:
            while args.max_train_updates is None or update_step < int(args.max_train_updates):
                before = update_step
                metrics = train_updates(1, final_env_step)
                if update_step == before:
                    raise RuntimeError("offline frozen-WM training could not draw a batch")
                if update_step % int(args.log_every) == 0:
                    elapsed = max(time.time() - start_time, 1e-6)
                    row = {
                        "env_step": final_env_step,
                        "update_step": update_step,
                        "fps": update_step / elapsed,
                        **metrics,
                    }
                    log_row(log_path, "frozen-wm", row)
            return

        assert env is not None and replay is not None and encoder is not None and processor is not None and obs is not None
        for env_step in range(int(resume_env_step) + 1, int(args.total_env_steps) + 1):
            final_env_step = int(env_step)
            with torch.no_grad():
                obs_embedding = obs_to_action_hidden(encoder, processor, obs, device, args.target_token_id)
                is_first = bool(obs.get("is_first", False)) or latent is None
                if is_first:
                    latent = world_model({"mode": "encode_latent", "hidden": obs_embedding})
                else:
                    assert prev_wm_action is not None
                    latent = world_model({
                        "mode": "observe_next",
                        "latent": latent,
                        "hidden": obs_embedding,
                        "actions": prev_wm_action,
                        "is_first": False,
                    })
                feat = world_model({"mode": "actor_input", "latent": latent}).float()
                action_chunk, _log_prob, _extra = policy({
                    "mode": "sample",
                    "hidden": feat,
                    "deterministic": bool(args.deterministic_collect),
                    "return_chunk": True,
                })
                policy_action = action_chunk.reshape(-1, action_chunk.shape[-1])[0, :7].detach().cpu().float().numpy()

            next_obs, reward, terminated, truncated, info = env.step(policy_action)
            done = bool(terminated or truncated)
            episode_len += 1
            episode_return += float(reward)
            if video_recorder is not None:
                video_recorder.capture(next_obs)
                saved_video = video_recorder.maybe_save(
                    env_step=env_step,
                    episode_len=episode_len,
                    episode_return=episode_return,
                    task_id=int(info.get("task_id", obs.get("task_id", 0))),
                    success=bool(terminated),
                )
                if saved_video:
                    print(f"[video] saved {saved_video}", flush=True)

            wm_action_np = np.asarray(info["wm_action"], dtype=np.float32).reshape(-1)[:7]
            episode.append({
                "image": np.asarray(obs["image"], dtype=np.uint8),
                "obs_embedding": obs_embedding.squeeze(0).detach().cpu().numpy().astype(np.float32),
                "policy_action": policy_action.astype(np.float32),
                "wm_action": wm_action_np.astype(np.float32),
                "reward": np.float32(reward),
                "done": np.float32(done),
                "is_terminal": np.float32(terminated),
                "is_last": np.float32(done),
            })
            prev_wm_action = torch.from_numpy(wm_action_np).to(device=device, dtype=obs_embedding.dtype).unsqueeze(0)
            obs = next_obs

            train_accum += float(args.train_ratio) / float(batch_steps)
            num_updates = int(train_accum)
            train_accum -= float(num_updates)
            metrics = train_updates(num_updates, env_step) if num_updates > 0 else {}

            if env_step % int(args.log_every) == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                row = {
                    "env_step": env_step,
                    "update_step": update_step,
                    "replay": replay.num_transitions,
                    "episode_len": episode_len,
                    "episode_return": episode_return,
                    "fps": env_step / elapsed,
                    **metrics,
                }
                log_row(log_path, "frozen-wm", row)

            if done:
                replay.add_episode(episode)
                print(
                    f"[episode] env_step={env_step} task={info.get('task_id')} "
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

            if args.max_train_updates is not None and update_step >= int(args.max_train_updates):
                break
    finally:
        save_checkpoint(
            out_dir,
            world_model=world_model,
            policy=policy,
            critic=critic,
            target_critic=target_critic,
            wm_optimizer=wm_optimizer,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            return_tracker=return_tracker,
            cfg=cfg,
            env_step=final_env_step,
            update_step=update_step,
        )
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
