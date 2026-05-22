#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.dreamer_vla import imagine_actor_critic_step, world_model_pretrain_step
from src.env.train_env import DreamerVLAOnlineTrainEnv
from src.models.critic.twohot_critic import ReturnPercentileTracker
from src.models.encoder import RynnVLAEncoder
from src.utils.fixed_step_video import FixedStepVideoRecorder
from src.utils.optim import build_optimizer
from src.utils.seed import set_seed
from src.utils.torch_utils import freeze_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online DreamerVLA training with pi0 action-query hidden inputs.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/dreamer_vla_libero_goal_pi0_action_hidden_head_actor.yaml"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--world-model-ckpt", required=True)
    parser.add_argument("--resume-ckpt", default=None)
    parser.add_argument("--vla-ckpt-path", default=str(PROJECT_ROOT / "data/ckpts/VLA_model_256/libero_goal"))
    parser.add_argument("--encoder-state-ckpt", default=str(PROJECT_ROOT / "data/ckpts/pi0_query_vla_libero_goal/epoch003_train_vla_loss1.255_success8of10.ckpt"))
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
    parser.add_argument("--min-replay", type=int, default=64)
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-token-id", type=int, default=10004)
    parser.add_argument("--rssm-action-scale", default="env", choices=["policy", "env"])
    parser.add_argument("--run-wm-phase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-actor-critic-phase", action=argparse.BooleanOptionalAction, default=True)
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
    return parser.parse_args()


def load_encoder_state(encoder: RynnVLAEncoder, ckpt_path: str) -> None:
    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"encoder state ckpt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dicts", {}).get("encoder")
    if state is None:
        raise RuntimeError(f"{path} has no state_dicts.encoder")
    dtype = next(encoder.parameters()).dtype
    state = {key: value.to(dtype=dtype) if torch.is_floating_point(value) else value for key, value in state.items()}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"[init] encoder loaded: tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def build_encoder(args: argparse.Namespace, device: torch.device) -> RynnVLAEncoder:
    action_head_type = str(getattr(args, "action_head_type", None) or "legacy")
    encoder = RynnVLAEncoder(
        model_path=args.vla_ckpt_path,
        tokenizer_path=str(PROJECT_ROOT / "data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"),
        text_tokenizer_path=str(PROJECT_ROOT / "data/ckpts/chameleon/tokenizer/text_tokenizer.json"),
        chameleon_vqgan_config=str(PROJECT_ROOT / "data/ckpts/chameleon/tokenizer/vqgan.yaml"),
        chameleon_vqgan_ckpt=str(PROJECT_ROOT / "data/ckpts/chameleon/tokenizer/vqgan.ckpt"),
        resolution=256,
        action_dim=7,
        time_horizon=5,
        action_head_type=action_head_type,
        pool="mean",
        freeze_backbone=True,
    ).to(device)
    enc_ckpt = getattr(args, "encoder_state_ckpt", None)
    if enc_ckpt:
        load_encoder_state(encoder, enc_ckpt)
    else:
        print(f"[init] encoder built with action_head_type={action_head_type}, no separate encoder_state_ckpt", flush=True)
    freeze_module(encoder)
    encoder.eval()
    return encoder


def load_world_model_state(world_model: torch.nn.Module, ckpt_path: str, reset_reward_head: bool = False) -> None:
    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"world model ckpt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dicts", {}).get("world_model") or payload.get("model")
    if state is None:
        raise RuntimeError(f"{path} has no state_dicts.world_model or model")
    model_state = world_model.state_dict()
    dtype = next(world_model.parameters()).dtype
    cleaned: dict[str, torch.Tensor] = {}
    skipped_reward = 0
    for raw_key, value in state.items():
        key = str(raw_key).removeprefix("module.")
        if reset_reward_head and key.startswith("reward_head."):
            skipped_reward += 1
            continue
        if key.startswith("reward_head.net.") and not key.startswith("reward_head.net.net."):
            candidate = key.replace("reward_head.net.", "reward_head.net.net.", 1)
            if candidate in model_state:
                key = candidate
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
            continue
        cleaned[key] = value.to(dtype=dtype) if torch.is_floating_point(value) else value
    missing, unexpected = world_model.load_state_dict(cleaned, strict=False)
    print(f"[init] world_model loaded: tensors={len(cleaned)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if skipped_reward:
        print(f"[init] skipped reward head tensors: {skipped_reward}", flush=True)


@torch.no_grad()
def obs_to_action_hidden(
    encoder: RynnVLAEncoder,
    processor: Any,
    obs: dict[str, Any],
    device: torch.device,
    target_token_id: int,
) -> torch.Tensor:
    record = obs["vla_record"]
    tokens = processor.process_item(record, training_mode=False)
    if isinstance(tokens, tuple):
        tokens = tokens[0]
    input_ids_list = [[int(tok) for tok in tokens]]
    labels = [[-100] * len(input_ids_list[0])]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*nested_from_padded CUDA kernels only support.*")
        warnings.filterwarnings("ignore", message=r".*PyTorch API of nested tensors is in prototype stage.*")
        _, _, _, hidden_states, _, _, _ = encoder.backbone(
            input_ids=input_ids_list,
            labels=labels,
            training=True,
            output_hidden_states=True,
            att_mask=False,
        )
    max_len = int(hidden_states.shape[1])
    seq = input_ids_list[0]
    row = [int(tok) for tok in seq[:max_len]] + [int(target_token_id)]
    mask = [1] * min(len(seq), max_len) + [1]
    target_len = max_len + 1
    if len(row) < target_len:
        row.extend([0] * (target_len - len(row)))
        mask.extend([0] * (target_len - len(mask)))
    input_ids = torch.tensor([row[:target_len]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask[:target_len]], dtype=torch.bool, device=device)
    action_hidden = encoder.extract_action_hidden(
        hidden_states=hidden_states,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_token_id=int(target_token_id),
        eval=True,
    )
    return action_hidden.reshape(action_hidden.shape[0], -1).float()


class OnlineReplay:
    def __init__(self, capacity: int, sequence_length: int) -> None:
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.episodes: deque[list[dict[str, Any]]] = deque()
        self._num_transitions = 0

    @property
    def num_transitions(self) -> int:
        return int(self._num_transitions)

    def add_episode(self, episode: list[dict[str, Any]]) -> None:
        if len(episode) < self.sequence_length:
            return
        self.episodes.append(episode)
        self._num_transitions += len(episode)
        while self._num_transitions > self.capacity and self.episodes:
            old = self.episodes.popleft()
            self._num_transitions -= len(old)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        valid = [episode for episode in self.episodes if len(episode) >= self.sequence_length]
        if not valid:
            raise RuntimeError("online replay has no full sequences")
        windows = []
        for _ in range(int(batch_size)):
            episode = random.choice(valid)
            start = random.randint(0, len(episode) - self.sequence_length)
            windows.append(episode[start:start + self.sequence_length])

        images = np.stack([[step["image"] for step in window] for window in windows], axis=0)
        obs_embedding = np.stack([[step["obs_embedding"] for step in window] for window in windows], axis=0)
        rewards = np.stack([[step["reward"] for step in window] for window in windows], axis=0)
        dones = np.stack([[step["done"] for step in window] for window in windows], axis=0)
        is_terminal = np.stack([[step["is_terminal"] for step in window] for window in windows], axis=0)
        is_last = np.stack([[step["is_last"] for step in window] for window in windows], axis=0)
        actions = np.zeros((len(windows), self.sequence_length, 7), dtype=np.float32)
        for batch_idx, window in enumerate(windows):
            for time_idx in range(1, self.sequence_length):
                actions[batch_idx, time_idx] = np.asarray(window[time_idx - 1]["wm_action"], dtype=np.float32)
        is_first = np.zeros((len(windows), self.sequence_length), dtype=np.bool_)
        is_first[:, 0] = True
        return {
            "images": torch.from_numpy(images).to(torch.float32),
            "obs_embedding": torch.from_numpy(obs_embedding).to(torch.float32),
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards.astype(np.float32, copy=False)),
            "dones": torch.from_numpy(dones.astype(np.float32, copy=False)),
            "is_terminal": torch.from_numpy(is_terminal.astype(np.float32, copy=False)),
            "is_last": torch.from_numpy(is_last.astype(np.float32, copy=False)),
            "is_first": torch.from_numpy(is_first),
        }


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
    optimizers = {
        "world_model_optimizer": wm_optimizer,
        "policy_optimizer": policy_optimizer,
        "critic_optimizer": critic_optimizer,
    }
    for key, module in modules.items():
        if key in state_dicts:
            use_strict = True if key != "policy" else bool(policy_strict)
            missing, unexpected = module.load_state_dict(state_dicts[key], strict=use_strict)
            if not use_strict and (missing or unexpected):
                print(
                    f"[resume] {key} loaded non-strict: "
                    f"missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}",
                    flush=True,
                )
    for key, optimizer in optimizers.items():
        if key in state_dicts:
            if key == "policy_optimizer" and not bool(load_policy_optimizer):
                print("[resume] skipping policy_optimizer state (fresh moments for new params)", flush=True)
                continue
            optimizer.load_state_dict(state_dicts[key])
    if "return_tracker" in state_dicts:
        return_tracker.load_state_dict(state_dicts["return_tracker"])
    env_step = int(payload.get("env_step", 0))
    update_step = int(payload.get("update_step", 0))
    print(f"[resume] loaded {path} env_step={env_step} update_step={update_step}", flush=True)
    return env_step, update_step


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

    print(f"[online-pi0] out_dir={out_dir}", flush=True)
    print(
        "[online-pi0] algo_params "
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
    print(f"[online-pi0] device={device} task_suite={args.task_suite} task_ids={args.task_ids}", flush=True)
    print(f"[online-pi0] wm_ckpt={args.world_model_ckpt}", flush=True)
    print(
        f"[online-pi0] episode_horizon={args.episode_horizon} "
        f"total_env_steps={args.total_env_steps} "
        f"max_train_updates={args.max_train_updates} "
        f"train_ratio={args.train_ratio}",
        flush=True,
    )
    print("[online-pi0] input=vla_policy history=2 state rotate180 action_query", flush=True)
    if int(args.video_every_env_steps) > 0:
        print(
            f"[online-pi0] video_every_env_steps={args.video_every_env_steps} "
            f"video_fps={args.video_fps} video_frame_key={args.video_frame_key}",
            flush=True,
        )
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    encoder = build_encoder(args, device)
    processor = encoder._build_processor(device)
    world_model = hydra.utils.instantiate(cfg.world_model).to(device=device, dtype=torch.bfloat16)
    load_world_model_state(
        world_model,
        args.world_model_ckpt,
        reset_reward_head=bool(OmegaConf.select(cfg, "init.reset_world_model_reward_head", default=False)),
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
        frozen_op = OmegaConf.select(cfg, "policy.freeze_output_projection", default=None)
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

    wm_optimizer = build_optimizer(world_model, cfg.optim.world_model)
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
        )

    task_ids = tuple(int(item) for item in str(args.task_ids).split(",") if item.strip())
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
        action_head_type="pi0_query",
    )
    replay = OnlineReplay(capacity=args.replay_size, sequence_length=args.sequence_length)
    obs, _info = env.reset(seed=args.seed)
    episode: list[dict[str, Any]] = []
    episode_return = 0.0
    episode_len = 0
    latent = None
    prev_wm_action: torch.Tensor | None = None
    update_step = int(resume_update_step)
    last_saved_update = int(resume_update_step) if int(resume_update_step) % int(args.save_every) == 0 else -1
    train_accum = 0.0
    batch_steps = max(int(args.batch_size) * int(args.sequence_length), 1)
    final_env_step = int(resume_env_step)
    stop_training = False
    if args.max_train_updates is not None and update_step >= int(args.max_train_updates):
        stop_training = True
    log_path = out_dir / "online_logs.json.txt"
    start_time = time.time()
    last_metrics: dict[str, float] = {}
    video_recorder = FixedStepVideoRecorder(
        every_steps=int(args.video_every_env_steps),
        output_dir=Path(args.video_dir).expanduser().resolve() if args.video_dir else out_dir / "videos",
        fps=int(args.video_fps),
        max_frames=int(args.video_max_frames),
        frame_key=str(args.video_frame_key),
    )

    try:
        for env_step in range(int(resume_env_step) + 1, int(args.total_env_steps) + 1):
            if stop_training:
                break
            final_env_step = int(env_step)
            obs_embedding = obs_to_action_hidden(encoder, processor, obs, device, args.target_token_id)
            is_first = bool(obs.get("is_first", False)) or latent is None
            with torch.no_grad():
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
            wm_action_np = np.asarray(info["wm_action"], dtype=np.float32).reshape(-1)[:7]
            episode.append({
                "image": np.asarray(obs["image"], dtype=np.uint8),
                "obs_embedding": obs_embedding.squeeze(0).detach().cpu().numpy().astype(np.float32),
                "policy_action": policy_action.astype(np.float32),
                "wm_action": wm_action_np.astype(np.float32),
                "reward": np.float32(reward),
                "done": np.float32(done),
                "is_first": bool(is_first),
                "is_terminal": np.float32(terminated),
                "is_last": np.float32(done),
            })
            episode_return += float(reward)
            episode_len += 1
            prev_wm_action = torch.from_numpy(wm_action_np).to(device=device, dtype=obs_embedding.dtype).unsqueeze(0)

            num_updates = 0
            if replay.num_transitions >= args.min_replay and replay.episodes:
                if args.train_every is not None:
                    if env_step % int(args.train_every) == 0:
                        num_updates = int(args.updates_per_train)
                else:
                    train_accum += float(args.train_ratio) / float(batch_steps)
                    num_updates = int(train_accum)
                    train_accum -= float(num_updates)

            if num_updates > 0:
                for _ in range(num_updates):
                    if args.max_train_updates is not None and update_step >= int(args.max_train_updates):
                        stop_training = True
                        break
                    batch = replay.sample(args.batch_size)
                    if args.run_wm_phase:
                        wm_metrics = world_model_pretrain_step(
                            policy=policy,
                            world_model=world_model,
                            optimizer=wm_optimizer,
                            batch=batch,
                            device=device,
                            optim_cfg=cfg.optim,
                        )
                    else:
                        wm_metrics = {"loss": 0.0}
                    if args.run_actor_critic_phase:
                        ac_metrics = imagine_actor_critic_step(
                            policy=policy,
                            world_model=world_model,
                            critic=critic,
                            target_critic=target_critic,
                            actor_optimizer=policy_optimizer,
                            critic_optimizer=critic_optimizer,
                            return_tracker=return_tracker,
                            obs={
                                "obs_embedding": batch["obs_embedding"],
                                "actions": batch["actions"],
                                "rewards": batch["rewards"],
                                "dones": batch["dones"],
                                "is_first": batch["is_first"],
                                "is_terminal": batch["is_terminal"],
                                "is_last": batch["is_last"],
                            },
                            device=device,
                            algorithm_cfg=cfg.algorithm,
                            optim_cfg=cfg.optim,
                            ref_policy=ref_policy,
                        )
                    else:
                        ac_metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "returns_mean": 0.0, "reward_mean": 0.0}
                    update_step += 1
                    if args.max_train_updates is not None and update_step >= int(args.max_train_updates):
                        stop_training = True
                    last_metrics = {
                        "wm": float(wm_metrics["loss"]),
                        "dyn": float(wm_metrics.get("dyn_kl", wm_metrics.get("dyn_loss", 0.0))),
                        "rec": float(wm_metrics.get("image_decoder_loss", wm_metrics.get("image_recon_mse_loss", 0.0))),
                        "rew": float(wm_metrics.get("reward_loss", 0.0)),
                        "actor": float(ac_metrics["actor_loss"]),
                        "critic": float(ac_metrics["critic_loss"]),
                        "G": float(ac_metrics["returns_mean"]),
                        "raw_G": float(ac_metrics.get("raw_returns_mean", ac_metrics["returns_mean"])),
                        "adv": float(ac_metrics.get("advantage_mean", 0.0)),
                        "kl": float(ac_metrics.get("ref_kl_mean", 0.0)),
                        "kl_coef": float(ac_metrics.get("kl_coef", 0.0)),
                        "reward_pred": float(ac_metrics["reward_mean"]),
                        "bc_ref": float(ac_metrics.get("actor_bc_ref_loss", 0.0)),
                        "bc_ref_scale": float(ac_metrics.get("actor_bc_ref_scale", 0.0)),
                        "repval": float(ac_metrics.get("repval_loss", 0.0)),
                    }
                    if stop_training:
                        break

            if env_step % args.log_every == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                row = {
                    "env_step": env_step,
                    "update_step": update_step,
                    "replay": replay.num_transitions,
                    "episode_len": episode_len,
                    "episode_return": episode_return,
                    "fps": env_step / elapsed,
                    **last_metrics,
                }
                print(
                    "[online-pi0] "
                    + " ".join(
                        f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
                        for key, value in row.items()
                    ),
                    flush=True,
                )
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            if update_step > 0 and update_step % args.save_every == 0 and update_step != last_saved_update:
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
            else:
                obs = next_obs

            if stop_training:
                print(
                    f"[online-pi0] reached max_train_updates={args.max_train_updates} "
                    f"at env_step={env_step}",
                    flush=True,
                )
                break

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
    finally:
        env.close()


if __name__ == "__main__":
    main()
