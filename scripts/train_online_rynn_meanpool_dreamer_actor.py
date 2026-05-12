#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from src.algorithms.dreamer_vla import imagine_actor_critic_step, world_model_pretrain_step
from src.env.libero_online_env import LIBEROOnlineEnv
from src.models.critic.twohot_critic import ReturnPercentileTracker
from src.models.encoder import RynnVLAEncoder
from src.utils.optim import build_optimizer
from src.utils.seed import set_seed
from src.utils.torch_utils import freeze_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online LIBERO DreamerVLA training with Rynn mean-pooled obs embedding and Dreamer actor."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/dreamer_vla_libero_goal_rynn_pixel_precomputed_actor.yaml"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--world-model-ckpt", required=True)
    parser.add_argument("--vla-ckpt-path", default=str(PROJECT_ROOT / "data/ckpts/VLA_model_256/libero_goal"))
    parser.add_argument(
        "--encoder-state-ckpt",
        default=str(
            PROJECT_ROOT
            / "data/outputs/vla/pretokenize_vla/pretokenize_vla_libero_goal_libero_goal_h5_20260508_060320"
            / "checkpoints/goal_h5_epoch000_train_vla_loss_1p323.ckpt"
        ),
    )
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-env-steps", type=int, default=200000)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--min-replay", type=int, default=64)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--updates-per-train", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--deterministic-collect", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_encoder_state(encoder: RynnVLAEncoder, ckpt_path: str) -> None:
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
    print(f"[init] encoder loaded: tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)}")


def _load_world_model_state(world_model: torch.nn.Module, ckpt_path: str, reset_reward_head: bool = False) -> None:
    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"world model ckpt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dicts", {}).get("world_model")
    if state is None:
        state = payload.get("model")
    if state is None:
        raise RuntimeError(f"{path} has no state_dicts.world_model or model")

    model_state = world_model.state_dict()
    dtype = next(world_model.parameters()).dtype
    cleaned: dict[str, torch.Tensor] = {}
    skipped_reward = 0
    for key, value in state.items():
        key = str(key).removeprefix("module.")
        if reset_reward_head and key.startswith("reward_head."):
            skipped_reward += 1
            continue
        if key.startswith("reward_head.net.") and not key.startswith("reward_head.net.net."):
            candidate = key.replace("reward_head.net.", "reward_head.net.net.", 1)
            if candidate in model_state:
                key = candidate
        if torch.is_floating_point(value):
            value = value.to(dtype=dtype)
        cleaned[key] = value

    mismatched = [
        (key, tuple(value.shape), tuple(model_state[key].shape))
        for key, value in cleaned.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    ]
    if mismatched:
        cleaned = {
            key: value
            for key, value in cleaned.items()
            if key not in model_state or tuple(value.shape) == tuple(model_state[key].shape)
        }
    missing, unexpected = world_model.load_state_dict(cleaned, strict=False)
    print(f"[init] world_model loaded: tensors={len(cleaned)} missing={len(missing)} unexpected={len(unexpected)}")
    if mismatched:
        print(f"[init] skipped shape mismatches: {mismatched[:5]}")
    if skipped_reward:
        print(f"[init] skipped reward head tensors: {skipped_reward}")


def _build_encoder(args: argparse.Namespace, device: torch.device) -> RynnVLAEncoder:
    encoder = RynnVLAEncoder(
        model_path=args.vla_ckpt_path,
        tokenizer_path=str(PROJECT_ROOT / "data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"),
        text_tokenizer_path=str(PROJECT_ROOT / "data/ckpts/chameleon/tokenizer/text_tokenizer.json"),
        chameleon_vqgan_config=str(PROJECT_ROOT / "data/ckpts/chameleon/tokenizer/vqgan.yaml"),
        chameleon_vqgan_ckpt=str(PROJECT_ROOT / "data/ckpts/chameleon/tokenizer/vqgan.ckpt"),
        resolution=256,
        action_dim=7,
        time_horizon=5,
        pool="mean",
        freeze_backbone=True,
    ).to(device)
    _load_encoder_state(encoder, args.encoder_state_ckpt)
    freeze_module(encoder)
    encoder.eval()
    return encoder


@torch.no_grad()
def _obs_to_embedding(
    encoder: RynnVLAEncoder,
    processor: Any,
    obs: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    images = []
    for third_pil, wrist_pil in obs["frame_history"]:
        images.extend([third_pil, wrist_pil])
    human = f"Finish the task: {obs['task_description']}." + "<|state|>" + "<|image|>" * len(images)
    record = {
        "conversations": [{"from": "human", "value": human}],
        "image": images,
        "action": [],
        "state": [np.asarray(obs["state"], dtype=np.float32)],
    }
    tokens = processor.process_item(record, training_mode=False)
    if isinstance(tokens, tuple):
        tokens = tokens[0]
    input_ids = [int(tok) for tok in tokens]
    labels = [[-100] * len(input_ids)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*nested_from_padded CUDA kernels only support.*")
        warnings.filterwarnings("ignore", message=r".*PyTorch API of nested tensors is in prototype stage.*")
        _, _, _, hidden_states, _, _, _ = encoder.backbone(
            input_ids=[input_ids],
            labels=labels,
            training=True,
            output_hidden_states=True,
            att_mask=False,
        )
    mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=device)
    mask[0, : len(input_ids)] = True
    weights = mask.to(hidden_states.dtype).unsqueeze(-1)
    return ((hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)).float()


def _sample_action(
    world_model: torch.nn.Module,
    policy: torch.nn.Module,
    latent: Any,
    deterministic: bool,
) -> tuple[np.ndarray, torch.Tensor]:
    with torch.no_grad():
        feat = world_model({"mode": "actor_input", "latent": latent}).float()
        action, _, _ = policy({"mode": "sample", "hidden": feat, "deterministic": deterministic})
    action = action.reshape(-1, action.shape[-1])[0, :7].float()
    action_np = action.detach().cpu().numpy().astype(np.float32)
    action_np = np.clip(action_np, -1.0, 1.0)
    return action_np, action.detach()


class OnlineReplay:
    def __init__(self, capacity: int, sequence_length: int) -> None:
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.episodes: deque[list[dict[str, Any]]] = deque()
        self._num_transitions = 0

    def add_episode(self, episode: list[dict[str, Any]]) -> None:
        if len(episode) < self.sequence_length:
            return
        self.episodes.append(episode)
        self._num_transitions += len(episode)
        while self._num_transitions > self.capacity and self.episodes:
            old = self.episodes.popleft()
            self._num_transitions -= len(old)

    @property
    def num_transitions(self) -> int:
        return self._num_transitions

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        valid = [ep for ep in self.episodes if len(ep) >= self.sequence_length]
        if not valid:
            raise RuntimeError("online replay has no sampleable sequences")
        rows = []
        for _ in range(int(batch_size)):
            ep = random.choice(valid)
            start = random.randint(0, len(ep) - self.sequence_length)
            rows.append(ep[start : start + self.sequence_length])

        def stack(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
            values = [[step[name] for step in row] for row in rows]
            tensor = torch.as_tensor(np.asarray(values))
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            return tensor

        return {
            "images": stack("image", torch.uint8),
            "obs_embedding": stack("obs_embedding", torch.float32),
            "actions": stack("action", torch.float32),
            "rewards": stack("reward", torch.float32),
            "dones": stack("done", torch.float32),
            "is_first": stack("is_first", torch.bool),
            "is_terminal": stack("is_terminal", torch.float32),
            "is_last": stack("is_last", torch.float32),
        }


def _save_checkpoint(
    out_dir: Path,
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
) -> None:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cfg": cfg,
        "env_step": int(env_step),
        "update_step": int(update_step),
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
    torch.save(payload, ckpt_dir / "latest.ckpt")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)
    cfg.init.vla_ckpt_path = args.vla_ckpt_path
    cfg.init.encoder_state_ckpt = args.encoder_state_ckpt
    cfg.init.world_model_state_ckpt = args.world_model_ckpt
    cfg.training.out_dir = str(out_dir)
    cfg.training.distributed_strategy = "single"

    print(f"[online] out_dir={out_dir}")
    print(f"[online] device={device} task_suite={args.task_suite} task_ids={args.task_ids}")
    print(f"[online] wm_ckpt={args.world_model_ckpt}")
    print(f"[online] encoder_ckpt={args.encoder_state_ckpt}")

    encoder = _build_encoder(args, device)
    processor = encoder._build_processor(device)

    world_model = hydra.utils.instantiate(cfg.world_model).to(device=device, dtype=torch.bfloat16)
    _load_world_model_state(
        world_model,
        args.world_model_ckpt,
        reset_reward_head=bool(OmegaConf.select(cfg, "init.reset_world_model_reward_head", default=False)),
    )
    policy = hydra.utils.instantiate(cfg.policy).to(device)
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

    task_ids = [int(item) for item in str(args.task_ids).split(",") if item.strip()]
    env = LIBEROOnlineEnv(
        task_suite_name=args.task_suite,
        task_id=task_ids[0],
        task_ids=task_ids,
        seed=args.seed,
        max_steps=args.max_episode_steps,
        action_input="normalized",
        history_length=1,
        task_sampling="sequential",
        init_state_sampling="sequential",
        sparse_success_reward=True,
    )
    replay = OnlineReplay(capacity=args.replay_size, sequence_length=args.sequence_length)

    obs, _info = env.reset(seed=args.seed)
    episode: list[dict[str, Any]] = []
    episode_return = 0.0
    episode_len = 0
    latent = None
    prev_action = None
    update_step = 0
    log_path = out_dir / "online_logs.json.txt"
    last_metrics: dict[str, float] = {}
    start_time = time.time()

    try:
        for env_step in range(1, args.max_env_steps + 1):
            obs_embedding = _obs_to_embedding(encoder, processor, obs, device)
            is_first = bool(obs.get("is_first", False)) or latent is None
            with torch.no_grad():
                if is_first:
                    latent = world_model({"mode": "encode_latent", "hidden": obs_embedding})
                else:
                    assert prev_action is not None
                    latent = world_model({
                        "mode": "observe_next",
                        "latent": latent,
                        "hidden": obs_embedding,
                        "actions": prev_action.to(device=device)[None],
                        "is_first": False,
                    })
            action_np, action_tensor = _sample_action(
                world_model,
                policy,
                latent,
                deterministic=bool(args.deterministic_collect),
            )
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)

            episode.append({
                "image": np.asarray(obs["image"], dtype=np.uint8),
                "obs_embedding": obs_embedding.squeeze(0).detach().cpu().numpy().astype(np.float32),
                "action": action_np.astype(np.float32),
                "reward": np.float32(reward),
                "done": np.float32(done),
                "is_first": bool(is_first),
                "is_terminal": np.float32(terminated),
                "is_last": np.float32(done),
            })
            episode_return += float(reward)
            episode_len += 1
            prev_action = action_tensor.detach()

            if (
                replay.num_transitions >= args.min_replay
                and env_step % args.train_every == 0
                and len(replay.episodes) > 0
            ):
                for _ in range(args.updates_per_train):
                    batch = replay.sample(args.batch_size)
                    wm_metrics = world_model_pretrain_step(
                        policy=policy,
                        world_model=world_model,
                        optimizer=wm_optimizer,
                        batch=batch,
                        device=device,
                        optim_cfg=cfg.optim,
                    )
                    ac_obs = {
                        "obs_embedding": batch["obs_embedding"].to(device),
                        "actions": batch["actions"].to(device),
                        "is_first": batch["is_first"].to(device),
                    }
                    ac_metrics = imagine_actor_critic_step(
                        policy=policy,
                        world_model=world_model,
                        critic=critic,
                        target_critic=target_critic,
                        actor_optimizer=policy_optimizer,
                        critic_optimizer=critic_optimizer,
                        return_tracker=return_tracker,
                        obs=ac_obs,
                        device=device,
                        algorithm_cfg=cfg.algorithm,
                        optim_cfg=cfg.optim,
                    )
                    update_step += 1
                    last_metrics = {
                        "wm": wm_metrics["loss"],
                        "dyn": wm_metrics.get("dyn_kl", wm_metrics.get("dyn_loss", 0.0)),
                        "rec": wm_metrics.get("image_decoder_loss", wm_metrics.get("image_recon_mse_loss", 0.0)),
                        "rew": wm_metrics.get("reward_loss", 0.0),
                        "actor": ac_metrics["actor_loss"],
                        "critic": ac_metrics["critic_loss"],
                        "G": ac_metrics["returns_mean"],
                        "reward_pred": ac_metrics["reward_mean"],
                    }

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
                    "[online] "
                    + " ".join(
                        f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
                        for key, value in row.items()
                    ),
                    flush=True,
                )
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            if update_step > 0 and update_step % args.save_every == 0:
                _save_checkpoint(
                    out_dir,
                    world_model,
                    policy,
                    critic,
                    target_critic,
                    wm_optimizer,
                    policy_optimizer,
                    critic_optimizer,
                    return_tracker,
                    cfg,
                    env_step,
                    update_step,
                )

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
                prev_action = None
            else:
                obs = next_obs

        _save_checkpoint(
            out_dir,
            world_model,
            policy,
            critic,
            target_critic,
            wm_optimizer,
            policy_optimizer,
            critic_optimizer,
            return_tracker,
            cfg,
            args.max_env_steps,
            update_step,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
