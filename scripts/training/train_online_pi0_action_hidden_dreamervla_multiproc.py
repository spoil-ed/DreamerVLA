#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.train_env import DreamerVLAOnlineTrainEnv

from scripts.training.train_online_pi0_action_hidden_dreamervla_multienv import (
    OnlineReplay,
    _init_distributed,
    build_encoder,
    load_world_model_state,
    obs_batch_to_action_hidden,
)


@dataclass
class ActionRequest:
    collector_id: int
    obs: dict[str, Any]
    is_first: bool


@dataclass
class TransitionMsg:
    collector_id: int
    obs: dict[str, Any]
    policy_action: np.ndarray
    wm_action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    task_id: int
    chunk_id: int
    chunk_index: int
    chunk_len: int


@dataclass
class CollectorDone:
    collector_id: int
    error: str | None = None


@dataclass
class ActionChunkResponse:
    chunk_id: int
    actions: np.ndarray


@dataclass
class LearnerCollectorState:
    latent: Any | None = None
    prev_wm_action: torch.Tensor | None = None
    current_obs_consumed: bool = True
    episode: list[dict[str, Any]] = field(default_factory=list)
    episode_return: float = 0.0
    episode_len: int = 0
    next_chunk_id: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental multiprocess online collector with batched VLA encoder/policy learner."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/online_wmpo_outcome_libero_goal.yaml"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--world-model-ckpt", required=True)
    parser.add_argument(
        "--vla-ckpt-path",
        default=str(PROJECT_ROOT / "data/ckpts/frozen_backbones/rynnvla_libero_goal_pi0_query/base_model"),
    )
    parser.add_argument("--encoder-state-ckpt", default="")
    parser.add_argument("--action-head-type", default="legacy", choices=["legacy", "pi0_query"])
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-ids", default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-horizon", type=int, default=80)
    parser.add_argument("--total-env-steps", type=int, default=200)
    parser.add_argument("--num-collectors-per-rank", type=int, default=2)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--encoder-batch-timeout-ms", type=float, default=20.0)
    parser.add_argument("--collect-chunk-steps", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--replay-size", type=int, default=2000)
    parser.add_argument("--replay-capacity-mode", default="per_task", choices=["per_task", "total_sharded"])
    parser.add_argument("--failure-prefix-steps", type=int, default=40)
    parser.add_argument("--failure-prefix-ratio", type=float, default=0.2)
    parser.add_argument("--task-balanced-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-token-id", type=int, default=10004)
    parser.add_argument("--rssm-action-scale", default="env", choices=["policy", "env"])
    parser.add_argument("--deterministic-collect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--freeze-log-std", action="store_true")
    parser.add_argument("--bc-to-ref", type=float, default=None)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collector_main(
    *,
    collector_id: int,
    global_collector_id: int,
    global_num_collectors: int,
    request_queue: Any,
    response_queue: Any,
    stop_event: Any,
    task_suite: str,
    task_ids: tuple[int, ...],
    seed: int,
    episode_horizon: int,
    action_head_type: str,
) -> None:
    env: DreamerVLAOnlineTrainEnv | None = None
    try:
        env = DreamerVLAOnlineTrainEnv(
            task_suite_name=task_suite,
            task_id=task_ids[global_collector_id % len(task_ids)],
            task_ids=task_ids,
            seed=seed,
            max_steps=episode_horizon,
            action_input="normalized",
            task_sampling="sequential",
            init_state_sampling="sequential",
            history_length=2,
            include_state=True,
            vla_rotate_180=True,
            obs_hidden_source="action_query",
            action_head_type=action_head_type,
        )
        task_pos = int(global_collector_id)
        current_task_id = int(task_ids[task_pos % len(task_ids)])
        obs, _info = env.reset(seed=seed, task_id=current_task_id)
        task_pos += int(global_num_collectors)
        request_queue.put(ActionRequest(collector_id=collector_id, obs=obs, is_first=True))
        while not stop_event.is_set():
            response = response_queue.get()
            if response is None:
                break
            if not isinstance(response, ActionChunkResponse):
                raise TypeError(f"collector {collector_id} received unexpected response {type(response)}")
            actions = np.asarray(response.actions, dtype=np.float32).reshape(-1, 7)
            for chunk_index, action in enumerate(actions):
                if stop_event.is_set():
                    break
                obs_before = obs
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                wm_action = np.asarray(info["wm_action"], dtype=np.float32).reshape(-1)[:7]
                request_queue.put(
                    TransitionMsg(
                        collector_id=collector_id,
                        obs=obs_before,
                        policy_action=np.asarray(action, dtype=np.float32).reshape(-1)[:7],
                        wm_action=wm_action,
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        task_id=int(info.get("task_id", obs_before.get("task_id", -1))),
                        chunk_id=int(response.chunk_id),
                        chunk_index=int(chunk_index),
                        chunk_len=int(actions.shape[0]),
                    )
                )
                if done:
                    current_task_id = int(task_ids[task_pos % len(task_ids)])
                    task_pos += int(global_num_collectors)
                    obs, _info = env.reset(task_id=current_task_id)
                    request_queue.put(ActionRequest(collector_id=collector_id, obs=obs, is_first=True))
                    break
                obs = next_obs
            else:
                request_queue.put(ActionRequest(collector_id=collector_id, obs=obs, is_first=False))
    except BaseException as exc:  # noqa: BLE001 - propagate worker failures to learner log.
        request_queue.put(CollectorDone(collector_id=collector_id, error=repr(exc)))
    finally:
        if env is not None:
            env.close()
        request_queue.put(CollectorDone(collector_id=collector_id, error=None))


def _drain_messages(request_queue: Any, *, max_items: int, timeout_s: float) -> list[Any]:
    messages: list[Any] = []
    deadline = time.time() + max(timeout_s, 0.0)
    while len(messages) < max_items:
        remaining = max(0.0, deadline - time.time())
        try:
            messages.append(request_queue.get(timeout=remaining if messages else max(timeout_s, 0.001)))
        except queue.Empty:
            break
    return messages


def _episode_finish_step(episode: list[dict[str, Any]]) -> int | None:
    for idx, step in enumerate(episode):
        if (
            bool(step.get("success", False))
            or float(step.get("is_terminal", 0.0)) > 0.5
            or float(step.get("reward", 0.0)) > 0.0
        ):
            return int(idx)
    return None


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, is_dist = _init_distributed()
    is_rank0 = rank == 0
    if args.device is None:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    out_dir = Path(args.out_dir).expanduser().resolve()
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    episode_log_f = (logs_dir / f"episodes_rank{rank}.jsonl").open("a", encoding="utf-8")
    online_log_f = (logs_dir / f"online_rank{rank}.jsonl").open("a", encoding="utf-8")

    cfg = OmegaConf.load(args.config)
    expected_obs_dim = 35840 if args.action_head_type == "legacy" else 5120
    cfg_obs_dim = int(OmegaConf.select(cfg, "world_model.obs_dim"))
    if cfg_obs_dim != expected_obs_dim:
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
    if is_rank0:
        OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
        print(
            f"[multiproc] rank={rank}/{world_size} out_dir={out_dir} "
            f"collectors_per_rank={args.num_collectors_per_rank} total_env_steps={args.total_env_steps}",
            flush=True,
        )

    encoder_args = argparse.Namespace(
        vla_ckpt_path=args.vla_ckpt_path,
        encoder_state_ckpt=args.encoder_state_ckpt,
        action_head_type=args.action_head_type,
    )
    encoder = build_encoder(encoder_args, device)
    processor = encoder._build_processor(device)
    world_model = hydra.utils.instantiate(cfg.world_model).to(device=device, dtype=torch.bfloat16)
    load_world_model_state(
        world_model,
        args.world_model_ckpt,
        reset_reward_head=bool(OmegaConf.select(cfg, "init.reset_world_model_reward_head", default=False)),
    )
    policy = hydra.utils.instantiate(cfg.policy).to(device).eval()
    world_model.eval()
    encoder.eval()

    task_ids = tuple(int(item) for item in str(args.task_ids).split(",") if item.strip())
    replay = OnlineReplay(
        capacity=int(args.replay_size),
        sequence_length=int(args.sequence_length),
        task_ids=task_ids,
        capacity_mode=str(args.replay_capacity_mode),
        failure_prefix_steps=int(args.failure_prefix_steps),
        failure_prefix_ratio=float(args.failure_prefix_ratio),
        task_balanced=bool(args.task_balanced_replay),
        rank=int(rank),
    )

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    request_queue = ctx.Queue(maxsize=int(args.queue_size) * max(1, int(args.num_collectors_per_rank)))
    response_queues = [ctx.Queue(maxsize=2) for _ in range(int(args.num_collectors_per_rank))]
    collectors: list[Any] = []
    global_num_collectors = int(world_size) * int(args.num_collectors_per_rank)
    for collector_id in range(int(args.num_collectors_per_rank)):
        global_collector_id = int(rank) * int(args.num_collectors_per_rank) + collector_id
        proc = ctx.Process(
            target=_collector_main,
            kwargs={
                "collector_id": collector_id,
                "global_collector_id": global_collector_id,
                "global_num_collectors": global_num_collectors,
                "request_queue": request_queue,
                "response_queue": response_queues[collector_id],
                "stop_event": stop_event,
                "task_suite": str(args.task_suite),
                "task_ids": task_ids,
                "seed": int(args.seed) + int(rank) * 100000 + collector_id * 1000,
                "episode_horizon": int(args.episode_horizon),
                "action_head_type": str(args.action_head_type),
            },
            daemon=True,
        )
        proc.start()
        collectors.append(proc)

    states = [LearnerCollectorState() for _ in range(int(args.num_collectors_per_rank))]
    active_collectors = set(range(int(args.num_collectors_per_rank)))
    env_step = 0
    start_time = time.time()
    encoder_latencies: list[float] = []
    encoder_batch_sizes: list[int] = []
    policy_latencies: list[float] = []

    try:
        while env_step < int(args.total_env_steps) and active_collectors:
            messages = _drain_messages(
                request_queue,
                max_items=max(1, int(args.encoder_batch_size)),
                timeout_s=float(args.encoder_batch_timeout_ms) / 1000.0,
            )
            if not messages:
                continue

            encode_messages: list[Any] = [
                msg for msg in messages if isinstance(msg, (ActionRequest, TransitionMsg))
            ]
            embeddings: dict[int, torch.Tensor] = {}
            if encode_messages:
                t0 = time.time()
                obs_batch = [msg.obs for msg in encode_messages]
                obs_embeddings = obs_batch_to_action_hidden(
                    encoder,
                    processor,
                    obs_batch,
                    device,
                    int(args.target_token_id),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                encoder_latencies.append(time.time() - t0)
                encoder_batch_sizes.append(len(encode_messages))
                for idx, msg in enumerate(encode_messages):
                    embeddings[id(msg)] = obs_embeddings[idx : idx + 1]

            action_requests: list[tuple[ActionRequest, torch.Tensor]] = []
            for msg in messages:
                if isinstance(msg, CollectorDone):
                    if msg.error:
                        raise RuntimeError(f"collector {msg.collector_id} failed: {msg.error}")
                    active_collectors.discard(int(msg.collector_id))
                    continue

                if isinstance(msg, ActionRequest):
                    state = states[msg.collector_id]
                    obs_embedding = embeddings[id(msg)]
                    with torch.no_grad():
                        if bool(msg.is_first) or state.latent is None:
                            state.latent = world_model({"mode": "encode_latent", "hidden": obs_embedding})
                        else:
                            if state.prev_wm_action is None:
                                raise RuntimeError(f"collector {msg.collector_id} missing prev_wm_action")
                            state.latent = world_model({
                                "mode": "observe_next",
                                "latent": state.latent,
                                "hidden": obs_embedding,
                                "actions": state.prev_wm_action,
                                "is_first": False,
                            })
                        feat = world_model({"mode": "actor_input", "latent": state.latent}).float()
                    state.current_obs_consumed = False
                    action_requests.append((msg, feat))
                    continue

                if isinstance(msg, TransitionMsg):
                    state = states[msg.collector_id]
                    obs_embedding = embeddings[id(msg)]
                    if state.current_obs_consumed:
                        with torch.no_grad():
                            if bool(msg.obs.get("is_first", False)) or state.latent is None:
                                state.latent = world_model({"mode": "encode_latent", "hidden": obs_embedding})
                            else:
                                if state.prev_wm_action is None:
                                    raise RuntimeError(f"collector {msg.collector_id} missing prev_wm_action")
                                state.latent = world_model({
                                    "mode": "observe_next",
                                    "latent": state.latent,
                                    "hidden": obs_embedding,
                                    "actions": state.prev_wm_action,
                                    "is_first": False,
                                })
                    else:
                        state.current_obs_consumed = True

                    done = bool(msg.terminated or msg.truncated)
                    state.episode.append({
                        "image": np.asarray(msg.obs["image"], dtype=np.uint8),
                        "obs_embedding": obs_embedding.squeeze(0).detach().cpu().numpy().astype(np.float32),
                        "policy_action": np.asarray(msg.policy_action, dtype=np.float32).reshape(-1)[:7],
                        "wm_action": np.asarray(msg.wm_action, dtype=np.float32).reshape(-1)[:7],
                        "reward": np.float32(msg.reward),
                        "done": np.float32(done),
                        "is_first": bool(msg.obs.get("is_first", False)),
                        "is_terminal": np.float32(msg.terminated),
                        "is_last": np.float32(done),
                        "task_id": int(msg.task_id),
                        "collect_chunk_id": int(msg.chunk_id),
                        "collect_chunk_index": int(msg.chunk_index),
                        "collect_chunk_len": int(msg.chunk_len),
                        "collector_id": int(msg.collector_id),
                    })
                    state.episode_return += float(msg.reward)
                    state.episode_len += 1
                    state.prev_wm_action = torch.from_numpy(
                        np.asarray(msg.wm_action, dtype=np.float32).reshape(-1)[:7]
                    ).to(device=device, dtype=obs_embedding.dtype).unsqueeze(0)
                    env_step += 1

                    if done:
                        replay_record = replay.add_episode(state.episode)
                        chunk_indices = [
                            int(step.get("collect_chunk_index", -1))
                            for step in state.episode
                            if int(step.get("collect_chunk_index", -1)) >= 0
                        ]
                        episode_entry = {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "ts_unix": time.time(),
                            "rank": int(rank),
                            "world_size": int(world_size),
                            "collector_id": int(msg.collector_id),
                            "env_step": int(env_step),
                            "task_id": int(msg.task_id),
                            "episode_id": None if replay_record is None else int(replay_record["episode_id"]),
                            "collection_index": None if replay_record is None else int(replay_record["collection_index"]),
                            "task_episode_index": None if replay_record is None else int(replay_record["task_episode_index"]),
                            "episode_len": int(state.episode_len),
                            "episode_return": float(state.episode_return),
                            "success": bool(msg.terminated),
                            "truncated": bool(msg.truncated),
                            "first_success_step": _episode_finish_step(state.episode),
                            "stored_in_replay": replay_record is not None,
                            "replay_transitions": int(replay.num_transitions),
                            "collect_chunk_index_counts": dict(Counter(chunk_indices)),
                        }
                        episode_log_f.write(json.dumps(_json_safe(episode_entry)) + "\n")
                        episode_log_f.flush()
                        print(
                            f"[episode] rank={rank} collector={msg.collector_id} env_step={env_step} "
                            f"task={msg.task_id} len={state.episode_len} "
                            f"return={state.episode_return:.3f} success={bool(msg.terminated)} "
                            f"replay={replay.num_transitions}",
                            flush=True,
                        )
                        state.latent = None
                        state.prev_wm_action = None
                        state.current_obs_consumed = True
                        state.episode = []
                        state.episode_return = 0.0
                        state.episode_len = 0

                    if is_rank0 and env_step > 0 and env_step % int(args.log_every) == 0:
                        elapsed = max(time.time() - start_time, 1e-6)
                        row = {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "rank": int(rank),
                            "env_step": int(env_step),
                            "fps": float(env_step / elapsed),
                            "replay": int(replay.num_transitions),
                            "active_collectors": int(len(active_collectors)),
                            "encoder_batch_mean": float(np.mean(encoder_batch_sizes[-50:])) if encoder_batch_sizes else 0.0,
                            "encoder_latency_ms_mean": (
                                float(np.mean(encoder_latencies[-50:]) * 1000.0) if encoder_latencies else 0.0
                            ),
                            "policy_latency_ms_mean": (
                                float(np.mean(policy_latencies[-50:]) * 1000.0) if policy_latencies else 0.0
                            ),
                            "replay_task_stats": replay.task_stats(task_ids),
                        }
                        print(
                            "[multiproc] "
                            + " ".join(
                                f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
                                for key, value in row.items()
                            ),
                            flush=True,
                        )
                        online_log_f.write(json.dumps(_json_safe(row)) + "\n")
                        online_log_f.flush()

            if action_requests:
                t0 = time.time()
                feats = torch.cat([feat for _msg, feat in action_requests], dim=0)
                with torch.no_grad():
                    action_chunk, _log_prob, _extra = policy({
                        "mode": "sample",
                        "hidden": feats,
                        "deterministic": bool(args.deterministic_collect),
                        "return_chunk": True,
                    })
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                policy_latencies.append(time.time() - t0)
                action_np = action_chunk.detach().cpu().float().numpy()
                for idx, (msg, _feat) in enumerate(action_requests):
                    state = states[msg.collector_id]
                    chunk = action_np[idx].reshape(-1, action_np.shape[-1])[:, :7]
                    collect_chunk_steps = int(args.collect_chunk_steps)
                    if collect_chunk_steps <= 0:
                        collect_chunk_steps = int(chunk.shape[0])
                    collect_chunk_steps = max(1, min(collect_chunk_steps, int(chunk.shape[0])))
                    state.next_chunk_id += 1
                    response_queues[msg.collector_id].put(
                        ActionChunkResponse(
                            chunk_id=int(state.next_chunk_id),
                            actions=np.asarray(chunk[:collect_chunk_steps], dtype=np.float32),
                        )
                    )

    finally:
        stop_event.set()
        for resp_q in response_queues:
            try:
                resp_q.put_nowait(None)
            except Exception:
                pass
        for proc in collectors:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        episode_log_f.close()
        online_log_f.close()
        if is_dist:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
