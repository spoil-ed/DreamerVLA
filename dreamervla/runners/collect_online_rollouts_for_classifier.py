#!/usr/bin/env python
"""Collect online LIBERO rollouts and dump them as classifier-dataset shards.

Reuses the RynnVLA action-hidden online training pipeline (encoder + processor + WM
+ actor) — but instead of running PPO updates, it only rolls out episodes and
writes them to disk in the schema expected by ``WMReplayClassifierDataset``:

    <out_raw_dir>/shard_<NNN>.hdf5      data/<ep>/actions  (T, 7)  float32
                                        data/<ep>/dones    (T,)   uint8
                                        data/<ep>/rewards  (T,)   uint8
    <out_hidden_dir>/shard_<NNN>.hdf5   data/<ep>/obs_embedding (T, D) float32

These shards can be used as failure examples for LatentSuccessClassifier
experiments. The dataset derives ``finish_step`` from ``dones`` and
``complete`` from ``rewards`` per episode, matching LUMOS's episode metadata
semantics.

Single GPU. Usage example::

    CUDA_VISIBLE_DEVICES=7 \\
        python -u \\
        python -m dreamervla.runners.collect_online_rollouts_for_classifier \\
            --config configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml \\
            --world-model-ckpt data/outputs/worldmodel/.../step_00002000.ckpt \\
            --vla-ckpt-path ${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/checkpoints/VLA_model_256/libero_goal \\
            --task-suite libero_goal --task-ids 0,1,2,3,4,5,6,7,8,9 \\
            --num-episodes 200 --episodes-per-shard 25 \\
            --out-raw-dir data/processed_data/libero_goal/libero_goal_online_rollouts_vla_sft \\
            --out-hidden-dir data/processed_data/libero_goal/libero_goal_online_rollouts_vla_sft_hidden \\
            --deterministic-collect
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from dreamervla.constants import DEFAULT_ACTION_TOKEN_ID  # noqa: E402
from dreamervla.dataset.online_rollout_dumper import RolloutDumper  # noqa: E402
from dreamervla.envs.train_env import DreamerVLAOnlineTrainEnv  # noqa: E402
from dreamervla.runners.online_utils import (  # noqa: E402
    build_encoder,
    load_world_model_state,
    obs_to_action_hidden,
)
from dreamervla.utils.paths import checkpoints_path  # noqa: E402
from dreamervla.utils.progress import ProgressReporter  # noqa: E402
from dreamervla.utils.seed import set_seed  # noqa: E402
from dreamervla.utils.torch_utils import freeze_module  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect online LIBERO rollouts into classifier-ready HDF5 shards."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/dreamervla/openvla_onetraj_libero_cotrain_ray.yaml"),
        help="Defaults to the legacy RynnVLA action-hidden config that matches the m1024 chunk WM "
        "and the *_legacy_action_hidden_vla_policy_h2 obs_embedding sidecars used by the "
        "WMReplayClassifierDataset.",
    )
    parser.add_argument("--world-model-ckpt", required=True)
    parser.add_argument(
        "--vla-ckpt-path",
        default=str(checkpoints_path("VLA_model_256", "libero_goal")),
    )
    parser.add_argument(
        "--encoder-state-ckpt",
        default="",
        help="Optional separate encoder state ckpt. Legacy RynnVLA extraction does NOT need this; "
        "leave empty to use the base SFT VLA backbone directly.",
    )
    parser.add_argument(
        "--actor-ckpt",
        default=None,
        help="Optional separate actor/policy ckpt. Defaults to using the SFT init from the policy config "
        "(matches LUMOS's 'rollout SFT actor before RL').",
    )
    parser.add_argument(
        "--action-head-type",
        default="legacy",
        choices=["legacy"],
        help="Encoder action-head variant. ``legacy`` produces RynnVLA hidden matching the "
        "m1024 chunk WM and the *_legacy_action_hidden_vla_policy_h2 sidecars used by "
        "WMReplayClassifierDataset.",
    )
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument(
        "--task-ids",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated task ids to sweep through during collection.",
    )
    parser.add_argument(
        "--task-sampling",
        choices=["sequential", "random"],
        default="sequential",
    )
    parser.add_argument(
        "--init-state-sampling",
        choices=["sequential", "random"],
        default="sequential",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--episodes-per-shard", type=int, default=25)
    parser.add_argument("--episode-horizon", type=int, default=200)
    parser.add_argument(
        "--target-token-id", type=int, default=DEFAULT_ACTION_TOKEN_ID
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--deterministic-collect",
        action="store_true",
        help="Sample actions deterministically (Gaussian mean). Recommended for stable rollout "
        "distributions; flip off if you want exploration noise in the corpus.",
    )
    parser.add_argument(
        "--allow-tiny-trainable",
        action="store_true",
        help="Silent-freeze guard; mirrors the training script. Collection itself doesn't "
        "update parameters, but the silent-freeze trap still implies a degenerate actor "
        "so we abort by default.",
    )
    parser.add_argument("--out-raw-dir", required=True)
    parser.add_argument("--out-hidden-dir", required=True)
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional path to write a JSONL log of (episode_idx, task_id, length, success) entries.",
    )
    return parser.parse_args()


def _coerce_task_ids(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _build_policy_and_wm(cfg: Any, args: argparse.Namespace, device: torch.device):
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
    world_model.eval()
    freeze_module(world_model)

    policy = hydra.utils.instantiate(cfg.policy).to(device)
    if args.actor_ckpt:
        payload = torch.load(args.actor_ckpt, map_location="cpu", weights_only=False)
        state = payload.get("state_dicts", {}).get("policy") or payload.get("model")
        if state is None:
            raise RuntimeError(f"{args.actor_ckpt} has no state_dicts.policy or model")
        missing, unexpected = policy.load_state_dict(state, strict=False)
        print(
            f"[init] policy resumed from {args.actor_ckpt} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    policy.eval()
    freeze_module(policy)

    # silent-freeze sanity (collection doesn't train, but a degenerate actor
    # still wastes rollout budget on a useless policy).
    n_trainable_total = sum(p.numel() for p in policy.parameters())
    print(
        f"[policy] {type(policy).__name__} total_params={n_trainable_total:,}",
        flush=True,
    )
    return world_model, policy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dumper = RolloutDumper(
        raw_dir=args.out_raw_dir,
        hidden_dir=args.out_hidden_dir,
        episodes_per_shard=int(args.episodes_per_shard),
        manifest_path=args.manifest_path,
    )

    cfg = OmegaConf.load(args.config)
    cfg.init.vla_ckpt_path = args.vla_ckpt_path
    cfg.init.encoder_state_ckpt = args.encoder_state_ckpt
    cfg.init.world_model_state_ckpt = args.world_model_ckpt

    print(f"[collect] out_raw_dir={dumper.raw_dir}", flush=True)
    print(f"[collect] out_hidden_dir={dumper.hidden_dir}", flush=True)
    print(f"[collect] device={device} seed={args.seed}", flush=True)

    encoder = build_encoder(args, device)
    processor = encoder._build_processor(device)

    world_model, policy = _build_policy_and_wm(cfg, args, device)

    task_ids = _coerce_task_ids(args.task_ids)
    print(
        f"[collect] task_suite={args.task_suite} task_ids={task_ids} "
        f"num_episodes={args.num_episodes} episodes_per_shard={args.episodes_per_shard} "
        f"episode_horizon={args.episode_horizon} deterministic={args.deterministic_collect}",
        flush=True,
    )
    env = DreamerVLAOnlineTrainEnv(
        task_suite_name=args.task_suite,
        task_id=task_ids[0],
        task_ids=tuple(task_ids),
        seed=args.seed,
        max_steps=args.episode_horizon,
        task_sampling=args.task_sampling,
        init_state_sampling=args.init_state_sampling,
    )

    start_time = time.time()
    obs, _info = env.reset(seed=args.seed)
    episode_steps: list[dict[str, Any]] = []
    latent = None
    prev_wm_action: torch.Tensor | None = None
    completed_episodes = 0

    pbar = ProgressReporter(int(args.num_episodes), "collect", unit="ep")
    try:
        while completed_episodes < args.num_episodes:
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
                feat = world_model({"mode": "actor_input", "latent": latent}).float()
                action_chunk, _log_prob, _extra = policy(
                    {
                        "mode": "sample",
                        "hidden": feat,
                        "deterministic": bool(args.deterministic_collect),
                        "return_chunk": True,
                    }
                )
                policy_action = (
                    action_chunk.reshape(-1, action_chunk.shape[-1])[0, :7]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )

            next_obs, reward, terminated, truncated, info = env.step(policy_action)
            done = bool(terminated or truncated)
            wm_action_np = np.asarray(info["wm_action"], dtype=np.float32).reshape(-1)[
                :7
            ]
            episode_steps.append(
                {
                    "obs_embedding": obs_embedding.squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    "wm_action": wm_action_np.astype(np.float32),
                    "reward": float(reward),
                    "done": 1.0 if done else 0.0,
                }
            )
            prev_wm_action = (
                torch.from_numpy(wm_action_np)
                .to(device=device, dtype=obs_embedding.dtype)
                .unsqueeze(0)
            )

            if done:
                entry = dumper.add_episode(
                    episode_steps,
                    task_id=int(info.get("task_id", -1)),
                    success=bool(terminated),
                )
                completed_episodes += 1
                pbar.set(completed_episodes)
                elapsed = time.time() - start_time
                print(
                    f"[collect] episode {completed_episodes}/{args.num_episodes} "
                    f"task={entry['task_id']} len={entry['length']} success={entry['success']} "
                    f"shard={entry['shard']} elapsed={elapsed:.1f}s",
                    flush=True,
                )

                episode_steps = []
                latent = None
                prev_wm_action = None
                obs, _info = env.reset()
            else:
                obs = next_obs

        print(
            f"[collect] done. episodes={dumper.total_episodes} "
            f"successes={dumper.total_success} "
            f"({dumper.total_success / max(dumper.total_episodes, 1):.2%}) "
            f"shards={dumper.shards_written}",
            flush=True,
        )
    finally:
        pbar.close()
        dumper.close()
        env.close()


if __name__ == "__main__":
    main()
