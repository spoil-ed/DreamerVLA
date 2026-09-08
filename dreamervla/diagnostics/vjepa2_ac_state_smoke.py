"""Run a bounded real-trajectory WM optimization and decoded-motion probe.

This diagnostic uses the production chunk loss and optimizer on one fixed RGB
trajectory. A second episode is held out. It is not a replacement training route
or an estimate of generalization; its artifacts make regression checks reviewable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.diagnostics.compare_wm_libero_rollout import (
    EncodedTrajectory,
    _decode_tokens,
    _encode_trajectories,
    _load_pixel_decoder,
    _load_trajectory,
    _rollout_closed_loop,
    _write_video,
)
from dreamervla.diagnostics.decoded_rollout_metrics import DecodedRolloutMetrics
from dreamervla.utils.optim import apply_optimizer_lr_schedule, build_optimizer


def main() -> None:
    """Encode two real episodes, run the configured loss, and save comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder-ckpt", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--encoded-cache", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="Weights-only warm start; NOT resume")
    parser.add_argument("--experiment", default="wm_pi05_collected_train")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--master-addr=127.0.0.1",
                "--master-port=29517",
                "--nproc-per-node=8",
                "--module",
                "dreamervla.diagnostics.vjepa2_ac_state_smoke",
                *sys.argv[1:],
            ],
            check=True,
        )
        return
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise ValueError("Training diagnostics require the default eight-GPU topology")
    if args.encoded_cache is None or not args.encoded_cache.is_file():
        raise ValueError(
            "DDP probe requires a verified pre-encoded cache; no concurrent cache writes"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group("nccl", device_id=device)
    root = args.output_dir.resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        (root / ".hydra").mkdir()
        (root / "checkpoints").mkdir()
    torch.distributed.barrier()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    register_dreamervla_resolvers()
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={args.experiment}",
                "task=pi05_libero_object",
                "profile=sinfra_pi05_object",
                "world_model.transition_type=vjepa2_ac",
                "world_model.transition_init=pretrained",
                *args.override,
            ],
        )
    if rank == 0:
        OmegaConf.save(cfg, root / ".hydra" / "resolved_config.yaml", resolve=True)
    length = (
        cfg.world_model.num_hist + cfg.world_model.chunk_size * cfg.world_model.chunk_rollout_chunks
    )
    data_dir = Path(cfg.offline_warmup.data_dir)
    rows = [
        json.loads(line) for line in (data_dir / "episode_index.jsonl").read_text().splitlines()
    ]
    selected = sorted(
        [
            r
            for r in rows
            if r["task_id"] == args.task_id and r["success"] and r["horizon"] >= length
        ],
        key=lambda r: r["episode_id"],
    )[:2]
    if len(selected) != 2:
        raise ValueError("Need two distinct successful episodes for the fixed/held-out probe")
    trajectories = [_load_trajectory(data_dir, row, length) for row in selected]
    encoder_contract = {
        "data_dir": str(data_dir.resolve()),
        "policy": OmegaConf.to_container(cfg.offline_warmup.online_latent.policy, resolve=True),
    }
    if args.encoded_cache is not None and args.encoded_cache.is_file():
        cached = torch.load(args.encoded_cache, map_location="cpu", weights_only=True)
        if (
            cached["episodes"] != selected
            or cached["length"] != length
            or cached.get("encoder_contract") != encoder_contract
        ):
            raise ValueError(
                "Diagnostic cache does not match trajectories and encoder configuration"
            )
        encoded = [EncodedTrajectory(**value) for value in cached["encoded"]]
    else:
        encoded = _encode_trajectories(
            trajectories,
            policy_cfg=cfg.offline_warmup.online_latent.policy,
            device=device,
            batch_size=4,
        )
        if args.encoded_cache is not None:
            args.encoded_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "episodes": selected,
                    "length": length,
                    "encoder_contract": encoder_contract,
                    "encoded": [
                        {"latent": value.latent, "attention_mask": value.attention_mask}
                        for value in encoded
                    ],
                },
                args.encoded_cache,
            )
    wm = instantiate(cfg.world_model).to(device)
    if args.init_checkpoint is not None:
        initial_state = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        from dreamervla.utils.legacy_wm_readout import discard_legacy_wm_readout

        state = discard_legacy_wm_readout(initial_state["state_dicts"]["world_model"])
        fresh = wm.state_dict()
        added = set(fresh) - set(state)
        new_adapter = {key for key in added if key.startswith("vjepa2_transition.condition_film.")}
        if added != new_adapter:
            raise ValueError(f"Warm start is missing transition parameters: {sorted(added)}")
        # Only explicitly selected new adapters may be absent. No decoder is
        # owned by this model or loaded into its training state.
        if rank == 0:
            print(
                json.dumps(
                    {
                        "new_trainable_adapter_keys": sorted(new_adapter),
                    }
                ),
                flush=True,
            )
        wm.load_state_dict({**{key: fresh[key] for key in added}, **state}, strict=True)
        del initial_state
    optimizer = build_optimizer(wm, cfg.optim.world_model)
    wrapped = DistributedDataParallel(
        wm, device_ids=[local_rank], static_graph=True, broadcast_buffers=False
    )
    batches = [
        dict(
            obs_embedding=enc.latent.to(device=device, dtype=torch.float32).unsqueeze(0),
            actions=torch.as_tensor(traj.actions, device=device).unsqueeze(0),
            proprio=torch.as_tensor(traj.proprio, device=device).unsqueeze(0),
            prefix_attention_mask=enc.attention_mask.to(device).unsqueeze(0),
        )
        for traj, enc in zip(trajectories, encoded, strict=True)
    ]
    results = {"episodes": [r["episode_id"] for r in selected], "training": [], "evaluations": []}
    results["world_size"] = world_size
    results["batch_contract"] = (
        "One identical fixed training episode per rank; NOT eight unique episodes"
    )
    results["init_checkpoint"] = None if args.init_checkpoint is None else str(args.init_checkpoint)
    results["evaluation_scope"] = (
        "Second episode is excluded only from THIS probe's updates; a warm-start "
        "checkpoint may already have trained on both episodes. Not a dataset holdout."
    )
    decoder = _load_pixel_decoder(args.decoder_ckpt, device)
    decoded_evaluator = DecodedRolloutMetrics(decoder=decoder, frame_stride=5).to(device)
    decoder.requires_grad_(False)
    if wm.pretrained_load_report is not None:
        results["pretrained_load_report"] = asdict(wm.pretrained_load_report)
    tracking = None
    if rank == 0:
        import wandb

        tracking = wandb.init(
            project=str(cfg.runner.logger.project_name),
            name=root.name,
            group="wm-dynamics-fixed-clip-ablation",
            mode=args.wandb_mode,
            dir=str(root),
            config={
                "experiment": args.experiment,
                "steps": args.steps,
                "world_size": world_size,
                "init_checkpoint": results["init_checkpoint"],
                "evaluation_scope": results["evaluation_scope"],
                "batch_contract": results["batch_contract"],
            },
        )

    @torch.no_grad()
    def evaluate(step: int) -> list[torch.Tensor]:
        wm.eval()
        predictions = []
        for index, batch in enumerate(batches):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred, target = _rollout_closed_loop(
                    wm,
                    batch["obs_embedding"][0],
                    batch["actions"][0],
                    cfg.world_model.chunk_rollout_chunks,
                    proprio=batch["proprio"][0],
                    attention_mask=batch["prefix_attention_mask"][0],
                )
                changed_actions = batch["actions"][0].clone()
                start = wm.num_hist - 1
                stop = start + wm.chunk_size * cfg.world_model.chunk_rollout_chunks
                changed_actions[start:stop] = changed_actions[start:stop].flip(0)
                changed, _ = _rollout_closed_loop(
                    wm,
                    batch["obs_embedding"][0],
                    changed_actions,
                    cfg.world_model.chunk_rollout_chunks,
                    proprio=batch["proprio"][0],
                    attention_mask=batch["prefix_attention_mask"][0],
                )
            valid = batch["prefix_attention_mask"][0, wm.num_hist :]
            visual = pred[..., : wm.token_dim].float()
            target_visual = target[..., : wm.token_dim].float()
            persistence = wm._normalize_raw_vision_tokens(batch["obs_embedding"])[
                0, wm.num_hist - 1
            ]
            metrics = dict(
                step=step,
                split="train" if index == 0 else "held_out",
                visual_mse=(visual[valid] - target_visual[valid]).square().mean().item(),
                changed_action_visual_mse=(
                    changed[..., : wm.token_dim].float()[valid] - target_visual[valid]
                )
                .square()
                .mean()
                .item(),
                persistence_mse=(persistence.expand_as(visual)[valid] - target_visual[valid])
                .square()
                .mean()
                .item(),
                action_response=(visual[valid] - changed[..., : wm.token_dim].float()[valid])
                .abs()
                .mean()
                .item(),
                proprio_mse=(
                    wm._raw_proprio_from_obs_tokens(pred, valid).float()
                    - batch["proprio"][0, wm.num_hist :]
                )
                .square()
                .mean()
                .item(),
            )
            metrics["correct_action_mse_advantage"] = (
                metrics["changed_action_visual_mse"] - metrics["visual_mse"]
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded_metrics = decoded_evaluator(
                    visual[None],
                    target_visual[None],
                    persistence[None, None],
                    token_mask=valid[None],
                    anchor_mask=batch["prefix_attention_mask"][:, wm.num_hist - 1 : wm.num_hist],
                )
            metrics.update({k: v.item() for k, v in decoded_metrics.items()})
            results["evaluations"].append(metrics)
            predictions.append(visual.cpu())
            if rank == 0:
                print(json.dumps(metrics), flush=True)
                tracking.log(
                    {
                        f"eval/{metrics['split']}/{k}": v
                        for k, v in metrics.items()
                        if k not in {"step", "split"}
                    },
                    step=step,
                )
        return predictions

    initial = evaluate(0)
    for step in range(args.steps):
        wm.train()
        schedule = apply_optimizer_lr_schedule(
            optimizer, cfg.optim.world_model, step=step, total_steps=args.steps
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = wrapped(batches[0])
        loss = output["_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        assert not decoder.training
        assert all(p.grad is None and not p.requires_grad for p in decoder.parameters())
        state_grad = wm.vjepa2_transition.state_output_adapter.weight.grad
        if state_grad is None or not torch.isfinite(state_grad).all() or state_grad.norm() == 0:
            raise RuntimeError(f"State prediction lost its training signal at step {step}")
        for group in optimizer.param_groups:
            if not group.get("update_enabled", True):
                for parameter in group["params"]:
                    parameter.grad = None
        grad_norm = torch.nn.utils.clip_grad_norm_(
            wm.parameters(), cfg.optim.grad_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        metrics = dict(
            step=step + 1,
            loss=loss.item(),
            proprio_mse=output["proprio_reconstruction_loss"].item(),
            grad_norm=grad_norm.item(),
            visual_motion_ratio=output["visual_motion_ratio"].item(),
            visual_delta_mse=output["visual_delta_mse"].item(),
            **{
                key: value.item()
                for key, value in output.items()
                if key.startswith(("decoded_", "teacher_forced_"))
            },
            **schedule,
        )
        results["training"].append(metrics)
        if rank == 0:
            print(json.dumps(metrics), flush=True)
            tracking.log({f"train/{key}": value for key, value in metrics.items()}, step=step + 1)
    # Check synchronization before releasing ranks that do not write artifacts.
    synced = wm.vjepa2_transition.output_adapter.weight.detach().clone()
    torch.distributed.broadcast(synced, src=0)
    torch.testing.assert_close(wm.vjepa2_transition.output_adapter.weight, synced, rtol=0, atol=0)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank != 0:
        return
    final = evaluate(args.steps)
    torch.save(
        {
            "state_dicts": {
                "world_model": wm.state_dict(),
                "world_model_optimizer": optimizer.state_dict(),
            },
            "global_step": args.steps,
        },
        root / "checkpoints" / "latest.ckpt",
    )
    saved = torch.load(
        root / "checkpoints" / "latest.ckpt", map_location="cpu", weights_only=True, mmap=True
    )
    wm.load_state_dict(saved["state_dicts"]["world_model"], strict=True)
    optimizer.load_state_dict(saved["state_dicts"]["world_model_optimizer"])
    results["strict_reload"] = True
    for i, batch in enumerate(batches):
        target = wm._normalize_raw_vision_tokens(batch["obs_embedding"])[0, wm.num_hist :].cpu()
        videos = [
            _decode_tokens(decoder, value, device=device, batch_size=4)
            for value in (target, initial[i], final[i])
        ]
        # Panels: encode→decode, initial rollout→decode, trained rollout→decode.
        frames = [torch.from_numpy(v).permute(0, 2, 1, 3, 4).flatten(2, 3).numpy() for v in videos]
        _write_video(
            root / f"episode_{selected[i]['episode_id']}_comparison.mp4",
            list(np.concatenate(frames, axis=1)),
            fps=10,
        )
        motion = [
            float(np.abs(v[1:].astype(float) - v[:-1].astype(float)).mean() / 255) for v in videos
        ]
        results.setdefault("decoded_motion", []).append(
            dict(
                split="train" if i == 0 else "held_out",
                reference=motion[0],
                initial=motion[1],
                trained=motion[2],
                initial_pixel_mse=float(
                    np.square((videos[1].astype(float) - videos[0]) / 255).mean()
                ),
                trained_pixel_mse=float(
                    np.square((videos[2].astype(float) - videos[0]) / 255).mean()
                ),
            )
        )
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["decoded_motion"]), flush=True)
    for row in results["decoded_motion"]:
        tracking.log(
            {f"eval/{row['split']}/{key}": value for key, value in row.items() if key != "split"},
            step=args.steps,
        )
    tracking.finish()


if __name__ == "__main__":
    main()
