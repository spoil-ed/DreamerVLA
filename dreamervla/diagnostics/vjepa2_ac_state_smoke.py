"""Run a bounded real-trajectory WM optimization and decoded-motion probe.

This diagnostic uses the production chunk loss and optimizer on one fixed RGB
trajectory. A second episode is held out. It is not a replacement training route
or an estimate of generalization; its artifacts make regression checks reviewable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

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
from dreamervla.utils.optim import apply_optimizer_lr_schedule, build_optimizer


def main() -> None:
    """Encode two real episodes, run the configured loss, and save comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder-ckpt", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--encoded-cache", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="Weights-only warm start; NOT resume")
    parser.add_argument("--experiment", default="wm_pi05_collected_train")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / ".hydra").mkdir()
    (root / "checkpoints").mkdir()
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
    OmegaConf.save(cfg, root / ".hydra" / "resolved_config.yaml", resolve=True)
    device = torch.device(args.device)
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
        state = initial_state["state_dicts"]["world_model"]
        fresh = wm.state_dict()
        added = set(fresh) - set(state)
        if any(not key.startswith("decoded_visual_loss.") for key in added):
            raise ValueError(f"Warm start is missing transition parameters: {sorted(added)}")
        # The readout was independently loaded strictly from its own checkpoint.
        # Every transition tensor must match exactly; only this new frozen
        # auxiliary component may be absent from the older WM checkpoint.
        print(json.dumps({"new_frozen_readout_keys": sorted(added)}), flush=True)
        wm.load_state_dict({**{key: fresh[key] for key in added}, **state}, strict=True)
        del initial_state
    optimizer = build_optimizer(wm, cfg.optim.world_model)
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
    results["init_checkpoint"] = None if args.init_checkpoint is None else str(args.init_checkpoint)
    results["evaluation_scope"] = (
        "Second episode is excluded only from THIS probe's updates; a warm-start "
        "checkpoint may already have trained on both episodes. Not a dataset holdout."
    )
    decoder = _load_pixel_decoder(args.decoder_ckpt, device)
    decoder.requires_grad_(False)
    if wm.pretrained_load_report is not None:
        results["pretrained_load_report"] = asdict(wm.pretrained_load_report)

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
            results["evaluations"].append(metrics)
            predictions.append(visual.cpu())
            print(json.dumps(metrics), flush=True)
        return predictions

    initial = evaluate(0)
    for step in range(args.steps):
        wm.train()
        schedule = apply_optimizer_lr_schedule(
            optimizer, cfg.optim.world_model, step=step, total_steps=args.steps
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = wm.chunk_loss(batches[0])
        loss = output["_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        if wm.decoded_visual_loss is not None:
            assert not wm.decoded_visual_loss.decoder.training
            assert all(
                p.grad is None and not p.requires_grad for p in wm.decoded_visual_loss.parameters()
            )
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
            **{key: value.item() for key, value in output.items() if key.startswith("decoded_")},
            **schedule,
        )
        results["training"].append(metrics)
        print(json.dumps(metrics), flush=True)
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


if __name__ == "__main__":
    main()
