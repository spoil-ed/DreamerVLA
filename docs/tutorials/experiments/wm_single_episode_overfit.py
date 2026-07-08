#!/usr/bin/env python3
"""Single-episode Chunk-WM overfit probe.

This is a diagnostic experiment for the OpenVLA-OFT token world model.  It keeps
one LIBERO episode fixed, trains only the world model on sliding windows from that
episode, and periodically compares imagined rollouts under true, zero, and random
action chunks.

The script is dry-run by default so it never occupies a GPU accidentally.  Pass
``--run`` to start training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_report_steps(text: str) -> set[int]:
    return {int(item) for item in text.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually train. Without this flag the script prints the resolved plan and exits.",
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=ROOT
        / "data/outputs/coldstart_warmup_cotrain/fixed_cls_wm_vla_eval_g7_component_20260707_205109/cotrain/resolved_config.yaml",
        help="Resolved cotrain config containing ray_components.world_model/classifier.",
    )
    parser.add_argument(
        "--wm-ckpt",
        type=Path,
        default=ROOT
        / "data/outputs/world_model_probe/current_actions_reward0_20260708_01/wm_probe_step1200.ckpt",
        help="World-model checkpoint with a top-level world_model state dict.",
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=Path,
        default=ROOT
        / "data/outputs/coldstart_warmup_cotrain/fixed_wm_wmpo_cls_mainline_20260707_01/init/fixed_wm_wmpo_cls_init.ckpt",
        help="Classifier checkpoint containing state_dicts.classifier.",
    )
    parser.add_argument(
        "--hidden-hdf5",
        type=Path,
        default=ROOT
        / "data/processed_data/libero_goal_no_noops_t_256_oft_input_token_embedding_vla_policy_h1/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
        help="Hidden sidecar HDF5 containing data/<demo>/obs_embedding and lang_emb.",
    )
    parser.add_argument(
        "--raw-hdf5",
        type=Path,
        default=ROOT
        / "data/processed_data/libero_goal_no_noops_t_256/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
        help="Raw reward HDF5 containing actions, rewards, and proprio observations.",
    )
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/outputs/world_model_probe/single_episode_overfit",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-6)
    parser.add_argument("--adam-eps", type=float, default=1.0e-20)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--report-steps",
        default="0,10,100,500,1000,1200",
        help="Comma-separated update steps that trigger eval records.",
    )
    parser.add_argument(
        "--probe-starts",
        default="0,32,64,96,120",
        help="Comma-separated episode start offsets for true/zero/random rollout comparison.",
    )
    return parser.parse_args()


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _append_json(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()


def _load_episode(
    hidden_hdf5: Path, raw_hdf5: Path, demo_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(hidden_hdf5, "r") as f:
        demo = f["data"][demo_key]
        hidden = np.asarray(demo["obs_embedding"], dtype=np.float32)
        lang = np.asarray(demo["lang_emb"], dtype=np.float32)
    with h5py.File(raw_hdf5, "r") as f:
        demo = f["data"][demo_key]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        rewards = np.asarray(demo["rewards"], dtype=np.float32)
        proprio = np.concatenate(
            [
                np.asarray(demo["obs"]["ee_pos"], dtype=np.float32),
                np.asarray(demo["obs"]["ee_ori"], dtype=np.float32),
                np.asarray(demo["obs"]["gripper_states"], dtype=np.float32),
            ],
            axis=-1,
        )
    return hidden, lang, actions, rewards, proprio


def main() -> None:
    args = parse_args()
    report_steps = _parse_report_steps(args.report_steps)
    probe_starts = [int(item) for item in args.probe_starts.split(",") if item.strip()]

    plan = {
        "resolved_config": str(args.resolved_config),
        "wm_ckpt": str(args.wm_ckpt),
        "classifier_ckpt": str(args.classifier_ckpt),
        "hidden_hdf5": str(args.hidden_hdf5),
        "raw_hdf5": str(args.raw_hdf5),
        "demo_key": args.demo_key,
        "out_dir": str(args.out_dir),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "adam_eps": args.adam_eps,
        "report_steps": sorted(report_steps),
        "probe_starts": probe_starts,
    }
    if not args.run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return

    for label, path in {
        "resolved config": args.resolved_config,
        "WM checkpoint": args.wm_ckpt,
        "classifier checkpoint": args.classifier_ckpt,
        "hidden HDF5": args.hidden_hdf5,
        "raw HDF5": args.raw_hdf5,
    }.items():
        _require_path(path, label)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    error_path = args.out_dir / "error.txt"

    try:
        _run_probe(args, report_steps, probe_starts, metrics_path)
    except BaseException:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _run_probe(
    args: argparse.Namespace,
    report_steps: set[int],
    probe_starts: list[int],
    metrics_path: Path,
) -> None:
    from dreamervla.envs.world_model.latent_world_model_env import _build_component

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(args.device)
    hidden, lang, actions, rewards, proprio = _load_episode(
        args.hidden_hdf5, args.raw_hdf5, args.demo_key
    )

    cfg = OmegaConf.load(args.resolved_config)
    wm_cfg = OmegaConf.to_container(cfg.ray_components.world_model, resolve=True)
    wm_cfg["kwargs"]["reward_loss_scale"] = 0.0
    wm_cfg["kwargs"]["chunk_rollout_chunks"] = 1
    wm_cfg["kwargs"]["chunk_rollout_loss_scale"] = 0.0
    cls_cfg = OmegaConf.to_container(cfg.ray_components.classifier, resolve=True)

    wm = _build_component(wm_cfg).to(device).train()
    classifier = _build_component(cls_cfg).to(device).eval()
    wm_payload = torch.load(args.wm_ckpt, map_location="cpu")
    cls_payload = torch.load(args.classifier_ckpt, map_location="cpu")
    wm.load_state_dict(wm_payload["world_model"])
    classifier.load_state_dict(cls_payload["state_dicts"]["classifier"])
    threshold = float(cls_payload.get("classifier_threshold", 0.95))

    optimizer = torch.optim.AdamW(
        wm.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=0.0,
    )
    history = int(wm.num_hist)
    chunk = int(wm.chunk_size)
    episode_len = int(hidden.shape[0])
    seq_len = history + chunk
    starts = np.arange(0, episode_len - seq_len + 1, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError(f"episode length {episode_len} is shorter than H+K={seq_len}")

    hidden_t = torch.as_tensor(hidden, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    proprio_t = torch.as_tensor(proprio, device=device)
    rewards_t = torch.as_tensor(rewards, device=device)
    lang_t = torch.as_tensor(lang, device=device)

    _append_json(
        metrics_path,
        {
            "event": "start",
            "episode_len": episode_len,
            "num_windows": int(len(starts)),
            "history": history,
            "chunk": chunk,
            "threshold": threshold,
            "lr": args.lr,
            "adam_eps": args.adam_eps,
            "max_steps": args.steps,
            "batch_size": args.batch_size,
            "source_wm": str(args.wm_ckpt),
        },
    )

    def make_batch(batch_starts: np.ndarray) -> dict[str, torch.Tensor]:
        idx = torch.as_tensor(batch_starts, device=device, dtype=torch.long)
        offsets = torch.arange(seq_len, device=device, dtype=torch.long)
        frame_idx = idx[:, None] + offsets[None]
        return {
            "obs_embedding": hidden_t.index_select(0, frame_idx.reshape(-1)).reshape(
                len(batch_starts), seq_len, *hidden_t.shape[1:]
            ),
            "current_actions": actions_t.index_select(0, frame_idx.reshape(-1)).reshape(
                len(batch_starts), seq_len, actions_t.shape[-1]
            ),
            "actions": torch.zeros(
                len(batch_starts), seq_len, actions_t.shape[-1], device=device
            ),
            "proprio": proprio_t.index_select(0, frame_idx.reshape(-1)).reshape(
                len(batch_starts), seq_len, proprio_t.shape[-1]
            ),
            "rewards": rewards_t.index_select(0, frame_idx.reshape(-1)).reshape(
                len(batch_starts), seq_len
            ),
            "lang_emb": lang_t[None].expand(len(batch_starts), -1),
        }

    def classifier_score_window(
        latents_flat: torch.Tensor, prop: torch.Tensor, end: int
    ) -> float:
        start = max(0, end - 7)
        window = latents_flat[start : end + 1]
        pwin = prop[start : end + 1]
        if window.shape[0] < 8:
            window = torch.cat(
                [window[:1].expand(8 - window.shape[0], -1), window], dim=0
            )
            pwin = torch.cat([pwin[:1].expand(8 - pwin.shape[0], -1), pwin], dim=0)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            score = torch.sigmoid(
                classifier(window[None], proprio=pwin[None], lang_emb=lang_t[None]).reshape(-1)
            )[0]
        return float(score.float().cpu())

    def rollout_scores(start: int, action_mode: str) -> tuple[list[float], float, float]:
        wm.eval()
        s = int(start)
        hist = hidden_t[s : s + history][None]
        prop0 = proprio_t[s + history - 1 : s + history]
        if action_mode == "true":
            action_chunk = actions_t[s + history - 1 : s + history - 1 + chunk][None]
        elif action_mode == "zero":
            action_chunk = torch.zeros(1, chunk, actions_t.shape[-1], device=device)
        elif action_mode == "random":
            gen = torch.Generator(device=device).manual_seed(1000 + s)
            action_chunk = torch.empty(
                1, chunk, actions_t.shape[-1], device=device
            ).uniform_(-1, 1, generator=gen)
        else:
            raise ValueError(action_mode)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            out = wm.predict_next_chunk(
                {"history": hist, "hidden": hist[:, -1], "lang": lang_t[None], "proprio": prop0},
                action_chunk,
            )
        visual = out["hidden_seq"][0, :, :, :4096].reshape(chunk, -1).float()
        pred_prop = out.get("proprio_seq", prop0[:, None].expand(-1, chunk, -1))[0].float()
        all_lat = torch.cat([hidden_t[s : s + history].reshape(history, -1).float(), visual])
        all_prop = torch.cat([proprio_t[s : s + history].float(), pred_prop])
        scores = [
            classifier_score_window(all_lat, all_prop, i)
            for i in range(history, history + chunk)
        ]
        target = hidden_t[s + history : s + history + chunk].reshape(chunk, -1).float()
        mse = float((visual - target).pow(2).mean().cpu())
        cos = float(
            torch.nn.functional.cosine_similarity(visual, target, dim=-1).mean().cpu()
        )
        wm.train()
        return scores, mse, cos

    def eval_overfit() -> dict[str, float]:
        eval_starts = np.linspace(0, len(starts) - 1, num=min(16, len(starts)), dtype=np.int64)
        losses: list[float] = []
        hidden_mse: list[float] = []
        hidden_cos: list[float] = []
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            for i in range(0, len(eval_starts), 4):
                batch = make_batch(starts[eval_starts[i : i + 4]])
                out = wm(batch)
                losses.append(float(out["_loss"].cpu()))
                hidden_mse.append(float(out["hidden_mse"].cpu()))
                hidden_cos.append(float(out["hidden_cosine_loss"].cpu()))
        curves: dict[str, float] = {}
        for s in probe_starts:
            if s < 0 or s > episode_len - seq_len:
                continue
            true_scores, true_mse, true_cos = rollout_scores(s, "true")
            zero_scores, zero_mse, zero_cos = rollout_scores(s, "zero")
            random_scores, random_mse, random_cos = rollout_scores(s, "random")
            curves[f"s{s}_true_max"] = max(true_scores)
            curves[f"s{s}_zero_max"] = max(zero_scores)
            curves[f"s{s}_random_max"] = max(random_scores)
            curves[f"s{s}_true_mse"] = true_mse
            curves[f"s{s}_zero_mse"] = zero_mse
            curves[f"s{s}_random_mse"] = random_mse
            curves[f"s{s}_true_cos"] = true_cos
            curves[f"s{s}_zero_cos"] = zero_cos
            curves[f"s{s}_random_cos"] = random_cos
        return {
            "loss": float(np.mean(losses)),
            "hidden_mse": float(np.mean(hidden_mse)),
            "hidden_cosine_loss": float(np.mean(hidden_cos)),
            **curves,
        }

    start_time = time.time()
    if 0 in report_steps:
        metric = eval_overfit()
        metric.update({"event": "report", "step": 0, "elapsed_s": 0.0})
        _append_json(metrics_path, metric)

    for step in range(1, args.steps + 1):
        offset = ((step - 1) * args.batch_size) % len(starts)
        batch_starts = np.take(
            starts, np.arange(offset, offset + args.batch_size) % len(starts)
        )
        batch = make_batch(batch_starts)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            out = wm(batch)
            loss = out["_loss"]
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(wm.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        if step in report_steps:
            metric = eval_overfit()
            metric.update(
                {
                    "event": "report",
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "grad_norm": float(torch.as_tensor(grad).cpu()),
                    "elapsed_s": time.time() - start_time,
                }
            )
            _append_json(metrics_path, metric)

    ckpt = args.out_dir / f"wm_single_episode_step{args.steps}.ckpt"
    torch.save({"world_model": wm.state_dict(), "steps": args.steps, "source": str(args.wm_ckpt)}, ckpt)
    _append_json(metrics_path, {"event": "saved", "ckpt": str(ckpt), "elapsed_s": time.time() - start_time})


if __name__ == "__main__":
    main()
