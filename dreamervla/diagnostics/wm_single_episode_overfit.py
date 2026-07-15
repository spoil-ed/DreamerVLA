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
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from dreamervla.utils.paths import data_path
from dreamervla.utils.run_config import load_run_config


def _parse_report_steps(text: str) -> set[int]:
    return {int(item) for item in text.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Execute GPU work for train/eval/all stages. Without this flag those "
            "stages print the resolved plan and exit."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("all", "check", "train", "eval"),
        default="all",
        help=(
            "Run the combined probe or one split stage. check validates inputs; "
            "train trains and saves a checkpoint; eval evaluates a trained checkpoint."
        ),
    )
    parser.add_argument(
        "--run-config",
        dest="run_config",
        type=Path,
        default=data_path(
            "outputs/coldstart_warmup_cotrain/"
            "fixed_cls_wm_vla_eval_g7_component_20260707_205109/"
            "cotrain/.hydra/config.yaml"
        ),
        help="Cotrain run config containing ray_components.world_model/classifier.",
    )
    parser.add_argument(
        "--resolved-config",
        dest="run_config",
        type=Path,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wm-ckpt",
        type=Path,
        default=data_path(
            "outputs/world_model_probe/current_actions_reward0_20260708_01/"
            "wm_probe_step1200.ckpt"
        ),
        help="World-model checkpoint with a top-level world_model state dict.",
    )
    parser.add_argument(
        "--trained-wm-ckpt",
        type=Path,
        default=None,
        help=(
            "Trained world-model checkpoint for --stage eval. Defaults to "
            "<out-dir>/wm_single_episode_step<steps>.ckpt."
        ),
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=Path,
        default=data_path(
            "outputs/coldstart_warmup_cotrain/"
            "fixed_wm_wmpo_cls_mainline_20260707_01/init/"
            "fixed_wm_wmpo_cls_init.ckpt"
        ),
        help="Classifier checkpoint containing state_dicts.classifier.",
    )
    parser.add_argument(
        "--hidden-hdf5",
        type=Path,
        default=data_path(
            "processed_data/"
            "OpenVLA_Onetraj_LIBERO_libero_goal/"
            "no_noops_t_256_oft_hidden_token_vla_policy_h1/"
            "open_the_middle_drawer_of_the_cabinet_demo.hdf5"
        ),
        help="Hidden sidecar HDF5 containing data/<demo>/obs_embedding and lang_emb.",
    )
    parser.add_argument(
        "--raw-hdf5",
        type=Path,
        default=data_path(
            "processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/"
            "no_noops_t_256_remaining_reward/"
            "open_the_middle_drawer_of_the_cabinet_demo.hdf5"
        ),
        help="Raw reward HDF5 containing actions, rewards, and proprio observations.",
    )
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=data_path("outputs/world_model_probe/single_episode_overfit"),
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


def _require_hdf5_datasets(path: Path, label: str, demo_key: str, datasets: list[str]) -> None:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{label} missing group: data")
        if demo_key not in f["data"]:
            raise KeyError(f"{label} missing demo: data/{demo_key}")
        demo = f["data"][demo_key]
        for dataset in datasets:
            node: Any = demo
            parts = dataset.split("/")
            for part in parts:
                if part not in node:
                    raise KeyError(f"{label} missing dataset: data/{demo_key}/{dataset}")
                node = node[part]


def _default_trained_wm_ckpt(args: argparse.Namespace) -> Path:
    return args.out_dir / f"wm_single_episode_step{args.steps}.ckpt"


def _required_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "run config": args.run_config,
        "classifier checkpoint": args.classifier_ckpt,
        "hidden HDF5": args.hidden_hdf5,
        "raw HDF5": args.raw_hdf5,
    }
    if args.stage == "eval":
        paths["trained WM checkpoint"] = args.trained_wm_ckpt
    else:
        paths["WM checkpoint"] = args.wm_ckpt
    return paths


def _validate_inputs(args: argparse.Namespace) -> None:
    for label, path in _required_paths(args).items():
        _require_path(path, label)
    _require_hdf5_datasets(
        args.hidden_hdf5,
        "hidden HDF5",
        args.demo_key,
        ["obs_embedding", "lang_emb"],
    )
    _require_hdf5_datasets(
        args.raw_hdf5,
        "raw HDF5",
        args.demo_key,
        ["actions", "rewards", "obs/ee_pos", "obs/ee_ori", "obs/gripper_states"],
    )


def _metrics_path_for_stage(out_dir: Path, stage: str) -> Path:
    if stage == "train":
        return out_dir / "train_metrics.jsonl"
    if stage == "eval":
        return out_dir / "eval_metrics.jsonl"
    return out_dir / "metrics.jsonl"


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
    if args.trained_wm_ckpt is None:
        args.trained_wm_ckpt = _default_trained_wm_ckpt(args)
    report_steps = _parse_report_steps(args.report_steps)
    probe_starts = [int(item) for item in args.probe_starts.split(",") if item.strip()]

    plan = {
        "stages": [
            {
                "name": "data_check",
                "checks": [
                    "run config exists",
                    "source WM checkpoint exists for train/all stages",
                    "trained WM checkpoint exists for eval stage",
                    "classifier checkpoint exists",
                    "hidden HDF5 contains demo obs_embedding/lang_emb",
                    "raw HDF5 contains demo actions/rewards/proprio",
                ],
            },
            {
                "name": "train",
                "description": "single-demo sliding-window Chunk-WM overfit",
            },
            {
                "name": "test",
                "description": "periodic true/zero/random action rollout comparison",
            },
            {
                "name": "outputs",
                "files": [
                    "metrics.jsonl",
                    "train_metrics.jsonl",
                    "train_summary.json",
                    "train_summary.md",
                    "eval_metrics.jsonl",
                    "summary.json",
                    "summary.md",
                    f"wm_single_episode_step{args.steps}.ckpt",
                ],
            },
        ],
        "stage": args.stage,
        "run_config": str(args.run_config),
        "wm_ckpt": str(args.wm_ckpt),
        "trained_wm_ckpt": str(args.trained_wm_ckpt),
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
    if args.stage == "check":
        _validate_inputs(args)
        print(json.dumps({"status": "ok", **plan}, indent=2, sort_keys=True))
        return

    if not args.run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return

    _validate_inputs(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = _metrics_path_for_stage(args.out_dir, args.stage)
    error_path = args.out_dir / "error.txt"
    metrics_path.write_text("", encoding="utf-8")
    if error_path.exists():
        error_path.unlink()

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

    cfg = load_run_config(args.run_config)
    wm_cfg = OmegaConf.to_container(cfg.ray_components.world_model, resolve=True)
    wm_cfg["kwargs"]["reward_loss_scale"] = 0.0
    wm_cfg["kwargs"]["chunk_rollout_chunks"] = 1
    wm_cfg["kwargs"]["chunk_rollout_loss_scale"] = 0.0
    cls_cfg = OmegaConf.to_container(cfg.ray_components.classifier, resolve=True)

    load_wm_ckpt = args.trained_wm_ckpt if args.stage == "eval" else args.wm_ckpt
    wm = _build_component(wm_cfg).to(device).train()
    classifier = _build_component(cls_cfg).to(device).eval()
    wm_payload = torch.load(load_wm_ckpt, map_location="cpu")
    cls_payload = torch.load(args.classifier_ckpt, map_location="cpu")
    wm.load_state_dict(wm_payload["world_model"])
    classifier.load_state_dict(cls_payload["state_dicts"]["classifier"])
    threshold = float(cls_payload.get("classifier_threshold", 0.95))

    optimizer = None
    if args.stage in {"all", "train"}:
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
            "source_wm": str(load_wm_ckpt),
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
    if args.stage == "eval":
        metric = eval_overfit()
        metric.update({"event": "report", "step": args.steps, "elapsed_s": 0.0})
        _append_json(metrics_path, metric)
        summary = _write_summary(
            args,
            metrics_path,
            load_wm_ckpt,
            probe_starts,
            source_wm_path=load_wm_ckpt,
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    if args.stage == "all" and 0 in report_steps:
        metric = eval_overfit()
        metric.update({"event": "report", "step": 0, "elapsed_s": 0.0})
        _append_json(metrics_path, metric)

    if optimizer is None:
        raise RuntimeError(f"optimizer was not initialized for stage {args.stage}")

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
            metric = {
                "event": "train_report",
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "grad_norm": float(torch.as_tensor(grad).cpu()),
                "elapsed_s": time.time() - start_time,
            }
            if args.stage == "all":
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
    torch.save(
        {"world_model": wm.state_dict(), "steps": args.steps, "source": str(args.wm_ckpt)},
        ckpt,
    )
    _append_json(
        metrics_path,
        {"event": "saved", "ckpt": str(ckpt), "elapsed_s": time.time() - start_time},
    )
    if args.stage == "train":
        summary = _write_train_summary(args, metrics_path, ckpt)
    else:
        summary = _write_summary(args, metrics_path, ckpt, probe_starts)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _action_sensitivity(
    record: dict[str, Any],
    probe_starts: list[int],
) -> tuple[list[dict[str, float]], float | None]:
    rows: list[dict[str, float]] = []
    gaps: list[float] = []
    for start in probe_starts:
        true_key = f"s{start}_true_mse"
        zero_key = f"s{start}_zero_mse"
        random_key = f"s{start}_random_mse"
        if true_key not in record or zero_key not in record or random_key not in record:
            continue
        true_mse = float(record[true_key])
        zero_mse = float(record[zero_key])
        random_mse = float(record[random_key])
        best_alt_mse = min(zero_mse, random_mse)
        advantage = best_alt_mse - true_mse
        rows.append(
            {
                "start": float(start),
                "true_mse": true_mse,
                "zero_mse": zero_mse,
                "random_mse": random_mse,
                "true_advantage_mse_vs_best_alt": advantage,
                "max_abs_mse_delta_vs_true": max(
                    abs(zero_mse - true_mse),
                    abs(random_mse - true_mse),
                ),
                "true_score_max": float(record.get(f"s{start}_true_max", float("nan"))),
                "zero_score_max": float(record.get(f"s{start}_zero_max", float("nan"))),
                "random_score_max": float(
                    record.get(f"s{start}_random_max", float("nan"))
                ),
            }
        )
        gaps.append(advantage)
    mean_gap = float(np.mean(gaps)) if gaps else None
    return rows, mean_gap


def _write_summary(
    args: argparse.Namespace,
    metrics_path: Path,
    ckpt_path: Path,
    probe_starts: list[int],
    source_wm_path: Path | None = None,
) -> dict[str, Any]:
    records = _read_jsonl(metrics_path)
    reports = [record for record in records if record.get("event") == "report"]
    if not reports:
        raise RuntimeError(f"no report records found in {metrics_path}")
    first = reports[0]
    final = reports[-1]
    action_rows, mean_action_gap = _action_sensitivity(final, probe_starts)
    summary: dict[str, Any] = {
        "status": "complete",
        "out_dir": str(args.out_dir),
        "metrics_path": str(metrics_path),
        "summary_json": str(args.out_dir / "summary.json"),
        "summary_md": str(args.out_dir / "summary.md"),
        "checkpoint": str(ckpt_path),
        "source_wm": str(source_wm_path or args.wm_ckpt),
        "demo_key": args.demo_key,
        "steps": int(args.steps),
        "first_report_step": int(first["step"]),
        "final_report_step": int(final["step"]),
        "hidden_mse_start": float(first["hidden_mse"]),
        "hidden_mse_final": float(final["hidden_mse"]),
        "hidden_mse_delta": float(final["hidden_mse"]) - float(first["hidden_mse"]),
        "hidden_cosine_loss_start": float(first["hidden_cosine_loss"]),
        "hidden_cosine_loss_final": float(final["hidden_cosine_loss"]),
        "mean_true_advantage_mse_vs_best_alt": mean_action_gap,
        "action_sensitivity": action_rows,
    }
    summary_json = args.out_dir / "summary.json"
    summary_md = args.out_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md.write_text(_render_summary_markdown(summary), encoding="utf-8")
    return summary


def _write_train_summary(
    args: argparse.Namespace,
    metrics_path: Path,
    ckpt_path: Path,
) -> dict[str, Any]:
    records = _read_jsonl(metrics_path)
    reports = [record for record in records if record.get("event") == "train_report"]
    final_report = reports[-1] if reports else {}
    saved = [record for record in records if record.get("event") == "saved"]
    if not saved:
        raise RuntimeError(f"no saved checkpoint record found in {metrics_path}")
    summary: dict[str, Any] = {
        "status": "complete",
        "stage": "train",
        "out_dir": str(args.out_dir),
        "metrics_path": str(metrics_path),
        "summary_json": str(args.out_dir / "train_summary.json"),
        "summary_md": str(args.out_dir / "train_summary.md"),
        "checkpoint": str(ckpt_path),
        "source_wm": str(args.wm_ckpt),
        "demo_key": args.demo_key,
        "steps": int(args.steps),
        "final_train_step": int(final_report.get("step", args.steps)),
        "final_train_loss": (
            float(final_report["train_loss"]) if "train_loss" in final_report else None
        ),
        "final_grad_norm": (
            float(final_report["grad_norm"]) if "grad_norm" in final_report else None
        ),
    }
    summary_json = args.out_dir / "train_summary.json"
    summary_md = args.out_dir / "train_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md.write_text(_render_train_summary_markdown(summary), encoding="utf-8")
    return summary


def _render_train_summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# WM Single-Episode Train Summary",
            "",
            f"- Status: `{summary['status']}`",
            f"- Stage: `{summary['stage']}`",
            f"- Steps: `{summary['steps']}`",
            f"- Source WM: `{summary['source_wm']}`",
            f"- Output dir: `{summary['out_dir']}`",
            f"- Checkpoint: `{summary['checkpoint']}`",
            f"- Final train loss: `{summary['final_train_loss']}`",
            f"- Final grad norm: `{summary['final_grad_norm']}`",
            f"- Raw metrics: `{summary['metrics_path']}`",
            f"- JSON summary: `{summary['summary_json']}`",
            "",
        ]
    )


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# WM Single-Episode Overfit Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Steps: `{summary['steps']}`",
        f"- Source WM: `{summary['source_wm']}`",
        f"- Output dir: `{summary['out_dir']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        "",
        "## Hidden Prediction",
        "",
        f"- Start hidden MSE: `{summary['hidden_mse_start']:.8g}`",
        f"- Final hidden MSE: `{summary['hidden_mse_final']:.8g}`",
        f"- Delta hidden MSE: `{summary['hidden_mse_delta']:.8g}`",
        f"- Start hidden cosine loss: `{summary['hidden_cosine_loss_start']:.8g}`",
        f"- Final hidden cosine loss: `{summary['hidden_cosine_loss_final']:.8g}`",
        "",
        "## Action Sensitivity",
        "",
        "Positive `true_advantage_mse_vs_best_alt` means true actions beat both zero and random actions.",
        "",
        "| start | true_mse | zero_mse | random_mse | true_advantage | max_abs_delta | true_score | zero_score | random_score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["action_sensitivity"]:
        lines.append(
            "| {start:.0f} | {true_mse:.8g} | {zero_mse:.8g} | {random_mse:.8g} | "
            "{true_advantage_mse_vs_best_alt:.8g} | {max_abs_mse_delta_vs_true:.8g} | "
            "{true_score_max:.8g} | {zero_score_max:.8g} | {random_score_max:.8g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"- Mean true-action MSE advantage: `{summary['mean_true_advantage_mse_vs_best_alt']}`",
            f"- Raw metrics: `{summary['metrics_path']}`",
            f"- JSON summary: `{summary['summary_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
