"""Episode-level evaluation harness for LatentSuccessClassifier ckpts.

Runs WMPO-protocol ``predict_success`` on real (or imagined) hidden trajectories
and reports episode-level F1 / precision / recall / finish_step MAE — the
metric that actually matches WMPO's downstream usage. Replaces the per-window
F1 sweep that's intrinsically capped near 0.20 on this task.

Protocol (matches WMPO ``robwm_rollout.py::predict_success`` + the inner
``LatentSuccessClassifier.predict_success`` in this repo):

    1. For each demo, materialize a [T, L] hidden trajectory.
    2. Slide a W-frame window at ``stride`` starting at ``max(W, min_steps+W)``.
    3. Apply the classifier; convert to p(success).
    4. The EARLIEST window with p ≥ threshold defines ``finish_step``; the
       episode is predicted complete iff any window fires.
    5. Compare ``complete_pred`` vs ``complete_true`` (from rewards.sum()>0) for
       episode-level F1/precision/recall. For TP episodes, also report
       |finish_step_pred − finish_step_true| MAE.

Usage:
    # one ckpt, WMPO-default protocol
    python -u scripts/eval_latent_classifier_episode.py \\
        --ckpt data/outputs/dreamervla/outcome_classifier/libero_goal/v2_wm_replay/best.ckpt \\
        --config configs/wmpo_classifier_libero_goal_v4_real_hidden.yaml \\
        --hidden-mode real \\
        --threshold 0.93 --stride 1 --min-steps 64 \\
        --out data/outputs/dreamervla/outcome_classifier/_compare_v0/v2_wm_replay_real.json

    # multiple ckpts at once, threshold sweep
    python -u scripts/eval_latent_classifier_episode.py \\
        --ckpt v1=...best.ckpt --ckpt v2=...best.ckpt --ckpt lr=...best.ckpt \\
        --config configs/wmpo_classifier_libero_goal_v4_real_hidden.yaml \\
        --hidden-mode real \\
        --threshold-sweep 0.5,0.7,0.9,0.93,0.95,0.99 --stride 1 --min-steps 64 \\
        --out data/outputs/dreamervla/outcome_classifier/_compare_v0/sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score

from src.dataloader.wm_replay_classifier_dataset import _find_demo_pairs
from src.models.reward import LatentSuccessClassifier, LatentSuccessClassifierConfig


def _parse_ckpt_arg(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        tag, path = raw.split("=", 1)
    else:
        path = raw
        tag = Path(path).parent.name
    return tag, Path(path)


def _load_classifier(ckpt_path: Path, override_cfg: dict | None, device: torch.device):
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Saved configs can be (a) dict written by v3 train script, (b) dict from LR
    # baseline, (c) older format with no `config`. Resolve preferentially.
    saved_cfg = payload.get("config")
    saved_classifier_cfg: dict | None = None
    if isinstance(saved_cfg, dict) and "classifier" in saved_cfg:
        saved_classifier_cfg = dict(saved_cfg["classifier"])

    # head_type must match the saved weights. Resolution order:
    #   1. payload["head_type"] (LR baseline writes it here explicitly)
    #   2. saved_classifier_cfg["head_type"] (newer ckpts have it under classifier)
    #   3. "transformer" — the legacy default; pre-head_type ckpts always used it.
    # The override_cfg's head_type is IGNORED for loading because it would
    # cause silent weight-shape mismatches on old ckpts (the bug this fixes).
    if "head_type" in payload:
        resolved_head = str(payload["head_type"])
    elif saved_classifier_cfg is not None and "head_type" in saved_classifier_cfg:
        resolved_head = str(saved_classifier_cfg["head_type"])
    else:
        resolved_head = "transformer"

    # Other architectural fields (latent_dim, window, hidden_dim, ...) also
    # must match the saved weights. Prefer saved → override → defaults.
    cfg_dict: dict = {}
    if saved_classifier_cfg is not None:
        cfg_dict.update(saved_classifier_cfg)
    elif override_cfg is not None:
        cfg_dict.update(override_cfg)
    else:
        raise RuntimeError(f"no classifier config available for {ckpt_path}")
    cfg_dict["head_type"] = resolved_head

    classifier_cfg = LatentSuccessClassifierConfig(**cfg_dict)
    model = LatentSuccessClassifier(classifier_cfg).to(device).eval()
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Only flag truly suspicious mismatches; transformer vs linear keys differ.
        print(f"  [load:{ckpt_path.name}] missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    return model, classifier_cfg, payload


def _load_demo_real_hidden(
    raw_p: Path, hid_p: Path, demo_key: str
) -> tuple[np.ndarray, int, bool]:
    with h5py.File(str(raw_p), "r") as fr:
        grp = fr[demo_key]
        dones = np.asarray(grp["dones"][...]) if "dones" in grp else None
        rewards = np.asarray(grp["rewards"][...]) if "rewards" in grp else None
    with h5py.File(str(hid_p), "r") as fh:
        obs = np.asarray(fh[f"{demo_key}/obs_embedding"][...], dtype=np.float32)
    T = int(obs.shape[0])
    obs = obs.reshape(T, -1)
    if dones is not None:
        dones = dones[:T]
        finish_step = int(np.argmax(dones)) + 1 if bool(dones.any()) else T
    else:
        finish_step = T
    if rewards is not None:
        complete = bool(rewards[:T].sum() > 0)
    else:
        complete = True
    return obs, finish_step, complete


def _collect_real_demos(
    cfg, val_only: bool = True
) -> list[tuple[str, str, np.ndarray, int, bool]]:
    """Return list of (split_tag, demo_id, hidden, finish_step, complete).

    With val_only=True (default), only the held-out tail demos are returned
    (val_demos_tail success + val_failure_tail failure). This matches v3's
    train/val split exactly.
    """
    out: list[tuple[str, str, np.ndarray, int, bool]] = []
    succ_pairs = _find_demo_pairs(cfg.wm_replay.raw_dir, cfg.wm_replay.hidden_dir)
    fail_pairs = _find_demo_pairs(cfg.wm_replay.failure_raw_dir, cfg.wm_replay.failure_hidden_dir)
    if val_only:
        v_succ = int(cfg.wm_replay.val_demos_tail)
        v_fail = int(cfg.wm_replay.val_failure_tail)
        succ_pairs = succ_pairs[-v_succ:]
        fail_pairs = fail_pairs[-v_fail:] if len(fail_pairs) > v_fail else fail_pairs
    print(f"  [demos] success={len(succ_pairs)} failure={len(fail_pairs)}", flush=True)
    for raw_p, hid_p, demo_key in succ_pairs:
        obs, fs, complete = _load_demo_real_hidden(raw_p, hid_p, demo_key)
        out.append(("succ", f"{raw_p.name}/{demo_key}", obs, fs, complete))
    for raw_p, hid_p, demo_key in fail_pairs:
        obs, fs, complete = _load_demo_real_hidden(raw_p, hid_p, demo_key)
        out.append(("fail", f"{raw_p.name}/{demo_key}", obs, fs, complete))
    return out


def _pool_chunks(hidden: np.ndarray, K: int, chunk_pool: str) -> np.ndarray:
    """Pool an env-step granular ``[T, L]`` trajectory to ``[T // K, L]`` chunk frames."""
    if K <= 1:
        return hidden
    T = int(hidden.shape[0])
    T_chunk = T // K
    if T_chunk < 1:
        return hidden[:0]
    reshaped = hidden[: T_chunk * K].reshape(T_chunk, K, hidden.shape[-1])
    if chunk_pool == "last":
        return reshaped[:, -1]
    if chunk_pool == "first":
        return reshaped[:, 0]
    return reshaped.mean(axis=1)


@torch.no_grad()
def _predict_episode(
    model, hidden: np.ndarray, W: int, stride: int, min_steps: int,
    batch_size: int, device: torch.device,
    K: int = 1, chunk_pool: str = "last",
) -> np.ndarray:
    """Return the per-window p(success) array, length = number of windows scanned.

    For chunk-level classifiers (``K > 1``) the env-step trajectory is pooled
    to chunk granularity once, and ``stride``/``min_steps`` are reinterpreted
    in chunk units (``min_steps`` is divided by K, rounded up).  Returned
    indices are window-end positions in the scan space (chunk-index for chunk
    classifiers; env-step-index for action classifiers).
    """
    scan_hidden = _pool_chunks(hidden, K, chunk_pool)
    T = int(scan_hidden.shape[0])
    scan_min_steps = (int(min_steps) + K - 1) // K if K > 1 else int(min_steps)
    first_end = max(W, scan_min_steps + W)
    if T < first_end:
        return np.zeros((0,), dtype=np.float32)
    windows: list[np.ndarray] = []
    for end in range(first_end, T + 1, stride):
        windows.append(scan_hidden[end - W : end])
    if not windows:
        return np.zeros((0,), dtype=np.float32)
    arr = np.stack(windows).astype(np.float32)  # [N, W, L]
    probs: list[np.ndarray] = []
    for i in range(0, arr.shape[0], batch_size):
        chunk = torch.from_numpy(arr[i:i + batch_size]).to(device)
        logits = model(chunk)
        probs.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs)


def _episode_decision(
    probs: np.ndarray, threshold: float, W: int, stride: int, min_steps: int, T: int,
    K: int = 1,
) -> tuple[bool, int]:
    """Earliest window-end index where probs >= threshold; else T-1.

    For chunk classifiers (``K > 1``) the returned finish_step is converted
    back to env-step index (chunk c -> env-step (c+1)*K - 1).
    """
    if probs.size == 0:
        return False, T - 1
    scan_min_steps = (int(min_steps) + K - 1) // K if K > 1 else int(min_steps)
    first_end = max(W, scan_min_steps + W)
    hit = np.where(probs >= threshold)[0]
    if hit.size == 0:
        return False, T - 1
    end = first_end + int(hit[0]) * stride
    finish_in_scan = end - 1  # window-end index in scan space
    if K > 1:
        # Map chunk index c to env-step (c + 1) * K - 1 (chunk-end env-step).
        finish_env = (finish_in_scan + 1) * K - 1
        return True, finish_env
    return True, finish_in_scan


def _aggregate(
    decisions: list[tuple[bool, int]],
    truths: list[tuple[bool, int]],
) -> dict:
    y_true = np.array([int(t[0]) for t in truths], dtype=np.int64)
    y_pred = np.array([int(d[0]) for d in decisions], dtype=np.int64)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    # finish_step MAE on TPs (both pred and true positive)
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    if len(tp_idx) > 0:
        fs_err = np.array(
            [abs(decisions[i][1] - truths[i][1]) for i in tp_idx], dtype=np.float32
        )
        fs_mae = float(fs_err.mean())
        fs_med = float(np.median(fs_err))
    else:
        fs_mae = float("nan")
        fs_med = float("nan")
    return {
        "f1": f1, "prec": prec, "rec": rec,
        "n_tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "n_fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "n_fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "n_tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "n_total": int(len(y_true)),
        "finish_step_mae_on_tp": fs_mae,
        "finish_step_median_err_on_tp": fs_med,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", required=True,
                        help="Either path or tag=path. Repeatable.")
    parser.add_argument("--config", required=True,
                        help="A wmpo_classifier_libero_goal_*.yaml — used for data paths "
                             "and val_demos_tail / val_failure_tail.")
    parser.add_argument("--hidden-mode", default="real", choices=["real"],
                        help="Currently only 'real'. Imagined-mode eval intentionally "
                             "omitted to enforce the WMPO-parity protocol.")
    parser.add_argument("--threshold", type=float, default=0.93,
                        help="WMPO default 0.93 (matches best_videomae_f10.1989_th0.93.pth).")
    parser.add_argument("--threshold-sweep", default=None,
                        help="Comma-separated list of thresholds. If set, overrides --threshold "
                             "and reports a sweep table.")
    parser.add_argument("--stride", type=int, default=1, help="WMPO default 1.")
    parser.add_argument("--min-steps", type=int, default=64,
                        help="WMPO predict_success min_steps; window-end must be >= W + min_steps.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-train-split", action="store_true",
                        help="If set, evaluate on ALL demos (training + val), not just the "
                             "held-out tail. Default: held-out val only.")
    parser.add_argument("--out", required=True, help="JSON output path.")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    override_cfg_classifier = OmegaConf.to_container(cfg.classifier, resolve=True)

    # ----- collect val episodes once; reuse across ckpts -----------------
    print(f"[1/3] collecting val demos (real hidden) under {args.config}", flush=True)
    demos = _collect_real_demos(cfg, val_only=not bool(args.use_train_split))
    truths = [(c, fs) for _, _, _, fs, c in demos]
    W = int(cfg.classifier.window)

    if args.threshold_sweep is not None:
        thresholds = [float(x) for x in args.threshold_sweep.split(",")]
    else:
        thresholds = [float(args.threshold)]

    out_payload: dict = {
        "config": args.config,
        "hidden_mode": args.hidden_mode,
        "stride": int(args.stride),
        "min_steps": int(args.min_steps),
        "W": W,
        "use_train_split": bool(args.use_train_split),
        "n_demos": len(demos),
        "n_pos": int(sum(1 for c, _ in truths if c)),
        "n_neg": int(sum(1 for c, _ in truths if not c)),
        "thresholds": thresholds,
        "ckpts": {},
    }

    # ----- run each ckpt -------------------------------------------------
    for raw in args.ckpt:
        tag, path = _parse_ckpt_arg(raw)
        print(f"\n[2/3] [{tag}] loading {path}", flush=True)
        model, classifier_cfg, payload = _load_classifier(
            path, override_cfg=override_cfg_classifier, device=device
        )
        K = int(getattr(classifier_cfg, "chunk_size", 1)) if str(getattr(classifier_cfg, "granularity", "action")) == "chunk" else 1
        chunk_pool = str(getattr(classifier_cfg, "chunk_pool", "last"))
        print(f"  head_type={classifier_cfg.head_type} latent_dim={classifier_cfg.latent_dim} "
              f"window={classifier_cfg.window} granularity={getattr(classifier_cfg, 'granularity', 'action')} "
              f"K={K} chunk_pool={chunk_pool}", flush=True)

        # Cache per-episode probs once; sweep thresholds in Python.
        per_demo: list[np.ndarray] = []
        for i, (split, demo_id, hidden, _, _) in enumerate(demos):
            probs = _predict_episode(
                model, hidden, W=W, stride=int(args.stride),
                min_steps=int(args.min_steps), batch_size=int(args.batch_size), device=device,
                K=K, chunk_pool=chunk_pool,
            )
            per_demo.append(probs)
            if (i + 1) % 10 == 0:
                print(f"  [{tag}] scored {i+1}/{len(demos)} demos", flush=True)

        per_threshold: dict[str, dict] = {}
        for th in thresholds:
            decisions = []
            for (split, demo_id, hidden, _, _), probs in zip(demos, per_demo):
                T = int(hidden.shape[0])
                decisions.append(_episode_decision(
                    probs, threshold=float(th), W=W,
                    stride=int(args.stride), min_steps=int(args.min_steps), T=T, K=K,
                ))
            per_threshold[f"th_{th:.3f}"] = _aggregate(decisions, truths)
            print(f"  [{tag}] th={th:.3f} → F1={per_threshold[f'th_{th:.3f}']['f1']:.4f}",
                  flush=True)

        # Per-demo dump (only emitted for the FIRST threshold, to keep file size sane)
        first_th = thresholds[0]
        per_demo_dump = []
        for (split, demo_id, hidden, fs_true, c_true), probs in zip(demos, per_demo):
            T = int(hidden.shape[0])
            pred_complete, pred_fs = _episode_decision(
                probs, threshold=float(first_th), W=W,
                stride=int(args.stride), min_steps=int(args.min_steps), T=T, K=K,
            )
            per_demo_dump.append({
                "split": split, "demo": demo_id,
                "T": T, "fs_true": int(fs_true), "complete_true": bool(c_true),
                "fs_pred@th0": int(pred_fs), "complete_pred@th0": bool(pred_complete),
                "max_prob": float(probs.max()) if probs.size else 0.0,
                "mean_prob": float(probs.mean()) if probs.size else 0.0,
            })

        out_payload["ckpts"][tag] = {
            "path": str(path),
            "head_type": classifier_cfg.head_type,
            "saved_window_f1": float(payload.get("f1", float("nan"))),
            "saved_threshold": float(payload.get("threshold", float("nan"))),
            "per_threshold": per_threshold,
            "per_demo_at_first_threshold": per_demo_dump,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"\n[3/3] wrote {out_path}", flush=True)

    # ----- compact stdout summary table ----------------------------------
    print("\n=== episode-level summary ===")
    print(f"{'ckpt':<32s} {'head':>12s} " + " ".join(f"{f'F1@{th:.2f}':>9s}" for th in thresholds))
    for tag, payload in out_payload["ckpts"].items():
        row = f"{tag:<32s} {payload['head_type']:>12s} "
        row += " ".join(
            f"{payload['per_threshold'][f'th_{th:.3f}']['f1']:>9.4f}" for th in thresholds
        )
        print(row)


if __name__ == "__main__":
    main()
