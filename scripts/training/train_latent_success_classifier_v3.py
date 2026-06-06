# ruff: noqa: E402
"""Train LatentSuccessClassifier — Phase 3 v3 / v4 (WM-replay + real failures + rollouts).

Same pipeline as v2 train script but threads through up to three extra sources:

    success demos → 401 imagined positive trajectories (complete=True)
    failure demos → 67 imagined real-failure trajectories (complete=False)
    rollout demos → real-policy SFT / online rollouts (label & finish_step
                    derived per-episode from rewards/dones). Closest analog
                    to WMPO's SFT-rollout corpus.
    [optional] swap-neg per success demo

Per-episode windowing matches WMPO/reward_model/videomae.py exactly:
    end window: label = int(complete), anchored at meta["finish_step"]
    1 random earlier window: label = 0

Single-GPU. Usage:
    CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
        python -u \
        scripts/train_latent_success_classifier_v3.py \
        --config configs/wmpo_classifier_libero_goal_v3_with_failures.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dreamer_vla.dataset.wm_replay_classifier_dataset import (
    WMReplayClassifierDataset,
    _find_demo_pairs,
)
from dreamer_vla.models.reward import LatentSuccessClassifier, LatentSuccessClassifierConfig
from dreamer_vla.models.world_model.rynn_dino_wm_chunk import ChunkAwareRynnDinoWMWorldModel


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg) -> dict:
    model.eval()
    probs_l, ys_l = [], []
    for xs, ys in loader:
        xs = xs.to(device, non_blocking=True)
        logits = model(xs)
        probs_l.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        ys_l.extend(ys.tolist())
    probs = np.asarray(probs_l, dtype=np.float32)
    ys = np.asarray(ys_l, dtype=np.int64)
    if len(ys) == 0:
        return {"metrics": {}, "best": {"f1": 0.0, "thresh": 0.5}, "n_val": 0}
    metrics: dict = {}
    best = {"f1": -1.0, "thresh": 0.5}
    thresholds = np.linspace(
        cfg.train.thresh_min, cfg.train.thresh_max, int(cfg.train.thresh_steps)
    )
    for th in thresholds:
        preds = (probs >= th).astype(int)
        f1 = float(f1_score(ys, preds, zero_division=0))
        metrics[f"th_{th:.2f}"] = {
            "f1": f1,
            "acc": float(accuracy_score(ys, preds)),
            "prec": float(precision_score(ys, preds, zero_division=0)),
            "rec": float(recall_score(ys, preds, zero_division=0)),
        }
        if f1 > best["f1"]:
            best = {"f1": f1, "thresh": float(th)}
    return {"metrics": metrics, "best": best, "n_val": int(len(ys))}


@torch.no_grad()
def _evaluate_episode_level(
    model: nn.Module,
    val_ds,
    device: torch.device,
    W: int,
    min_steps: int,
    stride: int,
    thresh_min: float,
    thresh_max: float,
    thresh_steps: int,
) -> dict:
    """WMPO-style episode-level F1.

    For each val episode, slide a W-frame window at the given stride from
    ``min_steps`` to the end. The episode is predicted as success iff ANY
    window has p(success) >= threshold (mirrors WMPO predict_success in
    WMPO/verl/workers/rollout/robwm_rollout.py and the F1 scan in
    WMPO/reward_model/find_thre.py).

    Sweeps thresholds and returns the best episode-level F1.
    """
    model.eval()
    # _all_labeled_trajs() returns list[(traj, complete, finish_step)] across
    # _pos_trajs / _neg_trajs / _failure_trajs / _rollout_trajs.
    all_trajs = list(val_ds._all_labeled_trajs())

    # Gather per-window max-probability per episode in a single forward pass batch.
    # We collect ALL windows for ALL episodes, tagged with episode idx + true label.
    flat_windows: list[torch.Tensor] = []
    flat_ep_idx: list[int] = []
    ep_true: list[int] = []
    for ep_idx, (traj, complete, finish_step) in enumerate(all_trajs):
        T = int(min(finish_step, traj.shape[0]))
        if T < W + min_steps:
            # too short to fire any window — count as predicted-0
            ep_true.append(int(bool(complete)))
            continue
        first_end = max(W, min_steps + W)
        for end in range(first_end, T + 1, stride):
            flat_windows.append(torch.from_numpy(traj[end - W : end]).float())
            flat_ep_idx.append(ep_idx)
        ep_true.append(int(bool(complete)))

    if not flat_windows:
        return {"metrics": {}, "best": {"f1": 0.0, "thresh": 0.5}, "n_ep": 0}

    # forward in mini-batches to bound GPU memory
    BATCH = 32
    ep_max_prob: dict[int, float] = {}
    for i in range(0, len(flat_windows), BATCH):
        chunk = torch.stack(flat_windows[i : i + BATCH]).to(device, non_blocking=True)
        logits = model(chunk)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        for p, idx in zip(probs, flat_ep_idx[i : i + BATCH]):
            if p > ep_max_prob.get(idx, -1.0):
                ep_max_prob[idx] = float(p)

    thresholds = np.linspace(thresh_min, thresh_max, int(thresh_steps))
    metrics: dict = {}
    best = {"f1": -1.0, "thresh": float(thresholds[0])}
    y_true = np.asarray(ep_true, dtype=np.int64)
    for th in thresholds:
        y_pred = np.array(
            [int(ep_max_prob.get(i, 0.0) >= th) for i in range(len(ep_true))],
            dtype=np.int64,
        )
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        metrics[f"th_{th:.2f}"] = {
            "f1": f1,
            "acc": float(accuracy_score(y_true, y_pred)),
            "prec": float(precision_score(y_true, y_pred, zero_division=0)),
            "rec": float(recall_score(y_true, y_pred, zero_division=0)),
        }
        if f1 > best["f1"]:
            best = {"f1": f1, "thresh": float(th)}
    return {"metrics": metrics, "best": best, "n_ep": int(len(ep_true))}


def _apply_ratio_knobs(
    ds,
    fail_oversample: int = 1,
    succ_cap: int | None = None,
    drop_swap_neg: bool = False,
    rollout_oversample: int = 1,
    tag: str = "ds",
) -> None:
    """Mutate _pos_trajs / _neg_trajs / _failure_trajs / _rollout_trajs in place.

    Called AFTER ``imagine_all()`` so the lists are populated. Each list entry
    contributes 2 windows during _train_yield (1 pos label=int(complete) +
    1 random earlier label=0), so duplicating an entry doubles its window
    contribution. Class balance composition (per epoch over the iterable):

        pos label=1 windows  =  #{_pos_trajs with complete=True}  =  len(_pos_trajs)*1
        neg label=0 windows  =  len(_pos_trajs)            (succ early)
                              + len(_neg_trajs)*2          (swap-perturbed, both windows label=0)
                              + len(_failure_trajs)*2      (failure, both windows label=0)
                              + len(_rollout_trajs)*[0..2] (depends on per-rollout complete flag)

    Effects:
      * ``succ_cap``: truncate _pos_trajs and the paired _neg_trajs (swap-neg
        is created per success demo, so they're 1-1 paired).
      * ``fail_oversample``: replicate failure trajs N times to amplify
        real-failure negative density.
      * ``drop_swap_neg``: clear _neg_trajs entirely, leaving only success-early
        and failure as negative sources.
      * ``rollout_oversample``: replicate rollout demos N times.
    """
    n_pos_before = len(ds._pos_trajs)
    n_neg_before = len(ds._neg_trajs)
    n_fail_before = len(ds._failure_trajs)
    n_roll_before = len(ds._rollout_trajs)

    if succ_cap is not None and succ_cap < n_pos_before:
        ds._pos_trajs = ds._pos_trajs[:succ_cap]
        ds._pos_meta = ds._pos_meta[:succ_cap]
        # swap-neg is 1-1 paired with success demos; cap to match
        if len(ds._neg_trajs) > succ_cap:
            ds._neg_trajs = ds._neg_trajs[:succ_cap]
            ds._neg_meta = ds._neg_meta[:succ_cap]

    if drop_swap_neg:
        ds._neg_trajs, ds._neg_meta = [], []

    if fail_oversample > 1 and ds._failure_trajs:
        ds._failure_trajs = list(ds._failure_trajs) * fail_oversample
        ds._failure_meta = list(ds._failure_meta) * fail_oversample

    if rollout_oversample > 1 and ds._rollout_trajs:
        ds._rollout_trajs = list(ds._rollout_trajs) * rollout_oversample
        ds._rollout_meta = list(ds._rollout_meta) * rollout_oversample

    print(
        f"[ratio:{tag}] pos {n_pos_before}->{len(ds._pos_trajs)}  "
        f"swap-neg {n_neg_before}->{len(ds._neg_trajs)}  "
        f"failure {n_fail_before}->{len(ds._failure_trajs)}  "
        f"rollout {n_roll_before}->{len(ds._rollout_trajs)}",
        flush=True,
    )


def _replace_with_real_hidden(ds, tag: str = "ds") -> None:
    """Bypass WM imagination: load REAL pi0 obs_embedding straight from HDF5.

    Use this to ablate "imagine drift" — see scripts/measure_real_vs_imagine.py
    for evidence that the chunk WM accumulates cosine drift to ~0.80 at the end
    of episodes, exactly where the positive window lives.

    Clears _neg_trajs (swap-perturbed) since they have no real-hidden analog.
    """

    def _do(pairs):
        out_trajs, out_meta = [], []
        for pair in pairs:
            obs, _act, finish_step, complete = ds._load_pair_at(pair)
            # obs may be [T, N, D] or [T, N*D]; flatten last dims
            arr = np.asarray(obs, dtype=np.float32).reshape(obs.shape[0], -1)
            out_trajs.append(arr)
            out_meta.append((bool(complete), int(finish_step)))
        return out_trajs, out_meta

    pos_trajs, pos_meta = _do(ds.pairs)
    fail_trajs, fail_meta = _do(ds.failure_pairs)
    roll_trajs, roll_meta = _do(getattr(ds, "rollout_pairs", []) or [])
    ds._pos_trajs, ds._pos_meta = pos_trajs, pos_meta
    ds._neg_trajs, ds._neg_meta = [], []
    ds._failure_trajs, ds._failure_meta = fail_trajs, fail_meta
    ds._rollout_trajs, ds._rollout_meta = roll_trajs, roll_meta
    print(
        f"[real-hidden:{tag}] pos={len(pos_trajs)} failure={len(fail_trajs)} "
        f"rollout={len(roll_trajs)} (swap-neg cleared)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    # ---- data ratio knobs (applied AFTER imagine_all) ----
    parser.add_argument(
        "--fail-oversample",
        type=int,
        default=1,
        help="Duplicate failure trajs N times after imagine_all (default 1=no oversample). "
        "Use >1 to push pos:neg ratio toward WMPO's ~1:5 regime.",
    )
    parser.add_argument(
        "--rollout-oversample",
        type=int,
        default=1,
        help="Duplicate rollout trajs N times after imagine_all.",
    )
    parser.add_argument(
        "--succ-cap",
        type=int,
        default=None,
        help="Cap the number of success trajs (and their paired swap-neg) used for training.",
    )
    parser.add_argument(
        "--drop-swap-neg",
        action="store_true",
        help="Empty _neg_trajs after imagine_all (kills the swap-perturbed negative class).",
    )
    # ---- ablate WM imagine drift ----
    parser.add_argument(
        "--use-real-hidden",
        action="store_true",
        help="Bypass imagine_all and use REAL pi0 obs_embedding from HDF5 instead. "
        "Lets us isolate imagine-drift as the failure source.",
    )
    # ---- WMPO-style episode-level eval ----
    parser.add_argument(
        "--episode-eval",
        action="store_true",
        help="Also report WMPO-style episode-level F1 (any window >= threshold => success).",
    )
    parser.add_argument(
        "--episode-min-steps",
        type=int,
        default=100,
        help="WMPO predict_success gate: earliest window-end frame.",
    )
    parser.add_argument(
        "--episode-stride",
        type=int,
        default=1,
        help="Sliding window stride for episode-level eval (WMPO default 1).",
    )
    parser.add_argument("--episode-thresh-min", type=float, default=0.3)
    parser.add_argument("--episode-thresh-max", type=float, default=0.99)
    parser.add_argument("--episode-thresh-steps", type=int, default=30)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)

    print(f"[1/4] loading chunk WM from {cfg.wm_replay.chunk_wm_ckpt}")
    chunk_wm = ChunkAwareRynnDinoWMWorldModel.from_rynn_dino_wm_ckpt(
        cfg.wm_replay.chunk_wm_ckpt,
        chunk_size=int(cfg.wm_replay.K),
        device=device,
        strict=True,
    ).eval()

    print("[2/4] discovering demo pairs (success + failure + rollout)")
    success_pairs = _find_demo_pairs(cfg.wm_replay.raw_dir, cfg.wm_replay.hidden_dir)
    failure_pairs = _find_demo_pairs(
        cfg.wm_replay.failure_raw_dir, cfg.wm_replay.failure_hidden_dir
    )
    rollout_raw_dir = OmegaConf.select(cfg, "wm_replay.rollout_raw_dir", default=None)
    rollout_hidden_dir = OmegaConf.select(
        cfg, "wm_replay.rollout_hidden_dir", default=None
    )
    if rollout_raw_dir and rollout_hidden_dir:
        rollout_pairs = _find_demo_pairs(rollout_raw_dir, rollout_hidden_dir)
    else:
        rollout_pairs = []
    print(
        f"  success demos: {len(success_pairs)}  failure demos: {len(failure_pairs)}"
        f"  rollout demos: {len(rollout_pairs)}"
    )

    val_succ_tail = int(cfg.wm_replay.val_demos_tail)
    val_fail_tail = int(cfg.wm_replay.val_failure_tail)
    val_rollout_tail = int(
        OmegaConf.select(cfg, "wm_replay.val_rollout_tail", default=0) or 0
    )
    train_succ = success_pairs[:-val_succ_tail]
    val_succ = success_pairs[-val_succ_tail:]
    train_fail = (
        failure_pairs[:-val_fail_tail]
        if len(failure_pairs) > val_fail_tail
        else failure_pairs[:]
    )
    val_fail = (
        failure_pairs[-val_fail_tail:] if len(failure_pairs) > val_fail_tail else []
    )
    if rollout_pairs and val_rollout_tail > 0 and len(rollout_pairs) > val_rollout_tail:
        train_rollout = rollout_pairs[:-val_rollout_tail]
        val_rollout = rollout_pairs[-val_rollout_tail:]
    else:
        train_rollout = rollout_pairs[:]
        val_rollout = []
    print(
        f"  train: {len(train_succ)} success + {len(train_fail)} failure + {len(train_rollout)} rollout"
    )
    print(
        f"  val:   {len(val_succ)} success + {len(val_fail)} failure + {len(val_rollout)} rollout"
    )

    def _build_ds(s_pairs, f_pairs, r_pairs, mode, seed):
        ds = WMReplayClassifierDataset(
            raw_dir=cfg.wm_replay.raw_dir,
            hidden_dir=cfg.wm_replay.hidden_dir,
            chunk_wm=chunk_wm,
            device=device,
            K=int(cfg.wm_replay.K),
            W=int(cfg.classifier.window),
            num_hist=int(cfg.wm_replay.num_hist),
            mode=mode,
            stride=int(
                cfg.train.stride_train if mode == "train" else cfg.train.stride_val
            ),
            neg_method=str(cfg.wm_replay.neg_method),
            noise_std=float(cfg.wm_replay.noise_std),
            swap_min_frac=float(cfg.wm_replay.swap_min_frac),
            swap_max_frac=float(cfg.wm_replay.swap_max_frac),
            include_swap_negatives=bool(cfg.wm_replay.include_swap_negatives),
            failure_raw_dir=cfg.wm_replay.failure_raw_dir,
            failure_hidden_dir=cfg.wm_replay.failure_hidden_dir,
            rollout_raw_dir=rollout_raw_dir,
            rollout_hidden_dir=rollout_hidden_dir,
            max_demos=None,
            max_failure_demos=None,
            max_rollout_demos=None,
            seed=seed,
        )
        ds.pairs = s_pairs
        ds.failure_pairs = f_pairs
        ds.rollout_pairs = r_pairs
        return ds

    tr_ds = _build_ds(train_succ, train_fail, train_rollout, mode="train", seed=42)
    va_ds = _build_ds(val_succ, val_fail, val_rollout, mode="val", seed=43)

    if args.use_real_hidden:
        print(
            "[3/4] --use-real-hidden: SKIPPING imagine_all, loading REAL hidden directly"
        )
        # imagine_all still has to run because _imagine_one initializes internal
        # buffers (and some setup state). But we immediately overwrite the trajs.
        tr_ds.imagine_all(verbose=False)
        va_ds.imagine_all(verbose=False)
        _replace_with_real_hidden(tr_ds, tag="train")
        _replace_with_real_hidden(va_ds, tag="val")
    else:
        print("[3/4] imagine_all() — running chunk WM (success + failure demos)")
        print("  imagining train demos...")
        tr_ds.imagine_all(verbose=True)
        print("  imagining val demos...")
        va_ds.imagine_all(verbose=True)

    # Apply ratio knobs ONLY to the training set; keep val composition unchanged
    # so per-window / episode-level metrics stay comparable across runs.
    _apply_ratio_knobs(
        tr_ds,
        fail_oversample=int(args.fail_oversample),
        succ_cap=args.succ_cap,
        drop_swap_neg=bool(args.drop_swap_neg),
        rollout_oversample=int(args.rollout_oversample),
        tag="train",
    )

    tr_ld = DataLoader(
        tr_ds,
        batch_size=int(cfg.train.batch_size),
        num_workers=0,
        collate_fn=_collate,
        pin_memory=True,
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=int(cfg.train.batch_size),
        num_workers=0,
        collate_fn=_collate,
        pin_memory=True,
    )

    classifier_cfg = LatentSuccessClassifierConfig(
        **OmegaConf.to_container(cfg.classifier)
    )
    model = LatentSuccessClassifier(classifier_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    ckpt_dir = Path(cfg.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, ckpt_dir / "config.yaml")
    log_path = ckpt_dir / "train_log.jsonl"

    print("[4/4] training classifier on WM-replayed success + real failures")
    best_f1 = -1.0
    step = 0
    tr_iter = iter(tr_ld)
    while step < int(cfg.train.max_steps):
        try:
            xs, ys = next(tr_iter)
        except StopIteration:
            tr_iter = iter(tr_ld)
            continue
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        model.train()
        logits = model(xs)
        loss = criterion(logits, ys)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        step += 1
        if step % int(cfg.train.log_every) == 0:
            line = {"step": step, "loss": float(loss.detach().item())}
            print(json.dumps(line))
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
        if step % int(cfg.train.eval_steps) == 0:
            out = _evaluate(model, va_ld, device, cfg)
            line = {"step": step, "best_val": out["best"], "n_val": out["n_val"]}
            print(json.dumps(line))
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")

            # WMPO-style episode-level F1 (any-window >= threshold)
            ep_out = None
            if bool(args.episode_eval):
                ep_out = _evaluate_episode_level(
                    model,
                    va_ds,
                    device,
                    W=int(cfg.classifier.window),
                    min_steps=int(args.episode_min_steps),
                    stride=int(args.episode_stride),
                    thresh_min=float(args.episode_thresh_min),
                    thresh_max=float(args.episode_thresh_max),
                    thresh_steps=int(args.episode_thresh_steps),
                )
                ep_line = {
                    "step": step,
                    "episode_best": ep_out["best"],
                    "n_ep": ep_out["n_ep"],
                }
                print(json.dumps(ep_line))
                with open(log_path, "a") as f:
                    f.write(json.dumps(ep_line) + "\n")

            # Keep best-checkpoint criterion the same as before (per-window F1).
            # If you'd rather select on episode-level F1, swap the line below.
            if out["best"]["f1"] > best_f1:
                best_f1 = float(out["best"]["f1"])
                payload = {
                    "model": model.state_dict(),
                    "threshold": float(out["best"]["thresh"]),
                    "f1": best_f1,
                    "step": step,
                    "config": OmegaConf.to_container(cfg),
                }
                if ep_out is not None:
                    payload["episode_best"] = ep_out["best"]
                torch.save(payload, ckpt_dir / "best.ckpt")
                print(
                    f"[best] step={step} f1={best_f1:.4f} thresh={out['best']['thresh']:.2f} → {ckpt_dir / 'best.ckpt'}"
                )
    print(f"done. best_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
