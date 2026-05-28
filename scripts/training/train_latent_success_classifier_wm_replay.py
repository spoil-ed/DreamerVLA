# ruff: noqa: E402
"""Train LatentSuccessClassifier on WM-replayed pos/neg trajectories.

Phase 3 v2 of the WMPO reproduction. Replaces demo-only shards with on-the-fly
chunk-WM imagined trajectories that match the PPO inference distribution.

Pipeline:
    1) load frozen chunk WM from existing m1024 ckpt
    2) build WMReplayClassifierDataset over libero_goal demo HDF5s
    3) imagine_all() → cached pos + neg trajectories on GPU
    4) plain torch DataLoader → classifier training loop (same as v1)

Single-GPU. Usage:
    CUDA_VISIBLE_DEVICES=6 \
        /home/user01/miniconda3/envs/dreamervla/bin/python \
        scripts/train_latent_success_classifier_wm_replay.py \
        --config configs/wmpo_classifier_libero_goal_wm_replay.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
def _evaluate(model, loader, device, cfg, max_batches: int | None = None):
    model.eval()
    probs_l, ys_l = [], []
    for i, (xs, ys) in enumerate(loader):
        xs = xs.to(device, non_blocking=True)
        logits = model(xs)
        probs_l.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        ys_l.extend(ys.tolist())
        if max_batches is not None and (i + 1) >= max_batches:
            break
    probs = np.asarray(probs_l, dtype=np.float32)
    ys = np.asarray(ys_l, dtype=np.int64)
    if len(ys) == 0:
        return {"metrics": {}, "best": {"f1": 0.0, "thresh": 0.5}, "n_val": 0}
    metrics = {}
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
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

    print("[2/4] building WMReplayClassifierDataset (train + val split)")
    all_pairs = _find_demo_pairs(cfg.wm_replay.raw_dir, cfg.wm_replay.hidden_dir)
    print(f"  total demo pairs: {len(all_pairs)}")
    val_tail = int(cfg.wm_replay.val_demos_tail)
    train_pairs = all_pairs[:-val_tail]
    val_pairs = all_pairs[-val_tail:]
    print(f"  train demos: {len(train_pairs)}  val demos: {len(val_pairs)}")

    def _build_ds(pairs, mode, seed):
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
            max_demos=None,
            seed=seed,
        )
        ds.pairs = pairs  # explicit split
        return ds

    tr_ds = _build_ds(train_pairs, mode="train", seed=42)
    va_ds = _build_ds(val_pairs, mode="val", seed=43)

    print("[3/4] imagine_all() — running chunk WM over train + val demos (GPU)")
    print(f"  imagining {len(tr_ds.pairs)} train demos...")
    tr_ds.imagine_all(verbose=True)
    print(f"  imagining {len(va_ds.pairs)} val demos...")
    va_ds.imagine_all(verbose=True)

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

    print("[4/4] training classifier on WM-replayed pos/neg")
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
            if out["best"]["f1"] > best_f1:
                best_f1 = float(out["best"]["f1"])
                torch.save(
                    {
                        "model": model.state_dict(),
                        "threshold": float(out["best"]["thresh"]),
                        "f1": best_f1,
                        "step": step,
                        "config": OmegaConf.to_container(cfg),
                    },
                    ckpt_dir / "best.ckpt",
                )
                print(
                    f"[best] step={step} f1={best_f1:.4f} thresh={out['best']['thresh']:.2f} → {ckpt_dir / 'best.ckpt'}"
                )
    print(f"done. best_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
