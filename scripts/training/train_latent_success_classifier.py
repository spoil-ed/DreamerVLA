# ruff: noqa: E402
"""Train LatentSuccessClassifier on libero_goal classifier shards.

Single-GPU by default (Phase 3 v1 data fits in one GPU easily — 299
episodes total). Multi-GPU DDP support deferred.

Usage:
    MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=4 \
        /home/user01/miniconda3/envs/dreamervla/bin/python \
        scripts/train_latent_success_classifier.py \
        --config configs/wmpo_classifier_libero_goal.yaml
"""

from __future__ import annotations

import argparse
import glob
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

from dreamer_vla.dataset.libero_sim_rollout_shards import LatentSuccessShardDataset
from dreamer_vla.models.reward import LatentSuccessClassifier, LatentSuccessClassifierConfig


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    train_shards = sorted(glob.glob(cfg.data.train_glob))
    val_shards = sorted(glob.glob(cfg.data.val_glob))
    print(f"train_shards={len(train_shards)} val_shards={len(val_shards)}")
    if not train_shards or not val_shards:
        raise SystemExit(
            f"shards missing: train glob={cfg.data.train_glob} val glob={cfg.data.val_glob}"
        )

    device = torch.device(args.device)
    tr_ds = LatentSuccessShardDataset(
        train_shards,
        window=cfg.classifier.window,
        stride=cfg.data.stride_train,
        mode="train",
        use_resample=True,
    )
    va_ds = LatentSuccessShardDataset(
        val_shards,
        window=cfg.classifier.window,
        stride=cfg.data.stride_val,
        mode="val",
        use_resample=False,
    )
    tr_ld = DataLoader(
        tr_ds,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.data.num_workers),
        collate_fn=_collate,
        pin_memory=True,
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.data.num_workers),
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
