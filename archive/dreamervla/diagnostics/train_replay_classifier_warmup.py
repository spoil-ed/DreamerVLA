"""Train the replay-window success classifier without building a world model.

This utility is intentionally narrow: it reuses the online replay seeding path and
``online_classifier_update_step`` so the classifier sees the same
``[B, W, token_count, token_dim]`` token grids, proprio windows, and language
sidecars as cotrain, while avoiding the WM build/train path entirely.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.runners.base_runner import _atomic_torch_save
from dreamervla.runners.offline_seed import seed_replay_from_offline
from dreamervla.runners.online_dreamervla import _unwrap, online_classifier_update_step
from dreamervla.runners.online_replay import OnlineReplay
from dreamervla.utils.checkpoint_util import TopKCheckpointManager

_PROGRESS_RE = re.compile(r"^classifier_step_(?P<step>\d+)\.ckpt$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_cfg(args: argparse.Namespace):
    overrides = [
        f"experiment={args.experiment}",
        f"task={args.task}",
        f"offline_warmup.data_dir={args.data_dir}",
        f"offline_warmup.hidden_dir={args.hidden_dir}",
        "offline_warmup.task_id=null",
        "env.task_ids=[0,1,2,3,4,5,6,7,8,9]",
        f"training.out_dir={args.out_dir}",
    ]
    overrides.extend(args.override or [])
    with initialize_config_dir(config_dir=str(_repo_root() / "configs"), version_base=None):
        return compose(config_name="train", overrides=overrides)


def _make_optimizer(classifier: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    opt = OmegaConf.select(cfg, "optim.classifier", default=None)
    name = str(OmegaConf.select(opt, "name", default="adamw")).lower() if opt else "adamw"
    lr = float(OmegaConf.select(opt, "lr", default=1.0e-4) if opt else 1.0e-4)
    betas = tuple(OmegaConf.select(opt, "betas", default=[0.9, 0.999]) if opt else [0.9, 0.999])
    eps = float(OmegaConf.select(opt, "eps", default=1.0e-8) if opt else 1.0e-8)
    weight_decay = float(
        OmegaConf.select(opt, "weight_decay", default=1.0e-4) if opt else 1.0e-4
    )
    if name == "adam":
        return torch.optim.Adam(classifier.parameters(), lr=lr, betas=betas, eps=eps)
    if name == "adamw":
        return torch.optim.AdamW(
            classifier.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
    raise ValueError(f"unsupported classifier optimizer: {name}")


def _progress_dir(out_dir: Path) -> Path:
    return out_dir / "ckpt" / "classifier_progress"


def _latest_progress_path(out_dir: Path) -> Path | None:
    latest_step = -1
    latest_path: Path | None = None
    progress = _progress_dir(out_dir)
    if not progress.is_dir():
        return None
    for path in progress.glob("classifier_step_*.ckpt"):
        match = _PROGRESS_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        if step > latest_step:
            latest_step = step
            latest_path = path
    return latest_path


def _save_checkpoint(
    *,
    out_dir: Path,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, float],
    cfg: Any,
    topk: TopKCheckpointManager | None,
    final: bool = False,
) -> None:
    payload = {
        "global_step": int(step),
        "classifier": _unwrap(classifier).state_dict(),
        "classifier_optimizer": optimizer.state_dict(),
        "classifier_threshold": 0.5,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "complete": bool(final),
    }
    if final:
        _atomic_torch_save(payload, out_dir / "ckpt" / "classifier_warmup.ckpt")
        return
    progress_path = _progress_dir(out_dir) / f"classifier_step_{int(step):08d}.ckpt"
    _atomic_torch_save(payload, progress_path)
    if topk is not None:
        data = {"step": int(step), **{key: float(value) for key, value in metrics.items()}}
        topk_path = topk.get_ckpt_path(data)
        if topk_path is not None:
            _atomic_torch_save(payload, Path(topk_path))


def _load_progress(
    *,
    out_dir: Path,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    path = _latest_progress_path(out_dir)
    if path is None:
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _unwrap(classifier).load_state_dict(payload["classifier"])
    if "classifier_optimizer" in payload:
        optimizer.load_state_dict(payload["classifier_optimizer"])
    step = int(payload.get("global_step", payload.get("step", 0)))
    print(f"[cls-only] resumed progress step={step} from {path}", flush=True)
    return step


@torch.no_grad()
def _evaluate(
    *,
    classifier: torch.nn.Module,
    replay: OnlineReplay,
    device: torch.device,
    batch_size: int,
    batches: int,
    early_neg_stride: int,
) -> dict[str, float]:
    module = _unwrap(classifier)
    cfg = module.cfg
    classifier.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    tp = fp = fn = 0
    pos = 0
    for _ in range(max(1, int(batches))):
        batch = replay.sample_classifier_windows(
            int(batch_size),
            window=int(cfg.window),
            chunk_size=int(getattr(cfg, "chunk_size", 1)),
            chunk_pool=str(getattr(cfg, "chunk_pool", "last")),
            early_neg_stride=int(early_neg_stride),
        )
        windows = batch["windows"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        kwargs: dict[str, Any] = {}
        if bool(getattr(module, "supports_proprio_conditioning", False)):
            kwargs["proprio"] = batch["proprio"].to(device, non_blocking=True)
        if bool(getattr(module, "supports_language_conditioning", False)):
            kwargs["lang_emb"] = batch["lang_emb"].to(device, non_blocking=True)
        logits = classifier(windows, **kwargs)
        losses.append(float(torch.nn.functional.cross_entropy(logits, labels).item()))
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        pred_pos = preds == 1
        true_pos = labels == 1
        tp += int((pred_pos & true_pos).sum().item())
        fp += int((pred_pos & ~true_pos).sum().item())
        fn += int((~pred_pos & true_pos).sum().item())
        pos += int(true_pos.sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "eval_loss": float(np.mean(losses)),
        "eval_acc": float(correct / max(total, 1)),
        "eval_f1": float(f1),
        "eval_pos_frac": float(pos / max(total, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="openvla_onetraj_libero_cotrain_noray")
    parser.add_argument("--task", default="openvla_onetraj_coldstart_libero")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--hidden-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buffer-size", type=int, default=160000)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--eval-first-step", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--topk-k", type=int, default=5)
    parser.add_argument("--early-neg-stride", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg(args)
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml")

    device = torch.device(args.device)
    replay = OnlineReplay(
        capacity=int(args.buffer_size),
        sequence_length=int(args.sequence_length),
        task_ids=tuple(range(10)),
        capacity_mode="total_sharded",
        rank=0,
    )
    n = seed_replay_from_offline(
        replay,
        data_dir=args.data_dir,
        hidden_dir=args.hidden_dir,
        default_task_id=None,
        max_episodes_per_task=None,
    )
    print(
        "[cls-only] replay loaded "
        f"episodes={n} transitions={replay.num_transitions} "
        f"classifier_windows={replay.classifier_window_count(window=8, chunk_size=8)}",
        flush=True,
    )

    classifier = hydra.utils.instantiate(cfg.classifier).to(device)
    optimizer = _make_optimizer(classifier, cfg)
    start_step = (
        _load_progress(out_dir=out_dir, classifier=classifier, optimizer=optimizer)
        if bool(args.resume)
        else 0
    )
    topk = (
        TopKCheckpointManager(
            save_dir=str(out_dir / "ckpt" / "classifier_topk"),
            monitor_key="eval_f1",
            mode="max",
            k=int(args.topk_k),
            format_str="classifier_step={step:08d}-f1={eval_f1:.6f}.ckpt",
        )
        if int(args.topk_k) > 0
        else None
    )

    log_path = out_dir / "classifier_train_log.jsonl"
    t0 = time.perf_counter()
    last_metrics: dict[str, float] = {}
    for step in range(int(start_step), int(args.steps)):
        train_metrics = online_classifier_update_step(
            classifier=classifier,
            optimizer=optimizer,
            replay=replay,
            device=device,
            batch_size=int(args.batch_size),
            early_neg_stride=int(args.early_neg_stride),
            grad_clip=float(args.grad_clip),
        )
        train = {
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "train_f1": float(train_metrics.get("f1", 0.0)),
            "train_pos_frac": float(train_metrics.get("pos_frac", 0.0)),
            "grad_norm": float(train_metrics.get("grad_norm", 0.0)),
        }
        current_step = step + 1
        do_eval = (
            bool(args.eval_first_step) and current_step == 1
        ) or current_step % int(args.checkpoint_every) == 0
        if do_eval:
            eval_metrics = _evaluate(
                classifier=classifier,
                replay=replay,
                device=device,
                batch_size=int(args.batch_size),
                batches=int(args.eval_batches),
                early_neg_stride=int(args.early_neg_stride),
            )
            last_metrics = {**train, **eval_metrics}
            _save_checkpoint(
                out_dir=out_dir,
                classifier=classifier,
                optimizer=optimizer,
                step=current_step,
                metrics=last_metrics,
                cfg=cfg,
                topk=topk,
            )
        elif current_step % int(args.log_every) == 0:
            last_metrics = dict(train)
        else:
            continue

        record = {
            "step": int(current_step),
            "elapsed_s": float(time.perf_counter() - t0),
            **last_metrics,
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            "[cls-only] "
            f"step={current_step}/{args.steps} "
            f"loss={train['train_loss']:.4f} acc={train['train_acc']:.3f} "
            f"f1={train['train_f1']:.3f} pos={train['train_pos_frac']:.3f} "
            f"eval_f1={last_metrics.get('eval_f1', float('nan')):.3f} "
            f"eval_acc={last_metrics.get('eval_acc', float('nan')):.3f}",
            flush=True,
        )

    final_metrics = _evaluate(
        classifier=classifier,
        replay=replay,
        device=device,
        batch_size=int(args.batch_size),
        batches=max(int(args.eval_batches), 50),
        early_neg_stride=int(args.early_neg_stride),
    )
    _save_checkpoint(
        out_dir=out_dir,
        classifier=classifier,
        optimizer=optimizer,
        step=int(args.steps),
        metrics=final_metrics,
        cfg=cfg,
        topk=None,
        final=True,
    )
    print(f"[cls-only] complete final={final_metrics}", flush=True)


if __name__ == "__main__":
    main()
