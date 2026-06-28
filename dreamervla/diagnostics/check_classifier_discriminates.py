"""Check that the outcome classifier can discriminate collected rollout latents.

This diagnostic uses the same classifier shape as the OpenVLA-OFT action-hidden
cotrain route, but streams a compact balanced window set from HDF5 instead of
materializing full episodes in OnlineReplay. Full episodes contain 56x4096
action-hidden tokens per env step; loading all 500 episodes into replay would
need far more host memory than this preflight check should consume.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from dreamervla.models.reward import build_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WindowExample:
    window: np.ndarray
    label: int
    episode_index: int


def _default_root() -> Path:
    return PROJECT_ROOT


def _default_data_root() -> Path:
    root = _default_root()
    return Path(__import__("os").environ.get("DVLA_DATA_ROOT", root / "data"))


def _success_and_finish(demo: h5py.Group) -> tuple[bool, int]:
    sparse = np.asarray(demo["sparse_rewards"][...], dtype=np.float32)
    hits = np.flatnonzero(sparse > 0.5)
    attr_success = bool(demo.attrs.get("episode_success", False))
    if hits.size:
        return True, int(hits[0]) + 1
    return attr_success, int(sparse.shape[0])


def _chunk_last_indices(end: int, *, window: int, chunk_size: int) -> np.ndarray:
    start = int(end) - int(window) * int(chunk_size)
    return np.arange(
        start + int(chunk_size) - 1,
        int(end),
        int(chunk_size),
        dtype=np.int64,
    )


def _load_window(
    hidden_demo: h5py.Group,
    *,
    end: int,
    window: int,
    chunk_size: int,
    token_count: int,
    token_dim: int,
) -> np.ndarray:
    indices = _chunk_last_indices(end, window=window, chunk_size=chunk_size)
    hidden = hidden_demo["obs_embedding"]
    frames = np.asarray(hidden[indices])
    if frames.ndim == 2:
        frames = frames.reshape(window, token_count, token_dim)
    elif frames.ndim != 3:
        raise ValueError(f"unexpected obs_embedding window shape: {frames.shape}")
    pooled = frames.astype(np.float32, copy=False).mean(axis=1)
    return np.asarray(pooled, dtype=np.float16)


def _iter_examples(
    reward_dir: Path,
    hidden_dir: Path,
    *,
    window: int,
    chunk_size: int,
    token_count: int,
    token_dim: int,
    early_neg_stride: int,
    max_episodes: int | None,
) -> list[WindowExample]:
    examples: list[WindowExample] = []
    episode_index = 0
    window_env = int(window) * int(chunk_size)
    for reward_path in sorted(reward_dir.glob("*.hdf5")):
        hidden_path = hidden_dir / reward_path.name
        if not hidden_path.exists():
            raise FileNotFoundError(f"missing hidden shard for {reward_path.name}: {hidden_path}")
        with h5py.File(reward_path, "r") as rf, h5py.File(hidden_path, "r") as hf:
            for demo_key in sorted(rf["data"].keys()):
                if max_episodes is not None and episode_index >= int(max_episodes):
                    return examples
                demo = rf["data"][demo_key]
                success, finish_step = _success_and_finish(demo)
                if finish_step < window_env:
                    episode_index += 1
                    continue
                hidden_demo = hf["data"][demo_key]
                examples.append(
                    WindowExample(
                        window=_load_window(
                            hidden_demo,
                            end=finish_step,
                            window=window,
                            chunk_size=chunk_size,
                            token_count=token_count,
                            token_dim=token_dim,
                        ),
                        label=int(success),
                        episode_index=episode_index,
                    )
                )

                early_end = finish_step - max(1, int(early_neg_stride))
                if early_end >= window_env:
                    examples.append(
                        WindowExample(
                            window=_load_window(
                                hidden_demo,
                                end=early_end,
                                window=window,
                                chunk_size=chunk_size,
                                token_count=token_count,
                                token_dim=token_dim,
                            ),
                            label=0,
                            episode_index=episode_index,
                        )
                    )
                episode_index += 1
    return examples


def _split_examples(
    examples: list[WindowExample], *, seed: int, val_fraction: float
) -> tuple[list[WindowExample], list[WindowExample]]:
    episode_ids = sorted({ex.episode_index for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(episode_ids)
    n_val = max(1, int(round(len(episode_ids) * float(val_fraction))))
    val_ids = set(episode_ids[:n_val])
    train = [ex for ex in examples if ex.episode_index not in val_ids]
    val = [ex for ex in examples if ex.episode_index in val_ids]
    return train, val


def _by_label(examples: list[WindowExample]) -> tuple[list[WindowExample], list[WindowExample]]:
    pos = [ex for ex in examples if int(ex.label) == 1]
    neg = [ex for ex in examples if int(ex.label) == 0]
    if not pos or not neg:
        raise RuntimeError(f"need both classes, got pos={len(pos)} neg={len(neg)}")
    return pos, neg


def _sample_balanced(
    pos: list[WindowExample],
    neg: list[WindowExample],
    *,
    batch_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = max(1, int(batch_size) // 2)
    picked = [rng.choice(pos) for _ in range(half)]
    picked.extend(rng.choice(neg) for _ in range(int(batch_size) - half))
    rng.shuffle(picked)
    windows = np.stack([ex.window for ex in picked], axis=0).astype(np.float32, copy=False)
    labels = np.asarray([ex.label for ex in picked], dtype=np.int64)
    return torch.from_numpy(windows), torch.from_numpy(labels)


@torch.no_grad()
def _eval_f1(
    classifier: torch.nn.Module,
    examples: list[WindowExample],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    classifier.eval()
    preds: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for start in range(0, len(examples), int(batch_size)):
        batch = examples[start : start + int(batch_size)]
        windows = torch.from_numpy(
            np.stack([ex.window for ex in batch], axis=0).astype(np.float32, copy=False)
        ).to(device)
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.long, device=device)
        logits = classifier(windows)
        preds.append(logits.argmax(dim=-1).detach().cpu())
        labels_all.append(labels.detach().cpu())
    pred = torch.cat(preds)
    label = torch.cat(labels_all)
    tp = ((pred == 1) & (label == 1)).sum().float()
    fp = ((pred == 1) & (label == 0)).sum().float()
    fn = ((pred == 0) & (label == 1)).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    acc = (pred == label).float().mean()
    return {
        "acc": float(acc.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(f1.item()),
        "pred_pos_frac": float((pred == 1).float().mean().item()),
    }


def _balanced_eval_subset(
    examples: list[WindowExample], *, seed: int
) -> list[WindowExample]:
    pos, neg = _by_label(examples)
    n = min(len(pos), len(neg))
    rng = random.Random(seed)
    subset = rng.sample(pos, n) + rng.sample(neg, n)
    rng.shuffle(subset)
    return subset


def _load_preprocess_config(hidden_dir: Path) -> dict:
    path = hidden_dir / "preprocess_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    data_root = _default_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward-dir",
        type=Path,
        default=data_root / "collected_rollouts/libero_goal/reward",
    )
    parser.add_argument(
        "--hidden-dir",
        type=Path,
        default=data_root / "collected_rollouts/libero_goal/hidden",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--early-neg-stride", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--min-f1", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    meta = _load_preprocess_config(args.hidden_dir)
    chunk_size = int(meta.get("chunk_size", meta.get("time_horizon", 8)))
    token_count = int(meta.get("token_count", 56))
    token_dim = int(meta.get("token_dim", 4096))
    device = torch.device(args.device)

    examples = _iter_examples(
        args.reward_dir,
        args.hidden_dir,
        window=args.window,
        chunk_size=chunk_size,
        token_count=token_count,
        token_dim=token_dim,
        early_neg_stride=args.early_neg_stride,
        max_episodes=args.max_episodes,
    )
    train_examples, val_examples = _split_examples(
        examples, seed=args.seed, val_fraction=0.2
    )
    train_pos, train_neg = _by_label(train_examples)
    val_balanced = _balanced_eval_subset(val_examples, seed=args.seed + 1)
    print(
        "[classifier-check] "
        f"examples={len(examples)} train_pos={len(train_pos)} train_neg={len(train_neg)} "
        f"val_balanced={len(val_balanced)} chunk_size={chunk_size} "
        f"token_count={token_count} token_dim={token_dim} device={device}",
        flush=True,
    )

    classifier = build_classifier(
        {
            "_target_": "dreamervla.models.reward.LatentSuccessClassifier",
            "latent_dim": token_dim,
            "action_dim": 7,
            "time_horizon": chunk_size,
            "token_dim": token_dim,
            "window": args.window,
            "head_type": "transformer",
            "hidden_dim": 1024,
            "num_layers": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "granularity": "chunk",
            "chunk_size": chunk_size,
            "chunk_pool": "last",
            "token_pool": "mean",
            "token_count": token_count,
        }
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=1.0e-4, betas=(0.9, 0.999), weight_decay=1.0e-4
    )
    rng = random.Random(args.seed)

    last = {}
    for step in range(1, int(args.steps) + 1):
        windows, labels = _sample_balanced(
            train_pos, train_neg, batch_size=args.batch_size, rng=rng
        )
        windows = windows.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        classifier.train()
        logits = classifier(windows)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == int(args.steps):
            last = _eval_f1(
                classifier,
                val_balanced,
                device=device,
                batch_size=args.eval_batch_size,
            )
            print(
                "[classifier-check] "
                f"step={step}/{args.steps} loss={float(loss.detach().cpu()):.4f} "
                f"val_acc={last['acc']:.3f} val_f1={last['f1']:.3f} "
                f"precision={last['precision']:.3f} recall={last['recall']:.3f} "
                f"pred_pos_frac={last['pred_pos_frac']:.3f}",
                flush=True,
            )

    f1 = float(last.get("f1", 0.0))
    if f1 < float(args.min_f1):
        raise SystemExit(
            f"classifier f1 {f1:.3f} < required {float(args.min_f1):.3f}"
        )


if __name__ == "__main__":
    main()
