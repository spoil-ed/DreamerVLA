"""Evaluate a success-classifier checkpoint on an external trajectory set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from dreamervla.runners.success_classifier_training_runner import (
    _success_probabilities_from_logits,
    _sweep_metrics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--hidden-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    return parser


def _pool_chunks(values: np.ndarray, *, chunk_size: int, pool: str) -> np.ndarray:
    count = int(values.shape[0]) // int(chunk_size)
    reshaped = values[: count * chunk_size].reshape(
        count,
        chunk_size,
        *values.shape[1:],
    )
    if pool == "last":
        return reshaped[:, -1]
    if pool == "first":
        return reshaped[:, 0]
    if pool == "mean":
        return reshaped.mean(axis=1)
    raise ValueError(f"unsupported chunk pool: {pool!r}")


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one checkpoint with trajectory-level max-window aggregation."""

    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    cfg = payload["cfg"]
    model = hydra.utils.instantiate(cfg.classifier)
    model.load_state_dict(payload["state_dicts"]["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    data_cfg = cfg.data
    dataset = hydra.utils.instantiate(
        cfg.task.classifier.dataset.validation,
        success_dir_raw=str(args.raw_dir),
        success_dir_hidden=str(args.hidden_dir),
        failure_dir_raw=None,
        failure_dir_hidden=None,
        window=int(data_cfg.window),
        stride=1,
        chunk_subsample=int(data_cfg.chunk_subsample),
        chunk_pool=str(data_cfg.chunk_pool),
        proprio_keys=OmegaConf.select(data_cfg, "proprio_keys", default=None),
        lang_emb_dir="__source_hidden__",
        lang_emb_key=str(OmegaConf.select(data_cfg, "lang_emb_key", default="lang_emb")),
        sampling_protocol=str(OmegaConf.select(data_cfg, "sampling_protocol", default="wmpo")),
        demo_split="all",
        val_fraction=0.2,
        split_seed=0,
        stratify_by_complete=False,
        distributed_rank=0,
        distributed_world_size=1,
        verbose=True,
    )

    window = int(data_cfg.window)
    chunk_size = int(data_cfg.chunk_subsample)
    chunk_pool = str(data_cfg.chunk_pool)
    batch_size = max(1, int(args.batch_size))
    scores: list[float] = []
    labels: list[int] = []
    identities: list[str] = []

    for trajectory in dataset.trajectories():
        obs, complete, finish_step, identity, extra = trajectory
        obs = _pool_chunks(
            obs[: int(finish_step)],
            chunk_size=chunk_size,
            pool=chunk_pool,
        )
        proprio = extra.get("proprio")
        if isinstance(proprio, np.ndarray):
            proprio = _pool_chunks(
                proprio[: int(finish_step)],
                chunk_size=chunk_size,
                pool=chunk_pool,
            )
        lang_emb = extra.get("lang_emb")
        maximum = 0.0
        for start in range(0, max(0, int(obs.shape[0]) - window + 1), batch_size):
            ends = range(start, min(start + batch_size, int(obs.shape[0]) - window + 1))
            windows = np.stack([obs[index : index + window] for index in ends])
            kwargs: dict[str, torch.Tensor] = {}
            if isinstance(proprio, np.ndarray):
                kwargs["proprio"] = (
                    torch.from_numpy(np.stack([proprio[index : index + window] for index in ends]))
                    .float()
                    .to(device)
                )
            if isinstance(lang_emb, np.ndarray):
                kwargs["lang_emb"] = (
                    torch.from_numpy(np.stack([lang_emb for _ in ends])).float().to(device)
                )
            logits = model(torch.from_numpy(windows).float().to(device), **kwargs)
            probabilities = _success_probabilities_from_logits(logits)
            maximum = max(maximum, float(probabilities.max().item()))
        scores.append(maximum)
        labels.append(int(bool(complete)))
        identities.append(str(identity))

    step = float(args.threshold_step)
    if not 0.0 < step <= 1.0:
        raise ValueError("--threshold-step must be within (0, 1]")
    thresholds = np.arange(step, 1.0, step, dtype=np.float64)
    metrics = _sweep_metrics(
        np.asarray(scores),
        np.asarray(labels),
        thresholds,
        tag="external_episode",
        selection_metric="macro_f1",
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_threshold": float(payload.get("classifier_threshold", 0.5)),
        "metrics": metrics,
        "score_summary": {
            "mean": float(np.mean(scores)),
            "p10": float(np.percentile(scores, 10)),
            "p50": float(np.percentile(scores, 50)),
            "p90": float(np.percentile(scores, 90)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },
        "trajectories": [
            {"identity": identity, "label": label, "score": score}
            for identity, label, score in zip(identities, labels, scores, strict=True)
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """CLI entry point."""

    args = _parser().parse_args()
    result = evaluate(args)
    print(json.dumps(result["metrics"], indent=2), flush=True)
    print(json.dumps(result["score_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
