"""Shared success-classifier update for standalone warmup and Ray cotrain."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from dreamervla.runners.success_classifier_training_runner import (
    _classifier_loss_and_predictions,
    _success_probabilities_from_logits,
)
from dreamervla.runtime.distributed import unwrap_module


class ClassifierReplay(Protocol):
    """Replay capability required by the classifier update."""

    def sample_classifier_windows(
        self,
        batch_size: int,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def online_classifier_update_step(
    *,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ClassifierReplay,
    device: torch.device,
    batch_size: int,
    early_neg_stride: int,
    grad_clip: float,
    loss_type: str | None = None,
    label_smoothing: float = 0.0,
    sampling_protocol: str = "lumos",
    balance_batches: bool = False,
) -> dict[str, Any]:
    """Run one classifier optimizer step from canonical replay windows."""

    module = unwrap_module(classifier)
    cfg = module.cfg
    cls_batch = replay.sample_classifier_windows(
        int(batch_size),
        window=int(cfg.window),
        chunk_size=int(getattr(cfg, "chunk_size", 1)),
        chunk_pool=str(getattr(cfg, "chunk_pool", "last")),
        early_neg_stride=int(early_neg_stride),
        sampling_protocol=str(sampling_protocol),
        balance_batches=bool(balance_batches),
    )
    windows = cls_batch["windows"].to(device, non_blocking=True)
    labels = cls_batch["labels"].to(device, non_blocking=True)
    task_ids = cls_batch.get("task_ids")
    pos_frac = labels.float().mean()
    forward_kwargs: dict[str, Any] = {}
    if bool(getattr(module, "supports_proprio_conditioning", False)):
        proprio = cls_batch.get("proprio")
        if not isinstance(proprio, torch.Tensor):
            raise ValueError(
                "classifier requires proprio conditioning, but replay "
                "sample_classifier_windows did not return proprio"
            )
        forward_kwargs["proprio"] = proprio.to(device, non_blocking=True)
    if bool(getattr(module, "supports_language_conditioning", False)):
        lang_emb = cls_batch.get("lang_emb")
        if not isinstance(lang_emb, torch.Tensor):
            raise ValueError(
                "classifier requires language conditioning, but replay "
                "sample_classifier_windows did not return lang_emb"
            )
        forward_kwargs["lang_emb"] = lang_emb.to(device, non_blocking=True)

    classifier.train()
    if bool(getattr(module, "supports_task_conditioning", False)) and isinstance(
        task_ids, torch.Tensor
    ):
        logits = classifier(
            windows,
            task_ids=task_ids.to(device, non_blocking=True),
            **forward_kwargs,
        )
    else:
        logits = classifier(windows, **forward_kwargs)

    resolved_loss_type = None if loss_type is None else str(loss_type).lower()
    if resolved_loss_type in {None, "", "auto"}:
        resolved_loss_type = "bce" if int(logits.shape[-1]) == 1 else "ce"
    loss, preds = _classifier_loss_and_predictions(
        logits,
        labels,
        loss_type=str(resolved_loss_type),
        label_smoothing=float(label_smoothing),
        class_weight=None,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        classifier.parameters(), max_norm=float(grad_clip)
    )
    optimizer.step()
    classifier.eval()
    module.eval()

    with torch.no_grad():
        probs = _success_probabilities_from_logits(logits.detach())
        preds = preds.detach()
        acc = (preds == labels).float().mean()
        pred_pos = preds == 1
        true_pos_label = labels == 1
        tp = (pred_pos & true_pos_label).sum().float()
        fp = (pred_pos & (~true_pos_label)).sum().float()
        fn = ((~pred_pos) & true_pos_label).sum().float()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = (2.0 * precision * recall) / (precision + recall).clamp_min(1.0e-12)

    return {
        "loss": float(loss.detach().cpu().item()),
        "acc": float(acc.detach().cpu().item()),
        "precision": float(precision.detach().cpu().item()),
        "recall": float(recall.detach().cpu().item()),
        "f1": float(f1.detach().cpu().item()),
        "tp": int(tp.detach().cpu().item()),
        "fp": int(fp.detach().cpu().item()),
        "fn": int(fn.detach().cpu().item()),
        "pos_frac": float(pos_frac.detach().cpu().item()),
        "prob_mean": float(probs.float().mean().detach().cpu().item()),
        "grad_norm": float(
            grad_norm.detach().cpu().item()
            if isinstance(grad_norm, torch.Tensor)
            else grad_norm
        ),
        "updated": 1.0,
        "skipped_single_class_batch": 0.0,
        "loss_type": str(resolved_loss_type),
        "sampling_protocol": str(sampling_protocol),
        "balance_batches": bool(balance_batches),
        "batch": _json_safe(
            {key: value for key, value in cls_batch.items() if key != "windows"}
        ),
    }


__all__ = ["ClassifierReplay", "online_classifier_update_step"]
