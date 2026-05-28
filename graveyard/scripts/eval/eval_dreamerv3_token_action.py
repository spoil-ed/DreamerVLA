from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_config(path_or_name: str) -> Path:
    path = Path(path_or_name).expanduser()
    if path.is_absolute():
        return path
    if path.suffix != ".yaml":
        path = path.with_suffix(".yaml")
    return PROJECT_ROOT / "configs" / path


def _resolve_path(path: str | None, default: Path | None = None) -> Path:
    if path is None:
        if default is None:
            raise ValueError("path is required")
        return default
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


def _load_model(cfg: Any, ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = hydra.utils.instantiate(cfg.world_model)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in {ckpt_path}")
    state = _strip_module_prefix(state)
    load_info: dict[str, Any] = {"strict": True, "missing_keys": [], "unexpected_keys": []}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        result = model.load_state_dict(state, strict=False)
        load_info = {
            "strict": False,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }
    model.to(device)
    model.eval()
    payload_dict = payload if isinstance(payload, dict) else {}
    payload_dict["_load_info"] = load_info
    return model, payload_dict


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _variant_actions(actions_next: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "real":
        return actions_next
    if variant == "zero":
        return torch.zeros_like(actions_next)
    if variant == "shuffled":
        flat = actions_next.reshape(-1, actions_next.shape[-1])
        perm = torch.randperm(flat.shape[0], device=flat.device)
        return flat[perm].reshape_as(actions_next)
    if variant == "random":
        flat = actions_next.reshape(-1, actions_next.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
        low = flat.amin(dim=0)
        high = flat.amax(dim=0)
        sample = mean + std * torch.randn_like(actions_next)
        return torch.minimum(torch.maximum(sample, low), high)
    raise ValueError(f"Unknown action variant: {variant}")


def _empty_stats() -> dict[str, float]:
    return {
        "ce_sum": 0.0,
        "correct": 0.0,
        "token_count": 0.0,
        "kl_sum": 0.0,
        "step_count": 0.0,
        "prior_entropy_sum": 0.0,
        "prob_l2_to_real_sum": 0.0,
        "deter_l2_to_real_sum": 0.0,
        "prior_argmax_match_real_sum": 0.0,
        "action_l2_to_real_sum": 0.0,
    }


def _finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    token_count = max(stats["token_count"], 1.0)
    step_count = max(stats["step_count"], 1.0)
    return {
        "token_ce": stats["ce_sum"] / token_count,
        "token_rec_sum_per_step": stats["ce_sum"] / step_count,
        "token_acc": stats["correct"] / token_count,
        "kl_post_prior_per_step": stats["kl_sum"] / step_count,
        "prior_entropy_per_step": stats["prior_entropy_sum"] / step_count,
        "prob_l2_to_real": stats["prob_l2_to_real_sum"] / step_count,
        "deter_l2_to_real": stats["deter_l2_to_real_sum"] / step_count,
        "prior_argmax_match_real": stats["prior_argmax_match_real_sum"] / step_count,
        "action_l2_to_real": stats["action_l2_to_real_sum"] / step_count,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    cfg_path = _resolve_config(args.config_name)
    cfg = OmegaConf.load(cfg_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = hydra.utils.instantiate(cfg.dataset)
    batch_size = int(args.batch_size or OmegaConf.select(cfg, "dataloader.batch_size", default=4))
    num_workers = int(args.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(torch.cuda.is_available()),
        drop_last=True,
        persistent_workers=bool(num_workers > 0),
        collate_fn=getattr(dataset, "collate_fn", None),
    )

    ckpt_path = _resolve_path(args.ckpt)
    model, payload = _load_model(cfg, ckpt_path, device)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if "real" not in variants:
        variants.insert(0, "real")
    stats = {variant: _empty_stats() for variant in variants}
    autocast_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    batches = 0
    for batch in loader:
        if batches >= args.num_batches:
            break
        batch = _move_batch(batch, device)
        tokens = batch["tokens"].long()
        actions = batch["actions"]
        is_first = batch["is_first"]
        bsz, seq_len = tokens.shape[:2]
        if seq_len < 2:
            continue

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            enc = model.encoder(tokens)
            seq = model.rssm.observe(enc, actions.to(dtype=enc.dtype), is_first)
            prev_deter = seq["deter"][:, :-1].reshape(-1, model.rssm.deter)
            prev_stoch = seq["stoch"][:, :-1].reshape(-1, model.rssm.stoch, model.rssm.classes)
            target_post_logits = seq["post_logits"][:, 1:].reshape(
                -1, model.rssm.stoch, model.rssm.classes
            )
            target_tokens = tokens[:, 1:]
            actions_next = actions[:, 1:].to(device=device, dtype=enc.dtype)
            flat_actions_real = actions_next.reshape(-1, actions_next.shape[-1])

            real_action = _variant_actions(actions_next, "real").reshape(-1, actions_next.shape[-1])
            real_deter = model.rssm._core(prev_deter, prev_stoch, real_action)
            real_prior_logits = model.rssm._prior(real_deter)
            real_prior_probs = model.rssm._probs(real_prior_logits).float()
            real_prior_argmax = real_prior_logits.argmax(dim=-1)

            for variant in variants:
                variant_next = _variant_actions(actions_next, variant)
                flat_action = variant_next.reshape(-1, variant_next.shape[-1])
                pred_deter = model.rssm._core(prev_deter, prev_stoch, flat_action)
                prior_logits = model.rssm._prior(pred_deter)
                prior_stoch = model.rssm._probs(prior_logits)
                logits = model.decoder(
                    pred_deter.reshape(bsz, seq_len - 1, -1),
                    prior_stoch.reshape(bsz, seq_len - 1, model.rssm.stoch, model.rssm.classes),
                )
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    target_tokens.reshape(-1).to(device=logits.device),
                    reduction="none",
                )
                pred = logits.argmax(dim=-1)
                correct = (pred == target_tokens.to(device=pred.device)).float().sum()
                kl = model.rssm._kl(target_post_logits, prior_logits)
                entropy = model.rssm._entropy(prior_logits)
                prior_probs = model.rssm._probs(prior_logits).float()
                prob_l2 = (prior_probs - real_prior_probs).pow(2).sum(dim=(-1, -2)).sqrt()
                deter_l2 = (pred_deter.float() - real_deter.float()).pow(2).mean(dim=-1).sqrt()
                argmatch = (prior_logits.argmax(dim=-1) == real_prior_argmax).float().mean(dim=-1)
                action_l2 = (flat_action.float() - flat_actions_real.float()).pow(2).mean(dim=-1).sqrt()

                cur = stats[variant]
                cur["ce_sum"] += float(ce.sum().detach().cpu())
                cur["correct"] += float(correct.detach().cpu())
                cur["token_count"] += float(target_tokens.numel())
                cur["kl_sum"] += float(kl.float().sum().detach().cpu())
                cur["step_count"] += float(kl.numel())
                cur["prior_entropy_sum"] += float(entropy.float().sum().detach().cpu())
                cur["prob_l2_to_real_sum"] += float(prob_l2.sum().detach().cpu())
                cur["deter_l2_to_real_sum"] += float(deter_l2.sum().detach().cpu())
                cur["prior_argmax_match_real_sum"] += float(argmatch.sum().detach().cpu())
                cur["action_l2_to_real_sum"] += float(action_l2.sum().detach().cpu())
        batches += 1

    metrics = {variant: _finalize_stats(value) for variant, value in stats.items()}
    real = metrics["real"]
    margins: dict[str, dict[str, float]] = {}
    for variant, value in metrics.items():
        if variant == "real":
            continue
        margins[f"real_vs_{variant}"] = {
            "ce_margin_variant_minus_real": value["token_ce"] - real["token_ce"],
            "rec_sum_margin_variant_minus_real": value["token_rec_sum_per_step"]
            - real["token_rec_sum_per_step"],
            "acc_margin_real_minus_variant": real["token_acc"] - value["token_acc"],
            "kl_margin_variant_minus_real": value["kl_post_prior_per_step"]
            - real["kl_post_prior_per_step"],
        }

    return {
        "config": str(cfg_path),
        "checkpoint": str(ckpt_path),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "load_info": payload.get("_load_info", {}),
        "num_batches": batches,
        "batch_size": batch_size,
        "sequence_length": int(getattr(dataset.data_spec, "sequence_length", -1)),
        "num_transition_steps": int(next(iter(stats.values()))["step_count"]),
        "variants": variants,
        "metrics": metrics,
        "margins": margins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DreamerV3-token action sensitivity.")
    parser.add_argument("--config-name", default="dreamerv3_token_libero_10")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--variants", default="real,zero,shuffled,random")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    default_out = (
        PROJECT_ROOT
        / "data/outputs/eval/eval_wm"
        / f"dreamerv3_token_action_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = _resolve_path(args.out_dir, default_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(args)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[eval] wrote {metrics_path}")


if __name__ == "__main__":
    main()
