"""Action sensitivity diagnostic for pretokenized WM checkpoints.

This is a localization experiment, not a model change.  It probes whether the
single-step prior is sensitive to action variants:

  real / scaled / zero / shuffled / random / previous-action / next-action

and reports both absolute latent distance and delta-latent distance.  For token
mode it also reports static/dynamic token reconstruction under each action.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.dataloader.pretokenize_dataset import PretokenizeDataset
from src.cli.eval_wm import (
    PROJECT_ROOT,
    extract_image_blocks,
    load_wm_state_dict,
    _strip_fsdp_prefix,
)


def build_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    configs_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _f(x: torch.Tensor | float) -> float:
    return float(x.float().mean().detach().cpu()) if isinstance(x, torch.Tensor) else float(x)


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x.float().norm(dim=-1)


def _parse_horizons(text: str) -> list[int]:
    horizons: list[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 1:
            raise ValueError(f"horizon must be >= 1, got {value}")
        horizons.append(value)
    return sorted(set(horizons))


def _parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    parts = str(value).replace(";", ",").split(",")
    out = [int(part.strip()) for part in parts if part.strip()]
    if not out:
        raise ValueError(f"expected at least one integer, got {value!r}")
    return out


def _infer_state_token_bounds(encoder: Any) -> tuple[int, int]:
    config = getattr(getattr(encoder, "backbone", None), "config", None)
    vocab_map = getattr(config, "vocabulary_map", None)
    if not isinstance(vocab_map, dict):
        raise ValueError("encoder config does not expose vocabulary_map for state-token ids")
    start_id = int(vocab_map["<reserved15500>"])
    end_id = int(vocab_map["<reserved16000>"])
    if end_id < start_id:
        raise ValueError(f"invalid state token range: {start_id}..{end_id}")
    return start_id, end_id


def _extract_state_bpe_ids(
    input_ids_list: list[list[int]],
    state_start_id: int,
    state_end_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    max_len = 0
    for seq in input_ids_list:
        tokens = [int(tok) for tok in seq]
        state_tokens: list[int] = []
        idx = 0
        while idx < len(tokens):
            if tokens[idx] != state_start_id:
                idx += 1
                continue
            end = idx + 1
            while end < len(tokens) and tokens[end] != state_end_id:
                end += 1
            if end < len(tokens):
                state_tokens.extend(tokens[idx : end + 1])
                idx = end + 1
            else:
                state_tokens.extend(tokens[idx:])
                break
        rows.append(state_tokens)
        max_len = max(max_len, len(state_tokens))

    ids = torch.zeros((len(rows), max_len), dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for row_idx, row in enumerate(rows):
        if not row:
            continue
        row_t = torch.tensor(row, dtype=torch.long)
        ids[row_idx, : row_t.numel()] = row_t
        mask[row_idx, : row_t.numel()] = True
    return ids, mask


def _load_neighbor_action(path: str, offset: int, action_dim: int) -> torch.Tensor | None:
    """Load action_{i+offset}.npy beside action_i.npy when available."""
    p = Path(path)
    m = re.match(r"action_(\d+)\.npy$", p.name)
    if not m:
        return None
    idx = int(m.group(1)) + int(offset)
    if idx < 0:
        return None
    npy = p.with_name(f"action_{idx}.npy")
    if not npy.is_file():
        return None
    arr = np.asarray(np.load(npy), dtype=np.float32).reshape(-1)
    if arr.shape[0] != action_dim:
        return None
    return torch.tensor(arr, dtype=torch.float32)


def _summarize_token_recon(
    logits: torch.Tensor,
    next_idx: torch.Tensor,
    cur_idx: torch.Tensor,
) -> dict[str, float]:
    # logits: [B, K, N_img, V]; idx: [B, K, N_img] or [B, N_img]
    if next_idx.ndim == 2:
        next_idx = next_idx.unsqueeze(1)
    if cur_idx.ndim == 2:
        cur_idx = cur_idx.unsqueeze(1)
    logits2 = logits
    ce = F.cross_entropy(
        logits2.reshape(-1, logits2.shape[-1]),
        next_idx.reshape(-1),
        reduction="none",
    ).view_as(next_idx)
    pred = logits2.argmax(dim=-1)
    correct = (pred == next_idx).float()
    dynamic = next_idx != cur_idx
    static = ~dynamic
    d_sum = dynamic.float().sum().clamp_min(1.0)
    s_sum = static.float().sum().clamp_min(1.0)
    return {
        "token_ce": _f(ce.mean()),
        "token_acc": _f(correct.mean()),
        "static_ce": _f((ce * static.float()).sum() / s_sum),
        "dynamic_ce": _f((ce * dynamic.float()).sum() / d_sum),
        "static_acc": _f((correct * static.float()).sum() / s_sum),
        "dynamic_acc": _f((correct * dynamic.float()).sum() / d_sum),
        "dynamic_fraction": _f(dynamic.float().mean()),
    }


def _summarize_token_recon_steps(
    logits: torch.Tensor,
    next_idx: torch.Tensor,
    cur_idx: torch.Tensor,
) -> dict[str, dict[str, float]]:
    if next_idx.ndim == 2:
        next_idx = next_idx.unsqueeze(1)
    if cur_idx.ndim == 2:
        cur_idx = cur_idx.unsqueeze(1)
    out: dict[str, dict[str, float]] = {}
    for t in range(logits.shape[1]):
        out[f"t{t+1}"] = _summarize_token_recon(
            logits[:, t : t + 1],
            next_idx[:, t : t + 1],
            cur_idx[:, t : t + 1],
        )
    return out


def _ridge_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    ridge: float = 10.0,
) -> dict[str, float]:
    """Small closed-form ridge probe with an 80/20 split.

    Reports test R2 against predicting the train-set mean target.  The value can
    be negative when the linear map is worse than the constant baseline.
    """
    x = x.detach().float().cpu()
    y = y.detach().float().cpu()
    n = int(x.shape[0])
    if n < 8:
        return {"r2": float("nan"), "mse": float("nan"), "baseline_mse": float("nan"), "n_train": 0, "n_test": 0}
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = max(1, int(0.8 * n))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    if int(test_idx.numel()) == 0:
        test_idx = train_idx

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    xs_train = (x_train - x_mean) / x_std
    ys_train = (y_train - y_mean) / y_std
    xs_test = (x_test - x_mean) / x_std

    xtx = xs_train.T @ xs_train
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype)
    w = torch.linalg.solve(xtx + ridge * eye, xs_train.T @ ys_train)
    y_pred = xs_test @ w * y_std + y_mean
    mse = ((y_pred - y_test) ** 2).mean()
    baseline_mse = ((y_mean - y_test) ** 2).mean()
    r2 = 1.0 - mse / baseline_mse.clamp_min(1e-12)
    return {
        "r2": float(r2.item()),
        "mse": float(mse.item()),
        "baseline_mse": float(baseline_mse.item()),
        "n_train": int(train_idx.numel()),
        "n_test": int(test_idx.numel()),
        "x_dim": int(x.shape[-1]),
        "y_dim": int(y.shape[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WM action sensitivity diagnostic")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-key", default="dataset_val_ind")
    parser.add_argument("--sequence-config-path", default=None)
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--horizons", default="1,2,3,5,7")
    parser.add_argument(
        "--which-blocks",
        default=None,
        help="Comma-separated image block indices. Defaults to viz.which_blocks or viz.which_block.",
    )
    parser.add_argument(
        "--tokens-per-image-block",
        type=int,
        default=None,
        help="Expected image tokens per selected block. Defaults to n_image_tokens / len(which_blocks).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    horizons = _parse_horizons(args.horizons)
    out_dir = (Path(args.out_dir) if args.out_dir else
               PROJECT_ROOT / "data" / "outputs" / "diagnose_wm" /
               datetime.now().strftime("action_sensitivity_%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[action-sens] config={args.config_name} ckpt={args.ckpt} out={out_dir}")
    cfg = build_cfg(args.config_name, args.overrides)

    print("[action-sens] building encoder ...")
    with open_dict(cfg):
        cfg.encoder.freeze_backbone = True
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print("[action-sens] building world model ...")
    hidden_dim = int(OmegaConf.select(cfg, "world_model.hidden_dim", default=4096))
    wm_kwargs: dict[str, Any] = {"hidden_dim": hidden_dim}
    if (str(OmegaConf.select(cfg, "world_model.io_mode", default="hidden")) == "token"
            and OmegaConf.select(cfg, "world_model.num_image_tokens_vocab") is None):
        wm_kwargs["num_image_tokens_vocab"] = len(
            encoder.backbone.model.vocabulary_mapping.bpe2img
        )
    state_start_id = state_end_id = None
    if bool(OmegaConf.select(cfg, "world_model.state_conditioning", default=False)):
        state_start_id, state_end_id = _infer_state_token_bounds(encoder)
        if OmegaConf.select(cfg, "world_model.state_token_offset") is None:
            wm_kwargs["state_token_offset"] = state_start_id
        if OmegaConf.select(cfg, "world_model.num_state_tokens_vocab") is None:
            wm_kwargs["num_state_tokens_vocab"] = state_end_id - state_start_id + 1
    world_model = hydra.utils.instantiate(cfg.world_model, **wm_kwargs)
    world_model = world_model.to(dtype=torch.bfloat16).to(device)
    state_dict = _strip_fsdp_prefix(load_wm_state_dict(Path(args.ckpt)))
    missing, unexpected = world_model.load_state_dict(state_dict, strict=False)
    print(f"[action-sens] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    world_model.eval()

    vocab_mapping = encoder.backbone.model.vocabulary_mapping
    image_token_bpe_ids = torch.tensor(
        sorted(vocab_mapping.bpe2img.keys()), dtype=torch.long, device=device,
    )
    if getattr(world_model, "spatial_codec", False):
        wm_io_mode = str(getattr(world_model, "io_mode", "hidden"))
        full_vocab_size = int(encoder.backbone.lm_head.weight.shape[0])
        world_model.attach_lm_head(
            encoder.backbone.lm_head if wm_io_mode == "hidden" else None,
            image_token_bpe_ids,
            full_vocab_size=full_vocab_size,
        )

    print("[action-sens] building dataset ...")
    sequence_mode = args.sequence_config_path is not None or args.sequence_length > 0
    if sequence_mode:
        seq_path = args.sequence_config_path
        if seq_path is None:
            seq_path = "/home/user01/liops/workspace/DreamerVLA/data/configs/libero_goal/his_2_third_view_wrist_w_state_1_256_pretokenize_seq_val_ind.yaml"
        dataset = PretokenizeDataset(
            seq_path,
            sequence_length=(args.sequence_length if args.sequence_length > 0 else None),
            stride=args.stride,
        )
    else:
        dataset_cfg = OmegaConf.select(cfg, args.dataset_key)
        if dataset_cfg is None:
            raise ValueError(f"dataset key not found in config: {args.dataset_key}")
        dataset = hydra.utils.instantiate(dataset_cfg)
    n_total = len(dataset)
    print(f"[action-sens] dataset size = {n_total}")

    n_image_tokens = int(getattr(world_model, "n_image_tokens", 256))
    image_bpe_set = set(image_token_bpe_ids.tolist())
    if args.which_blocks is not None:
        which_blocks = _parse_int_list(args.which_blocks)
    else:
        which_blocks_cfg = OmegaConf.select(cfg, "viz.which_blocks", default=None)
        which_blocks = (
            [int(block_idx) for block_idx in which_blocks_cfg]
            if which_blocks_cfg is not None
            else [int(OmegaConf.select(cfg, "viz.which_block", default=-2))]
        )
    if not which_blocks:
        raise ValueError("which_blocks must contain at least one image block index")
    if args.tokens_per_image_block is not None:
        n_img_tok_per_block = int(args.tokens_per_image_block)
    else:
        if n_image_tokens % len(which_blocks) != 0:
            raise ValueError(
                f"n_image_tokens={n_image_tokens} is not divisible by "
                f"len(which_blocks)={len(which_blocks)}"
            )
        n_img_tok_per_block = n_image_tokens // len(which_blocks)
    expected_total_image_tokens = n_img_tok_per_block * len(which_blocks)
    if expected_total_image_tokens != n_image_tokens:
        raise ValueError(
            f"selected blocks imply {expected_total_image_tokens} tokens "
            f"but world_model.n_image_tokens={n_image_tokens}"
        )
    print(
        f"[action-sens] image blocks={which_blocks}; "
        f"tokens_per_block={n_img_tok_per_block}; total={n_image_tokens}"
    )

    def extract_selected_blocks(seq):
        blocks = extract_image_blocks(list(seq))
        if not blocks:
            return None
        row: list[int] = []
        for block_idx in which_blocks:
            bidx = block_idx if block_idx >= 0 else len(blocks) + block_idx
            if bidx < 0 or bidx >= len(blocks):
                return None
            ids = [int(t) for t in blocks[bidx][2] if int(t) in image_bpe_set]
            if len(ids) != n_img_tok_per_block:
                return None
            row.extend(ids)
        return row

    indices = rng.choice(n_total, size=min(args.num_samples * 8, n_total), replace=False).tolist()
    obs_list: list[list[int]] = []
    next_obs_list: list[list[int]] = []
    seq_list: list[list[list[int]]] = []
    action_list: list[torch.Tensor] = []
    action_seq_list: list[torch.Tensor] = []
    prev_action_list: list[torch.Tensor | None] = []
    next_action_list: list[torch.Tensor | None] = []
    sample_task_names: list[str] = []
    sample_trajectory_keys: list[str] = []
    sample_files: list[str] = []
    input_id_seq_list: list[list[list[int]]] = []

    for idx in indices:
        n_collected = len(seq_list) if sequence_mode else len(obs_list)
        if n_collected >= args.num_samples:
            break
        sample = dataset[idx]
        if sequence_mode:
            seq_ids = sample.get("wm_obs_input_ids_seq")
            act_seq = sample.get("action_seq")
            if not isinstance(seq_ids, list) or not isinstance(act_seq, torch.Tensor):
                continue
            blocks = [extract_selected_blocks(step_ids) for step_ids in seq_ids]
            if any(block is None for block in blocks):
                continue
            if act_seq.ndim != 2 or act_seq.shape[0] != len(blocks) or act_seq.shape[0] < 2:
                continue
            seq_list.append([list(block) for block in blocks if block is not None])
            input_id_seq_list.append([list(step_ids) for step_ids in seq_ids])
            action_seq_list.append(act_seq.float())
            action_list.append(act_seq[1:].float().reshape(-1))
            prev_action_list.append(None)
            next_action_list.append(None)
            metas = sample.get("meta_seq", sample.get("meta", []))
            first_meta = metas[0] if isinstance(metas, list) and metas else {}
            sample_task_names.append(str(sample.get("task_name", first_meta.get("task_name", ""))))
            sample_trajectory_keys.append(str(first_meta.get("trajectory_key", "")))
            sample_files.append(str(first_meta.get("file", "")))
        else:
            cur_ids = list(sample.get("wm_obs_input_ids", []))
            next_ids = list(sample.get("wm_next_obs_input_ids", []))
            cur_block = extract_selected_blocks(cur_ids)
            next_block = extract_selected_blocks(next_ids)
            wm_action = sample.get("wm_action")
            if cur_block is None or next_block is None:
                continue
            if not isinstance(wm_action, torch.Tensor) or wm_action.numel() == 0:
                continue
            action = wm_action.float().mean(dim=0) if wm_action.ndim == 2 else wm_action.float()
            if action.ndim != 1:
                continue
            action_paths = sample.get("action", [])
            action_path = action_paths[0] if isinstance(action_paths, list) and action_paths else ""
            prev_action = _load_neighbor_action(str(action_path), -1, int(action.shape[0]))
            next_action = _load_neighbor_action(str(action_path), 1, int(action.shape[0]))
            obs_list.append(cur_block)
            next_obs_list.append(next_block)
            input_id_seq_list.append([cur_ids, next_ids])
            action_list.append(action)
            prev_action_list.append(prev_action)
            next_action_list.append(next_action)
            sample_task_names.append(str(sample.get("task_name", "")))
            meta = sample.get("meta", {})
            sample_trajectory_keys.append(str(meta.get("trajectory_key", "")) if isinstance(meta, dict) else "")
            sample_files.append(str(sample.get("file", "")))

    if sequence_mode and not seq_list:
        raise RuntimeError("no usable sequence samples")
    if not sequence_mode and not obs_list:
        raise RuntimeError("no usable samples")
    print(f"[action-sens] using {len(seq_list) if sequence_mode else len(obs_list)} samples")

    if sequence_mode:
        hidden_seq = torch.tensor(seq_list, dtype=torch.long, device=device)
        action_seq_full = torch.stack(action_seq_list, dim=0).to(device=device, dtype=torch.bfloat16)
        obs = hidden_seq[:, :-1]
        next_obs = hidden_seq[:, 1:]
        action_real = action_seq_full[:, 1:]
        B, K, A = action_real.shape
        action_seq_real = action_seq_full
    else:
        obs = torch.tensor(obs_list, dtype=torch.long, device=device)
        next_obs = torch.tensor(next_obs_list, dtype=torch.long, device=device)
        action_real = torch.stack(action_list, dim=0).to(device=device, dtype=torch.bfloat16)
        B, A = action_real.shape
        K = 1
        hidden_seq = torch.stack([obs, next_obs], dim=1)
        action_seq_real = torch.stack([torch.zeros_like(action_real), action_real], dim=1)

    with torch.no_grad():
        img_idx_seq = world_model._bpe_to_img_idx[hidden_seq]
        if (img_idx_seq < 0).any():
            raise RuntimeError("non-image BPE ids in extracted image blocks")
        per_tok = world_model.token_embedder(img_idx_seq)
        T_full = int(per_tok.shape[1])
        if bool(getattr(world_model, "state_conditioning", False)):
            if state_start_id is None or state_end_id is None:
                raise RuntimeError("state-token bounds unavailable for state-conditioned WM")
            flat_input_ids = [ids for seq in input_id_seq_list for ids in seq]
            state_ids_flat, state_mask_flat = _extract_state_bpe_ids(
                flat_input_ids, state_start_id, state_end_id,
            )
            state_ids_seq = state_ids_flat.view(B, T_full, -1).to(device=device)
            state_mask_seq = state_mask_flat.view(B, T_full, -1).to(device=device)
            per_tok = world_model._fuse_state_context(per_tok, state_ids_seq, state_mask_seq)
        bt = per_tok.shape[0] * per_tok.shape[1]
        obs_seq = world_model.conv_stem(per_tok.reshape(bt, *per_tok.shape[2:])).reshape(B, T_full, -1)

        post_mean, post_std, post_stoch, prior_mean_real, prior_std_real, prior_stoch_real, h_real = (
            world_model._infer_dreamer_seq(obs_seq.to(dtype=torch.bfloat16), action_seq_real)
        )

    z_prefix = post_stoch[:, :-1]
    target_z = post_stoch[:, 1:]
    target_feature_real = torch.cat([h_real, target_z], dim=-1)
    cur_img_idx = world_model._bpe_to_img_idx[obs]
    next_img_idx = world_model._bpe_to_img_idx[next_obs]

    variants: dict[str, torch.Tensor] = {
        "real": action_real,
        "scale_0.25": 0.25 * action_real,
        "scale_0.5": 0.5 * action_real,
        "scale_1.0": action_real,
        "scale_2.0": 2.0 * action_real,
        "zero": torch.zeros_like(action_real),
    }
    perm = torch.randperm(B, device=device)
    variants["shuffled"] = action_real[perm]
    mean = action_real.float().reshape(-1, A).mean(dim=0, keepdim=True)
    std = action_real.float().reshape(-1, A).std(dim=0, keepdim=True).clamp_min(1e-6)
    random = torch.randn(B, K, A, device=device) * std.view(1, 1, A) + mean.view(1, 1, A)
    variants["random_gaussian"] = random.to(dtype=torch.bfloat16)

    if sequence_mode:
        prev = action_seq_real[:, :-1]
        nxt = torch.cat([action_seq_real[:, 2:], action_seq_real[:, -1:]], dim=1)
        variants["prev_action"] = prev
        variants["next_action"] = nxt
        prev_valid_count = B * K
        next_valid_count = B * K
    else:
        prev_valid = [x is not None for x in prev_action_list]
        next_valid = [x is not None for x in next_action_list]
        if any(prev_valid):
            prev = []
            for valid, alt, cur in zip(prev_valid, prev_action_list, action_list):
                prev.append(alt if valid and alt is not None else cur)
            variants["prev_action_or_real"] = torch.stack(prev, dim=0).to(device=device, dtype=torch.bfloat16)
        if any(next_valid):
            nxt = []
            for valid, alt, cur in zip(next_valid, next_action_list, action_list):
                nxt.append(alt if valid and alt is not None else cur)
            variants["next_action_or_real"] = torch.stack(nxt, dim=0).to(device=device, dtype=torch.bfloat16)
        prev_valid_count = int(sum(prev_valid))
        next_valid_count = int(sum(next_valid))

    results: dict[str, dict[str, float]] = {}
    target_delta = target_z - z_prefix

    post_logits0 = post_mean[:, 0].float()
    post_logits1 = post_mean[:, 1:].float()
    delta_logits = post_logits1 - post_mean[:, :-1].float()
    delta_stoch = target_delta.float()
    action_float = action_real.float()
    linear_probe = {
        "delta_stoch_to_action": _ridge_probe(delta_stoch.reshape(B * K, -1), action_float.reshape(B * K, A), seed=args.seed + 11),
        "delta_logits_to_action": _ridge_probe(delta_logits.reshape(B * K, -1), action_float.reshape(B * K, A), seed=args.seed + 12),
        "z0_z1_stoch_to_action": _ridge_probe(
            torch.cat([z_prefix.float(), target_z.float()], dim=-1).reshape(B * K, -1),
            action_float.reshape(B * K, A),
            seed=args.seed + 13,
        ),
        "action_to_delta_stoch": _ridge_probe(action_float.reshape(B * K, A), delta_stoch.reshape(B * K, -1), seed=args.seed + 14),
        "action_to_delta_logits": _ridge_probe(action_float.reshape(B * K, A), delta_logits.reshape(B * K, -1), seed=args.seed + 15),
    }

    multi_step_probe: dict[str, dict[str, dict[str, float] | int | bool]] = {}
    for horizon in horizons:
        if horizon >= T_full:
            multi_step_probe[f"h{horizon}"] = {
                "valid": False,
                "num_windows": 0,
                "reason": f"sequence_length={T_full} is too short for horizon={horizon}",
            }
            continue
        dz_stoch_chunks: list[torch.Tensor] = []
        dz_logits_chunks: list[torch.Tensor] = []
        action_chunks: list[torch.Tensor] = []
        for start in range(T_full - horizon):
            dz_stoch_chunks.append((post_stoch[:, start + horizon] - post_stoch[:, start]).float())
            dz_logits_chunks.append((post_mean[:, start + horizon] - post_mean[:, start]).float())
            action_chunks.append(action_seq_real[:, start + 1 : start + horizon + 1].float().reshape(B, -1))
        dz_stoch_h = torch.cat(dz_stoch_chunks, dim=0)
        dz_logits_h = torch.cat(dz_logits_chunks, dim=0)
        action_h = torch.cat(action_chunks, dim=0)
        multi_step_probe[f"h{horizon}"] = {
            "valid": True,
            "num_windows": int(action_h.shape[0]),
            "delta_stoch_to_action_seq": _ridge_probe(dz_stoch_h, action_h, seed=args.seed + 100 + horizon),
            "delta_logits_to_action_seq": _ridge_probe(dz_logits_h, action_h, seed=args.seed + 200 + horizon),
            "action_seq_to_delta_stoch": _ridge_probe(action_h, dz_stoch_h, seed=args.seed + 300 + horizon),
            "action_seq_to_delta_logits": _ridge_probe(action_h, dz_logits_h, seed=args.seed + 400 + horizon),
        }

    def _rollout_from(
        start: int,
        action_window: torch.Tensor,  # [B, H, A], actions for start+1..start+H
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_roll = post_stoch[:, : start + 1].to(dtype=torch.bfloat16)
        action_prefix = action_seq_real[:, 1 : start + 1].to(dtype=torch.bfloat16)
        last_h: torch.Tensor | None = None
        for step in range(action_window.shape[1]):
            action_step = action_window[:, step : step + 1].to(dtype=torch.bfloat16)
            infer_action = torch.cat([action_prefix, action_step], dim=1)
            _, _, prior_step, h_step = world_model._infer_prior_seq(z_roll, infer_action)
            z_next = prior_step[:, -1:]
            last_h = h_step[:, -1]
            z_roll = torch.cat([z_roll, z_next], dim=1)
            action_prefix = torch.cat([action_prefix, action_step], dim=1)
        if last_h is None:
            raise RuntimeError("rollout horizon must be >= 1")
        return z_roll[:, -1], last_h

    horizon_sensitivity: dict[str, dict[str, Any]] = {}
    action_mean = action_seq_real[:, 1:].float().reshape(-1, A).mean(dim=0)
    action_std = action_seq_real[:, 1:].float().reshape(-1, A).std(dim=0).clamp_min(1e-6)
    with torch.no_grad():
        for horizon in horizons:
            if horizon >= T_full:
                horizon_sensitivity[f"h{horizon}"] = {
                    "valid": False,
                    "num_windows": 0,
                    "reason": f"sequence_length={T_full} is too short for horizon={horizon}",
                }
                continue
            variant_chunks: dict[str, dict[str, list[torch.Tensor]]] = {
                name: {"z_l2": [], "feature_l2": [], "delta_l2": [], "delta_cos": []}
                for name in ("real", "zero", "shuffled", "random_gaussian")
            }
            for start in range(T_full - horizon):
                real_window = action_seq_real[:, start + 1 : start + horizon + 1]
                perm_h = torch.randperm(B, device=device)
                windows = {
                    "real": real_window,
                    "zero": torch.zeros_like(real_window),
                    "shuffled": real_window[perm_h] if B > 1 else real_window,
                    "random_gaussian": (
                        torch.randn(B, horizon, A, device=device) * action_std.view(1, 1, A)
                        + action_mean.view(1, 1, A)
                    ).to(dtype=torch.bfloat16),
                }
                target_zh = post_stoch[:, start + horizon]
                anchor_z = post_stoch[:, start]
                target_delta_h = target_zh - anchor_z
                target_feature_h = torch.cat([h_real[:, start + horizon - 1], target_zh], dim=-1)
                for name, window in windows.items():
                    pred_zh, pred_hh = _rollout_from(start, window)
                    pred_delta_h = pred_zh - anchor_z
                    pred_feature_h = torch.cat([pred_hh, pred_zh], dim=-1)
                    variant_chunks[name]["z_l2"].append(_l2(pred_zh - target_zh))
                    variant_chunks[name]["feature_l2"].append(_l2(pred_feature_h - target_feature_h))
                    variant_chunks[name]["delta_l2"].append(_l2(pred_delta_h - target_delta_h))
                    variant_chunks[name]["delta_cos"].append(
                        F.cosine_similarity(pred_delta_h.float(), target_delta_h.float(), dim=-1)
                    )
            rows: dict[str, dict[str, float]] = {}
            for name, chunks in variant_chunks.items():
                z_l2_h = torch.cat(chunks["z_l2"], dim=0)
                feature_l2_h = torch.cat(chunks["feature_l2"], dim=0)
                delta_l2_h = torch.cat(chunks["delta_l2"], dim=0)
                delta_cos_h = torch.cat(chunks["delta_cos"], dim=0)
                rows[name] = {
                    "z_l2_mean": _f(z_l2_h),
                    "z_l2_median": _f(z_l2_h.median()),
                    "feature_l2_mean": _f(feature_l2_h),
                    "delta_l2_mean": _f(delta_l2_h),
                    "delta_cos_mean": _f(delta_cos_h),
                }
            real_h = rows["real"]
            for name, row in rows.items():
                row["z_l2_minus_real"] = row["z_l2_mean"] - real_h["z_l2_mean"]
                row["delta_l2_minus_real"] = row["delta_l2_mean"] - real_h["delta_l2_mean"]
            horizon_sensitivity[f"h{horizon}"] = {
                "valid": True,
                "num_windows": int((T_full - horizon) * B),
                "results": rows,
            }

    latent_distance: dict[str, float] = {}
    with torch.no_grad():
        latent_distance["nearby_frame_z_l2_mean"] = _f(_l2(post_stoch[:, 1:] - post_stoch[:, :-1]).reshape(-1))
        for horizon in horizons:
            if horizon < T_full:
                latent_distance[f"same_sequence_z_l2@{horizon}"] = _f(
                    _l2(post_stoch[:, horizon:] - post_stoch[:, :-horizon]).reshape(-1)
                )
        z0 = post_stoch[:, 0].float()
        if B >= 2:
            pair = torch.cdist(z0, z0, p=2)
            upper = torch.triu(torch.ones(B, B, dtype=torch.bool, device=device), diagonal=1)
            if upper.any():
                latent_distance["batch_pair_z0_l2_mean"] = _f(pair[upper])
            task_same = torch.tensor(
                [[sample_task_names[i] == sample_task_names[j] for j in range(B)] for i in range(B)],
                dtype=torch.bool,
                device=device,
            ) & upper
            task_diff = (~torch.tensor(
                [[sample_task_names[i] == sample_task_names[j] for j in range(B)] for i in range(B)],
                dtype=torch.bool,
                device=device,
            )) & upper
            traj_same = torch.tensor(
                [[bool(sample_trajectory_keys[i]) and sample_trajectory_keys[i] == sample_trajectory_keys[j] for j in range(B)] for i in range(B)],
                dtype=torch.bool,
                device=device,
            ) & upper
            if task_same.any():
                latent_distance["same_task_z0_l2_mean"] = _f(pair[task_same])
            if task_diff.any():
                latent_distance["different_task_z0_l2_mean"] = _f(pair[task_diff])
            if traj_same.any():
                latent_distance["same_trajectory_z0_l2_mean"] = _f(pair[traj_same])

    with torch.no_grad():
        for name, action in variants.items():
            action_seq = action if action.ndim == 3 else action.unsqueeze(1)
            pm, ps, pz, h = world_model._infer_prior_seq(z_prefix, action_seq)
            feature = torch.cat([h, pz], dim=-1)
            pred_delta = pz - z_prefix
            z_l2_all = _l2(pz - target_z)
            feature_l2_all = _l2(feature - target_feature_real)
            delta_l2_all = _l2(pred_delta - target_delta)
            z_l2 = z_l2_all.reshape(-1)
            feature_l2 = feature_l2_all.reshape(-1)
            delta_l2 = delta_l2_all.reshape(-1)
            delta_mse = ((pred_delta - target_delta).float() ** 2).mean(dim=-1).reshape(-1)
            pred_delta_flat = pred_delta.float().reshape(B * K, -1)
            target_delta_flat = target_delta.float().reshape(B * K, -1)
            delta_cos = F.cosine_similarity(pred_delta_flat, target_delta_flat, dim=-1)
            row = {
                "z_l2_mean": _f(z_l2),
                "z_l2_median": _f(z_l2.median()),
                "feature_l2_mean": _f(feature_l2),
                "feature_l2_median": _f(feature_l2.median()),
                "delta_l2_mean": _f(delta_l2),
                "delta_l2_median": _f(delta_l2.median()),
                "delta_mse_mean": _f(delta_mse),
                "pred_delta_l2_mean": _f(_l2(pred_delta_flat)),
                "target_delta_l2_mean": _f(_l2(target_delta_flat)),
                "delta_cos_mean": _f(delta_cos),
                "delta_cos_median": _f(delta_cos.median()),
                "prior_z_norm_mean": _f(_l2(pz).reshape(-1)),
                "prior_feature_norm_mean": _f(_l2(feature).reshape(-1)),
            }
            row["per_step"] = {
                f"t{t+1}": {
                    "z_l2_mean": _f(z_l2_all[:, t]),
                    "feature_l2_mean": _f(feature_l2_all[:, t]),
                    "delta_l2_mean": _f(delta_l2_all[:, t]),
                    "delta_cos_mean": _f(F.cosine_similarity(pred_delta[:, t].float(), target_delta[:, t].float(), dim=-1)),
                }
                for t in range(K)
            }
            if getattr(world_model, "image_decoder", None) is not None:
                logits = world_model.image_decoder(h, pz)
                row.update(_summarize_token_recon(logits, next_img_idx, cur_img_idx))
                row["per_step_token"] = _summarize_token_recon_steps(logits, next_img_idx, cur_img_idx)
            results[name] = row

    real = results["real"]
    for name, row in results.items():
        row["z_l2_minus_real"] = row["z_l2_mean"] - real["z_l2_mean"]
        row["delta_l2_minus_real"] = row["delta_l2_mean"] - real["delta_l2_mean"]
        if "dynamic_ce" in row and "dynamic_ce" in real:
            row["dynamic_ce_minus_real"] = row["dynamic_ce"] - real["dynamic_ce"]

    scale_order = ["zero", "scale_0.25", "scale_0.5", "real", "scale_2.0", "shuffled", "random_gaussian"]
    payload = {
        "ckpt": str(args.ckpt),
        "config": args.config_name,
        "dataset_key": args.dataset_key,
        "num_samples": B,
        "sequence_length": int(hidden_seq.shape[1]),
        "time_step_note": (
            f"Sequence mode: reports t=1..{K} for frame-to-frame prediction."
            if sequence_mode else
            "This pretokenized dataset provides one transition, so per-timestep output is only t=1."
        ),
        "action_alignment_note": {
            "current": "action_seq[1] is used to predict frame 1 from frame 0.",
            "prev_action_available": int(prev_valid_count),
            "next_action_available": int(next_valid_count),
        },
        "target_stats": {
            "target_z_norm_mean": _f(_l2(target_z).reshape(-1)),
            "target_delta_l2_mean": _f(_l2(target_delta).reshape(-1)),
            "action_norm_mean": _f(_l2(action_real).reshape(-1)),
            "action_norm_median": _f(_l2(action_real).reshape(-1).median()),
        },
        "linear_probe": linear_probe,
        "multi_step_probe": multi_step_probe,
        "horizon_sensitivity": horizon_sensitivity,
        "latent_distance": latent_distance,
        "scale_order": scale_order,
        "results": results,
        "sample_files_head": sample_files[:8],
    }

    out_path = out_dir / "action_sensitivity.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print("=" * 72)
    print("ACTION SENSITIVITY RESULT")
    print("=" * 72)
    for name in scale_order + ["prev_action_or_real", "next_action_or_real", "prev_action", "next_action"]:
        if name not in results:
            continue
        row = results[name]
        print(
            f"{name:>18s}  z_l2={row['z_l2_mean']:.4f}  "
            f"delta_l2={row['delta_l2_mean']:.4f}  "
            f"feature_l2={row['feature_l2_mean']:.4f}  "
            f"dyn_ce={row.get('dynamic_ce', float('nan')):.4f}"
        )
    print()
    print("HORIZON SENSITIVITY")
    print("-" * 72)
    for horizon in horizons:
        row = horizon_sensitivity.get(f"h{horizon}", {})
        if not row.get("valid"):
            print(f"h={horizon:<2d} skipped: {row.get('reason')}")
            continue
        res = row["results"]
        print(
            f"h={horizon:<2d} "
            f"real={res['real']['z_l2_mean']:.4f} "
            f"zero={res['zero']['z_l2_mean']:.4f} "
            f"shuf={res['shuffled']['z_l2_mean']:.4f} "
            f"rand={res['random_gaussian']['z_l2_mean']:.4f} "
            f"m_zero={res['zero']['z_l2_minus_real']:.4f} "
            f"m_shuf={res['shuffled']['z_l2_minus_real']:.4f}"
        )
    print(f"[action-sens] full report: {out_path}")


if __name__ == "__main__":
    main()
