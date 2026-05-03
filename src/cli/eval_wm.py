"""
World-model evaluation: rollout predictions and decode them back to pixel images.

Run:
    python -m scripts.eval_wm \
        --config-name pretokenize_wm_libero_10 \
        --ckpt data/outputs/pretokenize_wm/<run>/checkpoints/<file>.ckpt \
        --out-dir data/outputs/pretokenize_wm/<run>/eval \
        --num-samples 8 \
        --next-step 1

Architectural caveats you must know before reading the numbers:

  The TSSM world model trained by this repo operates on **pooled 4096-d hidden
  vectors** – one per frame, produced by mean-pooling the Chameleon LLM hidden
  states over the full prompt-plus-image-tokens sequence. Its output therefore
  has no spatial structure, so decoding it back to an image is necessarily a
  heuristic. This script uses the "broadcast-delta" trick:

        delta = pred_next_pooled - current_pooled        # [D]
        h'_i  = h_i (image-token hidden) + delta         # broadcast add
        bpe_i = argmax_{image-vocab} lm_head(h'_i)
        image = VQGAN.decode(bpe2vq(bpe_i))

  In effect the predicted image carries the *global* change the WM anticipates
  (lighting shift, average colour, rough motion) stamped onto the spatial
  template of the current frame. Treat the picture as a qualitative sanity
  check, not a pixel-accurate prediction.

  The second caveat: the pretokenized ``wm_next_obs_input_ids`` stored in the
  dataset is merely a **prompt template** (23 tokens, no real next-frame image
  tokens). That is what the WM was supervised on, so hidden-space MSE against
  that target is the *training* metric, not a true visual-prediction metric.
  For visual comparison we therefore also load the raw next-frame PNG from the
  trajectory (default: the frame 1 step after current).

What is written to ``out-dir``:

  * ``sample_###.png``  –  strip of [ GT current | GT next | predicted next ]
  * ``metrics.json``    –  per-sample and aggregate hidden-space metrics plus
                           a variance probe of the training target
"""
from __future__ import annotations

import argparse
import json
import math
import random
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
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Chameleon image-block delimiters (see modeling_xllmx_chameleon_ck_action_head.py:436-437).
# Between IMG_START_TOK ... IMG_END_TOK the payload is laid out as:
#     [IMG_HEADER, IMG_HEADER, row0_tok_0, .., row0_tok_{S-1}, ROW_SEP,
#                               row1_tok_0, .., row1_tok_{S-1}, ROW_SEP, ...,
#                               row{S-1}_tok_0, .., row{S-1}_tok_{S-1}, ROW_SEP]
# For the 256-resolution pretokenized data, S = 16 so every block has
# 2 + 16 * (16 + 1) = 274 tokens.
IMG_START_TOK = 8197
IMG_END_TOK = 8196
IMG_HEADER_TOK = 8812
ROW_SEP_TOK = 8803


# ── Token/image helpers ───────────────────────────────────────────────────────


def extract_image_blocks(token_ids: list[int]) -> list[tuple[int, int, list[int]]]:
    """Walk a BPE sequence and return every ``[start, end)`` image block plus ids."""
    blocks: list[tuple[int, int, list[int]]] = []
    i, n = 0, len(token_ids)
    while i < n:
        if token_ids[i] == IMG_START_TOK:
            start = i + 1
            j = start
            while j < n and token_ids[j] != IMG_END_TOK:
                j += 1
            if j > start:
                blocks.append((start, j, list(token_ids[start:j])))
            i = j + 1
        else:
            i += 1
    return blocks


def strip_image_block_delimiters(block_ids: list[int]) -> list[int]:
    """
    Drop the ``[8812, 8812]`` header and the trailing ``8803`` row-separator from
    every row, leaving only the S*S codebook-index BPE ids.
    """
    if len(block_ids) < 3:
        return list(block_ids)
    # Header: leading pair of IMG_HEADER_TOK. Keep stripping while the pattern holds.
    i = 0
    while i < len(block_ids) and block_ids[i] == IMG_HEADER_TOK:
        i += 1
    payload = block_ids[i:]
    # Each row is S image tokens followed by ROW_SEP_TOK; infer S from the first sep.
    try:
        first_sep = payload.index(ROW_SEP_TOK)
    except ValueError:
        return payload
    side = first_sep  # number of image tokens per row
    if side <= 0:
        return payload
    clean: list[int] = []
    row = 0
    pos = 0
    while pos < len(payload):
        row_end = pos + side
        if row_end > len(payload):
            break
        clean.extend(payload[pos:row_end])
        # Expect ROW_SEP_TOK right after each row; tolerate its absence on the last row.
        if row_end < len(payload) and payload[row_end] == ROW_SEP_TOK:
            pos = row_end + 1
        else:
            pos = row_end
        row += 1
    return clean


def decode_bpe_block_to_pil(
    bpe_ids: list[int],
    bpe2vq: torch.Tensor,
    vq_model: Any,
) -> Image.Image:
    from src.utils.vq_image_decoder import vq_tokens_to_pixels, tensor_to_pil

    clean_ids = strip_image_block_delimiters(bpe_ids)
    n = len(clean_ids)
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(
            f"image block token count {n} (after stripping delimiters) is not a perfect square"
        )
    bpe_tensor = torch.tensor(clean_ids, dtype=torch.long, device=bpe2vq.device)
    vq_ids = bpe2vq[bpe_tensor]
    if (vq_ids < 0).any():
        bad = int((vq_ids < 0).sum().item())
        raise ValueError(f"image block contains {bad} non-image bpe ids after stripping")
    pixels = vq_tokens_to_pixels(vq_ids.unsqueeze(0), vq_model, h_latent=side, w_latent=side)
    return tensor_to_pil(pixels[0])


_IMAGE_INDEX_RE = re.compile(r"(?P<stem>.*/image_)(?P<idx>\d+)(?P<ext>\.\w+)$")


def bump_image_path(path: str, offset: int) -> str | None:
    """``.../image_5.png`` + offset=3  →  ``.../image_8.png`` (if file exists)."""
    match = _IMAGE_INDEX_RE.match(path)
    if match is None:
        return None
    idx = int(match.group("idx")) + int(offset)
    if idx < 0:
        return None
    candidate = f"{match.group('stem')}{idx}{match.group('ext')}"
    return candidate if Path(candidate).is_file() else None


# ── Hidden-state helpers ──────────────────────────────────────────────────────


def encode_hidden(
    encoder, input_ids_list: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (pooled [B, D], full_hidden [B, T, D], mask [B, T]) for BPE inputs."""
    labels_list = [[-100] * len(seq) for seq in input_ids_list]
    lengths = [len(seq) for seq in input_ids_list]
    with torch.no_grad():
        _, _, _, hidden_states, _, _, _ = encoder.backbone(
            input_ids=input_ids_list,
            labels=labels_list,
            training=True,
            output_hidden_states=True,
            att_mask=False,
        )
    mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
    for idx, length in enumerate(lengths):
        mask[idx, :length] = True
    weights = mask.to(hidden_states.dtype).unsqueeze(-1)
    pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return pooled.float(), hidden_states.float(), mask


def hidden_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(pred, target).item()
    cos = F.cosine_similarity(pred, target, dim=-1).mean().item()
    tgt_norm = target.norm(dim=-1).mean().item()
    err_norm = (pred - target).norm(dim=-1).mean().item()
    rel_err = err_norm / max(tgt_norm, 1e-8)
    return {
        "mse": float(mse),
        "cosine": float(cos),
        "target_norm": float(tgt_norm),
        "error_norm": float(err_norm),
        "relative_error": float(rel_err),
    }


# ── Broadcast-delta image reconstruction ──────────────────────────────────────


def predicted_next_image_by_broadcast(
    current_pooled: torch.Tensor,       # [1, D]
    pred_next_pooled: torch.Tensor,     # [1, D]
    current_full_hidden: torch.Tensor,  # [1, T, D]
    current_input_ids: list[int],
    lm_head: torch.nn.Linear,
    image_token_bpe_ids: torch.Tensor,  # [|img_vocab|] long
    bpe2vq: torch.Tensor,
    vq_model: Any,
    which_block: int = -1,
) -> tuple[Image.Image | None, list[int]]:
    blocks = extract_image_blocks(current_input_ids)
    if not blocks:
        return None, []

    start, end, block_ids = blocks[which_block]
    # Keep only positions whose input id is a real image-codebook bpe id —
    # drop the [8812, 8812] header and every 8803 row-separator so the hidden
    # sequence matches the S*S clean layout that decode_bpe_block_to_pil wants.
    image_bpe_set = set(image_token_bpe_ids.tolist())
    keep_positions = [
        start + offset
        for offset, tok in enumerate(block_ids)
        if int(tok) in image_bpe_set
    ]
    if not keep_positions:
        return None, []
    img_hidden = current_full_hidden[0, torch.tensor(keep_positions, device=current_full_hidden.device)]

    delta = (pred_next_pooled[0] - current_pooled[0]).to(img_hidden.dtype)
    pred_img_hidden = img_hidden + delta.unsqueeze(0)                     # [S*S, D]

    lm_dtype = next(lm_head.parameters()).dtype
    logits = lm_head(pred_img_hidden.to(dtype=lm_dtype))                   # [S*S, V]
    image_vocab_logits = logits[:, image_token_bpe_ids]                    # [S*S, |img_vocab|]
    idx_in_img_vocab = image_vocab_logits.argmax(dim=-1)
    predicted_bpe = image_token_bpe_ids[idx_in_img_vocab].tolist()

    # predicted_bpe is already the clean S*S layout – bypass the strip inside
    # decode_bpe_block_to_pil by passing ids that contain no delimiter tokens.
    pil = decode_bpe_block_to_pil(predicted_bpe, bpe2vq, vq_model)
    return pil, predicted_bpe


# ── Layout ────────────────────────────────────────────────────────────────────


def save_comparison_grid(
    path: Path,
    panels: list[tuple[str, Image.Image | None]],
    cell_size: int = 256,
) -> None:
    from PIL import ImageDraw

    n = len(panels)
    header = 20
    canvas = Image.new("RGB", (cell_size * n, cell_size + header), color=(32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(panels):
        x0 = cell_size * i
        if img is not None:
            resized = img.convert("RGB").resize((cell_size, cell_size), Image.BILINEAR)
            canvas.paste(resized, (x0, header))
        else:
            draw.rectangle([x0, header, x0 + cell_size, header + cell_size], fill=(60, 20, 20))
            draw.text((x0 + 10, header + cell_size // 2), "(missing)", fill=(230, 230, 230))
        draw.text((x0 + 4, 4), label, fill=(230, 230, 230))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# ── Checkpoint loading ────────────────────────────────────────────────────────


def load_wm_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dicts" not in payload or "world_model" not in payload["state_dicts"]:
        raise ValueError(f"checkpoint {ckpt_path} has no world_model state dict")
    return payload["state_dicts"]["world_model"]


def _strip_fsdp_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned = key
        for prefix in ("_fsdp_wrapped_module.", "module."):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        out[cleaned] = value
    return out


def build_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    configs_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="World-model evaluation with image reconstruction")
    parser.add_argument("--config-name", default="pretokenize_wm_libero_10")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to data/outputs/eval_wm/<timestamp>.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--next-step",
        type=int,
        default=1,
        help="Frame offset to use for 'GT next' image (default 1 step ahead).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else PROJECT_ROOT / "data" / "outputs" / "eval_wm" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval_wm] config={args.config_name}  ckpt={args.ckpt}  out={out_dir}  next_step={args.next_step}")
    cfg = build_cfg(args.config_name, args.overrides)

    print("[eval_wm] building encoder ...")
    with open_dict(cfg):
        cfg.encoder.freeze_backbone = True
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print("[eval_wm] building world model ...")
    hidden_dim = int(OmegaConf.select(cfg, "world_model.hidden_dim", default=4096))
    wm_kwargs: dict[str, Any] = {"hidden_dim": hidden_dim}
    if (
        str(OmegaConf.select(cfg, "world_model.io_mode", default="hidden")) == "token"
        and OmegaConf.select(cfg, "world_model.num_image_tokens_vocab") is None
    ):
        wm_kwargs["num_image_tokens_vocab"] = len(
            encoder.backbone.model.vocabulary_mapping.bpe2img
        )
    world_model = hydra.utils.instantiate(cfg.world_model, **wm_kwargs)
    world_model = world_model.to(dtype=torch.bfloat16).to(device)
    state_dict = _strip_fsdp_prefix(load_wm_state_dict(Path(args.ckpt)))
    missing, unexpected = world_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[eval_wm] WARNING missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[eval_wm] WARNING unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    world_model.eval()

    print("[eval_wm] building VQGAN decoder ...")
    from src.utils.vq_image_decoder import load_vq_model, build_bpe2vq_tensor

    vq_model = load_vq_model(
        cfg_path=cfg.encoder.chameleon_vqgan_config,
        ckpt_path=cfg.encoder.chameleon_vqgan_ckpt,
        device=device,
    )
    inner_chameleon = encoder.backbone.model
    vocab_mapping = inner_chameleon.vocabulary_mapping
    bpe2vq_table = build_bpe2vq_tensor(vocab_mapping).to(device)
    image_token_bpe_ids = torch.tensor(
        sorted(vocab_mapping.bpe2img.keys()),
        dtype=torch.long,
        device=device,
    )
    lm_head = encoder.backbone.lm_head

    # Attach lm_head + image-vocab mapping to WM (required by encode_latent in
    # token mode and by spatial_codec recon).  Token mode produces logits via
    # its own decoder, so we pass lm_head=None for that path.
    if getattr(world_model, "spatial_codec", False):
        wm_io_mode = str(getattr(world_model, "io_mode", "hidden"))
        full_vocab_size = int(lm_head.weight.shape[0])
        world_model.attach_lm_head(
            lm_head if wm_io_mode == "hidden" else None,
            image_token_bpe_ids,
            full_vocab_size=full_vocab_size,
        )
        print(f"[eval_wm] attached vocab mapping (io_mode={wm_io_mode}, image_vocab={image_token_bpe_ids.numel()})")

    print("[eval_wm] building dataset ...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    n_total = len(dataset)
    print(f"[eval_wm] dataset size = {n_total}")

    rng = random.Random(args.seed)
    # sample deterministic indices spread across the dataset
    indices = rng.sample(range(n_total), k=min(args.num_samples * 4, n_total))

    all_metrics: list[dict[str, Any]] = []
    training_target_vectors: list[torch.Tensor] = []  # to probe target variance
    saved = 0

    for idx in indices:
        if saved >= args.num_samples:
            break
        sample = dataset[idx]

        wm_obs_ids = sample["wm_obs_input_ids"]
        wm_next_obs_ids = sample["wm_next_obs_input_ids"]
        if not extract_image_blocks(wm_obs_ids):
            continue

        # GT current frame = last image in sample.image (third view, i.e. index -2)
        image_paths = sample["image"]
        if len(image_paths) < 2:
            continue
        cur_path = image_paths[-2]  # third view of current frame (last 2 are cur frame's third+wrist)
        if not Path(cur_path).is_file():
            continue
        try:
            gt_cur_pil = Image.open(cur_path).convert("RGB")
        except Exception as exc:
            print(f"[eval_wm] idx={idx}: can't open cur image: {exc}")
            continue

        # GT next frame: bump the index in the path by --next-step
        next_path = bump_image_path(cur_path, args.next_step)
        gt_next_pil = Image.open(next_path).convert("RGB") if next_path else None

        # Action
        wm_action = sample["wm_action"]
        if not isinstance(wm_action, torch.Tensor) or wm_action.numel() == 0:
            continue
        action = wm_action.to(device=device, dtype=torch.bfloat16).unsqueeze(0)  # [1, T, A]

        # Encode hiddens.
        cur_pooled, cur_full, _ = encode_hidden(encoder, [wm_obs_ids])
        tgt_pooled, _, _ = encode_hidden(encoder, [wm_next_obs_ids])

        cur_pooled_b = cur_pooled.to(device=device, dtype=torch.bfloat16)
        tgt_pooled_b = tgt_pooled.to(device=device, dtype=torch.bfloat16)

        # In token mode, the WM's encode_latent expects raw image-vocab BPE
        # ids (not the Chameleon-encoder pooled hidden state).  Extract the
        # current-frame image block from `wm_obs_ids` and pass that instead.
        wm_io_mode = str(getattr(world_model, "io_mode", "hidden"))
        if wm_io_mode == "token":
            blocks = extract_image_blocks(wm_obs_ids)
            if not blocks or len(blocks) < 2:
                print(f"[eval_wm] idx={idx}: <2 image blocks; skip"); continue
            # which_block=-2 is the third-view of the current frame (matches viz code below)
            cur_block_ids = [t for t in blocks[-2][2] if t in image_token_bpe_ids.tolist()]
            n_img_tok = int(getattr(world_model, "n_image_tokens", 256))
            if len(cur_block_ids) != n_img_tok:
                print(f"[eval_wm] idx={idx}: got {len(cur_block_ids)} img tok, expected {n_img_tok}; skip"); continue
            wm_input_cur = torch.tensor([cur_block_ids], dtype=torch.long, device=device)  # [1, N_img]
        else:
            wm_input_cur = cur_pooled_b

        with torch.no_grad():
            latent = world_model.encode_latent(wm_input_cur)
            prior_next = world_model.predict_next(latent, action)
            pred_next_hidden = world_model.transition_head(
                torch.cat([prior_next.h, prior_next.stoch], dim=-1)
            )  # [1, D]
            # Token-mode predicted image-vocab logits: feed (h, z) of the
            # rolled-out NEXT step into the WM's image_decoder.  Output is
            # [1, N_img, vocab].  argmax → predicted next-frame image-vocab idx.
            pred_next_img_idx_tok = None
            if wm_io_mode == "token" and getattr(world_model, "image_decoder", None) is not None:
                # image_decoder expects [..., T, ...] shapes; pretrain_loss runs
                # it on [B, T, ...].  Add a singleton T axis.
                dec_h = prior_next.h.unsqueeze(1)        # [1, 1, d_model]
                dec_z = prior_next.stoch.unsqueeze(1)    # [1, 1, latent]
                logits = world_model.image_decoder(dec_h, dec_z)  # [1, 1, N_img, V]
                pred_next_img_idx_tok = logits.argmax(dim=-1).squeeze(0).squeeze(0)  # [N_img]

        # Hidden-mode dim check: in token mode `pred_next_hidden` is in
        # obs_dim space (1024) while `tgt_pooled` is the Chameleon LLM hidden
        # (4096).  They are not directly comparable; skip the MSE metric and
        # fall back to a token-level metric below.
        m_pred = None
        m_identity = None
        if pred_next_hidden.shape[-1] == tgt_pooled_b.shape[-1]:
            m_pred = hidden_metrics(pred_next_hidden.float(), tgt_pooled_b.float())
            m_identity = hidden_metrics(cur_pooled_b.float(), tgt_pooled_b.float())
        training_target_vectors.append(tgt_pooled.float().cpu().squeeze(0))

        # Token-mode prediction quality: per-token argmax accuracy of the
        # WM's predicted next-frame image vocab vs the GT next-frame block.
        m_token = None
        if wm_io_mode == "token" and pred_next_img_idx_tok is not None:
            try:
                next_blocks = extract_image_blocks(wm_next_obs_ids)
                # Pull the same view as `which_block`.  Fall back to last block.
                target_block_idx = -2 if len(next_blocks) >= 2 else -1
                gt_next_block_bpe = [t for t in next_blocks[target_block_idx][2] if t in image_token_bpe_ids.tolist()]
                if len(gt_next_block_bpe) == int(getattr(world_model, "n_image_tokens", 256)):
                    gt_next_block_bpe_t = torch.tensor(gt_next_block_bpe, dtype=torch.long, device=device)
                    gt_next_img_idx = world_model._bpe_to_img_idx[gt_next_block_bpe_t]  # [N_img]
                    if (gt_next_img_idx >= 0).all():
                        match = (pred_next_img_idx_tok == gt_next_img_idx).float()
                        m_token = {
                            "next_token_acc":   float(match.mean()),
                            "n_unique_pred":    int(pred_next_img_idx_tok.unique().numel()),
                            "n_unique_gt":      int(gt_next_img_idx.unique().numel()),
                        }
            except Exception as exc:
                print(f"[eval_wm] idx={idx}: token-acc compute failed: {exc}")

        # Which image block to visualise: the conversation stores
        # [third_{i-1}, wrist_{i-1}, third_i, wrist_i] so index -2 is the
        # third-person view of the *current* frame, matching image_paths[-2].
        which_block = -2

        # Broadcast-delta predicted image (using the same third-view block).
        pred_pil = None
        try:
            pred_pil, _ = predicted_next_image_by_broadcast(
                current_pooled=cur_pooled.to(device=device),
                pred_next_pooled=pred_next_hidden.float().to(device=device),
                current_full_hidden=cur_full.to(device=device),
                current_input_ids=wm_obs_ids,
                lm_head=lm_head,
                image_token_bpe_ids=image_token_bpe_ids,
                bpe2vq=bpe2vq_table,
                vq_model=vq_model,
                which_block=which_block,
            )
        except Exception as exc:
            print(f"[eval_wm] idx={idx}: broadcast-delta decode failed: {exc}")

        # Also reconstruct the *current* frame purely from its own image-block tokens –
        # a sanity check that the VQ decoder is wired up correctly.
        cur_blocks = extract_image_blocks(wm_obs_ids)
        cur_reconstructed = None
        try:
            cur_reconstructed = decode_bpe_block_to_pil(cur_blocks[which_block][2], bpe2vq_table, vq_model)
        except Exception as exc:
            print(f"[eval_wm] idx={idx}: cur VQ reconstruct failed: {exc}")

        grid_path = out_dir / f"sample_{saved:03d}_idx{idx}.png"
        save_comparison_grid(
            grid_path,
            panels=[
                ("GT current (PNG)",                 gt_cur_pil),
                (f"GT +{args.next_step} step (PNG)", gt_next_pil),
                ("cur via VQ (sanity)",              cur_reconstructed),
                ("pred next (broadcast-delta)",      pred_pil),
            ],
        )

        entry = {
            "sample_index": idx,
            "id": int(sample.get("id", idx)),
            "task_name": sample.get("task_name", ""),
            "cur_image_path": cur_path,
            "next_image_path": next_path,
            "prediction":        m_pred,
            "identity_baseline": m_identity,
            "token_metric":      m_token,
        }
        all_metrics.append(entry)
        saved += 1
        if m_pred is not None and m_identity is not None:
            print(
                f"[eval_wm] idx={idx}: "
                f"pred_mse={m_pred['mse']:.4f}  cos={m_pred['cosine']:.4f}  "
                f"rel_err={m_pred['relative_error']:.4f}  "
                f"(identity rel_err={m_identity['relative_error']:.4f})  →  {grid_path.name}"
            )
        elif m_token is not None:
            print(
                f"[eval_wm] idx={idx} (token mode): "
                f"next_token_acc={m_token['next_token_acc']:.4f}  "
                f"uniq_pred={m_token['n_unique_pred']}/{m_token['n_unique_gt']}  →  {grid_path.name}"
            )
        else:
            print(f"[eval_wm] idx={idx}: no prediction metric computed  →  {grid_path.name}")

    # Probe how much the training target actually varies across samples:
    target_variance_report: dict[str, float] = {}
    if training_target_vectors:
        targets = torch.stack(training_target_vectors, dim=0)   # [N, D]
        mean_vec = targets.mean(dim=0, keepdim=True)
        centered = targets - mean_vec
        per_dim_std = centered.std(dim=0).mean().item()
        total_norm = targets.norm(dim=-1).mean().item()
        diff_norm = centered.norm(dim=-1).mean().item()
        target_variance_report = {
            "mean_target_norm": float(total_norm),
            "mean_centered_norm": float(diff_norm),
            "avg_per_dim_std": float(per_dim_std),
            "relative_spread": float(diff_norm / max(total_norm, 1e-8)),
        }

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    pred_entries     = [m["prediction"]        for m in all_metrics if m.get("prediction")        is not None]
    identity_entries = [m["identity_baseline"] for m in all_metrics if m.get("identity_baseline") is not None]
    token_entries    = [m["token_metric"]      for m in all_metrics if m.get("token_metric")      is not None]

    summary = {
        "num_samples": len(all_metrics),
        "prediction": ({
            k: _mean([m[k] for m in pred_entries])
            for k in ("mse", "cosine", "relative_error")
        } if pred_entries else None),
        "identity_baseline": ({
            k: _mean([m[k] for m in identity_entries])
            for k in ("mse", "cosine", "relative_error")
        } if identity_entries else None),
        "token_metric": ({
            "next_token_acc": _mean([m["next_token_acc"] for m in token_entries]),
            "n_unique_pred":  _mean([m["n_unique_pred"]  for m in token_entries]),
            "n_unique_gt":    _mean([m["n_unique_gt"]    for m in token_entries]),
        } if token_entries else None),
        "training_target_variance": target_variance_report,
    }

    (out_dir / "metrics.json").write_text(json.dumps(
        {"summary": summary, "per_sample": all_metrics},
        indent=2,
    ))
    print("[eval_wm] done. summary:")
    print(json.dumps(summary, indent=2))
    print(f"[eval_wm] wrote {len(all_metrics)} comparison PNGs + metrics.json to {out_dir}")


if __name__ == "__main__":
    main()
