"""Reusable "decode WM prediction back to pixels" utility.

Extracts the broadcast-delta logic from scripts/eval_wm.py so it can be called
periodically during training to produce rec-vs-gt comparison strips.

Why broadcast-delta: the WM in this repo operates on a single mean-pooled
4096-d hidden per frame -- spatial structure is already discarded by the
encoder. To turn a predicted pooled hidden back into an image we (a) take the
current frame's per-token hidden states, (b) add the predicted delta
(pred_next_pooled - cur_pooled) to every image-token position, (c) project
through the LLM's lm_head to get image-token logits, and (d) decode via the
Chameleon VQGAN. The picture carries the global change the WM anticipates
stamped onto the spatial template of the current frame -- qualitative sanity
check, not a pixel-accurate prediction.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw


IMG_START_TOK = 8197
IMG_END_TOK = 8196
IMG_HEADER_TOK = 8812
ROW_SEP_TOK = 8803


def extract_image_blocks(token_ids: list[int]) -> list[tuple[int, int, list[int]]]:
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
    if len(block_ids) < 3:
        return list(block_ids)
    i = 0
    while i < len(block_ids) and block_ids[i] == IMG_HEADER_TOK:
        i += 1
    payload = block_ids[i:]
    try:
        first_sep = payload.index(ROW_SEP_TOK)
    except ValueError:
        return payload
    side = first_sep
    if side <= 0:
        return payload
    clean: list[int] = []
    pos = 0
    while pos < len(payload):
        row_end = pos + side
        if row_end > len(payload):
            break
        clean.extend(payload[pos:row_end])
        if row_end < len(payload) and payload[row_end] == ROW_SEP_TOK:
            pos = row_end + 1
        else:
            pos = row_end
    return clean


def _decode_bpe_block_to_pil(
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


def _save_panel_strip(
    path: Path,
    panels: list[tuple[str, Image.Image | None]],
    cell_size: int = 256,
) -> None:
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


class WorldModelImageVisualizer:
    """Wraps the VQGAN decoder and broadcast-delta pipeline.

    One instance, loaded once at training start. Call ``visualize_batch`` after
    running the WM on a batch to get a list of panel strips.

    Args:
        vqgan_config_path: path to Chameleon VQGAN yaml (same as encoder uses).
        vqgan_ckpt_path:   path to Chameleon VQGAN checkpoint.
        encoder:           the frozen RynnVLA encoder instance (already on GPU).
        device:            where the VQGAN runs (should match encoder's device).
        which_block:       which image block in wm_obs_input_ids to visualise.
                           -2 picks the third-view of the current frame under
                           the his_2_third_view_wrist_w_state_10_256 layout.
    """

    def __init__(
        self,
        vqgan_config_path: str,
        vqgan_ckpt_path: str,
        encoder: Any,
        device: torch.device | str,
        which_block: int = -2,
    ) -> None:
        from src.utils.vq_image_decoder import build_bpe2vq_tensor, load_vq_model

        self.device = torch.device(device)
        self.which_block = int(which_block)

        self.vq_model = load_vq_model(
            cfg_path=vqgan_config_path,
            ckpt_path=vqgan_ckpt_path,
            device=self.device,
        )
        inner_chameleon = encoder.backbone.model
        vocab_mapping = inner_chameleon.vocabulary_mapping
        self.bpe2vq = build_bpe2vq_tensor(vocab_mapping).to(self.device)
        self.image_token_bpe_ids = torch.tensor(
            sorted(vocab_mapping.bpe2img.keys()),
            dtype=torch.long,
            device=self.device,
        )
        self._image_bpe_set = set(self.image_token_bpe_ids.tolist())
        self.lm_head = encoder.backbone.lm_head
        self.encoder = encoder

    @torch.no_grad()
    def _encode_pooled_and_full(
        self, input_ids_list: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels_list = [[-100] * len(seq) for seq in input_ids_list]
        lengths = [len(seq) for seq in input_ids_list]
        _, _, _, hidden_states, _, _, _ = self.encoder.backbone(
            input_ids=input_ids_list,
            labels=labels_list,
            training=True,
            output_hidden_states=True,
            att_mask=False,
        )
        mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        for idx, length in enumerate(lengths):
            if length > 0:
                mask[idx, :length] = True
        weights = mask.to(hidden_states.dtype).unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled.float(), hidden_states.float()

    @torch.no_grad()
    def _broadcast_delta_decode(
        self,
        current_pooled: torch.Tensor,      # [D]
        pred_next_pooled: torch.Tensor,    # [D]
        current_full_hidden: torch.Tensor, # [T, D]
        current_input_ids: list[int],
    ) -> tuple[Image.Image | None, Image.Image | None]:
        blocks = extract_image_blocks(current_input_ids)
        if not blocks:
            return None, None
        idx = self.which_block if self.which_block >= 0 else len(blocks) + self.which_block
        if idx < 0 or idx >= len(blocks):
            return None, None
        start, _end, block_ids = blocks[idx]

        keep_positions = [
            start + offset
            for offset, tok in enumerate(block_ids)
            if int(tok) in self._image_bpe_set
        ]
        if not keep_positions:
            return None, None
        img_hidden = current_full_hidden[torch.tensor(keep_positions, device=current_full_hidden.device)]

        delta = (pred_next_pooled - current_pooled).to(img_hidden.dtype)
        pred_img_hidden = img_hidden + delta.unsqueeze(0)

        lm_dtype = next(self.lm_head.parameters()).dtype
        logits = self.lm_head(pred_img_hidden.to(dtype=lm_dtype))          # [S*S, V]
        image_vocab_logits = logits[:, self.image_token_bpe_ids]            # [S*S, |img_vocab|]
        idx_in_img_vocab = image_vocab_logits.argmax(dim=-1)
        predicted_bpe = self.image_token_bpe_ids[idx_in_img_vocab].tolist()

        try:
            cur_pil = _decode_bpe_block_to_pil(block_ids, self.bpe2vq, self.vq_model)
        except Exception:
            cur_pil = None
        try:
            pred_pil = _decode_bpe_block_to_pil(predicted_bpe, self.bpe2vq, self.vq_model)
        except Exception:
            pred_pil = None
        return cur_pil, pred_pil

    @torch.no_grad()
    def visualize_batch(
        self,
        world_model: Any,
        wm_obs_input_ids: list[list[int]],
        wm_next_obs_input_ids: list[list[int]],
        action: torch.Tensor,
        out_dir: Path,
        tag: str,
        num_samples: int = 4,
    ) -> list[Path]:
        """Decode up to ``num_samples`` frames from the batch and save panel strips.

        Returns the list of written PNG paths so the workspace can surface them.
        """
        out_dir = Path(out_dir)
        n = min(int(num_samples), len(wm_obs_input_ids), len(wm_next_obs_input_ids))
        if n == 0:
            return []

        cur_ids = wm_obs_input_ids[:n]
        nxt_ids = wm_next_obs_input_ids[:n]

        cur_pooled, cur_full = self._encode_pooled_and_full(cur_ids)
        nxt_pooled, _ = self._encode_pooled_and_full(nxt_ids)

        wm_dtype = next(world_model.parameters()).dtype
        wm_device = next(world_model.parameters()).device
        pooled_b = cur_pooled.to(device=wm_device, dtype=wm_dtype)
        action_b = action[:n].to(device=wm_device, dtype=wm_dtype)
        pred_pooled_b = world_model.predict_next_hidden(pooled_b, action_b)
        pred_pooled = pred_pooled_b.float().to(cur_pooled.device)

        # Pure-latent decoder path (no current-frame anchor). Only available
        # if the WM has an image_decoder head.
        pred_image_hiddens_b = None
        if getattr(world_model, "image_decoder", None) is not None:
            pred_image_hiddens_b = world_model.decode_pooled_to_image_hiddens(pred_pooled_b)

        saved: list[Path] = []
        for i in range(n):
            cur_pil, pred_pil = self._broadcast_delta_decode(
                current_pooled=cur_pooled[i],
                pred_next_pooled=pred_pooled[i],
                current_full_hidden=cur_full[i],
                current_input_ids=cur_ids[i],
            )
            # Also decode the ground-truth next frame block if it exists -- gives
            # something for pred to be compared against on the panel. In the
            # pretokenize data the next_obs tokens are often just a prompt with
            # no image, in which case this panel will be None.
            gt_next_pil = None
            try:
                nxt_blocks = extract_image_blocks(nxt_ids[i])
                if nxt_blocks:
                    bidx = self.which_block if self.which_block >= 0 else len(nxt_blocks) + self.which_block
                    if 0 <= bidx < len(nxt_blocks):
                        gt_next_pil = _decode_bpe_block_to_pil(
                            nxt_blocks[bidx][2], self.bpe2vq, self.vq_model,
                        )
            except Exception:
                gt_next_pil = None

            # Pure-latent decoder panel (no current-frame anchor): decode
            # predicted per-image-token hiddens directly through lm_head.
            pred_latent_pil = None
            if pred_image_hiddens_b is not None:
                try:
                    pred_latent_pil = self._decode_image_hiddens_to_pil(
                        pred_image_hiddens_b[i]
                    )
                except Exception:
                    pred_latent_pil = None

            path = out_dir / f"{tag}_sample{i:02d}.png"
            _save_panel_strip(
                path,
                panels=[
                    ("gt current (vq)",      cur_pil),
                    ("gt next (vq)",         gt_next_pil),
                    ("pred next (bcast)",    pred_pil),
                    ("pred next (latent)",   pred_latent_pil),
                ],
            )
            saved.append(path)
        return saved

    @torch.no_grad()
    def _decode_image_hiddens_to_pil(
        self, pred_image_hiddens: torch.Tensor
    ) -> Image.Image:
        """Decode [n_img_tok, obs_dim] → PIL via lm_head + VQGAN. No anchor."""
        n = pred_image_hiddens.shape[0]
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(f"n_image_tokens={n} is not a perfect square")
        lm_dtype = next(self.lm_head.parameters()).dtype
        logits = self.lm_head(pred_image_hiddens.to(dtype=lm_dtype))        # [n, V]
        image_vocab_logits = logits[:, self.image_token_bpe_ids]             # [n, |img_vocab|]
        idx_in_img_vocab = image_vocab_logits.argmax(dim=-1)
        predicted_bpe = self.image_token_bpe_ids[idx_in_img_vocab].tolist()
        return _decode_bpe_block_to_pil(predicted_bpe, self.bpe2vq, self.vq_model)


__all__ = ["WorldModelImageVisualizer"]
