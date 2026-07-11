"""Reusable "decode WM prediction back to pixels" utility.

This supports qualitative rec-vs-gt comparison strips for routes that keep
Chameleon image-token context around the WM observation. The visualization is a
sanity check, not a pixel-accurate prediction: it projects a predicted hidden
delta back through the VLA language head and decodes image-token logits with
the Chameleon VQGAN.
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
    from dreamervla.utils.vq_image_decoder import tensor_to_pil, vq_tokens_to_pixels

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
        raise ValueError(
            f"image block contains {bad} non-image bpe ids after stripping"
        )
    pixels = vq_tokens_to_pixels(
        vq_ids.unsqueeze(0), vq_model, h_latent=side, w_latent=side
    )
    return tensor_to_pil(pixels[0])


def _safe_decode(
    decode_fn,
    *,
    what: str | None,
    sample: int,
    view: int | None = None,
) -> Image.Image | None:
    """Run ``decode_fn()`` and return its PIL, or ``None`` on any exception.

    When ``what`` is given, a failure prints the standardized
    ``[viz] {what} decode failed ...`` diagnostic line (matching the existing
    per-route messages, including the optional ``view`` segment). When ``what``
    is ``None`` the failure is swallowed silently, matching the routes that
    previously caught without printing.
    """
    try:
        return decode_fn()
    except Exception as exc:  # noqa: BLE001 - diagnostic-only viz path
        if what is not None:
            view_seg = f" view {view}" if view is not None else ""
            print(f"[viz] {what} decode failed for sample {sample}{view_seg}: {exc}")
        return None


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
            draw.rectangle(
                [x0, header, x0 + cell_size, header + cell_size], fill=(60, 20, 20)
            )
            draw.text(
                (x0 + 10, header + cell_size // 2), "(missing)", fill=(230, 230, 230)
            )
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
        encoder:           the frozen VLA encoder instance (already on GPU).
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
        which_blocks: list[int] | tuple[int, ...] | None = None,
        which_block_labels: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        from dreamervla.utils.vq_image_decoder import build_bpe2vq_tensor, load_vq_model

        self.device = torch.device(device)
        self.which_block = int(which_block)
        self.which_blocks = (
            [int(block_idx) for block_idx in which_blocks]
            if which_blocks is not None
            else [self.which_block]
        )
        if not self.which_blocks:
            raise ValueError("which_blocks must contain at least one image block index")
        if which_block_labels is not None:
            self.which_block_labels = [str(label) for label in which_block_labels]
            if len(self.which_block_labels) != len(self.which_blocks):
                raise ValueError(
                    "which_block_labels length must match which_blocks length: "
                    f"{len(self.which_block_labels)} vs {len(self.which_blocks)}"
                )
        else:
            self.which_block_labels = [
                self._default_view_label(block_idx, view_idx)
                for view_idx, block_idx in enumerate(self.which_blocks)
            ]

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
        vocab_map = (
            getattr(getattr(encoder.backbone, "config", None), "vocabulary_map", {})
            or {}
        )
        self.state_start_token_id = int(vocab_map.get("<reserved15500>", 15504))
        self.state_end_token_id = int(vocab_map.get("<reserved16000>", 16004))

    @staticmethod
    def _default_view_label(block_idx: int, view_idx: int) -> str:
        if int(block_idx) == -2:
            return "third"
        if int(block_idx) == -1:
            return "wrist"
        return f"view{view_idx}"

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
        mask = torch.zeros(
            hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
        )
        for idx, length in enumerate(lengths):
            if length > 0:
                mask[idx, :length] = True
        weights = mask.to(hidden_states.dtype).unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(
            1.0
        )
        return pooled.float(), hidden_states.float()

    @torch.no_grad()
    def _broadcast_delta_decode(
        self,
        current_pooled: torch.Tensor,  # [D]
        pred_next_pooled: torch.Tensor,  # [D]
        current_full_hidden: torch.Tensor,  # [T, D]
        current_input_ids: list[int],
    ) -> tuple[Image.Image | None, Image.Image | None]:
        blocks = extract_image_blocks(current_input_ids)
        if not blocks:
            return None, None
        idx = (
            self.which_block
            if self.which_block >= 0
            else len(blocks) + self.which_block
        )
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
        img_hidden = current_full_hidden[
            torch.tensor(keep_positions, device=current_full_hidden.device)
        ]

        delta = (pred_next_pooled - current_pooled).to(img_hidden.dtype)
        pred_img_hidden = img_hidden + delta.unsqueeze(0)

        lm_dtype = next(self.lm_head.parameters()).dtype
        logits = self.lm_head(pred_img_hidden.to(dtype=lm_dtype))  # [S*S, V]
        image_vocab_logits = logits[:, self.image_token_bpe_ids]  # [S*S, |img_vocab|]
        idx_in_img_vocab = image_vocab_logits.argmax(dim=-1)
        predicted_bpe = self.image_token_bpe_ids[idx_in_img_vocab].tolist()

        cur_pil = _safe_decode(
            lambda: _decode_bpe_block_to_pil(block_ids, self.bpe2vq, self.vq_model),
            what=None,
            sample=0,
        )
        pred_pil = _safe_decode(
            lambda: _decode_bpe_block_to_pil(
                predicted_bpe, self.bpe2vq, self.vq_model
            ),
            what=None,
            sample=0,
        )
        return cur_pil, pred_pil

    @torch.no_grad()
    def _encode_per_image_token(
        self,
        input_ids_list: list[list[int]],
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """Run VLA backbone, then index the image-block's per-token hiddens.
        Returns ([B, N_img, C], list_of_block_ids).
        """
        labels_list = [[-100] * len(seq) for seq in input_ids_list]
        _, _, _, hidden_states, _, _, _ = self.encoder.backbone(
            input_ids=input_ids_list,
            labels=labels_list,
            training=True,
            output_hidden_states=True,
            att_mask=False,
        )
        per_sample: list[torch.Tensor] = []
        block_ids_list: list[list[int]] = []
        for idx, seq in enumerate(input_ids_list):
            blocks = extract_image_blocks(list(seq))
            if not blocks:
                raise ValueError(f"viz: sample {idx} has no image block")
            bidx = (
                self.which_block
                if self.which_block >= 0
                else len(blocks) + self.which_block
            )
            if not (0 <= bidx < len(blocks)):
                raise ValueError(f"viz: which_block={self.which_block} out of range")
            start, _end, block_ids = blocks[bidx]
            pos = [
                start + off
                for off, tok in enumerate(block_ids)
                if int(tok) in self._image_bpe_set
            ]
            pos_t = torch.tensor(pos, device=hidden_states.device)
            per_sample.append(hidden_states[idx].index_select(0, pos_t))
            block_ids_list.append(block_ids)
        return torch.stack(per_sample, dim=0).float(), block_ids_list

    def _extract_state_tokens(
        self,
        input_ids_list: list[list[int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[list[int]] = []
        max_len = 0
        for seq in input_ids_list:
            tokens = [int(tok) for tok in seq]
            state_tokens: list[int] = []
            idx = 0
            while idx < len(tokens):
                if tokens[idx] != self.state_start_token_id:
                    idx += 1
                    continue
                end = idx + 1
                while end < len(tokens) and tokens[end] != self.state_end_token_id:
                    end += 1
                if end < len(tokens):
                    state_tokens.extend(tokens[idx : end + 1])
                    idx = end + 1
                else:
                    state_tokens.extend(tokens[idx:])
                    break
            rows.append(state_tokens)
            max_len = max(max_len, len(state_tokens))

        ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
        mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
        for idx, row in enumerate(rows):
            if not row:
                continue
            row_t = torch.tensor(row, dtype=torch.long, device=device)
            ids[idx, : row_t.numel()] = row_t
            mask[idx, : row_t.numel()] = True
        return ids, mask

    def _decode_gt_next(
        self,
        nxt_seq: list[int],
        which_block: int,
        *,
        sample: int,
        view: int | None = None,
        what: str | None,
    ) -> Image.Image | None:
        """Decode the gt-next image block selected by ``which_block``.

        Returns ``None`` when the next frame has no matching block (or the
        index is out of range), and routes any decode failure through
        ``_safe_decode`` so the diagnostic/silent behavior matches the caller.
        """

        def _decode() -> Image.Image | None:
            nxt_blocks = extract_image_blocks(nxt_seq)
            if not nxt_blocks:
                return None
            bidx = which_block if which_block >= 0 else len(nxt_blocks) + which_block
            if not (0 <= bidx < len(nxt_blocks)):
                return None
            return _decode_bpe_block_to_pil(
                nxt_blocks[bidx][2],
                self.bpe2vq,
                self.vq_model,
            )

        return _safe_decode(_decode, what=what, sample=sample, view=view)

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

        Two rendering modes depending on the WM:
          - Route-B (spatial_codec=True): per-image-token hidden → conv stem
            → RSSM → conv deconv → lm_head → VQGAN.  Panels: cur / gt_next /
            pred_latent.
          - Route-0: hidden-delta visualization fallback
            and optional MLP image_decoder panel.
        """
        out_dir = Path(out_dir)
        n = min(int(num_samples), len(wm_obs_input_ids), len(wm_next_obs_input_ids))
        if n == 0:
            return []

        cur_ids = wm_obs_input_ids[:n]
        nxt_ids = wm_next_obs_input_ids[:n]

        wm_dtype = next(world_model.parameters()).dtype
        wm_device = next(world_model.parameters()).device
        action_b = action[:n].to(device=wm_device, dtype=wm_dtype)
        spatial_codec = bool(getattr(world_model, "spatial_codec", False))
        io_mode = str(getattr(world_model, "io_mode", "hidden"))

        if io_mode == "token":
            # Token-mode viz: WM predicts image BPE ids directly.  No
            # Chameleon forward, no lm_head — just decode gt current / gt
            # next / pred-next token ids via VQGAN.
            cur_block_ids: list[list[list[int]]] = []
            for sample_idx, seq in enumerate(cur_ids):
                blocks = extract_image_blocks(list(seq))
                selected: list[list[int]] = []
                for which_block in self.which_blocks:
                    bidx = (
                        which_block if which_block >= 0 else len(blocks) + which_block
                    )
                    if not (0 <= bidx < len(blocks)):
                        raise ValueError(
                            f"viz: sample {sample_idx} which_block={which_block} out of range"
                        )
                    selected.append(blocks[bidx][2])
                cur_block_ids.append(selected)
            cur_bpe = torch.tensor(
                [
                    [
                        int(t)
                        for block in sample_blocks
                        for t in block
                        if int(t) in self._image_bpe_set
                    ]
                    for sample_blocks in cur_block_ids
                ],
                dtype=torch.long,
                device=wm_device,
            )
            state_ids = state_mask = None
            if bool(getattr(world_model, "state_conditioning", False)):
                state_ids, state_mask = self._extract_state_tokens(cur_ids, wm_device)
            pred_bpe = world_model.predict_next_image_token_ids(
                cur_bpe,
                action_b,
                state_token_ids=state_ids,
                state_token_mask=state_mask,
            )  # [B, N_img]

            saved: list[Path] = []
            for i in range(n):
                panels: list[tuple[str, Image.Image | None]] = []
                offset = 0
                for view_idx, block_ids in enumerate(cur_block_ids[i]):
                    cur_pil = _safe_decode(
                        lambda block_ids=block_ids: _decode_bpe_block_to_pil(
                            block_ids,
                            self.bpe2vq,
                            self.vq_model,
                        ),
                        what="gt_cur",
                        sample=i,
                        view=view_idx,
                    )

                    gt_next_pil = self._decode_gt_next(
                        nxt_ids[i],
                        self.which_blocks[view_idx],
                        sample=i,
                        view=view_idx,
                        what="gt_next",
                    )

                    n_block_tokens = sum(
                        1 for tok in block_ids if int(tok) in self._image_bpe_set
                    )
                    pred_ids_i = pred_bpe[i, offset : offset + n_block_tokens].tolist()
                    offset += n_block_tokens
                    pred_next_pil = _safe_decode(
                        lambda pred_ids_i=pred_ids_i: _decode_bpe_block_to_pil(
                            pred_ids_i,
                            self.bpe2vq,
                            self.vq_model,
                        ),
                        what="pred_next",
                        sample=i,
                        view=view_idx,
                    )

                    label = self.which_block_labels[view_idx]
                    panels.extend(
                        [
                            (f"{label} cur", cur_pil),
                            (f"{label} gt_next", gt_next_pil),
                            (f"{label} pred", pred_next_pil),
                        ]
                    )

                path = out_dir / f"{tag}_sample{i:02d}.png"
                _save_panel_strip(path, panels=panels)
                saved.append(path)
            return saved

        if spatial_codec:
            cur_img_hiddens, cur_block_ids = self._encode_per_image_token(cur_ids)
            cur_img_hiddens_wm = cur_img_hiddens.to(device=wm_device, dtype=wm_dtype)
            # predict_next_hidden accepts [B, N_img, C_in] under spatial_codec
            pred_pooled_b = world_model.predict_next_hidden(
                cur_img_hiddens_wm, action_b
            )
            pred_image_hiddens_b = world_model.decode_pooled_to_image_hiddens(
                pred_pooled_b
            )
            # Keep on lm_head's device — _decode_image_hiddens_to_pil needs
            # to run lm_head which lives on GPU.
            lm_device = next(self.lm_head.parameters()).device
            pred_image_hiddens = pred_image_hiddens_b.to(device=lm_device)

            saved: list[Path] = []
            for i in range(n):
                # gt current (from raw block ids of current frame)
                cur_pil = _safe_decode(
                    lambda i=i: _decode_bpe_block_to_pil(
                        cur_block_ids[i],
                        self.bpe2vq,
                        self.vq_model,
                    ),
                    what="gt_cur",
                    sample=i,
                )
                # gt next (from raw block ids of next frame, if any)
                gt_next_pil = self._decode_gt_next(
                    nxt_ids[i],
                    self.which_block,
                    sample=i,
                    what="gt_next",
                )
                # predicted next from WM conv deconv
                pred_latent_pil = _safe_decode(
                    lambda i=i: self._decode_image_hiddens_to_pil(
                        pred_image_hiddens[i]
                    ),
                    what="pred_latent",
                    sample=i,
                )

                path = out_dir / f"{tag}_sample{i:02d}.png"
                _save_panel_strip(
                    path,
                    panels=[
                        ("gt current (vq)", cur_pil),
                        ("gt next (vq)", gt_next_pil),
                        ("pred next (latent)", pred_latent_pil),
                    ],
                )
                saved.append(path)
            return saved

        # ── Route-0 fallback path ────────────────────────────────────────────────
        cur_pooled, cur_full = self._encode_pooled_and_full(cur_ids)
        nxt_pooled, _ = self._encode_pooled_and_full(nxt_ids)

        pooled_b = cur_pooled.to(device=wm_device, dtype=wm_dtype)
        pred_pooled_b = world_model.predict_next_hidden(pooled_b, action_b)
        pred_pooled = pred_pooled_b.float().to(cur_pooled.device)

        pred_image_hiddens_b = None
        if getattr(world_model, "image_decoder", None) is not None:
            pred_image_hiddens_b = world_model.decode_pooled_to_image_hiddens(
                pred_pooled_b
            )

        saved: list[Path] = []
        for i in range(n):
            cur_pil, pred_pil = self._broadcast_delta_decode(
                current_pooled=cur_pooled[i],
                pred_next_pooled=pred_pooled[i],
                current_full_hidden=cur_full[i],
                current_input_ids=cur_ids[i],
            )
            gt_next_pil = self._decode_gt_next(
                nxt_ids[i],
                self.which_block,
                sample=i,
                what=None,
            )

            pred_latent_pil = None
            if pred_image_hiddens_b is not None:
                pred_latent_pil = _safe_decode(
                    lambda i=i: self._decode_image_hiddens_to_pil(
                        pred_image_hiddens_b[i]
                    ),
                    what=None,
                    sample=i,
                )

            path = out_dir / f"{tag}_sample{i:02d}.png"
            _save_panel_strip(
                path,
                panels=[
                    ("gt current (vq)", cur_pil),
                    ("gt next (vq)", gt_next_pil),
                    ("pred next (bcast)", pred_pil),
                    ("pred next (latent)", pred_latent_pil),
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
        logits = self.lm_head(pred_image_hiddens.to(dtype=lm_dtype))  # [n, V]
        image_vocab_logits = logits[:, self.image_token_bpe_ids]  # [n, |img_vocab|]
        idx_in_img_vocab = image_vocab_logits.argmax(dim=-1)
        predicted_bpe = self.image_token_bpe_ids[idx_in_img_vocab].tolist()
        return _decode_bpe_block_to_pil(predicted_bpe, self.bpe2vq, self.vq_model)


__all__ = ["WorldModelImageVisualizer"]
