"""WM image-token mapping helpers for EmbodiedEvalRunner.

Cohesive, closed group extracted from embodied_eval_runner.py (P3 god-file split,
mixin route): world-model IO-mode probing + image-BPE token extraction/mapping. These
methods call only each other and ``self`` attributes/models (no other runner methods),
so they live on a sibling mixin the runner inherits — zero call-site change.
Behaviour-preserving.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf


class EmbodiedEvalImageTokenMixin:
    def _attach_image_token_mapping(self) -> None:
        wm = getattr(self, "_unwrapped_world_model", None) or self.world_model
        if (
            wm is None
            or not getattr(wm, "spatial_codec", False)
            or self.encoder is None
        ):
            return
        lm_head = self.encoder.backbone.lm_head
        vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
        image_token_bpe_ids = torch.tensor(
            sorted(vocab_mapping.bpe2img.keys()), dtype=torch.long
        )
        full_vocab_size = int(lm_head.weight.shape[0])
        wm_io_mode = str(getattr(wm, "io_mode", "hidden"))
        wm.attach_lm_head(
            lm_head if wm_io_mode == "hidden" else None,
            image_token_bpe_ids,
            full_vocab_size=full_vocab_size,
        )
        if self.distributed.is_main_process:
            tag = "lm_head" if wm_io_mode == "hidden" else "vocab (token mode)"
            print(f"  [Eval] attached {tag} for image-token mapping.")

    def _wm_io_mode(self) -> str:
        wm = getattr(self, "_unwrapped_world_model", None) or getattr(
            self, "world_model", None
        )
        if wm is None:
            return "hidden"
        explicit = getattr(wm, "io_mode", None)
        if explicit is not None:
            return str(explicit)
        encoder = getattr(wm, "encoder", None)
        if (
            encoder is not None
            and encoder.__class__.__name__ == "DreamerV3TokenEncoder"
        ):
            return "token"
        return "hidden"

    def _wm_expects_image_vocab_tokens(self) -> bool:
        wm = getattr(self, "_unwrapped_world_model", None) or getattr(
            self, "world_model", None
        )
        encoder = getattr(wm, "encoder", None)
        return (
            encoder is not None
            and encoder.__class__.__name__ == "DreamerV3TokenEncoder"
        )

    def _wm_expects_pixel_images(self) -> bool:
        wm = getattr(self, "_unwrapped_world_model", None) or getattr(
            self, "world_model", None
        )
        encoder = getattr(wm, "encoder", None)
        return (
            encoder is not None
            and encoder.__class__.__name__ == "DreamerV3PixelEncoder"
        )

    def _get_image_bpe_set(self) -> set[int]:
        cached = getattr(self, "_image_bpe_set_cache", None)
        if cached is not None:
            return cached
        vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
        self._image_bpe_set_cache = set(vocab_mapping.bpe2img.keys())
        return self._image_bpe_set_cache

    def _extract_image_bpe_ids(self, input_ids_list: list[list[int]]) -> torch.Tensor:
        from dreamervla.utils.wm_image_viz import extract_image_blocks

        wm = getattr(self, "_unwrapped_world_model", None) or self.world_model
        wm_encoder = getattr(wm, "encoder", None)
        n_img_tok = int(
            getattr(wm, "n_image_tokens", getattr(wm_encoder, "n_image_tokens", 256))
        )
        which_blocks_cfg = OmegaConf.select(
            self.cfg, "eval.dreamer_which_image_blocks", default=None
        )
        if which_blocks_cfg is None:
            which_blocks = [
                int(
                    OmegaConf.select(
                        self.cfg, "eval.dreamer_which_image_block", default=-2
                    )
                )
            ]
        else:
            which_blocks = [int(item) for item in which_blocks_cfg]
        img_bpe = self._get_image_bpe_set()
        bpe2img = None
        if self._wm_expects_image_vocab_tokens():
            bpe2img = self.encoder.backbone.model.vocabulary_mapping.bpe2img
        rows: list[list[int]] = []
        for idx, seq in enumerate(input_ids_list):
            blocks = extract_image_blocks(list(seq))
            if not blocks:
                raise ValueError(
                    f"rollout sample {idx}: no image block found in tokens"
                )
            tok_ids: list[int] = []
            for which_block in which_blocks:
                bidx = which_block if which_block >= 0 else len(blocks) + which_block
                if not (0 <= bidx < len(blocks)):
                    raise ValueError(
                        f"rollout sample {idx}: image block {which_block} out of range"
                    )
                _start, _end, block_ids = blocks[bidx]
                tok_ids.extend(int(tok) for tok in block_ids if int(tok) in img_bpe)
            if len(tok_ids) != n_img_tok:
                raise ValueError(
                    f"rollout sample {idx}: image blocks {which_blocks} have {len(tok_ids)} image tokens, expected {n_img_tok}"
                )
            if bpe2img is not None:
                tok_ids = [int(bpe2img[int(tok)]) for tok in tok_ids]
            rows.append(tok_ids)
        return torch.tensor(rows, dtype=torch.long, device=self.device)
