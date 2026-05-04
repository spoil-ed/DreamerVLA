from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from src.dataloader.base_dataset import BaseDataset
from src.dataloader.pretokenize_dataset import PretokenizeDataset
from src.utils.wm_image_viz import extract_image_blocks


_IMG_TOKEN_NAME_RE = re.compile(r"^IMGIMG(.+)$")


@dataclass(frozen=True)
class LIBEROTokenSequenceSpec:
    config_path: str
    vocab_config_path: str
    num_windows: int
    sequence_length: int
    action_dim: int
    n_image_tokens: int
    num_views: int
    tokens_per_view: int
    num_image_tokens_vocab: int
    spatial_grid: tuple[int, int]
    which_blocks: tuple[int, ...]


def _image_name_to_index(name: str) -> int | None:
    match = _IMG_TOKEN_NAME_RE.match(name)
    if match is None:
        return None
    # Chameleon image-token names encode decimal digits as A..J.
    suffix = match.group(1)
    chars = {chr(ord("A") + idx): str(idx) for idx in range(10)}
    decoded = "".join(chars.get(ch, ch) for ch in suffix)
    # Some token names carry a trailing marker; keep only the numeric body.
    digits = "".join(ch for ch in decoded if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _load_bpe2img(vocab_config_path: str | Path) -> dict[int, int]:
    path = Path(vocab_config_path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    vocab_map = cfg.get("vocabulary_map", cfg)
    if not isinstance(vocab_map, dict):
        raise ValueError(f"{path} does not contain a vocabulary_map dict")
    mapping: dict[int, int] = {}
    for name, bpe_id in vocab_map.items():
        if not isinstance(name, str) or not name.startswith("IMGIMG"):
            continue
        idx = _image_name_to_index(name)
        if idx is not None:
            mapping[int(bpe_id)] = int(idx)
    if not mapping:
        raise ValueError(f"No IMGIMG entries found in {path}")
    return mapping


class LIBEROTokenSequenceDataset(BaseDataset):
    """DreamerV3-style token observation windows for LIBERO.

    This wraps the existing pretokenized sequence dataset and extracts the
    selected Chameleon image blocks as image-token indices:

      tokens:   [T, V, N] long, values in [0, num_image_tokens_vocab)
      actions:  [T, A] previous-action convention, actions[0] is zero
      rewards:  [T]
      dones:    [T]
      is_terminal: [T]
      is_last:  [T]
      is_first: [T], true for the first item of the sampled window

    The default two selected blocks are third view and wrist view. They are
    kept as separate views here because DreamerV3 concatenates multiple image
    observation keys along channels before the CNN, rather than stacking them
    into one taller spatial image.
    """

    def __init__(
        self,
        config_path: str | Path,
        vocab_config_path: str | Path,
        history: int = 1,
        batch_length: int = 7,
        replay_context: int = 1,
        sequence_length: int | None = None,
        stride: int = 1,
        sequence_next_obs_source: str = "current_obs",
        n_image_tokens: int = 512,
        spatial_grid: Sequence[int] = (16, 16),
        which_blocks: Sequence[int] = (-2, -1),
        num_image_tokens_vocab: int | None = None,
        max_windows: int | None = None,
    ) -> None:
        super().__init__()
        self.config_path = self.resolve_project_path(config_path)
        self.vocab_config_path = self.resolve_project_path(vocab_config_path)
        self.n_image_tokens = int(n_image_tokens)
        self.spatial_grid = (int(spatial_grid[0]), int(spatial_grid[1]))
        self.which_blocks = tuple(int(x) for x in which_blocks)
        if not self.which_blocks:
            raise ValueError("which_blocks must contain at least one image block")
        self.num_views = len(self.which_blocks)
        self.tokens_per_view = self.spatial_grid[0] * self.spatial_grid[1]
        expected_total = self.num_views * self.tokens_per_view
        if self.n_image_tokens != expected_total:
            raise ValueError(
                f"n_image_tokens={self.n_image_tokens} must equal "
                f"len(which_blocks) * spatial_grid product = {expected_total}"
            )
        self.n_image_tokens_per_block = self.tokens_per_view
        self.bpe2img = _load_bpe2img(self.vocab_config_path)
        self.img_bpe_set = set(self.bpe2img)
        inferred_vocab = max(self.bpe2img.values()) + 1
        self.num_image_tokens_vocab = (
            inferred_vocab if num_image_tokens_vocab is None else int(num_image_tokens_vocab)
        )
        if self.num_image_tokens_vocab < inferred_vocab:
            raise ValueError(
                f"num_image_tokens_vocab={self.num_image_tokens_vocab} is smaller "
                f"than inferred image vocab size {inferred_vocab}"
            )

        self.source = PretokenizeDataset(
            config_path=self.config_path,
            history=history,
            batch_length=batch_length,
            replay_context=replay_context,
            sequence_length=sequence_length,
            stride=stride,
            sequence_next_obs_source=sequence_next_obs_source,
        )
        self.max_windows = None if max_windows is None else int(max_windows)
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError("max_windows must be positive when provided")
        self.action_dim = int(getattr(self.source, "action_dim", 0) or 0)
        self._spec = LIBEROTokenSequenceSpec(
            config_path=str(self.config_path),
            vocab_config_path=str(self.vocab_config_path),
            num_windows=len(self),
            sequence_length=int(self.source.data_spec.sequence_length or 0),
            action_dim=self.action_dim,
            n_image_tokens=self.n_image_tokens,
            num_views=self.num_views,
            tokens_per_view=self.tokens_per_view,
            num_image_tokens_vocab=self.num_image_tokens_vocab,
            spatial_grid=self.spatial_grid,
            which_blocks=self.which_blocks,
        )

    @property
    def data_spec(self) -> LIBEROTokenSequenceSpec:
        return self._spec

    def get_normalizer(self) -> dict[str, Any]:
        return {}

    def __len__(self) -> int:
        n = len(self.source)
        return n if self.max_windows is None else min(n, self.max_windows)

    def _extract_image_indices(self, input_ids: list[int]) -> torch.Tensor:
        blocks = extract_image_blocks([int(x) for x in input_ids])
        if not blocks:
            raise ValueError("sample contains no image block")
        views: list[torch.Tensor] = []
        for which_block in self.which_blocks:
            block_idx = which_block if which_block >= 0 else len(blocks) + which_block
            if not (0 <= block_idx < len(blocks)):
                raise ValueError(
                    f"which_block={which_block} out of range for {len(blocks)} image blocks"
                )
            _start, _end, block_ids = blocks[block_idx]
            image_bpe = [int(tok) for tok in block_ids if int(tok) in self.img_bpe_set]
            if len(image_bpe) != self.n_image_tokens_per_block:
                raise ValueError(
                    f"block {which_block} has {len(image_bpe)} image tokens, "
                    f"expected {self.n_image_tokens_per_block}"
                )
            views.append(torch.tensor([self.bpe2img[tok] for tok in image_bpe], dtype=torch.long))
        return torch.stack(views, dim=0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.source[int(index)]
        seq_ids = item.get("wm_obs_input_ids_seq")
        if not isinstance(seq_ids, list):
            raise KeyError("Underlying PretokenizeDataset did not return wm_obs_input_ids_seq")
        tokens = torch.stack(
            [self._extract_image_indices([int(x) for x in step_ids]) for step_ids in seq_ids],
            dim=0,
        )
        actions = item["action_seq"].to(dtype=torch.float32)
        rewards = item["reward_seq"].to(dtype=torch.float32)
        dones = item["done_seq"].to(dtype=torch.float32)
        is_first = torch.zeros(tokens.shape[0], dtype=torch.bool)
        is_first[0] = True
        is_terminal = dones.to(dtype=torch.bool)
        is_last = torch.zeros_like(is_terminal)
        return {
            "tokens": tokens,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "is_terminal": is_terminal,
            "is_last": is_last,
            "is_first": is_first,
            "meta_seq": item.get("meta_seq", []),
            "file": item.get("file", ""),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "tokens": torch.stack([item["tokens"] for item in batch], dim=0),
            "actions": torch.stack([item["actions"] for item in batch], dim=0),
            "rewards": torch.stack([item["rewards"] for item in batch], dim=0),
            "dones": torch.stack([item["dones"] for item in batch], dim=0),
            "is_terminal": torch.stack([item["is_terminal"] for item in batch], dim=0),
            "is_last": torch.stack([item["is_last"] for item in batch], dim=0),
            "is_first": torch.stack([item["is_first"] for item in batch], dim=0),
            "meta_seq": [item.get("meta_seq", []) for item in batch],
            "file": [item.get("file", "") for item in batch],
        }


__all__ = ["LIBEROTokenSequenceDataset", "LIBEROTokenSequenceSpec"]
