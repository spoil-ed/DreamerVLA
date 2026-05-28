from __future__ import annotations

from typing import Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.n_bins = int(bins)
        self.min_action = min_action
        self.max_action = max_action
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, list[str]]:
        action = np.clip(
            action, a_min=float(self.min_action), a_max=float(self.max_action)
        )
        discretized_action = np.digitize(action, self.bins)
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(
                list(self.tokenizer.vocab_size - discretized_action)
            )
        return self.tokenizer.batch_decode(
            (self.tokenizer.vocab_size - discretized_action).tolist()
        )

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins


__all__ = ["ActionTokenizer"]
