from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    return {
        key: tree_map(fn, value) if isinstance(value, dict) else fn(value)
        for key, value in tree.items()
    }


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    return {
        key: tree_map_with_key(fn, value, (*keys, key))
        if isinstance(value, dict)
        else fn((*keys, key), value)
        for key, value in tree.items()
    }


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[dict]) -> dict:
        if self.padding_side != "right":
            raise AssertionError(
                f"Invalid Tokenizer padding_side={self.padding_side!r}"
            )
        input_ids = pad_sequence(
            [instance["input_ids"] for instance in instances],
            batch_first=True,
            padding_value=self.pad_token_id,
        )[:, : self.model_max_length]
        labels = pad_sequence(
            [instance["labels"] for instance in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        pixel_values = [instance["pixel_values"] for instance in instances]
        if not all(value is not None for value in pixel_values):
            raise AssertionError("Invalid VLA example with pixel_values=None")
        if not isinstance(pixel_values[0], torch.Tensor):
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values[0])}")
        if "pixel_values_wrist" in instances[0]:
            pixel_values = torch.cat(
                (
                    torch.stack(pixel_values),
                    torch.stack(
                        [instance["pixel_values_wrist"] for instance in instances]
                    ),
                ),
                dim=1,
            )
        else:
            pixel_values = torch.stack(pixel_values)

        actions = torch.stack(
            [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        )
        proprio = None
        if "proprio" in instances[0]:
            proprio = torch.tensor(
                np.squeeze(np.stack([instance["proprio"] for instance in instances]))
            )

        output = {
            "pixel_values": pixel_values,
            "proprio": proprio,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
        }
        if "dataset_name" in instances[0]:
            output["dataset_names"] = [
                instance["dataset_name"] for instance in instances
            ]
        return output


__all__ = ["PaddedCollatorForActionPrediction", "tree_map", "tree_map_with_key"]
