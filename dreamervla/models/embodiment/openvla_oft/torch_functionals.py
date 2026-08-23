# Copyright 2025 The DreamerVLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch
import torch.nn.functional as F


def compute_logprobs_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    op_type: str = "torch",
) -> torch.Tensor:
    if op_type != "torch":
        raise ValueError(
            "Only op_type='torch' is supported by the vendored DreamerVLA OpenVLA-OFT path."
        )
    batch_dim = logits.shape[:-1]
    last_dim = logits.shape[-1]
    logprobs = -F.cross_entropy(
        logits.reshape(-1, last_dim),
        target.reshape(-1),
        reduction="none",
    )
    return logprobs.view(*batch_dim).float()


def compute_entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=dim)
    p = logp.exp()
    entropy_term = torch.where(p > 0, p * logp, 0.0)
    return -entropy_term.sum(dim=dim)


def pad_tensor_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (
        (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    )
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)
