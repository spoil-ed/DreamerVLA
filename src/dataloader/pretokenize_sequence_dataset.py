"""Compatibility wrapper for the unified pretokenized dataset.

Sequence-window support now lives in ``pretokenize_dataset.PretokenizeDataset``.
This module keeps old imports working while avoiding a second implementation.
"""

from __future__ import annotations

from src.dataloader.pretokenize_dataset import PretokenizeDataSpec, PretokenizeDataset

PretokenizeSequenceDataSpec = PretokenizeDataSpec
PretokenizeSequenceDataset = PretokenizeDataset

__all__ = ["PretokenizeSequenceDataSpec", "PretokenizeSequenceDataset"]
