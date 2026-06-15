"""Iterable dataset over legacy libero_goal classifier shards.

Mirrors WMPO/reward_model/videomae.py::SuccessWindowDataset windowing
semantics: per episode emit 1 positive (last W frames) and 1 random
earlier-window negative. Validation: positive at end, all earlier windows
labeled negative at stride.

Shard schema (built by `python -m dreamervla.legacy.build_classifier_shards_from_demos`):
    <key>.latent.npy  shape [T, latent_dim], float16/float32
    <key>.meta.json   {"finish_step": int, "complete": bool, ...}
"""

from __future__ import annotations

import io
import json
import random
from collections.abc import Iterable, Iterator

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset


class LatentSuccessShardDataset(IterableDataset):
    def __init__(
        self,
        shard_globs: list[str],
        window: int = 8,
        stride: int = 8,
        mode: str = "train",
        use_resample: bool = True,
        shuffle_buf: int = 256,
        min_episode_len: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError(f"mode must be train/val, got {mode}")
        self.window = int(window)
        self.stride = int(stride)
        self.mode = mode
        self.min_episode_len = int(max(self.window, min_episode_len or self.window))

        if mode == "train":
            shard_source = (
                wds.ResampledShards(shard_globs, seed=42)
                if use_resample
                else wds.SimpleShardList(shard_globs)
            )
            pipeline = [
                shard_source,
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.to_tuple("latent.npy", "meta.json"),
                self._windows_train,
                wds.shuffle(shuffle_buf, initial=shuffle_buf),
            ]
        else:
            pipeline = [
                wds.SimpleShardList(shard_globs),
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.to_tuple("latent.npy", "meta.json"),
                self._windows_val,
            ]
        self.pipeline = wds.DataPipeline(*pipeline)

    def __iter__(self) -> Iterator:
        return iter(self.pipeline)

    @staticmethod
    def _load(latent_b: bytes, meta_b: bytes):
        latent = np.load(io.BytesIO(latent_b))  # [T, D]
        meta = json.loads(meta_b.decode())
        return latent, meta

    def _to_tensor(self, window: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(window.astype(np.float32))

    def _windows_train(self, stream: Iterable):
        W, S = self.window, self.stride
        for latent_b, meta_b in stream:
            latent, meta = self._load(latent_b, meta_b)
            finish_step = int(meta["finish_step"])
            T_valid = finish_step + 1  # frames 0..finish_step inclusive
            complete = bool(meta.get("complete", True))
            if T_valid < self.min_episode_len:
                continue

            end = T_valid
            yield self._to_tensor(latent[end - W : end]), int(complete)

            if T_valid - S >= W:
                neg_ends = list(range(T_valid - S, W - 1, -S))
                if not neg_ends:
                    neg_ends = list(range(T_valid - 1, W - 1, -1))
                if neg_ends:
                    end = random.choice(neg_ends)
                    yield self._to_tensor(latent[end - W : end]), 0

    def _windows_val(self, stream: Iterable):
        W, S = self.window, self.stride
        for latent_b, meta_b in stream:
            latent, meta = self._load(latent_b, meta_b)
            finish_step = int(meta["finish_step"])
            T_valid = finish_step + 1
            complete = bool(meta.get("complete", True))
            if T_valid < self.min_episode_len:
                continue
            end = T_valid
            yield self._to_tensor(latent[end - W : end]), int(complete)
            for end in range(T_valid - S, W - 1, -S):
                yield self._to_tensor(latent[end - W : end]), 0
