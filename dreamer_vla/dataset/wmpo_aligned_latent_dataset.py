"""WMPO-aligned latent W-frame classifier dataset.

Mirrors ``WMPO/reward_model/videomae.py::SuccessWindowDataset`` exactly,
but operates on **precomputed real pi0 action-hidden sidecar HDF5** instead
of raw video frames. The positive/negative protocol is intentionally
identical so that downstream eval thresholds (``WMPO/reward_model/find_thre.py``,
``WMPO/verl/.../robwm_rollout.py::predict_success``) transfer 1:1.

Per-demo (``finish_step = T``, ``complete ∈ {True, False}``, ``W = window``, ``S = stride``):

    Train (mirrors WMPO ``_windows``)
      * 1 "end" window  : ``obs[T-W : T]``, label = ``int(complete)``
      * 1 random earlier window from ``range(T-S, W-1, -S)``, label = 0
      * (success demo  → 1 pos + 1 neg ;
         failure demo  → 0 pos + 2 neg : the end window is label=0 because
                                          int(complete)=0)

    Val (mirrors WMPO ``_windows_val``)
      * 1 "end" window  : ``obs[T-W : T]``, label = ``int(complete)``
      * All earlier stride-S windows: label = 0
      * Deterministic, map-style indexing for reproducible eval.

The class balance per epoch is therefore fixed by the demo counts:
  pos windows = #success_demos
  neg windows = #success_demos + 2 × #failure_demos
For LIBERO-goal (433 success + 67 failure) → 433 : 567 ≈ 1 : 1.31.
(Verified 2026-05-25; matches the F1 ≈ 0.91 ceiling estimate in
[[dreamervla-classifier-ceiling]].)

Use ``dreamer_vla.dataset.wm_replay_classifier_dataset._find_demo_pairs`` to
discover ``(raw_path, hidden_path, demo_key)`` triples — the raw path is
read for ``rewards`` / ``dones`` (to compute ``finish_step`` + ``complete``),
the hidden path for ``obs_embedding``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from dreamer_vla.dataset.wm_replay_classifier_dataset import _find_demo_pairs


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DemoRecord:
    """One episode's frozen latent + label metadata."""

    obs: np.ndarray  # [T, L] float16, the real pi0 obs_embedding
    finish_step: int  # 1-based step where dones first fires (clamped to T)
    complete: bool  # rewards.sum() > 0
    eid: str  # stable episode id like "<file>/<demo_key>"


def _load_demo(raw_p: Path, hid_p: Path, demo_key: str) -> _DemoRecord | None:
    """Read one demo. Returns ``None`` if it lacks ``obs_embedding``."""
    with h5py.File(str(hid_p), "r") as hh:
        node = hh.get(f"{demo_key}/obs_embedding")
        if node is None:
            return None
        # keep fp16 in memory to halve footprint; cast at __getitem__.
        obs = np.asarray(node[...], dtype=np.float16)
    T = int(obs.shape[0])
    obs = obs.reshape(T, -1)

    with h5py.File(str(raw_p), "r") as fr:
        grp = fr[demo_key]
        dones = np.asarray(grp["dones"][...]) if "dones" in grp else None
        rewards = np.asarray(grp["rewards"][...]) if "rewards" in grp else None

    if dones is not None and bool(dones[:T].any()):
        finish_step = int(np.argmax(dones[:T])) + 1
    else:
        finish_step = T
    complete = bool(rewards[:T].sum() > 0) if rewards is not None else True

    eid = f"{raw_p.stem}/{demo_key.split('/')[-1]}"
    return _DemoRecord(obs=obs, finish_step=finish_step, complete=complete, eid=eid)


def _load_all(
    pairs: Sequence[tuple[Path, Path, str]],
    *,
    min_T: int,
    label: str,
) -> list[_DemoRecord]:
    out: list[_DemoRecord] = []
    skipped = 0
    for raw_p, hid_p, demo_key in pairs:
        rec = _load_demo(Path(raw_p), Path(hid_p), demo_key)
        if rec is None or rec.finish_step < min_T:
            skipped += 1
            continue
        out.append(rec)
    print(
        f"[wmpo-latent:{label}] loaded {len(out)} demos (skipped {skipped})", flush=True
    )
    return out


# ---------------------------------------------------------------------------
# Train: IterableDataset, infinite resampled stream
# ---------------------------------------------------------------------------


class WMPOAlignedLatentTrainDataset(IterableDataset):
    """Train-mode latent dataset. Each demo yields 1 end + 1 random earlier window.

    Stream is **infinite** (resampled, like WMPO's ``ResampledShards``) so the
    workspace controls duration purely via ``max_steps``.
    """

    def __init__(
        self,
        success_dir_raw: str | Path,
        success_dir_hidden: str | Path,
        failure_dir_raw: str | Path | None,
        failure_dir_hidden: str | Path | None,
        window: int = 8,
        stride: int = 8,
        seed: int = 0,
        verbose: bool = True,
        chunk_subsample: int = 1,
        chunk_pool: str = "last",
    ) -> None:
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if chunk_subsample < 1:
            raise ValueError(f"chunk_subsample must be >= 1, got {chunk_subsample}")
        if chunk_pool not in ("last", "first", "mean"):
            raise ValueError(f"chunk_pool must be last|first|mean, got {chunk_pool!r}")
        self.W = int(window)
        self.S = int(stride)
        self.seed = int(seed)
        # Chunk-level mode: each "frame" in the classifier window is pooled
        # from K=chunk_subsample consecutive env-step frames.  Window covers
        # W * K env-step frames total.  chunk_subsample=1 reduces to the
        # original env-step (action) granularity.
        self.K = int(chunk_subsample)
        self.chunk_pool = chunk_pool
        self.window_env = self.W * self.K  # env-step frames per window

        succ_pairs = _find_demo_pairs(success_dir_raw, success_dir_hidden)
        fail_pairs: list[tuple[Path, Path, str]] = []
        if failure_dir_raw is not None and failure_dir_hidden is not None:
            fail_pairs = _find_demo_pairs(failure_dir_raw, failure_dir_hidden)

        if verbose:
            print(
                f"[wmpo-latent:train] success pairs={len(succ_pairs)} "
                f"failure pairs={len(fail_pairs)} "
                f"granularity={'chunk' if self.K > 1 else 'action'} "
                f"W={self.W} K={self.K} pool={self.chunk_pool}",
                flush=True,
            )

        self._demos: list[_DemoRecord] = _load_all(
            succ_pairs, min_T=self.window_env, label="train-succ"
        ) + _load_all(fail_pairs, min_T=self.window_env, label="train-fail")
        if not self._demos:
            raise RuntimeError("WMPOAlignedLatentTrainDataset: no demos loaded")

        # composition summary — exact pos/neg windows per epoch
        # success demo: 1 pos (end) + 1 neg (random earlier)
        # failure demo: 2 neg (end is label=0 because complete=False, + random earlier)
        n_succ = sum(1 for d in self._demos if d.complete)
        n_fail = len(self._demos) - n_succ
        n_pos_windows = n_succ
        n_neg_windows = n_succ + 2 * n_fail
        if verbose:
            print(
                f"[wmpo-latent:train] per-epoch windows: "
                f"pos={n_pos_windows}  neg={n_neg_windows}  "
                f"ratio={n_pos_windows}:{n_neg_windows}",
                flush=True,
            )

    # ---- WebDataset-style infinite stream with per-worker shard ---------

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int]]:
        info = get_worker_info()
        if info is None:
            shard_idx, num_shards = 0, 1
            base_seed = self.seed
        else:
            shard_idx, num_shards = int(info.id), int(info.num_workers)
            base_seed = self.seed + 1000 * (shard_idx + 1)
        rng = np.random.default_rng(base_seed)
        # round-robin demo shard so each worker covers a disjoint subset
        demo_ids = list(range(shard_idx, len(self._demos), num_shards))
        if not demo_ids:
            return

        while True:
            for did in rng.permutation(demo_ids):
                rec = self._demos[int(did)]
                T = rec.finish_step
                obs = rec.obs
                # 1 end window — label = int(complete)
                end_window = obs[T - self.window_env : T]
                yield self._to_tensor(self._pool_window(end_window)), int(rec.complete)
                # 1 random earlier window — label = 0
                if T - self.S >= self.window_env:
                    ends = range(T - self.S, self.window_env - 1, -self.S)
                    ends_list = list(ends)
                    if ends_list:
                        end = int(rng.choice(ends_list))
                        yield (
                            self._to_tensor(
                                self._pool_window(obs[end - self.window_env : end])
                            ),
                            0,
                        )

    def _to_tensor(self, win: np.ndarray) -> torch.Tensor:
        # fp16 → fp32 here so the model sees a consistent dtype.
        return torch.from_numpy(np.ascontiguousarray(win)).float()

    def _pool_window(self, env_window: np.ndarray) -> np.ndarray:
        """Aggregate a ``[W*K, L]`` env-step window into a ``[W, L]`` classifier window.

        If ``self.K == 1`` this is a no-op (env-step / action granularity).
        """
        if self.K == 1:
            return env_window
        reshaped = env_window.reshape(self.W, self.K, env_window.shape[-1])
        if self.chunk_pool == "last":
            return reshaped[:, -1]
        if self.chunk_pool == "first":
            return reshaped[:, 0]
        return reshaped.mean(axis=1)

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.stack([b[0] for b in batch])  # [B, W, L]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys


# ---------------------------------------------------------------------------
# Val: Map-style Dataset, deterministic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ValSlot:
    demo_idx: int  # index into self._demos
    end_idx: int  # python end-exclusive window end (1-based)
    label: int  # 0 / 1
    is_end_window: bool


class WMPOAlignedLatentValDataset(Dataset):
    """Val-mode latent dataset (map-style, deterministic indexing).

    Each demo emits 1 "end" window (label = int(complete)) + all earlier
    stride-S windows (label = 0), matching WMPO's ``_windows_val``.

    ``self.demos`` is also exposed for **episode-level F1** eval: the
    workspace can slide stride-1 windows over each demo and aggregate
    with "any-positive over threshold" (the WMPO ``predict_success``
    protocol). See ``trajectories()`` for that hook.
    """

    def __init__(
        self,
        success_dir_raw: str | Path,
        success_dir_hidden: str | Path,
        failure_dir_raw: str | Path | None,
        failure_dir_hidden: str | Path | None,
        window: int = 8,
        stride: int = 1,
        verbose: bool = True,
        chunk_subsample: int = 1,
        chunk_pool: str = "last",
    ) -> None:
        super().__init__()
        if chunk_subsample < 1:
            raise ValueError(f"chunk_subsample must be >= 1, got {chunk_subsample}")
        if chunk_pool not in ("last", "first", "mean"):
            raise ValueError(f"chunk_pool must be last|first|mean, got {chunk_pool!r}")
        self.W = int(window)
        self.S = int(stride)
        self.K = int(chunk_subsample)
        self.chunk_pool = chunk_pool
        self.window_env = self.W * self.K

        succ_pairs = _find_demo_pairs(success_dir_raw, success_dir_hidden)
        fail_pairs: list[tuple[Path, Path, str]] = []
        if failure_dir_raw is not None and failure_dir_hidden is not None:
            fail_pairs = _find_demo_pairs(failure_dir_raw, failure_dir_hidden)
        if verbose:
            print(
                f"[wmpo-latent:val] success pairs={len(succ_pairs)} "
                f"failure pairs={len(fail_pairs)} "
                f"granularity={'chunk' if self.K > 1 else 'action'} "
                f"W={self.W} K={self.K} pool={self.chunk_pool}",
                flush=True,
            )

        self._demos: list[_DemoRecord] = _load_all(
            succ_pairs, min_T=self.window_env, label="val-succ"
        ) + _load_all(fail_pairs, min_T=self.window_env, label="val-fail")
        if not self._demos:
            raise RuntimeError("WMPOAlignedLatentValDataset: no demos loaded")

        # Precompute deterministic window slots
        slots: list[_ValSlot] = []
        for did, rec in enumerate(self._demos):
            T = rec.finish_step
            # end window
            slots.append(_ValSlot(did, T, int(rec.complete), is_end_window=True))
            # all earlier stride-S windows (env-step indexed; constrained so
            # the pooled window of W*K env-step frames fits before `end`)
            for end in range(T - self.S, self.window_env - 1, -self.S):
                slots.append(_ValSlot(did, end, 0, is_end_window=False))
        self._slots = slots
        if verbose:
            n_pos = sum(1 for s in slots if s.label == 1)
            n_neg = len(slots) - n_pos
            print(
                f"[wmpo-latent:val] total windows={len(slots)}  "
                f"pos={n_pos}  neg={n_neg}",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self._slots)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        slot = self._slots[idx]
        rec = self._demos[slot.demo_idx]
        env_window = rec.obs[slot.end_idx - self.window_env : slot.end_idx]
        if self.K > 1:
            reshaped = env_window.reshape(self.W, self.K, env_window.shape[-1])
            if self.chunk_pool == "last":
                window = reshaped[:, -1]
            elif self.chunk_pool == "first":
                window = reshaped[:, 0]
            else:
                window = reshaped.mean(axis=1)
        else:
            window = env_window
        meta = {
            "demo_idx": slot.demo_idx,
            "eid": rec.eid,
            "end_idx": int(slot.end_idx),
            "complete": bool(rec.complete),
            "finish_step": int(rec.finish_step),
            "is_end_window": bool(slot.is_end_window),
        }
        x = torch.from_numpy(np.ascontiguousarray(window)).float()
        return x, int(slot.label), meta

    # ---- episode-level hook for WMPO predict_success eval --------------

    def trajectories(self) -> Iterator[tuple[np.ndarray, bool, int, str]]:
        """Yield ``(obs[T,L], complete, finish_step, eid)`` per demo.

        Cast to fp32 once here so the consumer (episode-level eval) can
        slide stride-1 windows without re-casting.
        """
        for rec in self._demos:
            yield (
                rec.obs.astype(np.float32, copy=False),
                rec.complete,
                rec.finish_step,
                rec.eid,
            )

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, int, dict[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        xs = torch.stack([b[0] for b in batch])  # [B, W, L]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        metas = [b[2] for b in batch]
        return xs, ys, metas


__all__ = [
    "WMPOAlignedLatentTrainDataset",
    "WMPOAlignedLatentValDataset",
]
