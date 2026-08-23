"""LUMOS-aligned hidden-token W-frame classifier dataset.

Mirrors ``upstream reward_model/videomae.py::SuccessWindowDataset`` exactly,
but operates on canonical OpenVLA-OFT hidden-token sidecar HDF5 instead of raw
video frames. The positive/negative protocol is intentionally
identical so that downstream eval thresholds (``upstream reward_model/find_thre.py``,
``LUMOS/verl/.../robwm_rollout.py::predict_success``) transfer 1:1.

Per-demo (``finish_step = T``, ``complete ∈ {True, False}``, ``W = window``, ``S = stride``):

    Train (mirrors LUMOS ``_windows``)
      * 1 "end" window  : ``obs[T-W : T]``, label = ``int(complete)``
      * 1 random earlier window from ``range(T-S, W-1, -S)``, label = 0
      * (success demo  → 1 pos + 1 neg ;
         failure demo  → 0 pos + 2 neg : the end window is label=0 because
                                          int(complete)=0)

    Val (mirrors LUMOS ``_windows_val``)
      * 1 "end" window  : ``obs[T-W : T]``, label = ``int(complete)``
      * All earlier stride-S windows: label = 0
      * Deterministic, map-style indexing for reproducible eval.

The class balance per epoch is therefore fixed by the demo counts:
  pos windows = #success_demos
  neg windows = #success_demos + 2 × #failure_demos
For the processed LIBERO-goal h1 corpus (433 success + 67 failure), this is
433 : 567 ≈ 1 : 1.31. Exact counts depend on the selected raw/hidden dirs.

Use ``dreamervla.dataset.wm_replay_classifier_dataset._find_demo_pairs`` to
discover ``(raw_path, hidden_path, demo_key)`` triples — the raw path is
read for ``rewards`` / ``dones`` (to compute ``finish_step`` + ``complete``),
the hidden path for ``obs_embedding``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from dreamervla.dataset.wm_replay_classifier_dataset import _find_demo_pairs

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DemoRecord:
    """One episode's frozen latent + label metadata."""

    obs: np.ndarray  # [T, 256, 4096] float16 hidden_token
    proprio: np.ndarray | None  # [T, P] float32 raw proprio, optional
    lang_emb: np.ndarray | None  # [D] float32 episode-level language embedding, optional
    finish_step: int  # 1-based step where dones first fires (clamped to T)
    complete: bool  # rewards.sum() > 0
    eid: str  # stable episode id like "<file>/<demo_key>"


def _partition_demo_pairs(
    pairs: Sequence[tuple[Path, Path, str]],
    *,
    split: str,
    val_fraction: float,
    split_seed: int,
) -> list[tuple[Path, Path, str]]:
    """Return a stable trajectory-level train/validation partition."""

    split = str(split).lower()
    values = [(Path(raw), Path(hidden), str(key)) for raw, hidden, key in pairs]
    if split == "all":
        return values
    if split not in {"train", "val"}:
        raise ValueError(f"demo split must be all|train|val, got {split!r}")
    fraction = float(val_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"val_fraction must be within (0,1), got {fraction}")
    if len(values) < 2:
        raise ValueError(
            f"trajectory-level {split} split requires at least two demos, got {len(values)}"
        )

    def rank(pair: tuple[Path, Path, str]) -> tuple[str, str, str]:
        raw, hidden, demo_key = pair
        identity = f"{int(split_seed)}:{raw.name}:{hidden.name}:{demo_key}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return digest, raw.name, demo_key

    ranked = sorted(values, key=rank)
    num_val = max(1, min(len(ranked) - 1, int(round(len(ranked) * fraction))))
    return ranked[num_val:] if split == "train" else ranked[:num_val]


def _read_proprio(
    grp: h5py.Group,
    *,
    demo_ref: str,
    keys: Sequence[str],
    length: int,
) -> np.ndarray:
    obs_group = grp.get("obs")
    if not isinstance(obs_group, h5py.Group):
        raise KeyError(f"{demo_ref} missing obs group for proprio_keys={tuple(keys)}")
    arrays = []
    for key in keys:
        if key not in obs_group:
            raise KeyError(f"{demo_ref} missing obs/{key}")
        arr = np.asarray(obs_group[key][:length], dtype=np.float32).reshape(length, -1)
        arrays.append(arr)
    return np.concatenate(arrays, axis=-1)


def _read_lang_emb(
    hid_p: Path,
    demo_key: str,
    *,
    lang_emb_dir: str | Path | None,
    lang_emb_key: str,
) -> np.ndarray | None:
    if lang_emb_dir is None:
        return None
    if str(lang_emb_dir) == "__source_hidden__":
        lang_path = hid_p
    else:
        lang_path = Path(lang_emb_dir) / hid_p.name
    if not lang_path.is_file():
        raise FileNotFoundError(f"missing language sidecar: {lang_path}")
    with h5py.File(str(lang_path), "r") as handle:
        node = handle.get(f"{demo_key}/{lang_emb_key}")
        if node is None:
            raise KeyError(f"{lang_path}:{demo_key} missing {lang_emb_key}")
        lang = np.asarray(node[...], dtype=np.float32)
    if lang.ndim != 1:
        raise ValueError(f"{lang_emb_key} must be a per-demo vector, got {lang.shape}")
    return lang


def _load_demo(
    raw_p: Path,
    hid_p: Path,
    demo_key: str,
    *,
    proprio_keys: Sequence[str] | None = None,
    lang_emb_dir: str | Path | None = None,
    lang_emb_key: str = "lang_emb",
) -> _DemoRecord | None:
    """Read one demo. Returns ``None`` if it lacks ``obs_embedding``."""
    with h5py.File(str(hid_p), "r") as hh:
        node = hh.get(f"{demo_key}/obs_embedding")
        if node is None:
            return None
        # keep fp16 in memory to halve footprint; cast at __getitem__.
        obs = np.asarray(node[...], dtype=np.float16)
    T = int(obs.shape[0])

    with h5py.File(str(raw_p), "r") as fr:
        grp = fr[demo_key]
        raw_len = int(grp["actions"].shape[0]) if "actions" in grp else T
        if raw_len != T:
            raise ValueError(
                "raw/hidden length mismatch for "
                f"{raw_p}:{demo_key} and {hid_p}:{demo_key}: "
                f"raw actions length={raw_len}, hidden obs_embedding length={T}"
            )
        dones = np.asarray(grp["dones"][...]) if "dones" in grp else None
        # Prefer sparse_rewards (collector convention; rewards is all-zeros there),
        # fall back to rewards for canonical data — matches BalancedTerminalDataset /
        # CollectedRolloutClassifierDataset.
        _rk = "sparse_rewards" if "sparse_rewards" in grp else "rewards"
        rewards = np.asarray(grp[_rk][...]) if _rk in grp else None
        proprio = (
            _read_proprio(
                grp,
                demo_ref=f"{raw_p}:{demo_key}",
                keys=tuple(proprio_keys),
                length=T,
            )
            if proprio_keys
            else None
        )

    if dones is not None and bool(dones[:T].any()):
        finish_step = int(np.argmax(dones[:T])) + 1
    else:
        finish_step = T
    complete = bool(rewards[:T].sum() > 0) if rewards is not None else True
    lang_emb = _read_lang_emb(
        hid_p,
        demo_key,
        lang_emb_dir=lang_emb_dir,
        lang_emb_key=lang_emb_key,
    )

    eid = f"{raw_p.stem}/{demo_key.split('/')[-1]}"
    return _DemoRecord(
        obs=obs,
        proprio=proprio,
        lang_emb=lang_emb,
        finish_step=finish_step,
        complete=complete,
        eid=eid,
    )


def _load_all(
    pairs: Sequence[tuple[Path, Path, str]],
    *,
    min_T: int,
    label: str,
    proprio_keys: Sequence[str] | None = None,
    lang_emb_dir: str | Path | None = None,
    lang_emb_key: str = "lang_emb",
) -> list[_DemoRecord]:
    out: list[_DemoRecord] = []
    skipped = 0
    for raw_p, hid_p, demo_key in pairs:
        rec = _load_demo(
            Path(raw_p),
            Path(hid_p),
            demo_key,
            proprio_keys=proprio_keys,
            lang_emb_dir=lang_emb_dir,
            lang_emb_key=lang_emb_key,
        )
        if rec is None or rec.finish_step < min_T:
            skipped += 1
            continue
        out.append(rec)
    print(f"[lumos-latent:{label}] loaded {len(out)} demos (skipped {skipped})", flush=True)
    return out


# ---------------------------------------------------------------------------
# Train: IterableDataset, infinite resampled stream
# ---------------------------------------------------------------------------


class LumosAlignedLatentTrainDataset(IterableDataset):
    """Train-mode latent dataset. Each demo yields 1 end + 1 random earlier window.

    Stream is **infinite** (resampled, like LUMOS's ``ResampledShards``). The
    runner defines an epoch as one pass over the expected per-demo windows.
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
        proprio_keys: Sequence[str] | None = None,
        lang_emb_dir: str | Path | None = None,
        lang_emb_key: str = "lang_emb",
        sampling_protocol: str = "lumos",
        balance_batches: bool = False,
        demo_split: str = "all",
        val_fraction: float = 0.2,
        split_seed: int = 0,
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
        sampling_protocol = str(sampling_protocol)
        if sampling_protocol not in ("lumos", "wmpo"):
            raise ValueError(
                f"sampling_protocol must be 'lumos' or 'wmpo', got {sampling_protocol!r}"
            )
        self.W = int(window)
        self.S = int(stride)
        self.seed = int(seed)
        self.sampling_protocol = sampling_protocol
        self.balance_batches = bool(balance_batches)
        # Chunk-level mode: each "frame" in the classifier window is pooled
        # from K=chunk_subsample consecutive env-step frames.  Window covers
        # W * K env-step frames total.  chunk_subsample=1 reduces to the
        # original env-step (action) granularity.
        self.K = int(chunk_subsample)
        self.chunk_pool = chunk_pool
        self.window_env = self.W * self.K  # env-step frames per window
        self.proprio_keys = tuple(str(key) for key in (proprio_keys or ()))
        self.lang_emb_dir = Path(lang_emb_dir) if lang_emb_dir is not None else None
        self.lang_emb_key = str(lang_emb_key)
        self.demo_split = str(demo_split).lower()
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)

        succ_pairs = _partition_demo_pairs(
            _find_demo_pairs(success_dir_raw, success_dir_hidden),
            split=self.demo_split,
            val_fraction=self.val_fraction,
            split_seed=self.split_seed,
        )
        fail_pairs: list[tuple[Path, Path, str]] = []
        if failure_dir_raw is not None and failure_dir_hidden is not None:
            fail_pairs = _partition_demo_pairs(
                _find_demo_pairs(failure_dir_raw, failure_dir_hidden),
                split=self.demo_split,
                val_fraction=self.val_fraction,
                split_seed=self.split_seed,
            )

        if verbose:
            print(
                f"[lumos-latent:train] success pairs={len(succ_pairs)} "
                f"failure pairs={len(fail_pairs)} "
                f"granularity={'chunk' if self.K > 1 else 'action'} "
                f"W={self.W} K={self.K} pool={self.chunk_pool}",
                flush=True,
            )

        self._demos: list[_DemoRecord] = _load_all(
            succ_pairs,
            min_T=self.window_env,
            label="train-succ",
            proprio_keys=self.proprio_keys,
            lang_emb_dir=self.lang_emb_dir,
            lang_emb_key=self.lang_emb_key,
        ) + _load_all(
            fail_pairs,
            min_T=self.window_env,
            label="train-fail",
            proprio_keys=self.proprio_keys,
            lang_emb_dir=self.lang_emb_dir,
            lang_emb_key=self.lang_emb_key,
        )
        if not self._demos:
            raise RuntimeError("LumosAlignedLatentTrainDataset: no demos loaded")

        n_succ = sum(1 for d in self._demos if d.complete)
        n_fail = len(self._demos) - n_succ
        if self.sampling_protocol == "wmpo":
            self._ensure_wmpo_pools()
            n_pos_windows = n_succ
            n_neg_windows = (
                n_pos_windows if self.balance_batches else len(self._wmpo_negative_slots)
            )
        else:
            # composition summary — exact pos/neg windows per epoch
            # success demo: 1 pos (end) + 1 neg (random earlier)
            # failure demo: 2 neg (end is label=0 because complete=False, + random earlier)
            n_pos_windows = n_succ
            n_neg_windows = n_succ + 2 * n_fail
        self._epoch_windows = n_pos_windows + n_neg_windows
        if verbose:
            print(
                f"[lumos-latent:train] per-epoch windows: "
                f"pos={n_pos_windows}  neg={n_neg_windows}  "
                f"ratio={n_pos_windows}:{n_neg_windows} "
                f"protocol={self.sampling_protocol} "
                f"batch_balance={self.balance_batches}",
                flush=True,
            )

    def __len__(self) -> int:
        return int(self._epoch_windows)

    def summary(self) -> dict[str, int | str | bool]:
        n_succ = sum(1 for d in self._demos if d.complete)
        n_fail = len(self._demos) - n_succ
        if getattr(self, "sampling_protocol", "lumos") == "wmpo":
            self._ensure_wmpo_pools()
            n_pos_windows = n_succ
            n_neg_windows = (
                n_pos_windows if self.balance_batches else len(self._wmpo_negative_slots)
            )
        else:
            n_pos_windows = n_succ
            n_neg_windows = n_succ + 2 * n_fail
        return {
            "num_demos": int(len(self._demos)),
            "num_success_demos": int(n_succ),
            "num_failure_demos": int(n_fail),
            "epoch_pos_windows": int(n_pos_windows),
            "epoch_neg_windows": int(n_neg_windows),
            "sampling_protocol": str(getattr(self, "sampling_protocol", "lumos")),
            "balance_batches": bool(getattr(self, "balance_batches", False)),
            "window": int(self.W),
            "stride": int(self.S),
            "chunk_subsample": int(self.K),
            "chunk_pool": str(self.chunk_pool),
            "demo_split": str(self.demo_split),
        }

    def _ensure_wmpo_pools(self) -> None:
        if hasattr(self, "_wmpo_positive_ids") and hasattr(self, "_wmpo_negative_slots"):
            return
        positive_ids: list[int] = []
        negative_slots: list[tuple[int, int]] = []
        for did, rec in enumerate(self._demos):
            if rec.complete:
                positive_ids.append(did)
                for end in self._wmpo_success_negative_ends(finish_step=rec.finish_step):
                    negative_slots.append((did, end))
            else:
                for end in self._wmpo_failure_negative_ends(finish_step=rec.finish_step):
                    negative_slots.append((did, end))
        if not positive_ids:
            raise RuntimeError("WMPO sampling requires at least one successful trajectory")
        if not negative_slots:
            raise RuntimeError("WMPO sampling requires at least one negative clip")
        self._wmpo_positive_ids = positive_ids
        self._wmpo_negative_slots = negative_slots

    def _wmpo_success_negative_ends(self, *, finish_step: int) -> list[int]:
        """End indices for successful-trajectory negatives.

        WMPO samples negatives from ``L <= i <= N-L``. In token/chunk mode,
        ``window_env = L * chunk_subsample`` env-step frames, so successful
        negatives must end at least one full clip before the terminal clip.
        """
        max_end = int(finish_step) - int(self.window_env)
        if max_end < int(self.window_env):
            return []
        return list(range(int(self.window_env), max_end + 1, int(self.S)))

    def _wmpo_failure_negative_ends(self, *, finish_step: int) -> list[int]:
        max_end = int(finish_step)
        if max_end < int(self.window_env):
            return []
        return list(range(int(self.window_env), max_end + 1, int(self.S)))

    def _window_item(
        self,
        rec: _DemoRecord,
        *,
        end: int,
        label: int,
    ) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, dict[str, torch.Tensor]]:
        start = int(end) - int(self.window_env)
        window = rec.obs[start : int(end)]
        extra = self._window_extra(rec, start, int(end))
        item = (self._to_tensor(self._pool_window(window)), int(label))
        return (*item, extra) if extra else item

    def _iter_wmpo_balanced(
        self,
        *,
        rng: np.random.Generator,
        demo_ids: list[int],
    ) -> Iterator[tuple[torch.Tensor, int] | tuple[torch.Tensor, int, dict[str, torch.Tensor]]]:
        self._ensure_wmpo_pools()
        local_positive_ids = [did for did in self._wmpo_positive_ids if did in set(demo_ids)]
        local_negative_slots = [
            slot for slot in self._wmpo_negative_slots if int(slot[0]) in set(demo_ids)
        ]
        if not local_positive_ids:
            local_positive_ids = list(self._wmpo_positive_ids)
        if not local_negative_slots:
            local_negative_slots = list(self._wmpo_negative_slots)
        while True:
            pos_did = int(rng.choice(local_positive_ids))
            pos_rec = self._demos[pos_did]
            yield self._window_item(pos_rec, end=pos_rec.finish_step, label=1)

            neg_did, neg_end = local_negative_slots[int(rng.integers(len(local_negative_slots)))]
            yield self._window_item(self._demos[int(neg_did)], end=int(neg_end), label=0)

    # ---- WebDataset-style infinite stream with per-worker shard ---------

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, int] | tuple[torch.Tensor, int, dict[str, torch.Tensor]]]:
        info = get_worker_info()
        distributed_rank = int(getattr(self, "distributed_rank", 0) or 0)
        distributed_world_size = max(
            1,
            int(getattr(self, "distributed_world_size", 1) or 1),
        )
        if distributed_world_size == 1:
            if info is None:
                shard_idx, num_shards = 0, 1
                base_seed = self.seed
            else:
                shard_idx, num_shards = int(info.id), int(info.num_workers)
                base_seed = self.seed + 1000 * (shard_idx + 1)
        else:
            worker_id = 0 if info is None else int(info.id)
            num_workers = 1 if info is None else int(info.num_workers)
            shard_idx = distributed_rank * num_workers + worker_id
            num_shards = distributed_world_size * num_workers
            base_seed = self.seed + 1000 * (shard_idx + 1)
        rng = np.random.default_rng(base_seed)
        # round-robin demo shard so each worker covers a disjoint subset
        demo_ids = list(range(shard_idx, len(self._demos), num_shards))
        if not demo_ids:
            return

        if getattr(self, "sampling_protocol", "lumos") == "wmpo" and bool(
            getattr(self, "balance_batches", False)
        ):
            yield from self._iter_wmpo_balanced(rng=rng, demo_ids=demo_ids)
            return

        while True:
            for did in rng.permutation(demo_ids):
                rec = self._demos[int(did)]
                T = rec.finish_step
                # 1 end window — label = int(complete)
                yield self._window_item(rec, end=T, label=int(rec.complete))
                # 1 random earlier window — label = 0
                if self.window_env <= T - self.S:
                    ends = range(T - self.S, self.window_env - 1, -self.S)
                    ends_list = list(ends)
                    if ends_list:
                        end = int(rng.choice(ends_list))
                        yield self._window_item(rec, end=end, label=0)

    def _to_tensor(self, win: np.ndarray) -> torch.Tensor:
        # Sidecars are stored as fp16. Preserve that dtype through pinned host
        # memory so H2D moves half as many bytes; the classifier casts on GPU at
        # its projection boundary, producing the same fp32 values as a CPU cast.
        return torch.from_numpy(np.ascontiguousarray(win))

    def _pool_window(self, env_window: np.ndarray) -> np.ndarray:
        """Aggregate ``[W*K,...]`` env steps while preserving token-grid axes.

        If ``self.K == 1`` this is a no-op (env-step / action granularity).
        """
        if self.K == 1:
            return env_window
        reshaped = env_window.reshape(self.W, self.K, *env_window.shape[1:])
        if self.chunk_pool == "last":
            return reshaped[:, -1]
        if self.chunk_pool == "first":
            return reshaped[:, 0]
        return reshaped.mean(axis=1)

    def _window_extra(
        self,
        rec: _DemoRecord,
        start: int,
        end: int,
    ) -> dict[str, torch.Tensor]:
        extra: dict[str, torch.Tensor] = {}
        if rec.proprio is not None:
            proprio_window = rec.proprio[start:end]
            extra["proprio"] = torch.from_numpy(
                np.ascontiguousarray(self._pool_window(proprio_window))
            ).float()
        if rec.lang_emb is not None:
            extra["lang_emb"] = torch.from_numpy(np.ascontiguousarray(rec.lang_emb)).float()
        return extra

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, int] | tuple[torch.Tensor, int, dict[str, torch.Tensor]]],
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        xs = torch.stack([b[0] for b in batch])  # [B, W, 256, 4096]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        if len(batch[0]) < 3:
            return xs, ys
        extras = [b[2] for b in batch if len(b) >= 3]
        out: dict[str, torch.Tensor] = {}
        if extras and all("proprio" in extra for extra in extras):
            out["proprio"] = torch.stack([extra["proprio"] for extra in extras])
        if extras and all("lang_emb" in extra for extra in extras):
            out["lang_emb"] = torch.stack([extra["lang_emb"] for extra in extras])
        return (xs, ys, out) if out else (xs, ys)


# ---------------------------------------------------------------------------
# Val: Map-style Dataset, deterministic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ValSlot:
    demo_idx: int  # index into self._demos
    end_idx: int  # python end-exclusive window end (1-based)
    label: int  # 0 / 1
    is_end_window: bool


class LumosAlignedLatentValDataset(Dataset):
    """Val-mode latent dataset (map-style, deterministic indexing).

    Each demo emits 1 "end" window (label = int(complete)) + all earlier
    stride-S windows (label = 0), matching LUMOS's ``_windows_val``.

    ``self.demos`` is also exposed for **episode-level F1** eval: the
    workspace can slide stride-1 windows over each demo and aggregate
    with "any-positive over threshold" (the LUMOS ``predict_success``
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
        proprio_keys: Sequence[str] | None = None,
        lang_emb_dir: str | Path | None = None,
        lang_emb_key: str = "lang_emb",
        sampling_protocol: str = "lumos",
        demo_split: str = "all",
        val_fraction: float = 0.2,
        split_seed: int = 0,
    ) -> None:
        super().__init__()
        if chunk_subsample < 1:
            raise ValueError(f"chunk_subsample must be >= 1, got {chunk_subsample}")
        if chunk_pool not in ("last", "first", "mean"):
            raise ValueError(f"chunk_pool must be last|first|mean, got {chunk_pool!r}")
        sampling_protocol = str(sampling_protocol)
        if sampling_protocol not in ("lumos", "wmpo"):
            raise ValueError(
                f"sampling_protocol must be 'lumos' or 'wmpo', got {sampling_protocol!r}"
            )
        self.W = int(window)
        self.S = int(stride)
        self.K = int(chunk_subsample)
        self.chunk_pool = chunk_pool
        self.sampling_protocol = sampling_protocol
        self.window_env = self.W * self.K
        self.proprio_keys = tuple(str(key) for key in (proprio_keys or ()))
        self.lang_emb_dir = Path(lang_emb_dir) if lang_emb_dir is not None else None
        self.lang_emb_key = str(lang_emb_key)
        self.demo_split = str(demo_split).lower()
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)

        succ_pairs = _partition_demo_pairs(
            _find_demo_pairs(success_dir_raw, success_dir_hidden),
            split=self.demo_split,
            val_fraction=self.val_fraction,
            split_seed=self.split_seed,
        )
        fail_pairs: list[tuple[Path, Path, str]] = []
        if failure_dir_raw is not None and failure_dir_hidden is not None:
            fail_pairs = _partition_demo_pairs(
                _find_demo_pairs(failure_dir_raw, failure_dir_hidden),
                split=self.demo_split,
                val_fraction=self.val_fraction,
                split_seed=self.split_seed,
            )
        if verbose:
            print(
                f"[lumos-latent:val] success pairs={len(succ_pairs)} "
                f"failure pairs={len(fail_pairs)} "
                f"granularity={'chunk' if self.K > 1 else 'action'} "
                f"W={self.W} K={self.K} pool={self.chunk_pool}",
                flush=True,
            )

        self._demos: list[_DemoRecord] = _load_all(
            succ_pairs,
            min_T=self.window_env,
            label="val-succ",
            proprio_keys=self.proprio_keys,
            lang_emb_dir=self.lang_emb_dir,
            lang_emb_key=self.lang_emb_key,
        ) + _load_all(
            fail_pairs,
            min_T=self.window_env,
            label="val-fail",
            proprio_keys=self.proprio_keys,
            lang_emb_dir=self.lang_emb_dir,
            lang_emb_key=self.lang_emb_key,
        )
        if not self._demos:
            raise RuntimeError("LumosAlignedLatentValDataset: no demos loaded")

        # Precompute deterministic window slots
        slots: list[_ValSlot] = []
        for did, rec in enumerate(self._demos):
            T = rec.finish_step
            # end window
            slots.append(_ValSlot(did, T, int(rec.complete), is_end_window=True))
            if self.sampling_protocol == "wmpo" and rec.complete:
                earlier_ends = self._wmpo_success_negative_ends(finish_step=T)
            else:
                # LUMOS validation and failed WMPO trajectories enumerate all
                # earlier stride-S windows. Failed terminal is already included
                # above, so this starts at T-S.
                earlier_ends = list(range(T - self.S, self.window_env - 1, -self.S))
            for end in earlier_ends:
                slots.append(_ValSlot(did, int(end), 0, is_end_window=False))
        self._slots = slots
        if verbose:
            n_pos = sum(1 for s in slots if s.label == 1)
            n_neg = len(slots) - n_pos
            print(
                f"[lumos-latent:val] total windows={len(slots)}  pos={n_pos}  neg={n_neg}",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self._slots)

    def summary(self) -> dict[str, int | str]:
        n_succ = sum(1 for d in self._demos if d.complete)
        n_fail = len(self._demos) - n_succ
        n_pos_windows = sum(1 for s in self._slots if s.label == 1)
        n_neg_windows = len(self._slots) - n_pos_windows
        return {
            "num_demos": int(len(self._demos)),
            "num_success_demos": int(n_succ),
            "num_failure_demos": int(n_fail),
            "num_windows": int(len(self._slots)),
            "pos_windows": int(n_pos_windows),
            "neg_windows": int(n_neg_windows),
            "window": int(self.W),
            "stride": int(self.S),
            "chunk_subsample": int(self.K),
            "chunk_pool": str(self.chunk_pool),
            "sampling_protocol": str(getattr(self, "sampling_protocol", "lumos")),
            "demo_split": str(self.demo_split),
        }

    def _wmpo_success_negative_ends(self, *, finish_step: int) -> list[int]:
        max_end = int(finish_step) - int(self.window_env)
        if max_end < int(self.window_env):
            return []
        return list(range(int(self.window_env), max_end + 1, int(self.S)))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        slot = self._slots[idx]
        rec = self._demos[slot.demo_idx]
        env_window = rec.obs[slot.end_idx - self.window_env : slot.end_idx]
        if self.K > 1:
            reshaped = env_window.reshape(self.W, self.K, *env_window.shape[1:])
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
        if rec.proprio is not None:
            proprio_window = rec.proprio[slot.end_idx - self.window_env : slot.end_idx]
            if self.K > 1:
                reshaped_proprio = proprio_window.reshape(self.W, self.K, proprio_window.shape[-1])
                if self.chunk_pool == "last":
                    proprio_pooled = reshaped_proprio[:, -1]
                elif self.chunk_pool == "first":
                    proprio_pooled = reshaped_proprio[:, 0]
                else:
                    proprio_pooled = reshaped_proprio.mean(axis=1)
            else:
                proprio_pooled = proprio_window
            meta["proprio"] = torch.from_numpy(np.ascontiguousarray(proprio_pooled)).float()
        if rec.lang_emb is not None:
            meta["lang_emb"] = torch.from_numpy(np.ascontiguousarray(rec.lang_emb)).float()
        x = torch.from_numpy(np.ascontiguousarray(window)).float()
        return x, int(slot.label), meta

    # ---- episode-level hook for LUMOS predict_success eval --------------

    def trajectories(self) -> Iterator[tuple[np.ndarray, bool, int, str, dict[str, np.ndarray]]]:
        """Yield ``(obs[T,256,4096], complete, finish_step, eid, extra)`` per demo.

        Keep obs in its cached dtype. Token-grid sidecars are large, so
        episode-level eval casts only the current inference batch to fp32.
        """
        for rec in self._demos:
            extra: dict[str, np.ndarray] = {}
            if rec.proprio is not None:
                extra["proprio"] = rec.proprio.astype(np.float32, copy=False)
            if rec.lang_emb is not None:
                extra["lang_emb"] = rec.lang_emb.astype(np.float32, copy=False)
            yield (
                rec.obs,
                rec.complete,
                rec.finish_step,
                rec.eid,
                extra,
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
    "LumosAlignedLatentTrainDataset",
    "LumosAlignedLatentValDataset",
    "_partition_demo_pairs",
]
