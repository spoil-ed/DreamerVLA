"""Chunk-WM-replayed classifier dataset.

For each LIBERO demo:
  positive trajectory: imagined latents from chunk WM driven by the demo's
                       own GT actions (chunked at K=5)
  negative trajectory: imagined latents from chunk WM driven by perturbed
                       actions (swap / noise / random)

Both trajectories share the demo's first ``num_hist`` real obs_embedding
frames as init state; from there everything is WM imagination. This aligns
the classifier's training distribution to the PPO inference distribution:
chunk-structured imagined latents, with the WM's own drift/error baked in.

Two-stage usage:
    ds = WMReplayClassifierDataset(hdf5_pairs, chunk_wm, device, ...)
    ds.imagine_all()                  # GPU-bound, run on main process
    for window, label in DataLoader(ds, num_workers=2, ...):
        ...

``imagine_all()`` is GPU-bound and must run on the main process.
``__iter__`` consumes the cached trajectories and is CPU-only, so
``num_workers > 0`` is safe after caching.

WMPO finish_step alignment
--------------------------
Each demo's HDF5 carries ``dones`` and ``rewards`` arrays. We derive:
    finish_step (input coords) = argmax(dones)+1 if dones.any() else T_common
    complete                   = rewards.sum() > 0
matching WMPO/reward_model/videomae.py::SuccessWindowDataset's
``meta["finish_step"]`` / ``meta["complete"]``. The "end" window of every
trajectory is anchored at finish_step rather than at the full imagined
length — for offline LIBERO demos these usually coincide, but they
diverge for real policy rollouts where the episode terminates early.

Sources
-------
Three optional data sources can be combined freely:

1. ``raw_dir`` / ``hidden_dir`` (required) — success demos.
   - pos traj: imagined w/ GT actions, label=int(complete).
   - swap-neg: same init, perturbed actions, label=0 (synthetic neg).

2. ``failure_raw_dir`` / ``failure_hidden_dir`` (optional) — pi0 SFT
   FAILURE rollouts (replay-divergent demos). label=0 (real failures).

3. ``rollout_raw_dir`` / ``rollout_hidden_dir`` (optional) — real
   policy SFT / online rollouts with MIXED outcomes. label and
   finish_step are derived per-demo from rewards/dones. This is the
   closest analog to WMPO's SFT-rollout corpus.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def _find_demo_pairs(
    raw_dir: str | Path, hidden_dir: str | Path
) -> list[tuple[Path, Path, str]]:
    """Match raw demos (actions) with precomputed obs_embedding by filename.

    Returns a list of (raw_path, hidden_path, demo_key) — demo_key like
    'open_the_middle_drawer_..._demo.hdf5/data/demo_0'. Each raw file may
    contain many demo_groups (demo_0, demo_1, ...); each gets one tuple.
    """
    raw_dir = Path(raw_dir)
    hidden_dir = Path(hidden_dir)
    raw_files = sorted(raw_dir.glob("*.hdf5"))
    pairs: list[tuple[Path, Path, str]] = []
    for raw_p in raw_files:
        hid_p = hidden_dir / raw_p.name
        if not hid_p.exists():
            continue
        with h5py.File(str(hid_p), "r") as hh:
            if "data" not in hh:
                continue
            demo_keys = list(hh["data"].keys())
        # filter demos that actually have obs_embedding
        with h5py.File(str(hid_p), "r") as hh:
            keep_keys = [k for k in demo_keys if "obs_embedding" in hh[f"data/{k}"]]
        for k in keep_keys:
            pairs.append((raw_p, hid_p, f"data/{k}"))
    return pairs


class WMReplayClassifierDataset(IterableDataset):
    def __init__(
        self,
        *,
        raw_dir: str | Path,
        hidden_dir: str | Path,
        chunk_wm: nn.Module,
        device: torch.device | str,
        K: int = 5,
        W: int = 8,
        num_hist: int = 3,
        mode: str = "train",
        stride: int = 8,
        neg_method: str = "swap",
        noise_std: float = 0.05,
        swap_min_frac: float = 0.30,
        swap_max_frac: float = 0.80,
        max_demos: int | None = None,
        seed: int = 0,
        # Optional FAILURE data source (e.g., pi0 SFT sim rollouts that didn't
        # solve the task). When provided, each failure demo is imagined through
        # the chunk WM and contributes (end_window, label=0) + (earlier_window,
        # label=0) — matching WMPO's SuccessWindowDataset windowing for failed
        # episodes. Real failures are MORE INFORMATIVE than swap-perturbed
        # negatives because they capture the true distribution of "trajectories
        # that look plausible but don't reach the goal."
        failure_raw_dir: str | Path | None = None,
        failure_hidden_dir: str | Path | None = None,
        max_failure_demos: int | None = None,
        include_swap_negatives: bool = True,
        # Optional REAL POLICY ROLLOUT source — mixed-outcome rollouts from a
        # real policy (pi0 SFT inference, online RL exploration, ...). Each
        # demo's label and finish_step are derived per-episode from the
        # HDF5's ``rewards``/``dones`` arrays. This is the closest analog to
        # WMPO's training corpus (one episode = one real rollout, naturally
        # mixed success/failure).
        rollout_raw_dir: str | Path | None = None,
        rollout_hidden_dir: str | Path | None = None,
        max_rollout_demos: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError(f"mode must be train/val, got {mode}")
        if neg_method not in {"swap", "noise", "random"}:
            raise ValueError(
                f"neg_method must be one of swap/noise/random, got {neg_method}"
            )
        self.K = int(K)
        self.W = int(W)
        self.num_hist = int(num_hist)
        self.mode = mode
        self.stride = int(stride)
        self.neg_method = neg_method
        self.noise_std = float(noise_std)
        self.swap_min_frac = float(swap_min_frac)
        self.swap_max_frac = float(swap_max_frac)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.include_swap_negatives = bool(include_swap_negatives)

        # Chunk WM must be moved to device by the caller; we just hold the ref.
        self.chunk_wm = chunk_wm
        if (
            hasattr(self.chunk_wm, "chunk_size")
            and int(self.chunk_wm.chunk_size) != self.K
        ):
            raise ValueError(
                f"chunk_wm.chunk_size={self.chunk_wm.chunk_size} != dataset K={self.K}"
            )

        self.pairs = _find_demo_pairs(raw_dir, hidden_dir)
        if max_demos is not None:
            self.pairs = self.pairs[: int(max_demos)]
        if not self.pairs:
            raise RuntimeError(
                f"no demo pairs found under raw={raw_dir}, hidden={hidden_dir}"
            )

        self.failure_pairs: list[tuple[Path, Path, str]] = []
        if failure_raw_dir is not None and failure_hidden_dir is not None:
            self.failure_pairs = _find_demo_pairs(failure_raw_dir, failure_hidden_dir)
            if max_failure_demos is not None:
                self.failure_pairs = self.failure_pairs[: int(max_failure_demos)]

        self.rollout_pairs: list[tuple[Path, Path, str]] = []
        if rollout_raw_dir is not None and rollout_hidden_dir is not None:
            self.rollout_pairs = _find_demo_pairs(rollout_raw_dir, rollout_hidden_dir)
            if max_rollout_demos is not None:
                self.rollout_pairs = self.rollout_pairs[: int(max_rollout_demos)]

        # Filled by imagine_all(). For every list of imagined trajectories we
        # carry a parallel list of (complete, finish_step_imag) — both used by
        # the window yielder so that the "end" window matches WMPO's
        # meta["finish_step"] semantics rather than the full imagined length.
        self._pos_trajs: list[np.ndarray] | None = None  # imagined from success demos
        self._pos_meta: list[tuple[bool, int]] = []
        self._neg_trajs: list[np.ndarray] | None = (
            None  # swap-perturbed neg per success demo
        )
        self._neg_meta: list[tuple[bool, int]] = []
        self._failure_trajs: list[np.ndarray] | None = (
            None  # imagined from failure demos (real failures)
        )
        self._failure_meta: list[tuple[bool, int]] = []
        self._rollout_trajs: list[np.ndarray] | None = (
            None  # imagined from real policy rollouts (mixed)
        )
        self._rollout_meta: list[tuple[bool, int]] = []

    # ─── Data loading ──────────────────────────────────────────────────────
    def _load_demo(self, idx: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
        return self._load_pair_at(self.pairs[idx])

    def _load_pair_at(
        self, pair: tuple[Path, Path, str]
    ) -> tuple[np.ndarray, np.ndarray, int, bool]:
        """Load (obs_embedding, actions, finish_step_input, complete).

        ``finish_step_input`` follows WMPO's Python-slicing convention: it is
        one past the last index to include, derived from the first ``dones==1``
        entry. If ``dones`` is missing or all-zero we fall back to the full
        common length, which preserves the prior behaviour. ``complete`` is
        True iff the demo's ``rewards`` array contains any positive entry.
        """
        raw_p, hid_p, demo_key = pair
        with h5py.File(str(raw_p), "r") as fr:
            grp = fr[demo_key]
            actions = np.asarray(grp["actions"][...], dtype=np.float32)
            dones = np.asarray(grp["dones"][...]) if "dones" in grp else None
            rewards = np.asarray(grp["rewards"][...]) if "rewards" in grp else None
        with h5py.File(str(hid_p), "r") as fh:
            obs = np.asarray(fh[f"{demo_key}/obs_embedding"][...], dtype=np.float32)
        T_common = int(min(actions.shape[0], obs.shape[0]))
        obs = obs[:T_common]
        actions = actions[:T_common]
        if dones is not None:
            dones = dones[:T_common]
            finish_step = int(np.argmax(dones)) + 1 if bool(dones.any()) else T_common
        else:
            finish_step = T_common
        if rewards is not None:
            complete = bool(rewards[:T_common].sum() > 0)
        else:
            complete = True  # legacy demos with no rewards array → assume success
        return obs, actions, finish_step, complete

    # ─── Action perturbation ───────────────────────────────────────────────
    def _make_neg_actions(
        self,
        actions: np.ndarray,
        idx_self: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        T = int(actions.shape[0])
        if self.neg_method == "noise":
            return actions + rng.normal(0.0, self.noise_std, size=actions.shape).astype(
                np.float32
            )
        if self.neg_method == "random":
            low = actions.min(axis=0, keepdims=True)
            high = actions.max(axis=0, keepdims=True)
            return rng.uniform(low=low, high=high, size=actions.shape).astype(
                np.float32
            )
        # swap: random other demo, swap from random fraction onward
        if len(self.pairs) <= 1:
            # degenerate: fall back to noise
            return actions + rng.normal(0.0, self.noise_std, size=actions.shape).astype(
                np.float32
            )
        other_idx = int(rng.integers(0, len(self.pairs)))
        attempts = 0
        while other_idx == idx_self and attempts < 8:
            other_idx = int(rng.integers(0, len(self.pairs)))
            attempts += 1
        _, other_actions, _, _ = self._load_demo(other_idx)
        swap_at = int(
            rng.integers(
                int(self.swap_min_frac * T),
                max(int(self.swap_min_frac * T) + 1, int(self.swap_max_frac * T) + 1),
            )
        )
        swap_at = max(self.num_hist + self.K, min(swap_at, T - self.K))
        neg = actions.copy()
        T_other = int(other_actions.shape[0])
        tail_len = T - swap_at
        if tail_len <= 0:
            return neg
        if tail_len <= T_other:
            offset = int(rng.integers(0, max(1, T_other - tail_len + 1)))
            neg[swap_at:] = other_actions[offset : offset + tail_len]
        else:
            # other demo too short: tile
            reps = (tail_len + T_other - 1) // T_other
            tail = np.tile(other_actions, (reps, 1))[:tail_len]
            neg[swap_at:] = tail
        return neg.astype(np.float32)

    # ─── Imagine driver ────────────────────────────────────────────────────
    def _input_to_imag_idx(self, t_input: int) -> int:
        """Map an input-coord 'one-past-end' index to imagined-coord 'one-past-end'.

        Imagined seq layout (length 1 + num_chunks*K):
            idx 0           ← real obs[num_hist - 1]
            idx c*K+1..(c+1)*K ← WM-predicted for input timesteps H + c*K .. H + (c+1)*K - 1
        So input timestep ``t`` (0-indexed) maps to imagined index ``t - (H-1)``
        for ``t >= H-1``. A Python-slicing 'one-past-end' ``t_input`` therefore
        maps to one-past-end imagined index ``t_input - (H-1)``.
        """
        return max(0, int(t_input) - (self.num_hist - 1))

    @torch.no_grad()
    def _imagine_one(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return imagined latent sequence as float32 numpy.

        Rynn-DINO chunk WMs emit tokenized hidden ``[T_out,N,token_dim]``;
        older flat WMs continue to emit ``[T_out,obs_dim]``.
        """
        T = int(obs.shape[0])
        K = self.K
        H = self.num_hist
        if T < H + K:
            return obs[:T].astype(np.float32)

        device = self.device
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
        act_t = torch.from_numpy(actions).to(device=device, dtype=torch.float32)

        obs_init = obs_t[:H].unsqueeze(0)  # [1, H, obs_dim]
        act_init = act_t[:H].unsqueeze(0)  # [1, H, A]
        # observe_sequence returns history/actions of shape [B, steps, num_hist, ...].
        # predict_next* expects single-step latents of shape [B, num_hist, ...] —
        # so squeeze the per-step time dim by taking the LAST step.
        out = self.chunk_wm(
            {
                "mode": "observe_sequence",
                "obs_embedding": obs_init,
                "actions": act_init,
            }
        )
        latent_seq = out["latent"] if isinstance(out, dict) and "latent" in out else out
        latent = {
            "hidden": latent_seq["hidden"][:, -1],
            "history": latent_seq["history"][:, -1],
            "actions": latent_seq["actions"][:, -1],  # [B, num_hist, A]
        }

        if hasattr(self.chunk_wm, "obs_to_tokens"):
            real_seed = self.chunk_wm.obs_to_tokens(obs_init)[:, -1].squeeze(0)
            imagined: list[torch.Tensor] = [real_seed.unsqueeze(0)]
        else:
            imagined = [obs_t[H - 1 : H]]  # last real history frame
        T_apply = T - H
        num_chunks = T_apply // K
        for c in range(num_chunks):
            chunk_actions = act_t[H + c * K : H + (c + 1) * K].unsqueeze(0)  # [1, K, A]
            out = self.chunk_wm(
                {
                    "mode": "predict_next_chunk",
                    "latent": latent,
                    "actions": chunk_actions,
                }
            )
            imagined.append(out["hidden_seq"].squeeze(0))
            latent = {
                "history": out["history"],
                "actions": out["actions"],
                "hidden": out["hidden"],
            }
        seq = torch.cat(imagined, dim=0)
        return seq.detach().cpu().to(torch.float32).numpy()

    @torch.no_grad()
    def imagine_all(self, verbose: bool = True) -> None:
        """Pre-compute imagined trajectories for all demos. GPU-bound.

        Sources:
            - SUCCESS demos → pos_traj (demo actions) + neg_traj (swap-perturbed)
            - FAILURE demos → failure_traj (demo actions; real failure)
            - ROLLOUT demos → rollout_traj (per-demo derived label & finish_step)
        """
        self.chunk_wm.eval()
        rng = np.random.default_rng(self.seed)
        pos_trajs: list[np.ndarray] = []
        pos_meta: list[tuple[bool, int]] = []
        neg_trajs: list[np.ndarray] = []
        neg_meta: list[tuple[bool, int]] = []
        for i in range(len(self.pairs)):
            obs, actions, fs_in, complete = self._load_demo(i)
            pos = self._imagine_one(obs, actions)
            fs_imag = min(self._input_to_imag_idx(fs_in), int(pos.shape[0]))
            pos_trajs.append(pos)
            pos_meta.append((bool(complete), int(fs_imag)))
            if self.include_swap_negatives:
                neg_actions = self._make_neg_actions(actions, i, rng)
                neg = self._imagine_one(obs, neg_actions)
                # Swap-neg has no real "completion event"; reuse source demo's
                # finish_step so the end window aligns to the same time index.
                fs_imag_neg = min(fs_imag, int(neg.shape[0]))
                neg_trajs.append(neg)
                neg_meta.append((False, int(fs_imag_neg)))
            if verbose and (i + 1) % 25 == 0:
                last_neg_T = neg_trajs[-1].shape[0] if neg_trajs else 0
                print(
                    f"  imagine_all (success): {i + 1}/{len(self.pairs)} pos_T={pos.shape[0]} neg_T={last_neg_T} fs_imag={fs_imag}"
                )

        failure_trajs: list[np.ndarray] = []
        failure_meta: list[tuple[bool, int]] = []
        for i, pair in enumerate(self.failure_pairs):
            obs, actions, fs_in, complete = self._load_pair_at(pair)
            f_traj = self._imagine_one(obs, actions)
            fs_imag = min(self._input_to_imag_idx(fs_in), int(f_traj.shape[0]))
            failure_trajs.append(f_traj)
            failure_meta.append((bool(complete), int(fs_imag)))
            if verbose and (i + 1) % 10 == 0:
                print(
                    f"  imagine_all (failure): {i + 1}/{len(self.failure_pairs)} fail_T={f_traj.shape[0]} fs_imag={fs_imag}"
                )

        rollout_trajs: list[np.ndarray] = []
        rollout_meta: list[tuple[bool, int]] = []
        for i, pair in enumerate(self.rollout_pairs):
            obs, actions, fs_in, complete = self._load_pair_at(pair)
            r_traj = self._imagine_one(obs, actions)
            fs_imag = min(self._input_to_imag_idx(fs_in), int(r_traj.shape[0]))
            rollout_trajs.append(r_traj)
            rollout_meta.append((bool(complete), int(fs_imag)))
            if verbose and (i + 1) % 25 == 0:
                print(
                    f"  imagine_all (rollout): {i + 1}/{len(self.rollout_pairs)} "
                    f"T={r_traj.shape[0]} fs_imag={fs_imag} complete={complete}"
                )

        self._pos_trajs = pos_trajs
        self._pos_meta = pos_meta
        self._neg_trajs = neg_trajs
        self._neg_meta = neg_meta
        self._failure_trajs = failure_trajs
        self._failure_meta = failure_meta
        self._rollout_trajs = rollout_trajs
        self._rollout_meta = rollout_meta

    # ─── Window yielding ───────────────────────────────────────────────────
    def _all_labeled_trajs(self) -> list[tuple[np.ndarray, bool, int]]:
        """Unified view: (imagined_traj, complete, finish_step_imag).

        - pos_trajs (imagined success demos): complete from rewards (≈ True)
        - neg_trajs (swap-perturbed of success): complete=False (synthetic)
        - failure_trajs (imagined real failure demos): complete=False (real)
        - rollout_trajs (imagined real policy rollouts): complete per-episode
        """
        out: list[tuple[np.ndarray, bool, int]] = []
        if self._pos_trajs is not None:
            for traj, (complete, fs) in zip(
                self._pos_trajs, self._pos_meta, strict=True
            ):
                out.append((traj, bool(complete), int(fs)))
        if self._neg_trajs is not None:
            for traj, (complete, fs) in zip(
                self._neg_trajs, self._neg_meta, strict=True
            ):
                out.append((traj, bool(complete), int(fs)))
        if self._failure_trajs is not None:
            for traj, (complete, fs) in zip(
                self._failure_trajs, self._failure_meta, strict=True
            ):
                out.append((traj, bool(complete), int(fs)))
        if self._rollout_trajs is not None:
            for traj, (complete, fs) in zip(
                self._rollout_trajs, self._rollout_meta, strict=True
            ):
                out.append((traj, bool(complete), int(fs)))
        return out

    def __iter__(self) -> Iterator:
        if self._pos_trajs is None:
            raise RuntimeError("call imagine_all() before iterating")

        all_trajs = self._all_labeled_trajs()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            indices = list(range(len(all_trajs)))
        else:
            wid = int(worker_info.id)
            nw = int(worker_info.num_workers)
            indices = list(range(len(all_trajs)))[wid::nw]

        if self.mode == "train":
            rng = random.Random(
                (self.seed + (worker_info.id if worker_info else 0)) * 9973
            )
            rng.shuffle(indices)
            for it in self._train_yield(all_trajs, indices, rng):
                yield it
        else:
            for it in self._val_yield(all_trajs, indices):
                yield it

    def _train_yield(
        self,
        all_trajs: list[tuple[np.ndarray, bool, int]],
        indices: list[int],
        rng: random.Random,
    ) -> Iterable:
        """Per-episode 2 windows: (end, label=int(complete)) + (random earlier, label=0).

        Mirrors WMPO/reward_model/videomae.py SuccessWindowDataset._windows
        exactly: the end window is positive iff the episode succeeded, every
        earlier window is labeled negative regardless of success. The "end"
        anchor is the WMPO ``finish_step`` (in imagined-traj coords), not the
        full imagined length.
        """
        W, S = self.W, self.stride
        while True:  # resampled infinite stream
            for i in indices:
                traj, complete, finish_step = all_trajs[i]
                T = int(min(finish_step, traj.shape[0]))
                if T < W:
                    continue
                # end window — label is int(complete)
                yield torch.from_numpy(traj[T - W : T]).float(), int(complete)
                # random earlier window — always label 0
                ends = list(range(T - S, W - 1, -S)) or list(range(T - 1, W - 1, -1))
                if not ends:
                    continue
                end = rng.choice(ends)
                yield torch.from_numpy(traj[end - W : end]).float(), 0

    def _val_yield(
        self,
        all_trajs: list[tuple[np.ndarray, bool, int]],
        indices: list[int],
    ) -> Iterable:
        W, S = self.W, self.stride
        for i in indices:
            traj, complete, finish_step = all_trajs[i]
            T = int(min(finish_step, traj.shape[0]))
            if T < W:
                continue
            yield torch.from_numpy(traj[T - W : T]).float(), int(complete)
            for end in range(T - S, W - 1, -S):
                yield torch.from_numpy(traj[end - W : end]).float(), 0


__all__ = ["WMReplayClassifierDataset"]
