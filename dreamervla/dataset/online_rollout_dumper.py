"""Append-only HDF5 dumper for online LIBERO rollouts.

Writes one episode at a time into sharded HDF5 files that match the
``WMReplayClassifierDataset`` schema:

    <raw_dir>/shard_<NNN>.hdf5      data/<demo_key>/{actions, dones, rewards}
    <hidden_dir>/shard_<NNN>.hdf5   data/<demo_key>/obs_embedding

Designed to piggy-back on existing training loops: each completed episode is
already a ``list[dict]`` with ``obs_embedding`` / ``wm_action`` / ``reward``
/ ``done`` / ``is_terminal`` keys (see
the sync or Ray cotrain runner). Wire the dumper next
to ``replay.add_episode(episode)`` and rollouts get persisted to disk for free
— no extra GPU time, no separate collection pass.

The dataset derives ``finish_step`` from ``dones`` and ``complete`` from
``rewards.sum() > 0``, matching LUMOS's ``meta["finish_step"]`` /
``meta["complete"]`` semantics. The dumper writes ``dones[-1] = 1`` for every
episode (terminal OR timeout); successful episodes additionally carry positive
``rewards`` at the success step, so ``complete`` will be True iff the rollout
actually solved the task.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class RolloutDumper:
    """Shard-based HDF5 writer for online rollouts.

    Parameters
    ----------
    raw_dir, hidden_dir
        Output directories. Created on demand.
    episodes_per_shard
        Cut a new shard pair after this many episodes.
    shard_prefix
        Filename prefix; final filenames are ``{prefix}_{NNN}.hdf5``.
    manifest_path
        Optional JSONL log. One line per dumped episode with
        ``{shard, demo_key, task_id, length, success}``.
    """

    def __init__(
        self,
        *,
        raw_dir: str | Path,
        hidden_dir: str | Path,
        episodes_per_shard: int = 25,
        shard_prefix: str = "shard",
        manifest_path: str | Path | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir).expanduser().resolve()
        self.hidden_dir = Path(hidden_dir).expanduser().resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_shard = int(episodes_per_shard)
        self.shard_prefix = str(shard_prefix)

        self._shard_idx = 0
        self._ep_in_shard = 0
        self._total_episodes = 0
        self._total_success = 0
        self._raw_f: h5py.File | None = None
        self._hidden_f: h5py.File | None = None
        self._raw_grp: h5py.Group | None = None
        self._hidden_grp: h5py.Group | None = None
        self._closed = False

        self._manifest_handle = None
        if manifest_path is not None:
            self._manifest_handle = open(
                Path(manifest_path).expanduser().resolve(), "a", encoding="utf-8"
            )

    # ─── public API ────────────────────────────────────────────────────────
    def add_episode(
        self,
        episode: Iterable[Mapping[str, Any]],
        *,
        task_id: int | None = None,
        success: bool | None = None,
    ) -> dict[str, Any]:
        """Append one episode to the current shard. Rotates shards when full.

        ``episode`` is the list-of-dicts already produced by the online loop —
        each step dict must contain ``obs_embedding``, ``wm_action``,
        ``reward``, ``done``. Extra keys are ignored. ``success`` overrides
        the derived ``rewards.sum() > 0`` flag in the manifest log (purely
        informational; the dataset re-derives complete from rewards).
        """
        if self._closed:
            raise RuntimeError("RolloutDumper has been closed")
        steps = list(episode)
        T = len(steps)
        if T == 0:
            return {
                "shard": self._shard_idx,
                "demo_key": None,
                "length": 0,
                "success": False,
            }

        obs = np.stack(
            [np.asarray(s["obs_embedding"], dtype=np.float32) for s in steps], axis=0
        )
        actions = np.stack(
            [
                np.asarray(s["wm_action"], dtype=np.float32).reshape(-1)[:7]
                for s in steps
            ],
            axis=0,
        )
        rewards = np.array([float(s["reward"]) for s in steps], dtype=np.float32)
        dones = np.zeros((T,), dtype=np.uint8)
        dones[-1] = 1  # episode boundary — terminal OR timeout

        derived_success = bool(rewards.sum() > 0)
        if success is None:
            success = derived_success

        demo_key = f"demo_{self._total_episodes:05d}"
        if self._raw_grp is None or self._hidden_grp is None:
            self._open_shard()
        assert self._raw_grp is not None and self._hidden_grp is not None
        self._write(self._raw_grp, demo_key, "actions", actions)
        self._write(self._raw_grp, demo_key, "dones", dones)
        self._write(
            self._raw_grp, demo_key, "rewards", rewards.astype(np.uint8, copy=False)
        )
        self._write(self._hidden_grp, demo_key, "obs_embedding", obs)
        assert self._raw_f is not None and self._hidden_f is not None
        self._raw_f.flush()
        self._hidden_f.flush()

        log_entry = {
            "shard": int(self._shard_idx),
            "demo_key": demo_key,
            "task_id": int(task_id) if task_id is not None else -1,
            "length": int(T),
            "success": bool(success),
        }
        if self._manifest_handle is not None:
            self._manifest_handle.write(json.dumps(log_entry) + "\n")
            self._manifest_handle.flush()

        self._total_episodes += 1
        self._total_success += int(success)
        self._ep_in_shard += 1
        if self._ep_in_shard >= self.episodes_per_shard:
            self._close_shard()
            self._shard_idx += 1
            self._ep_in_shard = 0
            # New shard opens lazily on next add_episode — avoids leaving an
            # empty stub when the run ends right at a rotation boundary.
        return log_entry

    def close(self) -> None:
        if self._closed:
            return
        self._close_shard()
        if self._manifest_handle is not None:
            self._manifest_handle.close()
        self._closed = True

    @property
    def total_episodes(self) -> int:
        return int(self._total_episodes)

    @property
    def total_success(self) -> int:
        return int(self._total_success)

    @property
    def shard_index(self) -> int:
        return int(self._shard_idx)

    @property
    def shards_written(self) -> int:
        """Number of shard files that have at least one episode in them."""
        if self.episodes_per_shard <= 0:
            return 0
        full = self._total_episodes // self.episodes_per_shard
        partial = 1 if self._total_episodes % self.episodes_per_shard else 0
        return int(full + partial)

    def __enter__(self) -> RolloutDumper:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ─── internals ─────────────────────────────────────────────────────────
    def _open_shard(self) -> None:
        raw_path = self.raw_dir / f"{self.shard_prefix}_{self._shard_idx:03d}.hdf5"
        hidden_path = (
            self.hidden_dir / f"{self.shard_prefix}_{self._shard_idx:03d}.hdf5"
        )
        # If a previous run wrote this shard, append rather than truncate so
        # mid-training restarts don't clobber existing data.
        raw_mode = "a" if raw_path.exists() else "w"
        hidden_mode = "a" if hidden_path.exists() else "w"
        self._raw_f = h5py.File(str(raw_path), raw_mode)
        self._hidden_f = h5py.File(str(hidden_path), hidden_mode)
        self._raw_grp = self._raw_f.require_group("data")
        self._hidden_grp = self._hidden_f.require_group("data")

    def _close_shard(self) -> None:
        for handle_attr in ("_raw_f", "_hidden_f"):
            handle = getattr(self, handle_attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, handle_attr, None)
        self._raw_grp = None
        self._hidden_grp = None

    @staticmethod
    def _write(grp: h5py.Group, demo_key: str, name: str, value: np.ndarray) -> None:
        path = f"{demo_key}/{name}"
        if path in grp:
            del grp[path]
        grp.create_dataset(path, data=value, compression="gzip")


__all__ = ["RolloutDumper"]
