from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCHEME_NAME = "libero_noop_marking_v1"


def is_noop_action(
    action: np.ndarray,
    prev_action: np.ndarray | None = None,
    *,
    threshold: float = 1e-4,
) -> bool:
    """Return whether a LIBERO action is a filter-equivalent no-op."""
    action = np.asarray(action)
    if prev_action is None:
        return bool(np.linalg.norm(action[:-1]) < float(threshold))
    prev_action = np.asarray(prev_action)
    return bool(
        np.linalg.norm(action[:-1]) < float(threshold)
        and action[-1] == prev_action[-1]
    )


def compute_noop_mask(actions: np.ndarray, *, threshold: float = 1e-4) -> np.ndarray:
    """Mark no-op actions using the same previous-kept-action rule as filtering."""
    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(f"actions must be [T,A], got shape {actions.shape}")
    mask = np.zeros((actions.shape[0],), dtype=np.bool_)
    prev_kept: np.ndarray | None = None
    for idx, action in enumerate(actions):
        is_noop = is_noop_action(action, prev_kept, threshold=threshold)
        mask[idx] = is_noop
        if not is_noop:
            prev_kept = action
    return mask


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _copy_dataset(
    source: h5py.Dataset,
    dest_group: h5py.Group,
    name: str,
    *,
    indices: np.ndarray,
    episode_length: int,
) -> None:
    data = source[()]
    if source.shape and int(source.shape[0]) == int(episode_length):
        data = data[indices]
    created = dest_group.create_dataset(name, data=data, dtype=source.dtype)
    _copy_attrs(source.attrs, created.attrs)


def _copy_group(
    source: h5py.Group,
    dest: h5py.Group,
    *,
    indices: np.ndarray,
    episode_length: int,
) -> None:
    _copy_attrs(source.attrs, dest.attrs)
    for name, item in source.items():
        if name in {"noop_mask", "source_indices"}:
            continue
        if isinstance(item, h5py.Dataset):
            _copy_dataset(
                item,
                dest,
                name,
                indices=indices,
                episode_length=episode_length,
            )
        elif isinstance(item, h5py.Group):
            child = dest.create_group(name)
            _copy_group(
                item,
                child,
                indices=indices,
                episode_length=episode_length,
            )


def _demo_noop_mask(demo: h5py.Group, *, threshold: float) -> np.ndarray:
    if "noop_mask" in demo:
        return np.asarray(demo["noop_mask"], dtype=np.bool_).reshape(-1)
    if "actions" not in demo:
        raise KeyError(f"{demo.name} is missing actions and noop_mask")
    return compute_noop_mask(np.asarray(demo["actions"]), threshold=threshold)


def _demo_source_indices(demo: h5py.Group, episode_length: int) -> np.ndarray:
    if "source_indices" in demo:
        return np.asarray(demo["source_indices"], dtype=np.int64).reshape(-1)
    return np.arange(int(episode_length), dtype=np.int64)


def filter_marked_hdf5_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    filter_noops: bool = True,
    threshold: float = 1e-4,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Copy a marked LIBERO HDF5 file, optionally removing no-op frames."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.exists() and not bool(overwrite):
        raise FileExistsError(f"output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    frames_in = 0
    frames_out = 0
    noop_frames = 0
    demos = 0
    with h5py.File(source_path, "r", swmr=True, libver="latest") as src:
        with h5py.File(tmp_path, "w", libver="latest") as dst:
            _copy_attrs(src.attrs, dst.attrs)
            for name, item in src.items():
                if name != "data":
                    src.copy(item, dst, name=name)
                    continue
                data_out = dst.create_group("data")
                for demo_key in item.keys():
                    demo = item[demo_key]
                    noop_mask = _demo_noop_mask(demo, threshold=float(threshold))
                    episode_length = int(noop_mask.shape[0])
                    source_indices = _demo_source_indices(demo, episode_length)
                    if int(source_indices.shape[0]) != episode_length:
                        raise ValueError(
                            f"{source_path}:{demo_key} source_indices length "
                            f"{source_indices.shape[0]} != episode length {episode_length}"
                        )
                    keep = (
                        np.flatnonzero(~noop_mask)
                        if bool(filter_noops)
                        else np.arange(episode_length, dtype=np.int64)
                    )
                    demo_out = data_out.create_group(demo_key)
                    _copy_group(
                        demo,
                        demo_out,
                        indices=keep,
                        episode_length=episode_length,
                    )
                    filtered_noop_mask = noop_mask[keep]
                    demo_out.create_dataset(
                        "noop_mask", data=filtered_noop_mask.astype(np.bool_)
                    )
                    demo_out.create_dataset(
                        "source_indices", data=source_indices[keep].astype(np.int64)
                    )
                    demo_out.attrs["noop_marking_scheme"] = SCHEME_NAME
                    demo_out.attrs["noop_filtered"] = bool(filter_noops)
                    demo_out.attrs["source_episode_length"] = episode_length
                    demo_out.attrs["source_noop_frames"] = int(noop_mask.sum())
                    frames_in += episode_length
                    frames_out += int(keep.shape[0])
                    noop_frames += int(noop_mask.sum())
                    demos += 1

            dst.attrs["noop_marking_scheme"] = SCHEME_NAME
            dst.attrs["noop_filter_source_hdf5"] = str(source_path)
            dst.attrs["noop_filtered"] = bool(filter_noops)
            dst.attrs["noop_threshold"] = float(threshold)
            dst.attrs["noop_frames_in"] = int(frames_in)
            dst.attrs["noop_frames_out"] = int(frames_out)
            dst.attrs["noop_frames_removed"] = int(noop_frames if filter_noops else 0)

    tmp_path.replace(output_path)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "demos": demos,
        "frames_in": frames_in,
        "frames_out": frames_out,
        "noop_frames": noop_frames,
        "filter_noops": bool(filter_noops),
    }


def filter_marked_hdf5_dir(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    filter_noops: bool = True,
    threshold: float = 1e-4,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Filter every marked LIBERO HDF5 file under a directory."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    files = sorted(source_dir.glob("*.hdf5"))
    if not files:
        raise RuntimeError(f"No HDF5 files found under {source_dir}")
    records = []
    for source_path in files:
        records.append(
            filter_marked_hdf5_file(
                source_path,
                output_dir / source_path.name,
                filter_noops=filter_noops,
                threshold=threshold,
                overwrite=overwrite,
            )
        )
    return records
