#!/usr/bin/env python
"""Build OpenVLA-OFT hidden-token sidecars directly from LeRobot v2.1 episodes."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from dreamervla.dataset.vla_sft_lerobot_dataset import (
    LeRobotEpisode,
    decode_lerobot_video,
    format_lerobot_episode_path,
    load_lerobot_v21_episodes,
    select_lerobot_episodes,
)
from dreamervla.preprocess.preprocess_oft_hidden_token import (
    OBS_HIDDEN_SOURCE,
    _action_head_type_for_mode,
    _hidden_token_sidecar_dims,
    _init_distributed,
    _load_oft_components,
    _loaded_token_dim,
    _predict_hidden_token_images,
    _project_path,
    _resolve_num_images_in_input,
    resolve_oft_policy_mode,
)
from dreamervla.preprocess.sidecar_schema import (
    SIDECAR_SCHEMA_VERSION,
    required_demo_datasets,
    validate_hidden_token_preprocess_config,
)
from dreamervla.utils.hydra_config import script_namespace
from dreamervla.utils.progress import ProgressReporter

SOURCE_FORMAT = "lerobot_v2.1"


def _episode_filename(episode: LeRobotEpisode) -> str:
    return f"episode_{episode.episode_index:06d}.hdf5"


def _episode_group_key(episode: LeRobotEpisode) -> str:
    return f"episode_{episode.episode_index:06d}"


def _select_source_episodes(
    root: Path, args: SimpleNamespace
) -> tuple[dict[str, Any], list[LeRobotEpisode]]:
    info, episodes = load_lerobot_v21_episodes(
        root,
        required_features=(str(args.lerobot_image_key), str(args.action_key), str(args.state_key)),
        action_key=str(args.action_key),
        expected_action_dim=int(args.action_dim),
    )
    selected = select_lerobot_episodes(
        episodes,
        episodes_per_task=(None if args.episodes_per_task is None else int(args.episodes_per_task)),
        episode_selection=str(args.episode_selection),
        episode_selection_seed=int(args.episode_selection_seed),
    )
    if args.max_episodes is not None:
        selected = selected[: int(args.max_episodes)]
    if not selected:
        raise RuntimeError(f"No LeRobot episodes selected under {root}")
    return info, selected


def _validate_source_files(
    root: Path,
    info: dict[str, Any],
    episodes: Sequence[LeRobotEpisode],
    *,
    image_key: str,
) -> None:
    missing: list[str] = []
    for episode in episodes:
        data_path = format_lerobot_episode_path(root, info, "data_path", episode.episode_index)
        video_path = format_lerobot_episode_path(
            root,
            info,
            "video_path",
            episode.episode_index,
            video_key=image_key,
        )
        for path in (data_path, video_path):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Selected LeRobot episodes have missing files: " + ", ".join(missing)
        )


def _assign_episodes_by_frames(
    episodes: Sequence[LeRobotEpisode], *, world_size: int
) -> list[list[LeRobotEpisode]]:
    world_size = max(1, int(world_size))
    buckets: list[list[LeRobotEpisode]] = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]
    for episode in sorted(
        episodes, key=lambda item: (-item.length, item.task_index, item.episode_index)
    ):
        rank = min(range(world_size), key=lambda idx: (loads[idx], len(buckets[idx]), idx))
        buckets[rank].append(episode)
        loads[rank] += episode.length
    for bucket in buckets:
        bucket.sort(key=lambda item: item.episode_index)
    return buckets


def _is_complete_episode_sidecar(
    path: Path,
    episode: LeRobotEpisode,
    *,
    hidden_key: str,
    token_count: int,
    token_dim: int,
    output_dtype: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                return False
            if str(handle.attrs.get("source_format", "")) != SOURCE_FORMAT:
                return False
            if int(handle.attrs.get("source_episode_index", -1)) != episode.episode_index:
                return False
            demo = handle.get(f"data/{_episode_group_key(episode)}")
            if not isinstance(demo, h5py.Group) or not bool(demo.attrs.get("complete", False)):
                return False
            hidden = demo.get(hidden_key)
            lang = demo.get("lang_emb")
            return bool(
                isinstance(hidden, h5py.Dataset)
                and hidden.shape == (episode.length, int(token_count), int(token_dim))
                and hidden.dtype == np.dtype(output_dtype)
                and isinstance(lang, h5py.Dataset)
                and lang.shape == (int(token_dim),)
                and lang.dtype == np.dtype(output_dtype)
            )
    except (OSError, TypeError, ValueError):
        return False


def _estimated_bytes(
    episodes: Sequence[LeRobotEpisode],
    *,
    token_count: int,
    token_dim: int,
    output_dtype: str,
) -> int:
    itemsize = np.dtype(output_dtype).itemsize
    frames = sum(episode.length for episode in episodes)
    return itemsize * (frames * int(token_count) * int(token_dim) + len(episodes) * int(token_dim))


def build_lerobot_hidden_preprocess_config(
    args: SimpleNamespace,
    *,
    lerobot_dir: Path,
    out_hidden_token_dir: Path,
    world_size: int,
    token_count: int,
    episodes: Sequence[LeRobotEpisode],
) -> dict[str, Any]:
    """Build canonical sidecar metadata plus the LeRobot source identity."""

    config = vars(args).copy()
    config.update(
        source_format=SOURCE_FORMAT,
        lerobot_dir=str(lerobot_dir),
        out_hidden_token_dir=str(out_hidden_token_dir),
        selected_episode_indices=[episode.episode_index for episode in episodes],
        selected_task_indices=[episode.task_index for episode in episodes],
        selected_frames=sum(episode.length for episode in episodes),
        world_size=int(world_size),
        start_time=time.time(),
        model_path=str(_project_path(args.oft_ckpt)),
        encoder_state_ckpt="",
        action_head_type=_action_head_type_for_mode(args.resolved_policy_mode),
        obs_hidden_source=OBS_HIDDEN_SOURCE,
        token_count=int(token_count),
        hidden_dim=int(token_count) * int(args.token_dim),
        obs_embedding_shape=[int(token_count), int(args.token_dim)],
        hidden_storage_format="tokenized",
        sidecar_schema_version=SIDECAR_SCHEMA_VERSION,
        required_demo_datasets=required_demo_datasets(),
    )
    validate_hidden_token_preprocess_config(
        config,
        context="OpenVLA-OFT LeRobot preprocess config",
    )
    return config


def _write_attrs(
    handle: h5py.File,
    *,
    root: Path,
    info: dict[str, Any],
    episode: LeRobotEpisode,
    video_path: Path,
    components: dict[str, Any],
    args: SimpleNamespace,
) -> tuple[int, int]:
    token_count, hidden_dim = _hidden_token_sidecar_dims(
        components["vla"],
        image_keys=tuple(args.image_keys),
        token_dim=int(args.token_dim),
    )
    handle.attrs["complete"] = False
    handle.attrs["source_format"] = SOURCE_FORMAT
    handle.attrs["source_lerobot_root"] = str(root)
    handle.attrs["source_codebase_version"] = str(info["codebase_version"])
    handle.attrs["source_episode_index"] = episode.episode_index
    handle.attrs["source_task_index"] = episode.task_index
    handle.attrs["source_task"] = episode.task
    handle.attrs["source_video"] = str(video_path)
    handle.attrs["source_images_vla_aligned"] = bool(args.source_images_vla_aligned)
    handle.attrs["hidden_key"] = str(args.hidden_key)
    handle.attrs["hidden_dim"] = hidden_dim
    handle.attrs["obs_hidden_source"] = OBS_HIDDEN_SOURCE
    handle.attrs["obs_embedding_shape"] = np.asarray(
        [token_count, int(args.token_dim)], dtype=np.int64
    )
    handle.attrs["hidden_storage_format"] = "tokenized"
    handle.attrs["token_count"] = token_count
    handle.attrs["token_dim"] = int(args.token_dim)
    handle.attrs["output_dtype"] = str(np.dtype(args.output_dtype))
    handle.attrs["image_keys"] = json.dumps(list(args.image_keys))
    handle.attrs["lerobot_image_key"] = str(args.lerobot_image_key)
    handle.attrs["prompt_style"] = str(args.prompt_style)
    handle.attrs["history"] = int(args.history)
    handle.attrs["include_state"] = False
    handle.attrs["rotate_images_180"] = bool(args.rotate_images_180)
    handle.attrs["resolution"] = int(args.resolution)
    handle.attrs["model_path"] = str(_project_path(args.oft_ckpt))
    handle.attrs["action_head_type"] = _action_head_type_for_mode(args.resolved_policy_mode)
    handle.attrs["time_horizon"] = int(args.time_horizon)
    handle.attrs["chunk_size"] = int(args.chunk_size)
    return token_count, hidden_dim


def _write_episode_hidden(
    *,
    root: Path,
    info: dict[str, Any],
    episode: LeRobotEpisode,
    output_path: Path,
    components: dict[str, Any],
    args: SimpleNamespace,
    rank: int,
) -> None:
    video_path = format_lerobot_episode_path(
        root,
        info,
        "video_path",
        episode.episode_index,
        video_key=str(args.lerobot_image_key),
    )
    frames = decode_lerobot_video(video_path, expected_length=episode.length)
    if bool(args.rotate_images_180):
        frames = np.ascontiguousarray(frames[:, ::-1, ::-1])

    tmp_path = output_path.with_name(f"{output_path.name}.rank{rank}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    dtype = np.dtype(args.output_dtype)
    with h5py.File(tmp_path, "w", libver="latest") as output:
        token_count, hidden_dim = _write_attrs(
            output,
            root=root,
            info=info,
            episode=episode,
            video_path=video_path,
            components=components,
            args=args,
        )
        demo = output.create_group(f"data/{_episode_group_key(episode)}")
        demo.attrs["complete"] = False
        demo.attrs["length"] = episode.length
        demo.attrs["episode_index"] = episode.episode_index
        demo.attrs["task_index"] = episode.task_index
        demo.attrs["task_prompt"] = episode.task
        hidden = demo.create_dataset(
            str(args.hidden_key),
            shape=(episode.length, token_count, int(args.token_dim)),
            dtype=dtype,
            chunks=(1, token_count, int(args.token_dim)),
            compression=None,
        )
        hidden.attrs["hidden_dim"] = hidden_dim
        hidden.attrs["source_dtype"] = "float32"
        hidden.attrs["token_count"] = token_count
        hidden.attrs["token_dim"] = int(args.token_dim)
        hidden.attrs["hidden_storage_format"] = "tokenized"
        lang = demo.create_dataset(
            "lang_emb", shape=(int(args.token_dim),), dtype=dtype, compression=None
        )
        lang.attrs["hidden_dim"] = int(args.token_dim)
        lang.attrs["source_dtype"] = "float32"

        with ProgressReporter(
            episode.length,
            f"rank{rank} episode_{episode.episode_index:06d}",
            unit="frame",
        ) as progress:
            for start in range(0, episode.length, int(args.chunk_size)):
                end = min(start + int(args.chunk_size), episode.length)
                hidden_token, lang_emb = _predict_hidden_token_images(
                    components=components,
                    args=args,
                    images_by_frame=[[frame] for frame in frames[start:end]],
                    prompt=episode.task,
                )
                hidden[start:end] = hidden_token.astype(dtype, copy=False)
                if start == 0:
                    lang[...] = lang_emb[0].astype(dtype, copy=False)
                progress.update(end - start)
        demo.attrs["complete"] = True
        output.attrs["complete"] = True
    tmp_path.replace(output_path)


def _validate_outputs(
    out_dir: Path,
    episodes: Sequence[LeRobotEpisode],
    args: SimpleNamespace,
    *,
    token_count: int,
) -> None:
    config_path = out_dir / "preprocess_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing LeRobot hidden preprocess config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_hidden_token_preprocess_config(config, context=str(config_path))
    selected = [episode.episode_index for episode in episodes]
    if list(config.get("selected_episode_indices", [])) != selected:
        raise ValueError(
            "selected episode mismatch: "
            f"config={config.get('selected_episode_indices')!r}, expected={selected!r}"
        )
    invalid = [
        _episode_filename(episode)
        for episode in episodes
        if not _is_complete_episode_sidecar(
            out_dir / _episode_filename(episode),
            episode,
            hidden_key=str(args.hidden_key),
            token_count=token_count,
            token_dim=int(args.token_dim),
            output_dtype=str(args.output_dtype),
        )
    ]
    if invalid:
        raise RuntimeError(f"incomplete LeRobot hidden sidecars: {invalid}")


def parse_args() -> SimpleNamespace:
    """Resolve the Hydra-owned LeRobot hidden preprocessing configuration."""

    return script_namespace("preprocess_oft_hidden_token_lerobot")


def main() -> None:
    args = parse_args()
    if str(args.obs_hidden_source) != OBS_HIDDEN_SOURCE:
        raise SystemExit("OpenVLA-OFT preprocessing only supports obs_hidden_source=hidden_token")
    args.include_state = False
    args.resolved_policy_mode = (
        "discrete"
        if bool(args.fake_oft_components)
        else resolve_oft_policy_mode(_project_path(args.oft_ckpt), args.policy_mode)
    )
    if int(args.history) != 1 or int(_resolve_num_images_in_input(args)) != 1:
        raise SystemExit("OpenVLA-OFT preprocessing requires history=1 and one image")
    if not bool(args.source_images_vla_aligned):
        raise SystemExit("LeRobot source must declare source_images_vla_aligned=true")
    if bool(args.rotate_images_180):
        raise SystemExit("VLA-aligned LeRobot videos must not be rotated a second time")
    if int(args.chunk_size) < 1:
        raise SystemExit("chunk_size must be >= 1")

    root = _project_path(args.lerobot_dir)
    out_dir = _project_path(args.out_hidden_token_dir)
    info, episodes = _select_source_episodes(root, args)
    _validate_source_files(root, info, episodes, image_key=str(args.lerobot_image_key))
    token_count = int(args.patches_per_image) * len(tuple(args.image_keys))
    estimate = _estimated_bytes(
        episodes,
        token_count=token_count,
        token_dim=int(args.token_dim),
        output_dtype=str(args.output_dtype),
    )

    if str(args.command) == "plan":
        print(
            json.dumps(
                {
                    "source_format": SOURCE_FORMAT,
                    "source": str(root),
                    "output": str(out_dir),
                    "episodes": [episode.episode_index for episode in episodes],
                    "tasks": [episode.task for episode in episodes],
                    "frames": sum(episode.length for episode in episodes),
                    "estimated_bytes": estimate,
                    "estimated_gib": estimate / (1024**3),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if str(args.command) == "validate":
        _validate_outputs(out_dir, episodes, args, token_count=token_count)
        print(
            f"[oft-hidden-lerobot] valid episodes={len(episodes)} "
            f"frames={sum(episode.length for episode in episodes)} out={out_dir}"
        )
        return
    if str(args.command) != "preprocess":
        raise SystemExit("command must be preprocess|plan|validate")

    out_dir.mkdir(parents=True, exist_ok=True)
    rank, world_size, device = _init_distributed()
    pending: list[LeRobotEpisode] = []
    skipped: list[LeRobotEpisode] = []
    for episode in episodes:
        output = out_dir / _episode_filename(episode)
        if bool(args.overwrite) or not _is_complete_episode_sidecar(
            output,
            episode,
            hidden_key=str(args.hidden_key),
            token_count=token_count,
            token_dim=int(args.token_dim),
            output_dtype=str(args.output_dtype),
        ):
            pending.append(episode)
        else:
            skipped.append(episode)

    assignments = _assign_episodes_by_frames(pending, world_size=world_size)
    assigned = assignments[rank]
    if rank == 0:
        config = build_lerobot_hidden_preprocess_config(
            args,
            lerobot_dir=root,
            out_hidden_token_dir=out_dir,
            world_size=world_size,
            token_count=token_count,
            episodes=episodes,
        )
        (out_dir / "preprocess_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[oft-hidden-lerobot] episodes={len(episodes)} pending={len(pending)} "
            f"skipped={len(skipped)} frames={sum(e.length for e in episodes)} "
            f"estimated_gib={estimate / (1024**3):.3f} "
            f"loads_by_rank={[sum(e.length for e in bucket) for bucket in assignments]}"
        )
    if not assigned:
        print(f"[rank{rank}] done episodes=0 frames=0")
        return

    components = _load_oft_components(args, device)
    actual_patches = int(components["vla"].vision_backbone.get_num_patches())
    if actual_patches != int(args.patches_per_image):
        raise ValueError(
            f"patches_per_image mismatch: model={actual_patches}, "
            f"config={int(args.patches_per_image)}"
        )
    actual_token_dim = _loaded_token_dim(components["vla"])
    if actual_token_dim != int(args.token_dim):
        raise ValueError(
            f"token_dim mismatch: model={actual_token_dim}, config={int(args.token_dim)}"
        )

    frames_written = 0
    for episode in assigned:
        output = out_dir / _episode_filename(episode)
        if output.exists():
            output.unlink()
        _write_episode_hidden(
            root=root,
            info=info,
            episode=episode,
            output_path=output,
            components=components,
            args=args,
            rank=rank,
        )
        frames_written += episode.length
        print(
            f"[rank{rank}] wrote {_episode_filename(episode)} "
            f"task={episode.task_index} frames={episode.length}"
        )
    print(f"[rank{rank}] done episodes={len(assigned)} frames={frames_written}")


if __name__ == "__main__":
    main()
