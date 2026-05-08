#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models.encoder.protocol import EncoderInputBatch
from src.models.encoder.rynnvla_encoder import RynnVLAEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _default_ckpt_path(*parts: str) -> str:
    return str((PROJECT_ROOT / "data" / "ckpts" / Path(*parts)).resolve())


def _demo_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        try:
            return int(name.split("_")[-1]), name
        except ValueError:
            pass
    return 10**9, name


def _list_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted((str(key) for key in data_group.keys()), key=_demo_sort_key)


def _task_prompt_from_path(path: str | Path) -> str:
    stem = Path(path).name
    if stem.endswith("_demo.hdf5"):
        stem = stem[: -len("_demo.hdf5")]
    else:
        stem = Path(stem).stem
    return stem.replace("_", " ")


def _init_distributed() -> tuple[int, int, int, torch.device]:
    """Read torchrun rank env vars without creating a NCCL process group."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def _is_complete_hdf5(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return bool(handle.attrs.get("complete", False))
    except OSError:
        return False


def _hidden_dtype(name: str) -> np.dtype:
    normalized = str(name).lower()
    if normalized in {"fp16", "float16", "half"}:
        return np.dtype("float16")
    if normalized in {"fp32", "float32"}:
        return np.dtype("float32")
    raise ValueError(f"Unsupported output dtype: {name}")


def _compression(name: str) -> str | None:
    normalized = str(name).lower()
    if normalized in {"none", "null", "false", "0", ""}:
        return None
    if normalized not in {"lzf", "gzip"}:
        raise ValueError(f"Unsupported HDF5 compression: {name}")
    return normalized


def _source_stats(source_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    demos = 0
    frames = 0
    with h5py.File(source_path, "r", swmr=True, libver="latest") as handle:
        data_group = handle["data"]
        demo_keys = _list_demo_keys(data_group)
        if args.max_demos_per_file is not None:
            demo_keys = demo_keys[: int(args.max_demos_per_file)]
        for demo_key in demo_keys:
            demos += 1
            frames += int(data_group[demo_key]["actions"].shape[0])
    return {
        "source": str(source_path),
        "file": source_path.name,
        "demos": demos,
        "frames": frames,
    }


def _write_rank_progress(
    progress_dir: Path,
    rank: int,
    *,
    run_id: str,
    completed_demos: int,
    completed_frames: int,
    current_file: str | None = None,
    current_demo: str | None = None,
    done: bool = False,
) -> None:
    progress_dir.mkdir(parents=True, exist_ok=True)
    path = progress_dir / f"rank{rank:03d}.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "run_id": run_id,
        "rank": rank,
        "completed_demos": int(completed_demos),
        "completed_frames": int(completed_frames),
        "current_file": current_file,
        "current_demo": current_demo,
        "done": bool(done),
        "time": time.time(),
    }
    tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _read_progress(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_rank_progress(progress_dir: Path, rank: int, run_id: str) -> dict[str, Any] | None:
    progress = _read_progress(progress_dir / f"rank{rank:03d}.json")
    if not progress or progress.get("run_id") != run_id:
        return None
    return progress


def _all_ranks_done(progress_dir: Path, run_id: str, world_size: int) -> bool:
    for rank in range(world_size):
        progress = _read_rank_progress(progress_dir, rank, run_id)
        if not progress or not bool(progress.get("done", False)):
            return False
    return True


def _monitor_global_progress(
    progress_dir: Path,
    *,
    run_id: str,
    total_demos: int,
    total_frames: int,
    world_size: int,
    stop_event: threading.Event,
) -> None:
    if total_demos <= 0:
        return
    last_demos = 0
    with tqdm(
        total=total_demos,
        desc="total demos",
        position=world_size,
        leave=True,
        mininterval=1.0,
    ) as pbar:
        while not stop_event.is_set():
            completed_demos = 0
            completed_frames = 0
            for rank in range(world_size):
                progress = _read_rank_progress(progress_dir, rank, run_id)
                if not progress:
                    continue
                completed_demos += int(progress.get("completed_demos", 0))
                completed_frames += int(progress.get("completed_frames", 0))
            if completed_demos > last_demos:
                pbar.update(completed_demos - last_demos)
                last_demos = completed_demos
            pbar.set_postfix(
                frames=f"{completed_frames}/{total_frames}",
                refresh=False,
            )
            if completed_demos >= total_demos:
                break
            time.sleep(2.0)

        completed_demos = 0
        completed_frames = 0
        for rank in range(world_size):
            progress = _read_rank_progress(progress_dir, rank, run_id)
            if not progress:
                continue
            completed_demos += int(progress.get("completed_demos", 0))
            completed_frames += int(progress.get("completed_frames", 0))
        if completed_demos > last_demos:
            pbar.update(completed_demos - last_demos)
        pbar.set_postfix(frames=f"{completed_frames}/{total_frames}", refresh=True)


def _make_encoder(args: argparse.Namespace, device: torch.device) -> RynnVLAEncoder:
    encoder = RynnVLAEncoder(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        text_tokenizer_path=args.text_tokenizer_path,
        chameleon_vqgan_config=args.chameleon_vqgan_config,
        chameleon_vqgan_ckpt=args.chameleon_vqgan_ckpt,
        resolution=args.resolution,
        action_dim=args.action_dim,
        time_horizon=args.time_horizon,
        pool=args.pool,
        freeze_backbone=True,
    ).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def _encode_chunk(
    encoder: RynnVLAEncoder,
    prompt: str,
    obs_group: h5py.Group,
    image_keys: tuple[str, ...],
    start: int,
    end: int,
) -> np.ndarray:
    prompt_text: list[str] = []
    conversations: list[list[dict[str, str]]] = []
    image_batches: list[list[Any]] = []
    for tidx in range(start, end):
        views: list[Image.Image] = []
        for key in image_keys:
            image = np.asarray(obs_group[key][tidx], dtype=np.uint8)
            views.append(Image.fromarray(image))
        prompt_text.append(prompt)
        conversations.append([])
        image_batches.append(views)
    batch = EncoderInputBatch(
        prompt_text=prompt_text,
        conversations=conversations,
        images=image_batches,
        task_type=None,
    )
    with torch.no_grad():
        hidden = encoder.encode_inputs(batch).hidden.detach().cpu().numpy()
    return hidden


def _write_source_sidecar(
    source_path: Path,
    output_path: Path,
    encoder: RynnVLAEncoder,
    args: argparse.Namespace,
    rank: int,
    run_id: str,
    progress_dir: Path,
    completed_demos_base: int,
    completed_frames_base: int,
) -> dict[str, Any]:
    tmp_path = output_path.with_name(f"{output_path.name}.rank{rank}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    image_keys = tuple(args.image_keys)
    hidden_dtype = _hidden_dtype(args.output_dtype)
    compression = _compression(args.compression)
    prompt = _task_prompt_from_path(source_path)
    demos_written = 0
    frames_written = 0
    hidden_dim: int | None = None

    with h5py.File(source_path, "r", swmr=True, libver="latest") as source, h5py.File(tmp_path, "w", libver="latest") as out:
        data_group = source["data"]
        out_data = out.create_group("data")
        demo_keys = _list_demo_keys(data_group)
        if args.max_demos_per_file is not None:
            demo_keys = demo_keys[: int(args.max_demos_per_file)]

        iterator = tqdm(
            demo_keys,
            desc=f"rank{rank} {source_path.name}",
            position=rank,
            leave=False,
            mininterval=1.0,
        )
        for demo_key in iterator:
            demo = data_group[demo_key]
            obs_group = demo["obs"]
            for key in image_keys:
                if key not in obs_group:
                    raise KeyError(f"{source_path}:{demo_key} missing obs/{key}")
            length = int(demo["actions"].shape[0])
            demo_out = out_data.create_group(demo_key)
            demo_out.attrs["length"] = length
            demo_out.attrs["task_prompt"] = prompt
            dset = None
            for start in range(0, length, int(args.chunk_size)):
                end = min(start + int(args.chunk_size), length)
                hidden = _encode_chunk(
                    encoder=encoder,
                    prompt=prompt,
                    obs_group=obs_group,
                    image_keys=image_keys,
                    start=start,
                    end=end,
                )
                if dset is None:
                    hidden_dim = int(hidden.shape[-1])
                    dset = demo_out.create_dataset(
                        args.hidden_key,
                        shape=(length, hidden_dim),
                        dtype=hidden_dtype,
                        chunks=(min(max(1, int(args.chunk_size)), length), hidden_dim),
                        compression=compression,
                    )
                    dset.attrs["hidden_dim"] = hidden_dim
                    dset.attrs["source_dtype"] = "float32"
                dset[start:end] = hidden.astype(hidden_dtype, copy=False)
                frames_written += end - start
            demos_written += 1
            _write_rank_progress(
                progress_dir,
                rank,
                run_id=run_id,
                completed_demos=completed_demos_base + demos_written,
                completed_frames=completed_frames_base + frames_written,
                current_file=source_path.name,
                current_demo=demo_key,
            )

        out.attrs["complete"] = False
        out.attrs["source_hdf5"] = str(source_path)
        out.attrs["source_hdf5_dir"] = str(source_path.parent)
        out.attrs["hidden_key"] = str(args.hidden_key)
        out.attrs["hidden_dim"] = int(hidden_dim or 0)
        out.attrs["output_dtype"] = str(hidden_dtype)
        out.attrs["image_keys"] = json.dumps(list(image_keys))
        out.attrs["resolution"] = int(args.resolution)
        out.attrs["model_path"] = str(args.model_path)
        out.attrs["pool"] = str(args.pool)
        out.attrs["complete"] = True

    tmp_path.replace(output_path)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "demos": demos_written,
        "frames": frames_written,
        "hidden_dim": int(hidden_dim or 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute frozen RynnVLA hidden vectors for LIBERO pixel HDF5 files. "
            "The original image dataset is not modified; matching sidecar HDF5 "
            "files are written under --out-dir."
        )
    )
    parser.add_argument(
        "--hdf5-dir",
        default=str(PROJECT_ROOT / "data" / "processed_data" / "libero_goal_no_noops_t_256"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data" / "processed_data" / "libero_goal_no_noops_t_256_rynn_hidden"),
    )
    parser.add_argument("--image-keys", nargs="+", default=["agentview_rgb", "eye_in_hand_rgb"])
    parser.add_argument("--hidden-key", default="obs_embedding")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--output-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--compression", default="none", choices=["none", "lzf", "gzip"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-demos-per-file", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-global-progress", action="store_true")
    parser.add_argument("--model-path", default=_default_ckpt_path("VLA_model_256", "libero_10"))
    parser.add_argument("--tokenizer-path", default=_default_ckpt_path("models--Alpha-VLLM--Lumina-mGPT-7B-768"))
    parser.add_argument(
        "--text-tokenizer-path",
        default=_default_ckpt_path("chameleon", "tokenizer", "text_tokenizer.json"),
    )
    parser.add_argument(
        "--chameleon-vqgan-config",
        default=_default_ckpt_path("chameleon", "tokenizer", "vqgan.yaml"),
    )
    parser.add_argument(
        "--chameleon-vqgan-ckpt",
        default=_default_ckpt_path("chameleon", "tokenizer", "vqgan.ckpt"),
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--time-horizon", type=int, default=10)
    parser.add_argument("--pool", default="mean", choices=["mean", "last"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_dir = _project_path(args.hdf5_dir)
    out_dir = _project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank, world_size, _local_rank, device = _init_distributed()
    run_id = (
        os.environ.get("RYNN_HIDDEN_RUN_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or f"manual_{int(time.time())}"
    )

    files = sorted(hdf5_dir.glob("*.hdf5"))
    if args.max_files is not None:
        files = files[: int(args.max_files)]
    if not files:
        raise RuntimeError(f"No HDF5 files found under {hdf5_dir}")
    assigned = files[rank::world_size]
    assigned_stats = {stat["file"]: stat for stat in (_source_stats(path, args) for path in assigned)}
    progress_dir = out_dir / ".progress"
    total_demos = 0
    total_frames = 0
    stop_event: threading.Event | None = None
    monitor_thread: threading.Thread | None = None

    if rank == 0:
        all_stats = [_source_stats(path, args) for path in files]
        total_demos = sum(int(stat["demos"]) for stat in all_stats)
        total_frames = sum(int(stat["frames"]) for stat in all_stats)
        config_path = out_dir / "preprocess_config.json"
        config = vars(args).copy()
        config["hdf5_dir"] = str(hdf5_dir)
        config["out_dir"] = str(out_dir)
        config["num_source_files"] = len(files)
        config["world_size"] = world_size
        config["run_id"] = run_id
        config["total_demos"] = total_demos
        config["total_frames"] = total_frames
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        (out_dir / "preprocess_plan.json").write_text(
            json.dumps(
                {
                    "total_demos": total_demos,
                    "total_frames": total_frames,
                    "run_id": run_id,
                    "files": all_stats,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        progress_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[rynn-hidden] source={hdf5_dir} out={out_dir} "
            f"files={len(files)} demos={total_demos} frames={total_frames} "
            f"world_size={world_size} run_id={run_id}"
        )

    _write_rank_progress(
        progress_dir,
        rank,
        run_id=run_id,
        completed_demos=0,
        completed_frames=0,
    )
    if rank == 0 and not bool(args.no_global_progress):
        if total_demos <= 0:
            plan_path = out_dir / "preprocess_plan.json"
            if plan_path.is_file():
                plan = json.loads(plan_path.read_text())
                total_demos = int(plan.get("total_demos", 0))
                total_frames = int(plan.get("total_frames", 0))
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_global_progress,
            kwargs={
                "progress_dir": progress_dir,
                "run_id": run_id,
                "total_demos": total_demos,
                "total_frames": total_frames,
                "world_size": world_size,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        monitor_thread.start()

    encoder = _make_encoder(args, device=device)
    rank_records: list[dict[str, Any]] = []
    completed_demos = 0
    completed_frames = 0
    for source_path in assigned:
        output_path = out_dir / source_path.name
        source_stat = assigned_stats[source_path.name]
        if output_path.exists():
            if args.overwrite:
                output_path.unlink()
            elif _is_complete_hdf5(output_path):
                print(f"[rank{rank}] skip complete {output_path}")
                completed_demos += int(source_stat["demos"])
                completed_frames += int(source_stat["frames"])
                _write_rank_progress(
                    progress_dir,
                    rank,
                    run_id=run_id,
                    completed_demos=completed_demos,
                    completed_frames=completed_frames,
                    current_file=source_path.name,
                    done=False,
                )
                continue
            else:
                raise RuntimeError(
                    f"Refusing to use incomplete sidecar without --overwrite: {output_path}"
                )
        record = _write_source_sidecar(
            source_path=source_path,
            output_path=output_path,
            encoder=encoder,
            args=args,
            rank=rank,
            run_id=run_id,
            progress_dir=progress_dir,
            completed_demos_base=completed_demos,
            completed_frames_base=completed_frames,
        )
        completed_demos += int(record["demos"])
        completed_frames += int(record["frames"])
        _write_rank_progress(
            progress_dir,
            rank,
            run_id=run_id,
            completed_demos=completed_demos,
            completed_frames=completed_frames,
            current_file=source_path.name,
        )
        rank_records.append(record)
        print(
            f"[rank{rank}] wrote {output_path} "
            f"demos={record['demos']} frames={record['frames']} hidden_dim={record['hidden_dim']}"
        )

    _write_rank_progress(
        progress_dir,
        rank,
        run_id=run_id,
        completed_demos=completed_demos,
        completed_frames=completed_frames,
        done=True,
    )

    (out_dir / f"manifest_rank{rank:03d}.json").write_text(
        json.dumps(rank_records, indent=2, sort_keys=True) + "\n"
    )
    if rank == 0:
        if monitor_thread is not None:
            while not _all_ranks_done(progress_dir, run_id, world_size):
                time.sleep(2.0)
        if stop_event is not None:
            stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
        print("[rynn-hidden] done")


if __name__ == "__main__":
    main()
