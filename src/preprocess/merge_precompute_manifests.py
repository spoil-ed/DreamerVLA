from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge partitioned precompute manifests into a single manifest.")
    parser.add_argument("--root-output-dir", type=Path, required=True)
    parser.add_argument("--expected-parts", type=int, default=None)
    parser.add_argument("--cleanup-parts", action="store_true")
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return torch.load(path, map_location="cpu")


def main() -> None:
    args = _parse_args()
    root_output_dir = args.root_output_dir.expanduser().resolve()
    part_dirs = sorted(path for path in root_output_dir.glob("part_*") if path.is_dir())
    if not part_dirs:
        raise RuntimeError(f"No part directories found under {root_output_dir}")
    if args.expected_parts is not None and len(part_dirs) != int(args.expected_parts):
        raise RuntimeError(
            f"Expected {int(args.expected_parts)} part directories under {root_output_dir}, found {len(part_dirs)}"
        )

    part_manifests = []
    for part_dir in part_dirs:
        manifest_path = part_dir / "manifest.pt"
        manifest = _load_manifest(manifest_path)
        part_manifests.append((part_dir, manifest))

    first_manifest = part_manifests[0][1]
    merged_shards = []
    global_cursor = 0
    hidden_dim = int(first_manifest.get("hidden_dim", 0))
    shard_counter = 0

    for part_dir, manifest in sorted(part_manifests, key=lambda item: int(item[1].get("partition_index", 0))):
        if int(manifest.get("hidden_dim", hidden_dim)) != hidden_dim:
            raise ValueError(f"Hidden dim mismatch in {part_dir / 'manifest.pt'}")
        for shard in manifest["shards"]:
            shard_num = int(shard["num_samples"])
            source_path = part_dir / str(shard["file"])
            merged_file = f"shard_{shard_counter:05d}.pt"
            target_path = root_output_dir / merged_file
            if not source_path.is_file():
                raise FileNotFoundError(f"Shard not found: {source_path}")
            if target_path.exists():
                raise FileExistsError(f"Target shard already exists: {target_path}")
            shutil.move(str(source_path), str(target_path))
            merged_shard = dict(shard)
            merged_shard["file"] = merged_file
            merged_shard["start_index"] = global_cursor
            merged_shard["end_index"] = global_cursor + shard_num
            merged_shards.append(merged_shard)
            global_cursor += shard_num
            shard_counter += 1

    merged_manifest = dict(first_manifest)
    merged_manifest["output_dir"] = str(root_output_dir)
    merged_manifest["num_samples"] = global_cursor
    merged_manifest["num_shards"] = len(merged_shards)
    merged_manifest["shards"] = merged_shards
    merged_manifest["partitions"] = [
        {
            "partition_index": int(manifest.get("partition_index", 0)),
            "num_samples": int(manifest["num_samples"]),
            "manifest": str(Path(part_dir.name) / "manifest.pt"),
        }
        for part_dir, manifest in sorted(part_manifests, key=lambda item: int(item[1].get("partition_index", 0)))
    ]

    merged_manifest_path = root_output_dir / "manifest.pt"
    torch.save(merged_manifest, merged_manifest_path)
    print(
        f"[merge_precompute] merged_parts={len(part_manifests)} shards={len(merged_shards)} "
        f"samples={global_cursor} manifest={merged_manifest_path}"
    )

    if args.cleanup_parts:
        for part_dir, _ in part_manifests:
            shutil.rmtree(part_dir)


if __name__ == "__main__":
    main()
