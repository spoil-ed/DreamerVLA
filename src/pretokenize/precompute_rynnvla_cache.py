from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader.rynnvla_dataset import RynnVLALIBERODataset
from src.models.encoder.rynnvla_encoder import RynnVLAEncoder


DEFAULT_OUTPUT_DIR = Path("/home/yuxinglei/workspace/2026nips/Dreamer-VLA/data/pretokenized")
DEFAULT_CONFIG_PATH = Path("/home/yuxinglei/workspace/2026nips/Dreamer-VLA/configs/rynnvla_libero_object.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute RynnVLA encoder embeddings for world model training.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "rynnvla_libero_object")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--embedding-dtype", type=str, choices=("float16", "float32"), default="float32")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--without-state", action="store_true")
    parser.add_argument("--without-wrist", action="store_true")
    parser.add_argument("--without-action", action="store_true")
    parser.add_argument("--without-world-model", action="store_true")
    return parser.parse_args()


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already contains files: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _to_storage_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    return torch.float32


def _copy_tensor_list(tensors: list[torch.Tensor], dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.cat(tensors, dim=0)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.cpu().contiguous()


def _pad_action_tensors(actions: list[torch.Tensor], action_masks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_steps = max(int(tensor.shape[1]) for tensor in actions)
    action_dim = int(actions[0].shape[-1])
    total = sum(int(tensor.shape[0]) for tensor in actions)
    padded_actions = torch.zeros(total, max_steps, action_dim, dtype=torch.float32)
    padded_masks = torch.zeros(total, max_steps, dtype=torch.bool)
    cursor = 0
    for action_tensor, mask_tensor in zip(actions, action_masks):
        batch = int(action_tensor.shape[0])
        steps = int(action_tensor.shape[1])
        padded_actions[cursor:cursor + batch, :steps] = action_tensor.float()
        padded_masks[cursor:cursor + batch, :steps] = mask_tensor.bool()
        cursor += batch
    return padded_actions, padded_masks


def _save_shard(
    output_dir: Path,
    shard_idx: int,
    start_index: int,
    obs_embeddings: list[torch.Tensor],
    next_obs_embeddings: list[torch.Tensor],
    actions: list[torch.Tensor],
    action_masks: list[torch.Tensor],
    rewards: list[torch.Tensor],
    metas: list[dict[str, Any]],
    embedding_dtype: torch.dtype,
) -> dict[str, Any]:
    obs_tensor = _copy_tensor_list(obs_embeddings, dtype=embedding_dtype)
    next_obs_tensor = _copy_tensor_list(next_obs_embeddings, dtype=embedding_dtype)
    action_tensor, action_mask_tensor = _pad_action_tensors(actions, action_masks)
    reward_tensor = _copy_tensor_list(rewards)

    shard_name = f"shard_{shard_idx:05d}.pt"
    shard_path = output_dir / shard_name
    payload = {
        "start_index": int(start_index),
        "end_index": int(start_index + obs_tensor.shape[0]),
        "obs_embedding": obs_tensor,
        "next_obs_embedding": next_obs_tensor,
        "action": action_tensor,
        "action_mask": action_mask_tensor,
        "reward": reward_tensor,
        "meta": list(metas),
    }
    torch.save(payload, shard_path)
    return {
        "file": shard_name,
        "start_index": int(payload["start_index"]),
        "end_index": int(payload["end_index"]),
        "num_samples": int(obs_tensor.shape[0]),
    }


def main() -> None:
    args = _parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    config_path = args.config_path.expanduser().resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    dataset = RynnVLALIBERODataset(
        config_path=config_path,
        resolution=int(args.resolution),
        with_state=not bool(args.without_state),
        with_wrist=not bool(args.without_wrist),
        with_action=not bool(args.without_action),
        with_world_model=not bool(args.without_world_model),
    )
    if args.max_samples is not None:
        sample_count = min(int(args.max_samples), len(dataset))
        dataset_for_loader = Subset(dataset, list(range(sample_count)))
    else:
        sample_count = len(dataset)
        dataset_for_loader = dataset

    dataloader = DataLoader(
        dataset_for_loader,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=dataset.collate_fn,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    encoder = RynnVLAEncoder().to(args.device)
    encoder.eval()
    storage_dtype = _to_storage_dtype(args.embedding_dtype)

    manifest: dict[str, Any] = {
        "version": 1,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "device": str(args.device),
        "num_samples": int(sample_count),
        "embedding_dtype": str(args.embedding_dtype),
        "normalizer": dataset.get_normalizer(),
        "source_data_spec": asdict(dataset.data_spec),
        "shards": [],
    }

    obs_embeddings: list[torch.Tensor] = []
    next_obs_embeddings: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    action_masks: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    metas: list[dict[str, Any]] = []
    shard_idx = 0
    shard_start = 0
    processed = 0
    hidden_dim: int | None = None

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            obs_hidden = encoder.encode(batch["obs"]).detach().cpu()
            next_obs_hidden = encoder.encode(batch["next_obs"]).detach().cpu()
            if hidden_dim is None:
                hidden_dim = int(obs_hidden.shape[-1])
                manifest["hidden_dim"] = hidden_dim

            obs_embeddings.append(obs_hidden.to(dtype=storage_dtype))
            next_obs_embeddings.append(next_obs_hidden.to(dtype=storage_dtype))
            actions.append(batch["action"].detach().cpu())
            action_masks.append(batch["action_mask"].detach().cpu())
            rewards.append(batch["reward"].detach().cpu())
            metas.extend(list(batch["meta"]))

            processed += int(obs_hidden.shape[0])
            current_shard_size = sum(tensor.shape[0] for tensor in obs_embeddings)
            if current_shard_size >= int(args.shard_size):
                shard_info = _save_shard(
                    output_dir=output_dir,
                    shard_idx=shard_idx,
                    start_index=shard_start,
                    obs_embeddings=obs_embeddings,
                    next_obs_embeddings=next_obs_embeddings,
                    actions=actions,
                    action_masks=action_masks,
                    rewards=rewards,
                    metas=metas,
                    embedding_dtype=storage_dtype,
                )
                manifest["shards"].append(shard_info)
                shard_idx += 1
                shard_start = processed
                obs_embeddings = []
                next_obs_embeddings = []
                actions = []
                action_masks = []
                rewards = []
                metas = []

            if processed % max(int(args.batch_size), 1) == 0:
                elapsed = time.perf_counter() - start_time
                samples_per_sec = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[precompute] processed={processed}/{sample_count} "
                    f"elapsed={elapsed:.1f}s samples_per_sec={samples_per_sec:.3f}"
                )

    if obs_embeddings:
        shard_info = _save_shard(
            output_dir=output_dir,
            shard_idx=shard_idx,
            start_index=shard_start,
            obs_embeddings=obs_embeddings,
            next_obs_embeddings=next_obs_embeddings,
            actions=actions,
            action_masks=action_masks,
            rewards=rewards,
            metas=metas,
            embedding_dtype=storage_dtype,
        )
        manifest["shards"].append(shard_info)

    elapsed = time.perf_counter() - start_time
    manifest["elapsed_sec"] = float(elapsed)
    manifest["samples_per_sec"] = float(processed / elapsed) if elapsed > 0 else 0.0
    manifest["num_shards"] = int(len(manifest["shards"]))

    torch.save(manifest, output_dir / "manifest.pt")
    print(
        f"[precompute] done samples={processed} shards={manifest['num_shards']} "
        f"elapsed={elapsed/3600:.2f}h output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
