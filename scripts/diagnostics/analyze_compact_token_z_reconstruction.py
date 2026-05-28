#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dreamer_vla.models.world_model.dreamerv3_torch import CompactTokenSequenceAutoencoder


class FullseqFrameDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        sidecar_dir: str | Path,
        max_files: int | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.sidecar_dir = Path(sidecar_dir).expanduser().resolve()
        paths = sorted(self.sidecar_dir.glob("*.hdf5"))
        if max_files is not None:
            paths = paths[: int(max_files)]
        if not paths:
            raise FileNotFoundError(
                f"No complete .hdf5 fullseq sidecar files found under {self.sidecar_dir}"
            )
        self.paths = paths
        self._files: dict[int, h5py.File] = {}
        self.index: list[tuple[int, str, int]] = []
        for file_idx, path in enumerate(paths):
            with h5py.File(path, "r") as handle:
                for demo_key in sorted(handle["data"].keys()):
                    group = handle["data"][demo_key]
                    if "actor_hidden_states" not in group:
                        continue
                    length = int(group["actor_hidden_states"].shape[0])
                    for frame_idx in range(length):
                        self.index.append((file_idx, demo_key, frame_idx))
                        if max_frames is not None and len(self.index) >= int(
                            max_frames
                        ):
                            return
        if not self.index:
            raise RuntimeError(f"No actor_hidden_states found under {self.sidecar_dir}")

    def __len__(self) -> int:
        return len(self.index)

    def _file(self, file_idx: int) -> h5py.File:
        handle = self._files.get(file_idx)
        if handle is None:
            handle = h5py.File(self.paths[file_idx], "r")
            self._files[file_idx] = handle
        return handle

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, demo_key, frame_idx = self.index[int(idx)]
        group = self._file(file_idx)["data"][demo_key]
        hidden = np.asarray(group["actor_hidden_states"][frame_idx], dtype=np.float32)
        if "actor_attention_mask" in group:
            mask = np.asarray(
                group["actor_attention_mask"][frame_idx][:-1], dtype=np.bool_
            )
            mask = mask[: hidden.shape[0]]
        elif "actor_seq_lens" in group:
            valid = int(group["actor_seq_lens"][frame_idx])
            mask = np.zeros((hidden.shape[0],), dtype=np.bool_)
            mask[:valid] = True
        else:
            mask = np.ones((hidden.shape[0],), dtype=np.bool_)
        return {
            "hidden": torch.from_numpy(hidden),
            "mask": torch.from_numpy(mask),
        }


def collate_fullseq(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(int(item["hidden"].shape[0]) for item in batch)
    hidden_dim = int(batch[0]["hidden"].shape[-1])
    hidden = torch.zeros(len(batch), max_len, hidden_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for idx, item in enumerate(batch):
        length = int(item["hidden"].shape[0])
        hidden[idx, :length] = item["hidden"]
        mask[idx, :length] = item["mask"][:length]
    return {"hidden": hidden, "mask": mask}


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask_f = mask.to(dtype=pred.dtype).unsqueeze(-1)
    denom = mask_f.sum().clamp_min(1.0) * pred.shape[-1]
    return ((pred - target).square() * mask_f).sum() / denom


def masked_cosine_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    pred_n = F.normalize(pred.float(), dim=-1)
    target_n = F.normalize(target.float(), dim=-1)
    per = 1.0 - (pred_n * target_n).sum(dim=-1)
    mask_f = mask.float()
    return (per * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact token-z autoencoder on VLA full hidden sequences."
    )
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--latent-tokens", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--target-tokens", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    dataset = FullseqFrameDataset(
        args.sidecar_dir, max_files=args.max_files, max_frames=args.max_frames
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=True,
        collate_fn=collate_fullseq,
    )
    first = dataset[0]["hidden"]
    device = torch.device(
        args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu"
    )
    model = CompactTokenSequenceAutoencoder(
        in_dim=int(first.shape[-1]),
        latent_tokens=int(args.latent_tokens),
        latent_dim=int(args.latent_dim),
        target_tokens=int(args.target_tokens),
        num_heads=int(args.num_heads),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    print(
        f"[compact-z] frames={len(dataset)} hidden_dim={int(first.shape[-1])} "
        f"latent=({args.latent_tokens},{args.latent_dim}) target_tokens={args.target_tokens} device={device}"
    )
    iterator = iter(loader)
    last_metrics: dict[str, float] = {}
    for step in range(int(args.steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        hidden = batch["hidden"].to(device)
        mask = batch["mask"].to(device)
        out = model(hidden, mask)
        loss = masked_mse(out["reconstruction"], out["target"], out["target_mask"])
        cos = masked_cosine_loss(
            out["reconstruction"], out["target"], out["target_mask"]
        )
        if step > 0:
            optim.zero_grad(set_to_none=True)
            (loss + 0.1 * cos).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optim.step()
        last_metrics = {
            "mse": float(loss.detach().cpu()),
            "cos": float(cos.detach().cpu()),
            "rmse": float(loss.detach().sqrt().cpu()),
        }
        if step % 20 == 0 or step == int(args.steps):
            print(
                f"[compact-z] step={step:04d} mse={last_metrics['mse']:.6f} "
                f"rmse={last_metrics['rmse']:.4f} cos={last_metrics['cos']:.4f}"
            )
    print(f"[compact-z] final {last_metrics}")


if __name__ == "__main__":
    main()
