"""
Sequence dataset for TransDreamer-style world model training.

Each sample provides a contiguous sequence of T frames from the same episode:
    obs_embedding_seq:  [T, obs_dim]    - LLM hidden states
    action_seq:         [T, action_dim] - single action per step (not chunk)
    reward_seq:         [T]
    done_seq:           [T]

The episode structure is preserved in the manifest so consecutive frames
from the same episode can be grouped.

Preprocess requirement:
    The preencode script must save a manifest with episode groupings.
    Each shard entry should have:
        {
          "episode_id": str,
          "step_index": int,   <- position within the episode
          "obs_embedding": Tensor[obs_dim],
          "action": Tensor[action_dim],   <- single action (not chunk)
          "reward": float,
          "done": bool,
        }
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.dataloader.base_dataset import BaseDataset


@dataclass(frozen=True)
class SequencePreencodeSFTDataSpec:
    manifest_path: str
    num_sequences: int
    seq_len: int
    hidden_dim: int
    action_dim: int


class SequencePreencodeSFTDataset(BaseDataset):
    """
    Loads pre-encoded embeddings and groups them into fixed-length sequences.

    Each __getitem__ returns T consecutive frames from the SAME episode.
    If an episode is shorter than seq_len, it is skipped.

    Expected manifest format (saved by preprocess script):
        manifest.pt = {
            "episodes": {
                episode_id: {
                    "obs_embedding":  Tensor[L, obs_dim],
                    "action":         Tensor[L, action_dim],  # single action/step
                    "reward":         Tensor[L],
                    "done":           Tensor[L],
                }
            },
            "hidden_dim":  int,
            "action_dim":  int,
        }
    """

    def __init__(
        self,
        manifest_path: str | Path,
        seq_len: int = 16,
        stride: int = 1,
    ) -> None:
        """
        Args:
            manifest_path: path to manifest.pt produced by the sequence preprocess script.
            seq_len:  number of consecutive frames T per training sample.
            stride:   step size when sliding the window over each episode.
        """
        super().__init__()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if self.manifest_path.is_dir():
            self.manifest_path = self.manifest_path / "manifest_seq.pt"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Sequence manifest not found: {self.manifest_path}\n"
                "Run the sequence preencode script first."
            )

        manifest = torch.load(self.manifest_path, map_location="cpu")
        self.hidden_dim: int = int(manifest["hidden_dim"])
        self.action_dim: int = int(manifest["action_dim"])
        self.seq_len = int(seq_len)

        # Build flat index: list of (episode_id, start_step)
        self._index: list[tuple[str, int]] = []
        self._episodes: dict[str, dict[str, torch.Tensor]] = {}

        for episode_id, ep_data in manifest["episodes"].items():
            ep_len = int(ep_data["obs_embedding"].shape[0])
            if ep_len < self.seq_len:
                continue
            self._episodes[episode_id] = {
                k: v.float() if v.dtype.is_floating_point else v
                for k, v in ep_data.items()
            }
            for start in range(0, ep_len - self.seq_len + 1, stride):
                self._index.append((episode_id, start))

        self._data_spec = SequencePreencodeSFTDataSpec(
            manifest_path=str(self.manifest_path),
            num_sequences=len(self._index),
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
        )

    @property
    def data_spec(self) -> SequencePreencodeSFTDataSpec:
        return self._data_spec

    def get_normalizer(self) -> dict[str, Any]:
        return {}

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, start = self._index[index]
        ep = self._episodes[episode_id]
        end = start + self.seq_len
        return {
            # [T, obs_dim]
            "obs_embedding_seq": ep["obs_embedding"][start:end],
            # [T, action_dim]  - single action per timestep
            "action_seq": ep["action"][start:end],
            # [T]
            "reward_seq": ep["reward"][start:end],
            # [T]
            "done_seq": ep["done"][start:end].float(),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "obs_embedding_seq": torch.stack([b["obs_embedding_seq"] for b in batch]),  # [B, T, obs_dim]
            "action_seq":        torch.stack([b["action_seq"]        for b in batch]),  # [B, T, action_dim]
            "reward_seq":        torch.stack([b["reward_seq"]        for b in batch]),  # [B, T]
            "done_seq":          torch.stack([b["done_seq"]          for b in batch]),  # [B, T]
        }


__all__ = ["SequencePreencodeSFTDataset", "SequencePreencodeSFTDataSpec"]
