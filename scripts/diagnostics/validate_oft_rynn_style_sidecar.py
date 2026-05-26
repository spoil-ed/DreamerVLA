#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _first_complete_hdf5(hidden_dir: Path) -> Path:
    for path in sorted(hidden_dir.glob("*.hdf5")):
        try:
            with h5py.File(path, "r") as handle:
                if bool(handle.attrs.get("complete", False)):
                    return path
        except OSError:
            continue
    raise FileNotFoundError(f"No complete sidecar hdf5 found under {hidden_dir}")


def _validate_hdf5(path: Path, expected_dim: int, expected_tokens: int, expected_token_dim: int) -> None:
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)
        assert bool(attrs.get("complete", False)), f"{path} is not complete"
        assert str(attrs.get("hidden_key")) == "obs_embedding"
        assert str(attrs.get("obs_hidden_source")) == "action_query"
        assert str(attrs.get("prompt_style")) == "vla_policy"
        assert int(attrs.get("history")) == 2
        assert bool(attrs.get("include_state")) is True
        assert bool(attrs.get("rotate_images_180")) is True
        assert int(attrs.get("hidden_dim")) == int(expected_dim)
        assert int(attrs.get("action_hidden_sequence_dim")) == int(expected_tokens)
        assert int(attrs.get("action_hidden_dim")) == int(expected_token_dim)
        demo_key = sorted(handle["data"].keys())[0]
        demo = handle["data"][demo_key]
        obs = demo["obs_embedding"]
        action_hidden = demo["action_hidden_states"]
        assert obs.shape[-1] == int(expected_dim), obs.shape
        assert action_hidden.shape[1:] == (int(expected_tokens), int(expected_token_dim)), action_hidden.shape
        assert obs.shape[0] == action_hidden.shape[0]
        print(
            f"[hdf5-ok] {path.name} demo={demo_key} "
            f"obs_embedding={tuple(obs.shape)} action_hidden_states={tuple(action_hidden.shape)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OFT sidecars against the rynn-style dataloader contract.")
    parser.add_argument("--config-name", required=True, help="Config name under configs/, without .yaml")
    parser.add_argument("--sample-batch", action="store_true", help="Instantiate dataset and read one sample window.")
    args = parser.parse_args()

    config_dir = _project_path("configs")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=args.config_name)
    dataset_cfg = cfg.dataset
    hidden_dir = _project_path(dataset_cfg.hidden_dir)
    expected_dim = int(cfg.world_model.obs_dim)
    expected_tokens = int(cfg.world_model.token_count)
    expected_token_dim = int(cfg.world_model.token_dim)
    sidecar_path = _first_complete_hdf5(hidden_dir)
    _validate_hdf5(sidecar_path, expected_dim, expected_tokens, expected_token_dim)

    if args.sample_batch:
        dataset = instantiate(dataset_cfg, max_files=1, max_windows=None)
        item = dataset[0]
        obs = item["obs_embedding"]
        assert isinstance(obs, torch.Tensor)
        assert tuple(obs.shape) == (int(dataset_cfg.sequence_length), expected_dim), tuple(obs.shape)
        print(
            f"[dataset-ok] len={len(dataset)} obs_embedding={tuple(obs.shape)} "
            f"actions={tuple(item['actions'].shape)} rewards={tuple(item.get('rewards', torch.empty(0)).shape)}"
        )


if __name__ == "__main__":
    main()
