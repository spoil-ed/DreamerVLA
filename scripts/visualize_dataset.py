from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import BaseDataset


def _to_uint8_image(image: torch.Tensor) -> Image.Image:
    tensor = image.detach().cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def _save_preview(sample: dict, output_path: Path) -> None:
    obs_image = _to_uint8_image(sample["obs"]["image"])
    next_obs_image = _to_uint8_image(sample["next_obs"]["image"])
    canvas = Image.new(
        "RGB",
        (obs_image.width + next_obs_image.width, max(obs_image.height, next_obs_image.height)),
    )
    canvas.paste(obs_image, (0, 0))
    canvas.paste(next_obs_image, (obs_image.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@hydra.main(
    config_path="../configs",
    config_name="debug",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
    assert isinstance(dataset, BaseDataset), "Dataset must be an instance of BaseDataset"
    dataloader = DataLoader(dataset, **cfg.dataloader)

    sample_index = 0
    sample = dataset[sample_index]
    batch = next(iter(dataloader))

    preview_dir = PROJECT_ROOT / "tmp_files" / "dataset_preview"
    preview_path = preview_dir / f"sample_{sample_index:06d}.png"
    stats_path = preview_dir / f"sample_{sample_index:06d}.json"

    _save_preview(sample, preview_path)

    stats = {
        "data_spec": {
            "train_config_path": dataset.data_spec.train_config_path,
            "hdf5_path": dataset.data_spec.hdf5_path,
            "task_name": dataset.data_spec.task_name,
            "num_transitions": dataset.data_spec.num_transitions,
            "action_dim": dataset.data_spec.action_dim,
            "proprio_dim": dataset.data_spec.proprio_dim,
            "vocab_size": dataset.data_spec.vocab_size,
            "max_text_length": dataset.data_spec.max_text_length,
            "image_key": dataset.data_spec.image_key,
            "wrist_image_key": dataset.data_spec.wrist_image_key,
            "low_dim_keys": list(dataset.data_spec.low_dim_keys),
        },
        "sample": {
            "index": sample_index,
            "meta": sample["meta"],
            "obs_image_shape": list(sample["obs"]["image"].shape),
            "next_obs_image_shape": list(sample["next_obs"]["image"].shape),
            "text_tokens": sample["obs"]["text"].tolist(),
            "text_attention_mask": sample["obs"]["text_attention_mask"].tolist(),
            "proprio": sample["obs"]["proprio"].tolist(),
            "action": sample["action"].tolist(),
            "reward": sample["reward"].tolist(),
            "obs_image_min": float(sample["obs"]["image"].min()),
            "obs_image_max": float(sample["obs"]["image"].max()),
        },
        "batch_shapes": {
            "obs.image": list(batch["obs"]["image"].shape),
            "obs.text": list(batch["obs"]["text"].shape),
            "obs.proprio": list(batch["obs"]["proprio"].shape),
            "action": list(batch["action"].shape),
            "reward": list(batch["reward"].shape),
        },
        "normalizer": {
            key: {
                sub_key: value.tolist()
                for sub_key, value in values.items()
            }
            for key, values in dataset.get_normalizer().items()
        },
        "preview_path": str(preview_path),
    }

    preview_dir.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print(f"preview_image={preview_path}")
    print(f"preview_stats={stats_path}")
    print(f"dataset_size={len(dataset)}")
    print(
        "batch_shapes "
        f"image={tuple(batch['obs']['image'].shape)} "
        f"text={tuple(batch['obs']['text'].shape)} "
        f"proprio={tuple(batch['obs']['proprio'].shape)} "
        f"action={tuple(batch['action'].shape)} "
        f"reward={tuple(batch['reward'].shape)}"
    )
    print(f"sample_meta={sample['meta']}")
    print(f"sample_action={sample['action'].tolist()}")
    print(f"sample_reward={sample['reward'].tolist()}")


if __name__ == "__main__":
    main()
