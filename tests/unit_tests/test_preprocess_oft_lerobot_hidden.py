from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from dreamervla.dataset.vla_sft_lerobot_dataset import LeRobotEpisode
from dreamervla.preprocess.preprocess_oft_hidden_token import _load_oft_components
from dreamervla.preprocess.preprocess_oft_hidden_token_lerobot import (
    _validate_outputs,
    _write_episode_hidden,
    build_lerobot_hidden_preprocess_config,
)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        command="preprocess",
        lerobot_dir=str(tmp_path / "source"),
        out_hidden_token_dir=str(tmp_path / "hidden"),
        oft_ckpt=str(tmp_path / "checkpoint"),
        obs_hidden_source="hidden_token",
        unnorm_key="libero_goal_no_noops",
        policy_mode="discrete",
        resolved_policy_mode="discrete",
        image_keys=["agentview_rgb"],
        lerobot_image_key="observation.images.image",
        action_key="action",
        state_key="observation.state",
        num_images_in_input=1,
        source_images_vla_aligned=True,
        include_state=False,
        center_crop=False,
        history=1,
        rotate_images_180=False,
        resolution=256,
        prompt_style="vla_policy",
        load_in_8bit=False,
        load_in_4bit=False,
        hidden_key="obs_embedding",
        time_horizon=8,
        action_dim=7,
        patches_per_image=256,
        token_dim=4096,
        chunk_size=2,
        output_dtype="float16",
        episodes_per_task=1,
        episode_selection="first",
        episode_selection_seed=0,
        max_episodes=None,
        overwrite=False,
        fake_oft_components=True,
        fake_num_patches=256,
    )


def test_lerobot_hidden_writer_emits_canonical_episode_sidecar(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    episode = LeRobotEpisode(
        episode_index=7,
        task_index=2,
        task="put the bowl on the plate",
        length=3,
    )
    info = {
        "codebase_version": "v2.1",
        "chunks_size": 1000,
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
    }
    frames = np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3)
    monkeypatch.setattr(
        "dreamervla.preprocess.preprocess_oft_hidden_token_lerobot.decode_lerobot_video",
        lambda _path, *, expected_length: frames[:expected_length],
    )
    seen_first_pixels: list[int] = []

    def fake_predict(*, components, args, images_by_frame, prompt):
        _ = components, prompt
        seen_first_pixels.extend(int(images[0][0, 0, 0]) for images in images_by_frame)
        batch = len(images_by_frame)
        return (
            np.zeros((batch, 256, args.token_dim), dtype=np.float32),
            np.zeros((batch, args.token_dim), dtype=np.float32),
        )

    monkeypatch.setattr(
        "dreamervla.preprocess.preprocess_oft_hidden_token_lerobot._predict_hidden_token_images",
        fake_predict,
    )
    components = _load_oft_components(args, torch.device("cpu"))
    output_dir = tmp_path / "hidden"
    output_dir.mkdir()
    output_path = output_dir / "episode_000007.hdf5"

    _write_episode_hidden(
        root=tmp_path / "source",
        info=info,
        episode=episode,
        output_path=output_path,
        components=components,
        args=args,
        rank=0,
    )

    assert seen_first_pixels == [0, 48, 96]
    with h5py.File(output_path, "r") as handle:
        assert bool(handle.attrs["complete"])
        assert handle.attrs["source_format"] == "lerobot_v2.1"
        assert bool(handle.attrs["source_images_vla_aligned"])
        assert not bool(handle.attrs["rotate_images_180"])
        assert handle.attrs["source_episode_index"] == 7
        demo = handle["data/episode_000007"]
        assert bool(demo.attrs["complete"])
        assert demo.attrs["task_prompt"] == episode.task
        assert demo["obs_embedding"].shape == (3, 256, 4096)
        assert demo["obs_embedding"].dtype == np.dtype("float16")
        assert demo["lang_emb"].shape == (4096,)

    config = build_lerobot_hidden_preprocess_config(
        args,
        lerobot_dir=tmp_path / "source",
        out_hidden_token_dir=output_dir,
        world_size=1,
        token_count=256,
        episodes=[episode],
    )
    (output_dir / "preprocess_config.json").write_text(json.dumps(config), encoding="utf-8")
    _validate_outputs(output_dir, [episode], args, token_count=256)


def test_lerobot_hidden_config_records_exact_episode_selection(tmp_path: Path) -> None:
    args = _args(tmp_path)
    episodes = [
        LeRobotEpisode(episode_index=0, task_index=0, task="zero", length=11),
        LeRobotEpisode(episode_index=9, task_index=1, task="one", length=13),
    ]

    config = build_lerobot_hidden_preprocess_config(
        args,
        lerobot_dir=tmp_path / "source",
        out_hidden_token_dir=tmp_path / "hidden",
        world_size=2,
        token_count=256,
        episodes=episodes,
    )

    assert config["source_format"] == "lerobot_v2.1"
    assert config["selected_episode_indices"] == [0, 9]
    assert config["selected_task_indices"] == [0, 1]
    assert config["selected_frames"] == 24
    assert config["source_images_vla_aligned"] is True
    assert config["rotate_images_180"] is False
