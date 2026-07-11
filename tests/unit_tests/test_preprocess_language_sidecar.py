from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch

from dreamervla.preprocess.preprocess_oft_input_tokens import (
    _load_oft_components,
    _write_source_input_tokens,
)


def test_oft_input_token_preprocess_writes_per_demo_language_sidecar(
    tmp_path: Path,
) -> None:
    source = tmp_path / "put_the_bowl_on_the_plate_demo.hdf5"
    length = 3
    with h5py.File(source, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((length, 7), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset(
            "agentview_rgb",
            data=np.zeros((length, 16, 16, 3), dtype=np.uint8),
        )
        obs.create_dataset(
            "eye_in_hand_rgb",
            data=np.zeros((length, 16, 16, 3), dtype=np.uint8),
        )
        obs.create_dataset("ee_pos", data=np.zeros((length, 3), dtype=np.float32))
        obs.create_dataset("ee_ori", data=np.zeros((length, 3), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.zeros((length, 2), dtype=np.float32))

    args = Namespace(
        fake_oft_components=True,
        fake_num_patches=2,
        num_images_in_input=None,
        history=2,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
        oft_ckpt=str(tmp_path / "fake_ckpt"),
        center_crop=False,
        unnorm_key="fake",
        include_state=True,
        rotate_images_180=False,
        hidden_key="obs_embedding",
        time_horizon=2,
        action_dim=3,
        token_dim=8,
        output_dtype="float16",
        chunk_size=2,
        prompt_style="vla_policy",
        resolution=256,
        resolved_policy_mode="discrete",
        max_demos_per_file=None,
    )
    components = _load_oft_components(args, torch.device("cpu"))
    out_input = tmp_path / "input" / source.name
    out_input.parent.mkdir()

    stats = _write_source_input_tokens(
        source_path=source,
        out_input_token_path=out_input,
        components=components,
        args=args,
        rank=0,
    )

    assert stats == {"demos": 1, "frames": length}
    with h5py.File(out_input, "r") as handle:
        assert bool(handle.attrs["complete"])
        demo = handle["data"]["demo_0"]
        assert demo["obs_embedding"].shape == (length, 4, 8)
        assert demo["lang_emb"].shape == (8,)
        assert demo["lang_emb"].dtype == np.dtype("float16")
        assert demo["lang_emb"].attrs["hidden_dim"] == 8
