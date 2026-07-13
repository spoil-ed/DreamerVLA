from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class _TinyImageProcessor:
    def apply_transform(self, image):
        _ = image
        return torch.zeros(3, 224, 224)


class _TinyTokenizer:
    model_max_length = 128
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True):
        _ = add_special_tokens
        return SimpleNamespace(input_ids=list(range(1, min(len(text), 120) + 1)))


class _TinyActionTokenizer:
    def __call__(self, actions):
        arr = np.asarray(actions)
        if arr.ndim == 1:
            return "A" * int(arr.shape[0])
        return ["A" * int(arr.shape[1]) for _ in range(arr.shape[0])]

    def decode_token_ids_to_actions(self, token_ids):
        return np.asarray(token_ids, dtype=np.float32)


class _TinyVLA:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class _TinyVisionBackbone:
    def __init__(self, *, patches: int = 256, images: int = 1) -> None:
        self.patches = patches
        self.images = images

    def get_num_patches(self) -> int:
        return self.patches

    def get_num_images_in_input(self) -> int:
        return self.images


class _TinyForwardVLA(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int = 4096,
        patches: int = 256,
        images: int = 1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.vision_backbone = _TinyVisionBackbone(
            patches=patches,
            images=images,
        )

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        total_len = self.vision_backbone.get_num_patches() + seq_len
        logits = torch.zeros(batch_size, total_len, 32, device=input_ids.device)
        hidden = torch.zeros(batch_size, total_len, 4, device=input_ids.device)
        return CausalLMOutputWithPast(
            loss=input_ids.float().sum() * 0.0 + torch.tensor(1.25, device=input_ids.device),
            logits=logits,
            hidden_states=(hidden,),
        )


class _TinyProcessor:
    image_processor = _TinyImageProcessor()
    tokenizer = _TinyTokenizer()

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class _TinyDistributed:
    is_main_process = True

    def unwrap_module(self, module):
        return module


def _write_demo_file(path: Path, num_demos: int = 3, length: int = 2) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for demo_idx in range(num_demos):
            demo = data.create_group(f"demo_{demo_idx}")
            demo.create_dataset("actions", data=np.zeros((length, 7), dtype=np.float32))
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=np.zeros((length, 4, 4, 3), dtype=np.uint8))
            obs.create_dataset("eye_in_hand_rgb", data=np.zeros((length, 4, 4, 3), dtype=np.uint8))
            obs.create_dataset("ee_states", data=np.zeros((length, 6), dtype=np.float32))
            obs.create_dataset("gripper_states", data=np.zeros((length, 2), dtype=np.float32))


def test_vla_sft_hdf5_dataset_randomly_keeps_one_demo_per_file(tmp_path: Path) -> None:
    from dreamervla.dataset.vla_sft_hdf5_dataset import VLASFTHDF5Dataset

    _write_demo_file(tmp_path / "task_alpha_demo.hdf5")
    _write_demo_file(tmp_path / "task_beta_demo.hdf5")
    stats = {
        "action": {"q01": [-1.0] * 7, "q99": [1.0] * 7, "mask": [True] * 7},
        "proprio": {"q01": [-1.0] * 8, "q99": [1.0] * 8, "mask": [True] * 8},
    }

    first = VLASFTHDF5Dataset(
        hdf5_dir=tmp_path,
        processor=_TinyProcessor(),
        action_tokenizer=_TinyActionTokenizer(),
        dataset_statistics=stats,
        action_horizon=2,
        demos_per_task=1,
        demo_selection_seed=5,
    )
    second = VLASFTHDF5Dataset(
        hdf5_dir=tmp_path,
        processor=_TinyProcessor(),
        action_tokenizer=_TinyActionTokenizer(),
        dataset_statistics=stats,
        action_horizon=2,
        demos_per_task=1,
        demo_selection_seed=5,
    )

    first_keys = {(Path(sample.file_path).name, sample.demo_key) for sample in first.samples}
    second_keys = {(Path(sample.file_path).name, sample.demo_key) for sample in second.samples}

    assert len(first) == 4
    assert first_keys == second_keys
    assert {file_name for file_name, _demo_key in first_keys} == {
        "task_alpha_demo.hdf5",
        "task_beta_demo.hdf5",
    }
    assert len({demo_key for _file_name, demo_key in first_keys}) <= 2
    assert first.data_spec.one_trajectory_sft is True
    assert first.data_spec.demos_per_task == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"image_keys": ("agentview_rgb", "eye_in_hand_rgb")}, "image_keys"),
        ({"use_wrist_image": True}, "wrist image"),
        ({"use_proprio": True}, "VLA-side proprio"),
    ],
)
def test_vla_sft_hdf5_dataset_rejects_non_mainline_inputs(
    tmp_path: Path,
    kwargs: dict[str, object],
    match: str,
) -> None:
    from dreamervla.dataset.vla_sft_hdf5_dataset import VLASFTHDF5Dataset

    _write_demo_file(tmp_path / "task_demo.hdf5", num_demos=1, length=1)
    stats = {
        "action": {"q01": [-1.0] * 7, "q99": [1.0] * 7, "mask": [True] * 7},
        "proprio": {"q01": [-1.0] * 8, "q99": [1.0] * 8, "mask": [True] * 8},
    }

    with pytest.raises(ValueError, match=match):
        VLASFTHDF5Dataset(
            hdf5_dir=tmp_path,
            processor=_TinyProcessor(),
            action_tokenizer=_TinyActionTokenizer(),
            dataset_statistics=stats,
            **kwargs,
        )
def test_openvla_oft_lm_head_mode_computes_token_loss_without_action_head() -> None:
    from dreamervla.models.embodiment.openvla_oft_policy import OpenVLAOFTPolicy

    policy = OpenVLAOFTPolicy.from_modules(
        vla=_TinyForwardVLA(),
        action_head=None,
        action_tokenizer=_TinyActionTokenizer(),
        num_patches=256,
        use_l1_regression=False,
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "labels": torch.tensor([[-100, 10, 11, 12]], dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 224, 224),
        "actions": torch.zeros(1, 1, 3),
        "action_token_mask": torch.tensor([[True, True, True]]),
    }

    loss, metrics = policy.compute_loss(batch, device=torch.device("cpu"))

    assert loss.item() == 1.25
    assert metrics["loss_value"] == 1.25


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"use_l1_regression": True}, "L1/action-query"),
        ({"use_proprio": True}, "does not include proprio"),
        ({"use_film": True}, "does not use FiLM"),
    ],
)
def test_openvla_oft_policy_constructor_rejects_removed_routes(
    kwargs: dict[str, object],
    match: str,
) -> None:
    from dreamervla.models.embodiment.openvla_oft_policy import OpenVLAOFTPolicy

    with pytest.raises(ValueError, match=match):
        OpenVLAOFTPolicy(model_path="/does/not/matter", **kwargs)


@pytest.mark.parametrize(
    ("component_name", "match"),
    [
        ("action_head--1_checkpoint.pt", "L1/action-query"),
        ("proprio_projector--1_checkpoint.pt", "does not include proprio"),
    ],
)
def test_openvla_oft_policy_constructor_rejects_checkpoint_components(
    tmp_path: Path,
    component_name: str,
    match: str,
) -> None:
    from dreamervla.models.embodiment.openvla_oft_policy import OpenVLAOFTPolicy

    (tmp_path / component_name).write_bytes(b"")

    with pytest.raises(ValueError, match=match):
        OpenVLAOFTPolicy(model_path=str(tmp_path))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"action_head": nn.Linear(1, 1), "use_l1_regression": True}, "L1/action-query"),
        ({"proprio_projector": nn.Linear(1, 1), "use_proprio": True}, "does not include proprio"),
        ({"use_diffusion": True}, "diffusion checkpoints"),
        ({"use_film": True}, "does not use FiLM"),
    ],
)
def test_openvla_oft_from_modules_rejects_removed_components(
    kwargs: dict[str, object],
    match: str,
) -> None:
    from dreamervla.models.embodiment.openvla_oft_policy import OpenVLAOFTPolicy

    base = {
        "vla": _TinyForwardVLA(),
        "action_head": None,
        "action_tokenizer": _TinyActionTokenizer(),
        "num_patches": 256,
        "use_l1_regression": False,
        "use_proprio": False,
        "proprio_projector": None,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        OpenVLAOFTPolicy.from_modules(**base)


@pytest.mark.parametrize(
    ("vla", "num_patches", "token_dim", "match"),
    [
        (_TinyForwardVLA(), 56, None, "token_count expected=56 loaded=256"),
        (_TinyForwardVLA(patches=56), 256, None, "token_count expected=256 loaded=56"),
        (_TinyForwardVLA(token_dim=1024), 256, 4096, "token_dim expected=4096 loaded=1024"),
        (_TinyForwardVLA(images=2), 256, None, "token_count expected=256 loaded=512"),
    ],
)
def test_openvla_oft_from_modules_rejects_metadata_geometry_mismatch(
    vla: nn.Module,
    num_patches: int,
    token_dim: int | None,
    match: str,
) -> None:
    from dreamervla.models.embodiment.openvla_oft_policy import OpenVLAOFTPolicy

    with pytest.raises(ValueError, match=match):
        OpenVLAOFTPolicy.from_modules(
            vla=vla,
            action_head=None,
            action_tokenizer=_TinyActionTokenizer(),
            num_patches=num_patches,
            token_dim=token_dim,
        )


@pytest.mark.parametrize(
    ("vla", "token_count", "token_dim"),
    [
        (_TinyForwardVLA(patches=56, token_dim=1024), 56, 1024),
        (_TinyForwardVLA(images=2), 512, 4096),
    ],
)
def test_openvla_oft_from_modules_accepts_loaded_backbone_geometry(
    vla: nn.Module,
    token_count: int,
    token_dim: int,
) -> None:
    from dreamervla.models.embodiment.openvla_oft_policy import OpenVLAOFTPolicy

    policy = OpenVLAOFTPolicy.from_modules(
        vla=vla,
        action_head=None,
        action_tokenizer=_TinyActionTokenizer(),
        num_patches=token_count,
        token_dim=token_dim,
    )

    assert policy.token_count == token_count
    assert policy.token_dim == token_dim
