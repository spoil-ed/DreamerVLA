from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
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


class _TinyForwardVLA(nn.Module):
    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, 32, device=input_ids.device)
        hidden = torch.zeros(batch_size, seq_len, 4, device=input_ids.device)
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


def test_openvla_oft_hdf5_dataset_randomly_keeps_one_demo_per_file(tmp_path: Path) -> None:
    from dreamer_vla.dataset.openvla_oft_hdf5_dataset import OpenVLAOFTHDF5Dataset

    _write_demo_file(tmp_path / "task_alpha_demo.hdf5")
    _write_demo_file(tmp_path / "task_beta_demo.hdf5")
    stats = {
        "action": {"q01": [-1.0] * 7, "q99": [1.0] * 7, "mask": [True] * 7},
        "proprio": {"q01": [-1.0] * 8, "q99": [1.0] * 8, "mask": [True] * 8},
    }

    first = OpenVLAOFTHDF5Dataset(
        hdf5_dir=tmp_path,
        processor=_TinyProcessor(),
        action_tokenizer=_TinyActionTokenizer(),
        dataset_statistics=stats,
        action_horizon=2,
        demos_per_task=1,
        demo_selection_seed=5,
    )
    second = OpenVLAOFTHDF5Dataset(
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


def test_openvla_oft_component_save_allows_lm_head_mode_without_action_head(tmp_path: Path) -> None:
    from dreamer_vla.runners.openvla_oft_runner import OpenVLAOFTTrainingRunner

    workspace = OpenVLAOFTTrainingRunner.__new__(OpenVLAOFTTrainingRunner)
    workspace._output_dir = str(tmp_path)
    workspace.distributed = _TinyDistributed()
    workspace.policy = SimpleNamespace(
        vla=_TinyVLA(),
        processor=_TinyProcessor(),
        action_head=None,
        proprio_projector=None,
    )

    workspace._save_oft_components(step=3)

    save_dir = tmp_path / "openvla_oft_components--3"
    assert (save_dir / "lora_adapter").is_dir()
    assert not (save_dir / "action_head--3_checkpoint.pt").exists()


def test_openvla_oft_lm_head_mode_computes_token_loss_without_action_head() -> None:
    from dreamer_vla.models.encoder.openvla_oft_policy import OpenVLAOFTPolicy

    policy = OpenVLAOFTPolicy.from_modules(
        vla=_TinyForwardVLA(),
        action_head=None,
        action_tokenizer=_TinyActionTokenizer(),
        num_patches=0,
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
