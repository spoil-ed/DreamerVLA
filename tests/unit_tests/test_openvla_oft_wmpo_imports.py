from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDORED_OPENVLA_OFT = PROJECT_ROOT / "third_party" / "openvla-oft"


class _TinyTokenizer:
    vocab_size = 32000
    model_max_length = 32
    pad_token_id = 0

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(0xE000 + int(token_id) % 256) for token_id in token_ids)

    def batch_decode(self, token_ids: list[list[int]]) -> list[str]:
        return [self.decode(row) for row in token_ids]


def test_openvla_oft_path_stays_inside_dreamervla_tree() -> None:
    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    selected = ensure_openvla_oft_on_path()

    assert selected == VENDORED_OPENVLA_OFT
    assert (selected / "prismatic").is_dir()


def test_official_openvla_oft_default_root_lives_under_dreamervla() -> None:
    from dreamervla.diagnostics.openvla_oft_obs_action_policy import default_openvla_oft_root

    assert default_openvla_oft_root() == VENDORED_OPENVLA_OFT


def test_prismatic_imports_resolve_to_dreamervla_tree() -> None:
    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()
    modules = [
        "prismatic.extern.hf.configuration_prismatic",
        "prismatic.extern.hf.modeling_prismatic",
        "prismatic.extern.hf.processing_prismatic",
        "prismatic.models.action_heads",
        "prismatic.models.projectors",
        "prismatic.models.backbones.llm.prompting",
        "prismatic.training.train_utils",
        "prismatic.util.data_utils",
        "prismatic.vla.action_tokenizer",
        "prismatic.vla.constants",
        "prismatic.vla.datasets",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert Path(module.__file__).resolve().is_relative_to(VENDORED_OPENVLA_OFT)


def test_action_tokenizer_action_head_and_collator_interfaces() -> None:
    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    tokenizer = ActionTokenizer(_TinyTokenizer())
    encoded = tokenizer(np.zeros((ACTION_DIM,), dtype=np.float32))
    decoded = tokenizer.decode_token_ids_to_actions(np.array([31999, 31998], dtype=np.int64))

    assert isinstance(encoded, str)
    assert decoded.shape == (2,)

    head = L1RegressionActionHead(input_dim=4, hidden_dim=8, action_dim=ACTION_DIM)
    hidden = torch.zeros(2, NUM_ACTIONS_CHUNK * ACTION_DIM, 4)
    action = head.predict_action(hidden)

    assert action.shape == (2, NUM_ACTIONS_CHUNK, ACTION_DIM)

    collator = PaddedCollatorForActionPrediction(model_max_length=16, pad_token_id=0)
    batch = collator(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([-100, 2, 3]),
                "pixel_values": torch.zeros(3, 224, 224),
                "actions": np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32),
                "proprio": np.zeros((8,), dtype=np.float32),
                "dataset_name": "tiny",
            },
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([-100, 2]),
                "pixel_values": torch.ones(3, 224, 224),
                "actions": np.ones((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32),
                "proprio": np.ones((8,), dtype=np.float32),
                "dataset_name": "tiny",
            },
        ]
    )

    assert batch["input_ids"].shape == (2, 3)
    assert batch["pixel_values"].shape == (2, 3, 224, 224)
    assert batch["actions"].shape == (2, NUM_ACTIONS_CHUNK, ACTION_DIM)
    assert batch["dataset_names"] == ["tiny", "tiny"]
