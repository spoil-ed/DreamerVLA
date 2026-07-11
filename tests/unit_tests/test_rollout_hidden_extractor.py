"""OpenVLA-OFT hidden-token rollout extraction contracts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from dreamervla.runners.rollout_hidden_extractor import (
    OFTBatchedDecoder,
    OFTRolloutHiddenExtractor,
    hidden_token_from_projected,
)

TOKEN_COUNT = 256
TOKEN_DIM = 4096
EXPECTED_TOKEN_SHAPE = (TOKEN_COUNT, TOKEN_DIM)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONE_TRAJ_DISCRETE_CKPT = (
    PROJECT_ROOT
    / "data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
)
_SKIP_GPU = "needs CUDA GPU and DVLA_REAL_MODEL_UNIT=1"


class _MinimalPolicy:
    class cfg:
        center_crop = True
        use_proprio = False
        use_film = False

    vla = None
    processor = None
    action_head = None
    proprio_projector = None


def test_batched_decoder_rejects_l1_action_head() -> None:
    class _Policy:
        vla = object()
        action_head = object()
        proprio_projector = None

    with pytest.raises(ValueError, match="L1/action-query checkpoints are closed"):
        OFTBatchedDecoder(_Policy(), "libero_goal_no_noops")


def test_batched_decoder_rejects_proprio_projector() -> None:
    class _Policy:
        vla = object()
        action_head = None
        proprio_projector = object()

    with pytest.raises(ValueError, match="does not include proprio"):
        OFTBatchedDecoder(_Policy(), "libero_goal_no_noops")


def test_extractor_rejects_multiview_and_history_aliases() -> None:
    with pytest.raises(ValueError, match="one image and history=1"):
        OFTRolloutHiddenExtractor(
            _MinimalPolicy(),
            image_keys=["agentview_rgb", "eye_in_hand_rgb"],
            history=1,
        )
    with pytest.raises(ValueError, match="one image and history=1"):
        OFTRolloutHiddenExtractor(
            _MinimalPolicy(),
            image_keys=["agentview_rgb"],
            history=2,
        )


def test_history_buffer_is_single_frame_on_mainline() -> None:
    extractor = OFTRolloutHiddenExtractor(
        _MinimalPolicy(),
        image_keys=["agentview_rgb"],
        history=1,
        rotate_images_180=False,
        unnorm_key="dummy",
    )
    first = np.zeros((4, 4, 3), dtype=np.uint8)
    second = np.full((4, 4, 3), 99, dtype=np.uint8)

    first_history = extractor._get_history("agentview_rgb", first)
    second_history = extractor._get_history("agentview_rgb", second)
    assert len(first_history) == 1 and np.array_equal(first_history[0], first)
    assert len(second_history) == 1 and np.array_equal(second_history[0], second)


def test_hidden_token_accepts_only_canonical_projected_shape() -> None:
    projected = torch.zeros(2, TOKEN_COUNT, TOKEN_DIM)
    projected[:, -1, -1] = 7
    hidden = hidden_token_from_projected(
        projected,
        image_keys=["agentview_rgb"],
        patches_per_image=TOKEN_COUNT,
    )

    assert torch.equal(hidden, projected)
    assert hidden.shape == (2, *EXPECTED_TOKEN_SHAPE)

    with pytest.raises(ValueError, match="one image"):
        hidden_token_from_projected(
            projected,
            image_keys=["agentview_rgb", "eye_in_hand_rgb"],
            patches_per_image=TOKEN_COUNT,
        )
    with pytest.raises(ValueError, match=r"\[B,256,4096\]"):
        hidden_token_from_projected(
            torch.zeros(2, TOKEN_COUNT, 1024),
            image_keys=["agentview_rgb"],
            patches_per_image=TOKEN_COUNT,
        )


def test_rollout_hidden_extractor_docs_use_role_based_wm_wording() -> None:
    source = (
        PROJECT_ROOT / "dreamervla" / "runners" / "rollout_hidden_extractor.py"
    ).read_text(encoding="utf-8")
    assert ("DINO" + "-WM") not in source
    assert ("dino" + "_wm") not in source.lower()
    assert ("dino" + "wm") not in source.lower()


def test_left_pad_batch_pads_left_and_repositions_bos() -> None:
    from dreamervla.runners.rollout_hidden_extractor import _left_pad_batch

    bos, pad = 1, 0
    short = torch.tensor([[bos, 10, 11]])
    long = torch.tensor([[bos, 20, 21, 22, 23]])
    short_mask = torch.ones(1, 3, dtype=torch.long)
    long_mask = torch.ones(1, 5, dtype=torch.long)

    ids, mask = _left_pad_batch(
        [short, long], [short_mask, long_mask], pad, bos
    )

    assert ids.tolist() == [[bos, pad, pad, 10, 11], [bos, 20, 21, 22, 23]]
    assert mask.tolist() == [[1, 0, 0, 1, 1], [1, 1, 1, 1, 1]]
    positions = (mask[0].cumsum(0) - 1).clamp(min=0).tolist()
    assert [positions[3], positions[4]] == [1, 2]


def test_left_pad_batch_same_length_is_noop() -> None:
    from dreamervla.runners.rollout_hidden_extractor import _left_pad_batch

    first = torch.tensor([[1, 10, 11]])
    second = torch.tensor([[1, 20, 21]])
    ones = torch.ones(1, 3, dtype=torch.long)

    ids, mask = _left_pad_batch([first, second], [ones, ones.clone()], 0, 1)

    assert ids.tolist() == [[1, 10, 11], [1, 20, 21]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_batched_forward_rejects_empty() -> None:
    from dreamervla.runners.rollout_hidden_extractor import batched_forward

    with pytest.raises(ValueError):
        batched_forward(object(), [], "dummy")


def _gpu_available() -> bool:
    return os.environ.get("DVLA_REAL_MODEL_UNIT") == "1" and torch.cuda.is_available()


def _discrete_ckpt_available() -> bool:
    return (
        ONE_TRAJ_DISCRETE_CKPT.is_dir()
        and not any(ONE_TRAJ_DISCRETE_CKPT.glob("action_head--*_checkpoint.pt"))
        and any(ONE_TRAJ_DISCRETE_CKPT.glob("*.safetensors"))
    )


def _load_discrete_policy():
    import json

    from dreamervla.models.embodiment.openvla_oft_policy import OpenVLAOFTPolicy

    policy = OpenVLAOFTPolicy(
        model_path=str(ONE_TRAJ_DISCRETE_CKPT),
        torch_dtype="bf16",
        num_images_in_input=1,
        use_lora=False,
        use_l1_regression=False,
        use_diffusion=False,
        use_proprio=False,
        use_film=False,
        freeze_vla_backbone=True,
    )
    policy.eval()
    policy.to(torch.device("cuda:0"))
    with (ONE_TRAJ_DISCRETE_CKPT / "dataset_statistics.json").open() as handle:
        policy.vla.norm_stats = json.load(handle)
    return policy


@pytest.fixture(scope="module")
def oft_discrete_policy():
    if not (_discrete_ckpt_available() and _gpu_available()):
        pytest.skip("one-trajectory discrete checkpoint or opt-in GPU unavailable")
    return _load_discrete_policy()


def _make_discrete_extractor(policy):
    return OFTRolloutHiddenExtractor(
        policy,
        image_keys=["agentview_rgb"],
        history=1,
        rotate_images_180=True,
        center_crop=True,
        unnorm_key="libero_goal_no_noops",
    )


def _random_obs(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {
        "agentview_rgb": rng.randint(0, 256, (256, 256, 3), dtype=np.uint8),
    }


@pytest.mark.skipif(
    not _discrete_ckpt_available(), reason="one-trajectory discrete checkpoint not found"
)
@pytest.mark.skipif(not _gpu_available(), reason=_SKIP_GPU)
def test_batched_forward_discrete_headless(oft_discrete_policy) -> None:
    from dreamervla.runners.rollout_hidden_extractor import batched_forward

    assert oft_discrete_policy.action_head is None
    task = "open the middle drawer of the cabinet"
    obs_a, obs_b = _random_obs(31), _random_obs(32)

    def single(obs):
        extractor = _make_discrete_extractor(oft_discrete_policy)
        extractor.reset()
        return extractor.step(obs, task)

    def prepared(obs):
        extractor = _make_discrete_extractor(oft_discrete_policy)
        extractor.reset()
        return extractor.prepare(obs, task)

    chunk_a, hidden_a = single(obs_a)
    assert hidden_a.shape == EXPECTED_TOKEN_SHAPE
    assert hidden_a.dtype == torch.float16
    assert np.asarray(chunk_a[0]).shape == (7,)

    batch_one = batched_forward(
        oft_discrete_policy, [prepared(obs_a)], "libero_goal_no_noops"
    )
    torch.testing.assert_close(batch_one[0][1], hidden_a, rtol=0, atol=0)
    np.testing.assert_allclose(batch_one[0][0][0], chunk_a[0], rtol=0, atol=0)

    batch_two = batched_forward(
        oft_discrete_policy,
        [prepared(obs_a), prepared(obs_b)],
        "libero_goal_no_noops",
    )
    assert batch_two[0][1].shape == EXPECTED_TOKEN_SHAPE
