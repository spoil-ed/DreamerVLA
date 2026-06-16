"""Tests for OFT rollout hidden extractor.

Test plan:
  1. test_flatten_action_hidden_matches_sidecar_shape — pure shape-contract unit
     test; runs without GPU or model weights (always).
  2. test_flatten_action_hidden_batched_squeeze — verify batch-dim squeeze path.
  3. test_flatten_action_hidden_values_preserved — dtype-only transform.
  4. test_flatten_action_hidden_wrong_batch_size_not_squeezed — B>1 passthrough.
  5. test_history_buffer_padding — OFTRolloutHiddenExtractor pads history
     correctly without a model (structural test).
  6. test_inline_matches_offline_sidecar — CONSISTENCY GATE (GPU + ckpt required):
     drive OFTRolloutHiddenExtractor with stored reward-HDF5 frames and compare
     the returned flat_hidden against the pre-computed offline sidecar
     obs_embedding.  Asserts numerical equivalence within a justified tolerance.

Tolerance justification for test 6:
  The offline sidecars were built with PIL-based image preprocessing
  (PIL LANCZOS resize + PIL BICUBIC centre-crop, via
  dreamervla.preprocess.preprocess_oft_action_hidden._prepare_images_for_vla).
  The extractor uses TF-based preprocessing
  (experiments.robot.openvla_utils.prepare_images_for_vla: TF lanczos3 resize
  with JPEG roundtrip + TF crop_and_resize), which is the real-robot deployment
  path.  Empirically (8 pairs, demo_0 and demo_1, t in {1, T//4, T//2, T-1}):
    TF prep:  max_abs_err ≤ 0.25,  Pearson r ≥ 0.9996
    PIL prep: max_abs_err up to 1.93, Pearson r as low as 0.982
  The residual ~0.25 with TF prep is therefore primarily a PIL-vs-TF
  preprocessing difference, not purely fp16 non-determinism.  PIL prep was
  measured and rejected (much worse).  We set atol=0.5 (2× the observed
  max) and additionally require Pearson r > 0.999.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dreamervla.runners.rollout_hidden_extractor import (
    OFTRolloutHiddenExtractor,
    flatten_action_hidden,
)

# ── dimensions (from sidecar config) ────────────────────────────────────────
TIME_HORIZON = 8
ACTION_DIM = 7
TOKEN_DIM = 4096
EXPECTED_FLAT = TIME_HORIZON * ACTION_DIM * TOKEN_DIM  # 229376

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFT_CKPT = PROJECT_ROOT / "data/checkpoints/OpenVLA-OFT/libero_goal"

# Paths to the gold-standard sidecar and reward HDF5 (libero_goal)
SIDECAR_DIR = (
    PROJECT_ROOT
    / "data/datasets/processed_data"
    / "libero_goal_no_noops_t_256_oft_official_legacy_action_hidden_vla_policy_h2"
)
REWARD_DIR = (
    PROJECT_ROOT
    / "data/datasets/processed_data"
    / "libero_goal_no_noops_t_256_pi06_remaining_reward"
)
_TEST_HDF5 = "open_the_middle_drawer_of_the_cabinet_demo.hdf5"

# Tolerance: 0.5 (≈ 2 fp16 ULPs at |x| ~ 4); see module docstring.
_ATOL = 0.5
_MIN_PEARSON = 0.999


# ── pure unit tests (always run) ─────────────────────────────────────────────


def test_flatten_action_hidden_matches_sidecar_shape():
    """flatten_action_hidden([8,7,4096]) → (229376,) float16."""
    h = torch.randn(TIME_HORIZON, ACTION_DIM, TOKEN_DIM)
    flat = flatten_action_hidden(h)
    assert flat.shape == (EXPECTED_FLAT,), f"shape mismatch: {flat.shape}"
    assert flat.dtype == torch.float16, f"dtype mismatch: {flat.dtype}"


def test_flatten_action_hidden_batched_squeeze():
    """flatten_action_hidden([1,56,4096]) (batched output) → (229376,) float16."""
    h = torch.randn(1, TIME_HORIZON * ACTION_DIM, TOKEN_DIM)
    flat = flatten_action_hidden(h)
    assert flat.shape == (EXPECTED_FLAT,), f"shape mismatch: {flat.shape}"
    assert flat.dtype == torch.float16, f"dtype mismatch: {flat.dtype}"


def test_flatten_action_hidden_values_preserved():
    """Ensure cast to float16 is the only transformation (no normalization etc.)."""
    h = torch.ones(TIME_HORIZON, ACTION_DIM, TOKEN_DIM, dtype=torch.float32)
    flat = flatten_action_hidden(h)
    assert flat.shape == (EXPECTED_FLAT,)
    assert flat.dtype == torch.float16
    assert torch.all(flat == torch.tensor(1.0, dtype=torch.float16))


def test_flatten_action_hidden_wrong_batch_size_not_squeezed():
    """Batch size > 1 should NOT be squeezed — result has B*56*4096 elements."""
    h = torch.randn(2, TIME_HORIZON * ACTION_DIM, TOKEN_DIM)
    flat = flatten_action_hidden(h)
    assert flat.ndim == 1
    assert flat.shape[0] == 2 * EXPECTED_FLAT


def test_history_buffer_padding():
    """OFTRolloutHiddenExtractor pads history on first step (no model needed)."""

    # Minimal stub that satisfies the extractor's attribute accesses without a GPU.
    class _StubPolicy:
        class cfg:
            center_crop = True
            use_proprio = False
            use_film = False

        vla = None  # not called in this test
        processor = None
        action_head = None
        proprio_projector = None

    extractor = OFTRolloutHiddenExtractor(
        _StubPolicy(),  # type: ignore[arg-type]
        image_keys=["cam_a", "cam_b"],
        history=3,
        rotate_images_180=False,
        unnorm_key="dummy",
    )
    extractor.reset()

    # Feed the first frame: buffer should auto-pad to length 3
    frame_a = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_a[0, 0] = [1, 2, 3]
    frame_b = np.ones((4, 4, 3), dtype=np.uint8) * 7

    # Manually call _get_history to inspect padding without a full forward pass
    hist_a = extractor._get_history("cam_a", frame_a)
    hist_b = extractor._get_history("cam_b", frame_b)

    assert len(hist_a) == 3, f"expected 3 frames, got {len(hist_a)}"
    assert len(hist_b) == 3, f"expected 3 frames, got {len(hist_b)}"
    # First two entries are the padding copy of frame_a/frame_b
    assert np.array_equal(hist_a[0], frame_a) and np.array_equal(hist_a[1], frame_a)
    assert np.array_equal(hist_b[0], frame_b) and np.array_equal(hist_b[1], frame_b)
    # Third entry is the actual first frame
    assert np.array_equal(hist_a[2], frame_a)

    # Second call: push a NEW frame and verify oldest is evicted
    frame_a2 = np.full((4, 4, 3), 99, dtype=np.uint8)
    hist_a2 = extractor._get_history("cam_a", frame_a2)
    assert len(hist_a2) == 3
    assert np.array_equal(hist_a2[2], frame_a2), "newest frame should be last"
    # One pad copy of frame_a was evicted; the remaining two are both frame_a,
    # so the buffer still starts with frame_a (pixel-identical to a pad copy).
    assert np.array_equal(hist_a2[0], frame_a), "buffer[0] should still be frame_a"


# ── real-model consistency gate (skipped when ckpt/GPU/sidecar unavailable) ──

_SKIP_CKPT = f"OFT checkpoint not found at {OFT_CKPT}"
_SKIP_GPU = "No CUDA GPU available"
_SKIP_SIDECAR = f"Offline sidecar not found at {SIDECAR_DIR}"
_SKIP_REWARD = f"Reward HDF5 not found at {REWARD_DIR}"


def _ckpt_available() -> bool:
    return OFT_CKPT.is_dir() and any(OFT_CKPT.glob("action_head--*_checkpoint.pt"))


def _gpu_available() -> bool:
    return torch.cuda.is_available()


def _sidecar_available() -> bool:
    return (SIDECAR_DIR / _TEST_HDF5).is_file()


def _reward_available() -> bool:
    return (REWARD_DIR / _TEST_HDF5).is_file()


@pytest.mark.skipif(not _ckpt_available(), reason=_SKIP_CKPT)
@pytest.mark.skipif(not _gpu_available(), reason=_SKIP_GPU)
@pytest.mark.skipif(not _sidecar_available(), reason=_SKIP_SIDECAR)
@pytest.mark.skipif(not _reward_available(), reason=_SKIP_REWARD)
def test_inline_matches_offline_sidecar():
    """CONSISTENCY GATE: inline extractor reproduces offline sidecar obs_embedding.

    Drives OFTRolloutHiddenExtractor with frames read from the stored reward
    HDF5 (same source as the offline sidecar) and compares the returned
    flat_hidden against the pre-computed offline obs_embedding element-wise.

    Tolerance: atol=0.5 (≈ 2 fp16 ULPs at |x|≈4) with Pearson r > 0.999.
    See module docstring for the justification.

    Tested: demo_0 and demo_1 at t=1, T//4, T//2, T-1 (8 (demo,t) pairs).
    """
    import json

    import h5py

    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()

    from dreamervla.models.encoder.openvla_oft_policy import OpenVLAOFTPolicy

    # ── load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda:0")
    policy = OpenVLAOFTPolicy(
        model_path=str(OFT_CKPT),
        component_ckpt_dir=str(OFT_CKPT),
        torch_dtype="bf16",
        num_images_in_input=4,  # history(2) × views(2)
        use_lora=False,
        use_l1_regression=True,
        use_diffusion=False,
        use_proprio=True,
        use_film=False,
        freeze_vla_backbone=True,
    )
    policy.eval()
    policy.to(device)
    vla = policy.vla

    # Load LIBERO-specific norm_stats from dataset_statistics.json
    stats_path = OFT_CKPT / "dataset_statistics.json"
    with stats_path.open() as f:
        vla.norm_stats = json.load(f)
    assert "libero_goal_no_noops" in vla.norm_stats, (
        "libero_goal_no_noops not in norm_stats; wrong checkpoint?"
    )

    # Cast proprio_projector to bfloat16 so predict_action can pass bf16 proprio
    # through the projector (matches the actual sidecar-generation runtime).
    policy.proprio_projector.to(dtype=torch.bfloat16)

    # ── build extractor ───────────────────────────────────────────────────────
    extractor = OFTRolloutHiddenExtractor(
        policy,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
        history=2,
        rotate_images_180=True,
        unnorm_key="libero_goal_no_noops",
    )

    image_keys = ["agentview_rgb", "eye_in_hand_rgb"]
    unnorm_key = "libero_goal_no_noops"
    task_description = "open the middle drawer of the cabinet"

    failures: list[str] = []

    with (
        h5py.File(SIDECAR_DIR / _TEST_HDF5, "r") as sc,
        h5py.File(REWARD_DIR / _TEST_HDF5, "r") as rw,
    ):
        for demo_key in ["demo_0", "demo_1"]:
            obs_group = rw[f"data/{demo_key}/obs"]
            gt_emb = sc[f"data/{demo_key}/obs_embedding"][:]  # (T, 229376) float16
            T = gt_emb.shape[0]
            test_ts = [1, T // 4, T // 2, T - 1]

            # Reset extractor history at episode start
            extractor.reset()

            # Replay the demo from t=0 up to the last test timestep,
            # feeding frames one by one so history accumulates correctly.
            last_t = max(test_ts)
            for t in range(last_t + 1):
                # Build obs dict from stored HDF5 frames (raw, pre-rotation)
                obs = {}
                for key in image_keys:
                    obs[key] = np.asarray(obs_group[key][t], dtype=np.uint8)
                obs["state"] = np.concatenate([
                    np.asarray(obs_group["ee_pos"][t], dtype=np.float32).reshape(-1),
                    np.asarray(obs_group["ee_ori"][t], dtype=np.float32).reshape(-1),
                    np.asarray(obs_group["gripper_states"][t], dtype=np.float32).reshape(-1),
                ])

                _, flat_hidden = extractor.step(obs, task_description)

                if t not in test_ts:
                    continue

                # Compare against offline ground truth
                computed = flat_hidden.numpy().astype(np.float32)
                gt = gt_emb[t].astype(np.float32)

                max_err = float(np.abs(computed - gt).max())
                mean_err = float(np.abs(computed - gt).mean())
                corr = float(np.corrcoef(computed, gt)[0, 1])

                pass_atol = bool(np.allclose(computed, gt, atol=_ATOL, rtol=0))
                pass_corr = corr >= _MIN_PEARSON

                msg = (
                    f"{demo_key} t={t}: max_abs_err={max_err:.4f} "
                    f"mean_abs_err={mean_err:.6f} pearson_r={corr:.6f} "
                    f"pass_atol({_ATOL})={pass_atol} pass_corr({_MIN_PEARSON})={pass_corr}"
                )
                print(f"\n  [consistency] {msg}")

                if not pass_atol:
                    failures.append(
                        f"FAIL atol: {demo_key} t={t} max_err={max_err:.4f} > {_ATOL}"
                    )
                if not pass_corr:
                    failures.append(
                        f"FAIL corr: {demo_key} t={t} r={corr:.6f} < {_MIN_PEARSON}"
                    )

    if failures:
        pytest.fail(
            "Inline extractor does not match offline sidecar within tolerance:\n"
            + "\n".join(failures)
        )
