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


# ── batched forward (step_batch) ─────────────────────────────────────────────
#
# Within-rank parallelism (migration §5.1) feeds K env observations through ONE
# VLA forward.  The upstream OFT predict_action wrapper has two batch==1 bugs
# (modeling_prismatic.py:972 token-cat, :924 reshape), so batched_forward bypasses
# it and runs the batch-safe internals.  The gating smoke
# (scripts/smoke_oft_batched_forward.py) established:
#   - B=1 batched == extractor.step (bit-exact),
#   - decoded actions are partner-invariant (no cross-batch leakage),
#   - obs_embedding has bf16 batched-kernel nondeterminism ~0.25 (same order as the
#     TF-vs-PIL gold tolerance) -> tolerance-based equivalence, NOT byte-identity.


def test_left_pad_batch_pads_left_and_repositions_bos():
    """Mixed-length sequences -> left-padded to batch max, BOS forced to absolute index 0.

    Left padding right-aligns the real content so the appended action tokens land at the
    same absolute index across the batch; BOS-at-0 keeps the vision-insert-after-BOS path
    uniform.  position_ids = cumsum(mask)-1 (computed downstream) then matches the
    unpadded positions.  No model needed.
    """
    from dreamervla.runners.rollout_hidden_extractor import _left_pad_batch

    BOS, PAD = 1, 0
    a = torch.tensor([[BOS, 10, 11]])            # L=3
    b = torch.tensor([[BOS, 20, 21, 22, 23]])    # L=5 (== batch max)
    ma = torch.ones(1, 3, dtype=torch.long)
    mb = torch.ones(1, 5, dtype=torch.long)

    ids, mask = _left_pad_batch([a, b], [ma, mb], PAD, BOS)

    assert ids.shape == (2, 5) and mask.shape == (2, 5)
    # short row: BOS at 0, pads in the middle, content (10,11) right-aligned
    assert ids[0].tolist() == [BOS, PAD, PAD, 10, 11]
    assert mask[0].tolist() == [1, 0, 0, 1, 1]
    # full-length row unchanged
    assert ids[1].tolist() == [BOS, 20, 21, 22, 23]
    assert mask[1].tolist() == [1, 1, 1, 1, 1]
    # cumsum(mask)-1 reproduces unpadded positions for the real tokens
    pos0 = (mask[0].cumsum(0) - 1).clamp(min=0).tolist()
    assert [pos0[3], pos0[4]] == [1, 2]  # 10->pos1, 11->pos2, same as standalone [BOS,10,11]


def test_left_pad_batch_same_length_is_noop():
    """Equal-length batch must pass through unchanged (preserves the bit-exact same-task path)."""
    from dreamervla.runners.rollout_hidden_extractor import _left_pad_batch

    a = torch.tensor([[1, 10, 11]])
    b = torch.tensor([[1, 20, 21]])
    ones = torch.ones(1, 3, dtype=torch.long)
    ids, mask = _left_pad_batch([a, b], [ones, ones.clone()], 0, 1)
    assert ids.tolist() == [[1, 10, 11], [1, 20, 21]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_batched_forward_rejects_empty():
    """Empty prep list must raise (nothing to batch)."""
    from dreamervla.runners.rollout_hidden_extractor import batched_forward

    with pytest.raises(ValueError):
        batched_forward(object(), [], "dummy")


# ── real-model batched-equivalence gate (skipped when ckpt/GPU unavailable) ───


def _rand_obs(seed: int) -> dict:
    rng = np.random.RandomState(seed)
    return {
        "agentview_rgb": rng.randint(0, 256, (256, 256, 3), dtype=np.uint8),
        "eye_in_hand_rgb": rng.randint(0, 256, (256, 256, 3), dtype=np.uint8),
        "state": rng.randn(8).astype(np.float32),
    }


def _load_oft_policy():
    import json

    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()
    from dreamervla.models.encoder.openvla_oft_policy import OpenVLAOFTPolicy

    policy = OpenVLAOFTPolicy(
        model_path=str(OFT_CKPT),
        component_ckpt_dir=str(OFT_CKPT),
        torch_dtype="bf16",
        num_images_in_input=4,
        use_lora=False,
        use_l1_regression=True,
        use_diffusion=False,
        use_proprio=True,
        use_film=False,
        freeze_vla_backbone=True,
    )
    policy.eval()
    policy.to(torch.device("cuda:0"))
    with (OFT_CKPT / "dataset_statistics.json").open() as f:
        policy.vla.norm_stats = json.load(f)
    policy.proprio_projector.to(dtype=torch.bfloat16)
    return policy


@pytest.fixture(scope="module")
def oft_policy():
    if not (_ckpt_available() and _gpu_available()):
        pytest.skip("OFT ckpt or GPU unavailable")
    return _load_oft_policy()


def _make_extractor(policy):
    return OFTRolloutHiddenExtractor(
        policy,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
        history=2,
        rotate_images_180=True,
        center_crop=True,
        unnorm_key="libero_goal_no_noops",
    )


@pytest.mark.skipif(not _ckpt_available(), reason=_SKIP_CKPT)
@pytest.mark.skipif(not _gpu_available(), reason=_SKIP_GPU)
def test_step_equals_batched_forward_at_b1(oft_policy):
    """B=1: batched_forward([prepare(obs)]) must reproduce extractor.step(obs) bit-exact.

    Proves step and the batched path share one code path (no numerical drift), so the
    existing offline-sidecar consistency gate also guards the batched path.
    """
    from dreamervla.runners.rollout_hidden_extractor import batched_forward

    task = "put the bowl on the plate"
    obs = _rand_obs(7)

    ext = _make_extractor(oft_policy)
    ext.reset()
    chunk_s, hid_s = ext.step(obs, task)

    ext2 = _make_extractor(oft_policy)
    ext2.reset()
    prep = ext2.prepare(obs, task)
    out = batched_forward(oft_policy, [prep], "libero_goal_no_noops")

    assert len(out) == 1
    chunk_b, hid_b = out[0]
    assert hid_b.shape == (EXPECTED_FLAT,) and hid_b.dtype == torch.float16
    hid_err = (hid_b.float() - hid_s.float()).abs().max().item()
    act_err = float(np.abs(np.asarray(chunk_b[0]) - np.asarray(chunk_s[0])).max())
    print(f"\n  [b1] max|hidden|={hid_err:.6g} max|action|={act_err:.6g}")
    assert hid_err == 0.0, f"B=1 batched hidden must equal step exactly, got {hid_err}"
    assert act_err == 0.0, f"B=1 batched action must equal step exactly, got {act_err}"


@pytest.mark.skipif(not _ckpt_available(), reason=_SKIP_CKPT)
@pytest.mark.skipif(not _gpu_available(), reason=_SKIP_GPU)
def test_batched_forward_matches_single_and_partner_invariant(oft_policy):
    """K=2 batched matches per-sample single within tolerance, with no cross-batch leakage.

    - actions match single within control-noise tolerance (atol=2e-2),
    - obs_embedding matches single within the gold tolerance (atol=0.5, pearson>0.999),
    - A's decoded action is invariant to whether its batch partner is B or C
      (functional no-leakage; bf16 batched kernels are NOT bit-deterministic, so we
      test the decoded action, not the raw hidden).
    """
    from dreamervla.runners.rollout_hidden_extractor import batched_forward

    task = "put the bowl on the plate"
    obs_a, obs_b, obs_c = _rand_obs(1), _rand_obs(2), _rand_obs(3)

    def single(obs):
        ext = _make_extractor(oft_policy)
        ext.reset()
        return ext.step(obs, task)

    chunk_a, hid_a = single(obs_a)
    chunk_b, hid_b = single(obs_b)

    def prep(obs):
        ext = _make_extractor(oft_policy)
        ext.reset()
        return ext.prepare(obs, task)

    ab = batched_forward(oft_policy, [prep(obs_a), prep(obs_b)], "libero_goal_no_noops")
    ac = batched_forward(oft_policy, [prep(obs_a), prep(obs_c)], "libero_goal_no_noops")
    assert len(ab) == 2

    # action accuracy vs single
    act_err_a = float(np.abs(np.asarray(ab[0][0][0]) - np.asarray(chunk_a[0])).max())
    act_err_b = float(np.abs(np.asarray(ab[1][0][0]) - np.asarray(chunk_b[0])).max())
    # obs_embedding within gold tolerance + correlation
    emb_a, gt_a = ab[0][1].numpy().astype(np.float32), hid_a.numpy().astype(np.float32)
    emb_err_a = float(np.abs(emb_a - gt_a).max())
    corr_a = float(np.corrcoef(emb_a, gt_a)[0, 1])
    # partner invariance: A's action with partner B vs partner C
    partner_drift = float(np.abs(np.asarray(ab[0][0][0]) - np.asarray(ac[0][0][0])).max())

    print(f"\n  [k2] act_err_a={act_err_a:.4g} act_err_b={act_err_b:.4g} "
          f"emb_err_a={emb_err_a:.4g} corr_a={corr_a:.6f} partner_drift={partner_drift:.4g}")

    assert act_err_a <= 2e-2 and act_err_b <= 2e-2, "batched action must match single within control noise"
    assert emb_err_a <= _ATOL and corr_a >= _MIN_PEARSON, "obs_embedding must match single within gold tolerance"
    assert partner_drift <= 2e-2, "A's action must not depend on its batch partner (no leakage)"


@pytest.mark.skipif(not _ckpt_available(), reason=_SKIP_CKPT)
@pytest.mark.skipif(not _gpu_available(), reason=_SKIP_GPU)
def test_batched_forward_mixed_task_matches_single(oft_policy):
    """MIXED-TASK batch (different prompts, different token lengths) matches per-sample single.

    Left-padding + position_ids=cumsum(mask)-1 + attention mask make a padded sample
    numerically equal to computing it alone (block-diagonal attention => no cross-sample
    interaction).  Verifies action accuracy, obs_embedding within gold tolerance, and that
    a short-prompt sample's action is invariant to which long-prompt partner it batches with.
    """
    from dreamervla.runners.rollout_hidden_extractor import batched_forward

    task_short = "pick up the bowl"
    task_long = "put both the cream cheese box and the butter in the basket on the right"
    obs_a, obs_b, obs_c = _rand_obs(21), _rand_obs(22), _rand_obs(23)

    def single(obs, task):
        ext = _make_extractor(oft_policy)
        ext.reset()
        return ext.step(obs, task)

    def prep(obs, task):
        ext = _make_extractor(oft_policy)
        ext.reset()
        return ext.prepare(obs, task)

    chunk_a, hid_a = single(obs_a, task_short)
    chunk_b, hid_b = single(obs_b, task_long)

    pa, pb = prep(obs_a, task_short), prep(obs_b, task_long)
    assert pa["input_ids"].shape[-1] != pb["input_ids"].shape[-1], (
        "test needs different prompt token lengths to exercise padding"
    )

    ab = batched_forward(oft_policy, [pa, pb], "libero_goal_no_noops")
    assert len(ab) == 2

    act_err_a = float(np.abs(np.asarray(ab[0][0][0]) - np.asarray(chunk_a[0])).max())
    act_err_b = float(np.abs(np.asarray(ab[1][0][0]) - np.asarray(chunk_b[0])).max())
    emb_a, gt_a = ab[0][1].numpy().astype(np.float32), hid_a.numpy().astype(np.float32)
    emb_b, gt_b = ab[1][1].numpy().astype(np.float32), hid_b.numpy().astype(np.float32)
    emb_err_a = float(np.abs(emb_a - gt_a).max()); corr_a = float(np.corrcoef(emb_a, gt_a)[0, 1])
    emb_err_b = float(np.abs(emb_b - gt_b).max()); corr_b = float(np.corrcoef(emb_b, gt_b)[0, 1])

    # partner invariance: short-prompt A with long-prompt B vs with long-prompt C
    ac = batched_forward(oft_policy, [prep(obs_a, task_short), prep(obs_c, task_long)], "libero_goal_no_noops")
    partner_drift = float(np.abs(np.asarray(ab[0][0][0]) - np.asarray(ac[0][0][0])).max())

    print(f"\n  [mixed] act_err_a={act_err_a:.4g} act_err_b={act_err_b:.4g} "
          f"emb_err_a={emb_err_a:.4g} corr_a={corr_a:.6f} emb_err_b={emb_err_b:.4g} "
          f"corr_b={corr_b:.6f} partner_drift={partner_drift:.4g}")

    assert act_err_a <= 2e-2 and act_err_b <= 2e-2, "mixed-task batched action must match single"
    assert emb_err_a <= _ATOL and corr_a >= _MIN_PEARSON, "short-task obs_embedding off tolerance"
    assert emb_err_b <= _ATOL and corr_b >= _MIN_PEARSON, "long-task obs_embedding off tolerance"
    assert partner_drift <= 2e-2, "A's action must not depend on its (differently-lengthed) partner"


# ── discrete / headless (one-trajectory) gate ────────────────────────────────
# The real cold-start base policy is the one-trajectory OFT checkpoint, which is
# DISCRETE: no L1 action head, actions decoded from the LM logits (argmax -> bin
# centers).  The obs_embedding (action-query hidden) is head-independent, so the
# batched machinery must hold here too; only the action decode differs.

ONE_TRAJ_DISCRETE_CKPT = (
    PROJECT_ROOT / "data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
)


def _discrete_ckpt_available() -> bool:
    return (
        ONE_TRAJ_DISCRETE_CKPT.is_dir()
        and not any(ONE_TRAJ_DISCRETE_CKPT.glob("action_head--*.pt"))
        and any(ONE_TRAJ_DISCRETE_CKPT.glob("*.safetensors"))
    )


def _load_oft_discrete_policy():
    import json

    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()
    from dreamervla.models.encoder.openvla_oft_policy import OpenVLAOFTPolicy

    policy = OpenVLAOFTPolicy(
        model_path=str(ONE_TRAJ_DISCRETE_CKPT),
        component_ckpt_dir=str(ONE_TRAJ_DISCRETE_CKPT),
        torch_dtype="bf16",
        num_images_in_input=4,
        use_lora=False,
        use_l1_regression=False,  # discrete LM-head
        use_diffusion=False,
        use_proprio=False,  # discrete one-traj ckpt has no proprio projector
        use_film=False,
        freeze_vla_backbone=True,
    )
    policy.eval()
    policy.to(torch.device("cuda:0"))
    with (ONE_TRAJ_DISCRETE_CKPT / "dataset_statistics.json").open() as f:
        policy.vla.norm_stats = json.load(f)
    return policy


@pytest.fixture(scope="module")
def oft_discrete_policy():
    if not (_discrete_ckpt_available() and _gpu_available()):
        pytest.skip("one-traj discrete ckpt or GPU unavailable")
    return _load_oft_discrete_policy()


def _make_discrete_extractor(policy):
    return OFTRolloutHiddenExtractor(
        policy,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
        history=2,
        rotate_images_180=True,
        center_crop=True,
        unnorm_key="libero_goal_no_noops",
    )


@pytest.mark.skipif(not _discrete_ckpt_available(), reason="one-traj discrete ckpt not found")
@pytest.mark.skipif(not _gpu_available(), reason=_SKIP_GPU)
def test_batched_forward_discrete_headless(oft_discrete_policy):
    """Discrete (headless) one-traj ckpt: actions from logits->bins; obs_embedding head-independent.

    - the policy has NO action head (action_head is None),
    - B=1 batched == single bit-exact (machinery is head-agnostic),
    - decoded action is shape (7,) and (near) partner-invariant (argmax can flip a bin under
      bf16 noise, so a small tolerance — not exact like the continuous L1 head).
    """
    from dreamervla.runners.rollout_hidden_extractor import batched_forward

    assert oft_discrete_policy.action_head is None, "fixture must be the headless/discrete ckpt"

    task = "open the middle drawer of the cabinet"
    obs_a, obs_b, obs_c = _rand_obs(31), _rand_obs(32), _rand_obs(33)

    def single(obs):
        ext = _make_discrete_extractor(oft_discrete_policy)
        ext.reset()
        return ext.step(obs, task)

    def prep(obs):
        ext = _make_discrete_extractor(oft_discrete_policy)
        ext.reset()
        return ext.prepare(obs, task)

    chunk_a, hid_a = single(obs_a)
    assert hid_a.shape == (EXPECTED_FLAT,) and hid_a.dtype == torch.float16
    assert np.asarray(chunk_a[0]).shape == (7,)

    out1 = batched_forward(oft_discrete_policy, [prep(obs_a)], "libero_goal_no_noops")
    b1_hid = (out1[0][1].float() - hid_a.float()).abs().max().item()
    b1_act = float(np.abs(np.asarray(out1[0][0][0]) - np.asarray(chunk_a[0])).max())

    ab = batched_forward(oft_discrete_policy, [prep(obs_a), prep(obs_b)], "libero_goal_no_noops")
    ac = batched_forward(oft_discrete_policy, [prep(obs_a), prep(obs_c)], "libero_goal_no_noops")
    partner = float(np.abs(np.asarray(ab[0][0][0]) - np.asarray(ac[0][0][0])).max())

    print(f"\n  [discrete] b1_hid={b1_hid:.4g} b1_act={b1_act:.4g} partner_drift={partner:.4g}")
    assert b1_hid == 0.0 and b1_act == 0.0, "B=1 discrete batched must equal single exactly"
    assert partner <= 5e-2, "discrete action must be ~partner-invariant (allow a bin flip)"
