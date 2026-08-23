# OpenVLA-OFT Action-Token Decode Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore exact OpenVLA-OFT discrete action-token semantics so base evaluation, real rollout, SFT, and PPO retain all 256 action-token classes while decoding the terminal class to the final one of 255 bin centers.

**Architecture:** Keep the native OpenVLA decoder and its differentiable categorical interface. Derive the categorical vocabulary from `ActionTokenizer.vocab_size` (256), not `len(model.bin_centers)` (255), and centralize class-to-bin-center clipping so rollout and actor sampling share upstream-compatible endpoint behavior.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face/OpenVLA-OFT, pytest.

---

### Task 1: Preserve the terminal OpenVLA action token

**Files:**
- Modify: `dreamervla/models/embodiment/openvla_oft_policy.py`
- Modify: `dreamervla/runners/rollout_hidden_extractor.py`
- Test: `tests/unit_tests/test_openvla_oft_policy_native_forward.py`

- [x] **Step 1: Write the failing tests**

Add tests proving that the native action vocabulary contains all 256 checkpoint action tokens, including `model.vocab_size - 256`, and that both terminal action classes decode to the final bin center.

```python
def test_native_action_vocabulary_includes_terminal_checkpoint_token() -> None:
    policy = _policy()
    output = policy.forward_action_tokens(
        input_ids=torch.tensor([[1, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        pixel_values=torch.ones((1, 3, 2, 2)),
    )
    assert output.action_logits.shape[-1] == 256
    assert output.action_token_ids[-1].item() == policy.vla.vocab_size - 256


def test_terminal_action_class_clips_to_last_bin_center() -> None:
    policy = _policy()
    classes = torch.tensor([254, 255])
    decoded = policy.action_classes_to_normalized_actions(classes)
    assert decoded[0].item() == decoded[1].item() == policy.vla.bin_centers[-1]
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
pytest -q tests/unit_tests/test_openvla_oft_policy_native_forward.py
```

Expected: failures showing 255 logits instead of 256 and absence of terminal-class clipping.

- [x] **Step 3: Implement the minimal fix**

In `OpenVLAOFTPolicy`, add a helper that derives the 256-class count from `self.action_tokenizer.vocab_size`, validates the checkpoint geometry, and maps class indices to `bin_centers` using `clamp(max=len(bin_centers)-1)`. Use it in native forward and actor action decoding. Use the same helper from `OFTBatchedDecoder` so base/real rollout and actor sampling cannot diverge.

- [x] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
pytest -q tests/unit_tests/test_openvla_oft_policy_native_forward.py tests/unit_tests/test_rollout_hidden_extractor.py tests/unit_tests/test_openvla_oft_base_eval_runner.py
```

Expected: all tests pass.

- [x] **Step 5: Verify real-checkpoint parity**

Run the opt-in CUDA parity diagnostic on the one-trajectory checkpoint. Compare native deterministic token IDs and normalized actions against the upstream full-vocabulary argmax plus `vocab_size - token_id - 1` clipping for all 56 action positions.

Expected: 56/56 token IDs and actions agree, including gripper positions that emit `vocab_size - 256`.

- [x] **Step 6: Run regression suite and commit**

Run:

```bash
pytest -q tests/unit_tests/test_openvla_oft_policy_native_forward.py \
  tests/unit_tests/test_rollout_hidden_extractor.py \
  tests/unit_tests/test_openvla_oft_base_eval_runner.py \
  tests/unit_tests/test_manual_cotrain_stage_order.py \
  tests/unit_tests/test_encoder_sft_phase.py
git diff --check
```

Then commit with:

```bash
git add docs/superpowers/plans/2026-07-13-openvla-oft-action-token-decode-fix.md \
  dreamervla/models/embodiment/openvla_oft_policy.py \
  dreamervla/runners/rollout_hidden_extractor.py \
  tests/unit_tests/test_openvla_oft_policy_native_forward.py
git commit -s -m "fix: preserve terminal OFT action token"
```
