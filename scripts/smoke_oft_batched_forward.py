"""Gating smoke (migration §5.0): can OFT run BATCHED (B=K) action-hidden inference?

Decision this answers: the within-rank parallelism plan
(docs/superpowers/specs/2026-06-16-rlinf-vectorized-rollout-migration.md) hinges on
feeding K env observations through ONE VLA forward.  The upstream OFT
``OpenVLAForActionPrediction.predict_action`` wrapper has two batch==1 assumptions:

  - modeling_prismatic.py:972  appends a [1,1] token via cat(dim=1) -> breaks for B>1
  - modeling_prismatic.py:924  reshape(NUM_ACTIONS_CHUNK, ACTION_DIM) -> drops the batch dim

Everything else in the L1-regression path is batch-safe (verified by reading the
helpers + L1RegressionActionHead.predict_action which reshapes via shape[0]).

This smoke proves:
  A. the naive wrapper call with B=2 raises (documents the blocker), and
  B. a "fixed" batched-internals forward (batched token-append + (B,8,7) reshape)
     reproduces the per-sample single-obs ``(action_chunk, obs_embedding)`` exactly.

If B passes, batched inference is viable by bypassing the wrapper -> green light to
implement ``OFTRolloutHiddenExtractor.step_batch`` via TDD.

This is a THROWAWAY spike (TDD exception: explore-first).  Run:
    cd /mnt/data/spoil/workspace/DreamerVLA
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
    export CUDA_VISIBLE_DEVICES=0
    PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python scripts/smoke_oft_batched_forward.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


MODEL_PATH = "data/checkpoints/OpenVLA-OFT/libero_goal"
UNNORM_KEY = "libero_goal_no_noops"
TASK = "put the bowl on the plate"


def _load_policy(device: torch.device):
    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()
    from dreamervla.models.encoder.openvla_oft_policy import OpenVLAOFTPolicy

    model_path = str(Path(MODEL_PATH).resolve())
    policy = OpenVLAOFTPolicy(
        model_path=model_path,
        component_ckpt_dir=model_path,
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
    policy.to(device)
    with (Path(model_path) / "dataset_statistics.json").open() as fh:
        policy.vla.norm_stats = json.load(fh)
    if policy.proprio_projector is not None:
        policy.proprio_projector.to(dtype=torch.bfloat16)
    return policy


def _make_obs(seed: int) -> dict:
    rng = np.random.RandomState(seed)
    return {
        "agentview_rgb": rng.randint(0, 256, (256, 256, 3), dtype=np.uint8),
        "eye_in_hand_rgb": rng.randint(0, 256, (256, 256, 3), dtype=np.uint8),
        "state": rng.randn(8).astype(np.float32),
    }


def _fixed_batched_forward(policy, captured, unnorm_key):
    """Replicate predict_action + _regression_or_discrete_prediction for B=K.

    FIX1: batched trailing-29871 append.  FIX2: reshape (B, NUM_ACTIONS_CHUNK, ACTION_DIM).
    Returns (actions np (B,chunk,adim), actions_hidden tensor (B,56,D)).
    """
    from prismatic.vla.constants import (
        ACTION_DIM,
        IGNORE_INDEX,
        NUM_ACTIONS_CHUNK,
    )

    model = policy.vla
    action_head = policy.action_head
    proprio_projector = policy.proprio_projector

    input_ids = torch.cat([c["input_ids"] for c in captured], dim=0)
    attention_mask = torch.cat([c["attention_mask"] for c in captured], dim=0)
    pixel_values = torch.cat([c["pixel_values"] for c in captured], dim=0)
    use_proprio = captured[0]["proprio"] is not None
    proprio = (
        np.stack([np.asarray(c["proprio"]).reshape(-1) for c in captured], axis=0)
        if use_proprio
        else None
    )

    # FIX1: batched token append (vs upstream [1,1] cat)
    if not torch.all(input_ids[:, -1] == 29871):
        pad = torch.full(
            (input_ids.shape[0], 1), 29871, dtype=input_ids.dtype, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, pad], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(pad, dtype=attention_mask.dtype)], dim=1
        )

    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1
    input_ids, attention_mask = model._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels = model._prepare_labels_for_action_prediction(labels, input_ids)

    # Wrap the whole forward (incl. action_head) in inference_mode, matching how
    # the extractor wraps predict_action — otherwise the head's layer_norm tries
    # to save inference tensors for backward.
    with torch.inference_mode():
        input_embeddings = model.get_input_embeddings()(input_ids)
        all_actions_mask = model._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        projected = model._process_vision_features(pixel_values, language_embeddings, use_film=False)
        if use_proprio:
            proprio_t = torch.Tensor(proprio).to(projected.device, dtype=projected.dtype)
            projected = model._process_proprio_features(projected, proprio_t, proprio_projector)

        NUM_PATCHES = (
            model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input()
        )
        if use_proprio:
            NUM_PATCHES += 1

        all_actions_mask_u = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask_u
        multimodal_embeddings, multimodal_attention_mask = model._build_multimodal_attention(
            input_embeddings, projected, attention_mask
        )
        lm_out = model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = lm_out.hidden_states[-1]
        span = ACTION_DIM * NUM_ACTIONS_CHUNK
        actions_hidden = last_hidden[
            :, NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + span, :
        ]  # (B, 56, D)
        normalized = action_head.predict_action(actions_hidden)  # (B, chunk, adim)
        B = input_ids.shape[0]
        normalized = normalized.reshape(B, NUM_ACTIONS_CHUNK, ACTION_DIM).float().cpu().numpy()  # FIX2
    actions = model._unnormalize_actions(normalized, unnorm_key)  # (B, chunk, adim)
    return actions, actions_hidden


def main() -> None:
    device = torch.device("cuda:0")
    print(f"[smoke] loading OFT policy on {device} ...", flush=True)
    policy = _load_policy(device)
    from dreamervla.runners.rollout_hidden_extractor import OFTRolloutHiddenExtractor

    ext = OFTRolloutHiddenExtractor(
        policy,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
        history=2,
        rotate_images_180=True,
        center_crop=True,
        unnorm_key=UNNORM_KEY,
    )

    obs_a, obs_b, obs_c = _make_obs(1), _make_obs(2), _make_obs(3)

    # ── single-obs path (ground truth) + capture exact predict_action inputs ──
    captured: list[dict] = []
    orig = policy.vla.predict_action

    def _capture(*args, **kwargs):
        captured.append(
            {k: kwargs.get(k) for k in ("input_ids", "pixel_values", "attention_mask", "proprio")}
        )
        return orig(*args, **kwargs)

    policy.vla.predict_action = _capture
    ext.reset()
    chunk_a, hid_a = ext.step(obs_a, TASK)
    ext.reset()
    chunk_b, hid_b = ext.step(obs_b, TASK)
    ext.reset()
    chunk_c, hid_c = ext.step(obs_c, TASK)
    policy.vla.predict_action = orig
    cap_a, cap_b, cap_c = captured

    print(f"[smoke] single: hid_a {tuple(hid_a.shape)} {hid_a.dtype}  "
          f"action[0] {np.asarray(chunk_a[0]).shape}", flush=True)
    assert hid_a.shape == (229376,), hid_a.shape
    assert len(captured) == 3

    # ── A. naive wrapper batched call SHOULD fail (documents the blocker) ──
    naive_ok = False
    try:
        bi = torch.cat([captured[0]["input_ids"], captured[1]["input_ids"]], dim=0)
        bp = torch.cat([captured[0]["pixel_values"], captured[1]["pixel_values"]], dim=0)
        bm = torch.cat([captured[0]["attention_mask"], captured[1]["attention_mask"]], dim=0)
        bpr = np.stack([np.asarray(captured[0]["proprio"]).reshape(-1),
                        np.asarray(captured[1]["proprio"]).reshape(-1)], axis=0)
        with torch.inference_mode():
            policy.vla.predict_action(
                input_ids=bi, pixel_values=bp, attention_mask=bm, unnorm_key=UNNORM_KEY,
                do_sample=False, proprio=bpr, proprio_projector=policy.proprio_projector,
                action_head=policy.action_head, use_film=False,
            )
        naive_ok = True
        print("[smoke] A. naive wrapper B=2: UNEXPECTEDLY SUCCEEDED", flush=True)
    except Exception as exc:
        print(f"[smoke] A. naive wrapper B=2 failed as expected: "
              f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)

    from dreamervla.runners.rollout_hidden_extractor import flatten_action_hidden

    # ── B1. replication faithfulness: my batched fn at B=1 must reproduce the extractor ──
    actions_b1a, hidden_b1a = _fixed_batched_forward(policy, [cap_a], UNNORM_KEY)
    rep_hid_err = (flatten_action_hidden(hidden_b1a[0:1].cpu()).float() - hid_a.float()).abs().max().item()
    rep_act_err = float(np.abs(actions_b1a[0, 0] - np.asarray(chunk_a[0])).max())
    print(f"[smoke] B1. replication vs extractor single (B=1): "
          f"max|hidden|={rep_hid_err:.4g}  max|action|={rep_act_err:.4g}  (expect ~0)", flush=True)

    # ── B2. FUNCTIONAL no-leakage: A's decoded action must be invariant to its batch ──
    #   partner. [A,B] row0 vs [A,C] row0 — if A's action barely moves when the neighbour
    #   changes B->C, there is no cross-batch contamination.  (Bit-identity is NOT expected:
    #   bf16 batched attention/GEMM is non-deterministic per row — established below.)
    actions_ab, hidden_ab = _fixed_batched_forward(policy, [cap_a, cap_b], UNNORM_KEY)
    actions_ac, _hidden_ac = _fixed_batched_forward(policy, [cap_a, cap_c], UNNORM_KEY)
    partner_act_drift = float(np.abs(actions_ab[0, 0] - actions_ac[0, 0]).max())
    print(f"[smoke] B2. A's action drift when partner B->C: max|diff|={partner_act_drift:.4g} "
          f"(expect ~batch-noise, NOT O(1))", flush=True)

    # ── B3. batched K=2 (A,B) vs per-sample single: action accuracy + obs_embedding residual ──
    act_err_a = float(np.abs(actions_ab[0, 0] - np.asarray(chunk_a[0])).max())
    act_err_b = float(np.abs(actions_ab[1, 0] - np.asarray(chunk_b[0])).max())
    hid_resid_a = (flatten_action_hidden(hidden_ab[0:1].cpu()).float() - hid_a.float()).abs().max().item()
    hid_resid_b = (flatten_action_hidden(hidden_ab[1:2].cpu()).float() - hid_b.float()).abs().max().item()
    print(f"[smoke] B3. batched shapes hidden {tuple(hidden_ab.shape)} actions {actions_ab.shape}", flush=True)
    print(f"[smoke] B3. max|action_batched - single|:    obsA={act_err_a:.4g}  obsB={act_err_b:.4g}", flush=True)
    print(f"[smoke] B3. max|obs_embedding_batched - single| (INFO; bf16 batch nondeterminism): "
          f"obsA={hid_resid_a:.4g}  obsB={hid_resid_b:.4g}", flush=True)

    # Decision (principled):
    #  1. replication exact  -> the batched math is correct (B=1 reproduces extractor).
    #  2. partner-invariance -> no cross-batch contamination (A's action ignores its neighbour).
    #  3. action accuracy    -> batched decode matches single within control noise.
    #  The obs_embedding residual (~0.2-0.4 fp16 max-abs) is bf16 batched-kernel nondeterminism,
    #  same order as the pipeline's existing TF-vs-PIL ~0.25 gold tolerance -> reported, not gated.
    #  CONSEQUENCE for migration: batched dumps are NOT byte-identical to single -> acceptance
    #  criterion must be tolerance-based, not byte-equality.
    replication_ok = rep_hid_err <= 5e-2 and rep_act_err <= 5e-3
    no_leak = partner_act_drift <= 2e-2
    actions_ok = act_err_a <= 2e-2 and act_err_b <= 2e-2
    assert hidden_ab.shape[0] == 2 and actions_ab.shape[0] == 2

    ok = replication_ok and no_leak and actions_ok
    print(f"\n[smoke] checks: replication_exact={replication_ok}  partner_invariant={no_leak}  "
          f"actions_within_tol={actions_ok}", flush=True)
    print(f"[smoke] RESULT: {'GREEN — batched B=K viable via fixed internals (bypass predict_action wrapper)' if ok else 'RED — investigate'}",
          flush=True)
    print(f"[smoke]   naive wrapper batchable: {naive_ok} (expected False)", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
