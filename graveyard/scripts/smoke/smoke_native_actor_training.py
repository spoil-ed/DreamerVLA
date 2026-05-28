"""Smoke test: HEAD's VLAActionHeadActor with all params trainable.

Goal — make it visually unambiguous that the *original* (HEAD-committed)
VLA actor can actually learn:

  1. Build VLAActionHeadActor with default kwargs (44M params, all
     requires_grad=True by default; no residual_mlp, no Pi0ActionHiddenActor).
  2. deepcopy() into a frozen reference.
  3. Pick a fixed eval input (random WM feat).
  4. Train the live actor for a handful of SGD steps on a dummy MSE loss
     against a random target chunk.
  5. At each step, print the action_chunk(eval_input) for both the
     trained and the frozen copies and the MSE between them.

If the printed numbers move while frozen stays constant, the original
actor's parameters are actually updating — i.e. there is no silent
freeze-bug in the original implementation.

This script intentionally does NOT touch the running A/B/C training
runs, and runs on CPU (small enough; ~44M params, batch=2).
"""

from __future__ import annotations

import copy
import sys

import torch

sys.path.insert(0, "/mnt/data/spoil/workspace/DreamerVLA")

from src.models.vla_actor import VLAActionHeadActor


def chunk_from(actor: VLAActionHeadActor, wm_feat: torch.Tensor) -> torch.Tensor:
    out = actor(
        {
            "mode": "sample",
            "hidden": wm_feat,
            "deterministic": True,
            "return_chunk": True,
        }
    )
    _, _, extra = out
    return extra["action_chunk"]


def fmt_vec(t: torch.Tensor) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in t.tolist()) + "]"


def main() -> None:
    torch.manual_seed(7)
    device = "cpu"

    actor = VLAActionHeadActor().to(device)
    frozen = copy.deepcopy(actor).to(device)
    for p in frozen.parameters():
        p.requires_grad = False
    frozen.eval()

    n_train = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    print(f"trained actor: trainable params = {n_train:,}")
    print(
        f"frozen  actor: trainable params = {n_frozen:,}  (deepcopy, requires_grad=False)"
    )
    print()

    eval_feat = torch.randn(2, 768, device=device)
    train_feat = torch.randn(2, 768, device=device)
    target_chunk = torch.randn(2, 10, 7, device=device)

    opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=1e-3)

    header = (
        f"{'step':>4} {'loss':>10} "
        f"{'mean|train-frozen|':>20} {'max|train-frozen|':>20} "
        f" trained chunk[0,0,:]                                        "
        f" frozen  chunk[0,0,:]"
    )
    print(header)
    print("-" * len(header))

    def report(step: int, loss_val: float) -> None:
        was_training = actor.training
        actor.eval()
        with torch.no_grad():
            tc = chunk_from(actor, eval_feat)
            fc = chunk_from(frozen, eval_feat)
            diff = (tc - fc).abs()
            mean_abs = float(diff.mean())
            max_abs = float(diff.max())
        if was_training:
            actor.train()
        tv = tc[0, 0, :].cpu()
        fv = fc[0, 0, :].cpu()
        print(
            f"{step:>4} {loss_val:>10.6f} {mean_abs:>20.6f} {max_abs:>20.6f}  "
            f"{fmt_vec(tv)}  {fmt_vec(fv)}"
        )

    # Step 0: nothing trained yet -> should match frozen exactly
    report(0, float("nan"))

    for step in range(1, 31):
        chunk = chunk_from(actor, train_feat)
        loss = (chunk - target_chunk).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step <= 5 or step % 5 == 0:
            report(step, float(loss))

    print()
    print("--- sanity check: are weights actually different from frozen? ---")
    n_same = 0
    n_diff = 0
    for (na, pa), (_, pf) in zip(actor.named_parameters(), frozen.named_parameters()):
        if torch.allclose(pa.detach(), pf.detach()):
            n_same += 1
        else:
            n_diff += 1
    print(f"trained-vs-frozen: {n_diff} param tensors changed, {n_same} unchanged")


if __name__ == "__main__":
    main()
