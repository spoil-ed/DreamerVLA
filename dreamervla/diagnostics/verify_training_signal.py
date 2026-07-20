"""Verify an imagined-success SFT checkpoint without loading its models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dreamervla.runtime.training_signal import evaluate_imagined_success_sft_signal
from dreamervla.utils.hf_checkpoint import load_runner_payload
from dreamervla.utils.run_paths import resolve_resume_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    """Print a JSON verdict and return nonzero when the signal contract fails."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="cotrain checkpoint, checkpoints directory, or owning run root",
    )
    args = parser.parse_args(argv)
    checkpoint = resolve_resume_checkpoint(args.checkpoint)
    payload = load_runner_payload(checkpoint)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    cfg = payload.get("cfg", {})
    manual_cotrain = cfg.get("manual_cotrain", {}) if isinstance(cfg, dict) else {}
    training_mode = (
        str(manual_cotrain.get("training_mode", ""))
        if isinstance(manual_cotrain, dict)
        else ""
    )
    result = evaluate_imagined_success_sft_signal(
        metrics,
        training_mode=training_mode,
        policy_initial_hash=str(payload.get("policy_initial_hash", "")),
        policy_final_hash=str(payload.get("policy_final_hash", "")),
        applied_policy_steps=int(payload.get("applied_policy_steps", 0) or 0),
    )
    rendered = {"checkpoint": str(checkpoint), **result.to_dict()}
    print(json.dumps(rendered, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
