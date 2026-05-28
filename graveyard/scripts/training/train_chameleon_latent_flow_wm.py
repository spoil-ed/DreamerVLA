from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description="Train LaDiWM-style Chameleon latent-flow WM.")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES value, e.g. '0' or '0,1'.")
    parser.add_argument("--config-name", default="chameleon_latent_action_wm_libero_10")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Extra Hydra-style overrides.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "src.cli.train",
        "--config-name",
        args.config_name,
    ]
    if args.batch_size is not None:
        cmd.append(f"dataloader.batch_size={args.batch_size}")
    if args.output_dir is not None:
        cmd.append(f"training.out_dir={args.output_dir}")
    if args.debug:
        cmd.extend(
            [
                "training.debug=true",
                "training.max_train_steps=1",
                "dataloader.batch_size=1",
                "dataloader.num_workers=0",
                "dataloader.persistent_workers=false",
                "dataset_val_ind=null",
                "dataset_val_ood=null",
            ]
        )
    cmd.extend(args.overrides)
    raise SystemExit(subprocess.call(cmd, cwd=repo, env=env))


if __name__ == "__main__":
    main()
