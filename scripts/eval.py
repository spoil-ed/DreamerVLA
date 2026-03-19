import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dreamer_vla.config import load_config
from dreamer_vla.trainer.main_ppo import build_trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the minimal Dreamer-VLA stack.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config override.")
    args = parser.parse_args()

    trainer = build_trainer(config=load_config(args.config))
    trainer.init_workers()
    metrics = trainer._validate(global_steps=0)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
