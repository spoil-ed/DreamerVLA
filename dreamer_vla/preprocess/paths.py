from __future__ import annotations

from pathlib import Path

from dreamer_vla.utils.paths import checkpoints_path, data_root, processed_data_path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DATA_ROOT = processed_data_path()
DEFAULT_CONVS_DIR = DEFAULT_DATA_ROOT / "convs"
DEFAULT_TOKENS_DIR = DEFAULT_DATA_ROOT / "tokens"
DEFAULT_CONCATE_DIR = DEFAULT_DATA_ROOT / "concate_tokens"
DEFAULT_TOKENIZER_PATH = checkpoints_path("models--Alpha-VLLM--Lumina-mGPT-7B-768")
DEFAULT_CHAMELEON_TOKENIZER_DIR = checkpoints_path("chameleon", "tokenizer")

__all__ = [
    "DEFAULT_CHAMELEON_TOKENIZER_DIR",
    "DEFAULT_CONCATE_DIR",
    "DEFAULT_CONVS_DIR",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_TOKENIZER_PATH",
    "DEFAULT_TOKENS_DIR",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "data_root",
]
