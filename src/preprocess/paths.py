from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed_data"
DEFAULT_CONVS_DIR = DEFAULT_DATA_ROOT / "convs"
DEFAULT_TOKENS_DIR = DEFAULT_DATA_ROOT / "tokens"
DEFAULT_CONCATE_DIR = DEFAULT_DATA_ROOT / "concate_tokens"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "data" / "ckpts" / "models--Alpha-VLLM--Lumina-mGPT-7B-768"
DEFAULT_CHAMELEON_TOKENIZER_DIR = PROJECT_ROOT / "data" / "ckpts" / "chameleon" / "tokenizer"
