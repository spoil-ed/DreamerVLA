"""Project-wide constants with a single canonical definition (X-03).

Kept dependency-free so any module can import it without cycles.
"""

from __future__ import annotations

# OpenVLA-OFT / Chameleon action-token start id: the vocab position where the
# discrete action-token range begins. It is a property of the pretrained VLA
# tokenizer, not a free hyperparameter — override the per-route ``target_token_id``
# config / CLI key only when the backbone's vocab actually differs. Every
# first-party site reads this constant instead of repeating the literal 10004.
DEFAULT_ACTION_TOKEN_ID = 10004
