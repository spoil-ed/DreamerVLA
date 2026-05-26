from __future__ import annotations

from src.workspace.pretokenize_vla_workspace import PretokenizeVLAWorkspace


class VLASFTWorkspace(PretokenizeVLAWorkspace):
    """Route-specific VLA SFT workspace.

    The training loop lives in ``PretokenizeVLAWorkspace`` for compatibility
    with existing checkpoints.  This class gives the route an explicit home so
    configs can distinguish VLA SFT from generic pretokenized data plumbing.
    """

    workspace_name = "vla_sft"
    workspace_status = "current"
    workspace_family = "vla"


__all__ = ["VLASFTWorkspace"]
