from __future__ import annotations

from src.workspace.base_workspace import BaseWorkspace
from src.workspace.chameleon_latent_action_wm_workspace import (
    ChameleonLatentActionWMWorkspace as _ChameleonLatentActionWMWorkspace,
)
from src.workspace.dreamer_vla_workspace import DreamerVLAWorkspace as _DreamerVLAWorkspace
from src.workspace.dreamerv3_pixel_workspace import DreamerV3PixelWorkspace as _DreamerV3PixelWorkspace
from src.workspace.dreamerv3_token_workspace import DreamerV3TokenWorkspace as _DreamerV3TokenWorkspace
from src.workspace.eval_libero_vla_workspace import EvalLiberoVLAWorkspace as _EvalLiberoVLAWorkspace
from src.workspace.pretokenize_vla_workspace import PretokenizeVLAWorkspace as _PretokenizeVLAWorkspace
from src.workspace.rynn_backbone_dreamerv3_wm_workspace import (
    RynnBackboneDreamerV3WMWorkspace as _RynnBackboneDreamerV3WMWorkspace,
)


class ActionHiddenWMWorkspace(_RynnBackboneDreamerV3WMWorkspace):
    workspace_name = "action_hidden_wm"
    workspace_status = "current"
    workspace_family = "world_model"


class PixelWMWorkspace(_DreamerV3PixelWorkspace):
    workspace_name = "pixel_wm"
    workspace_status = "secondary"
    workspace_family = "world_model"


class TokenWMWorkspace(_DreamerV3TokenWorkspace):
    workspace_name = "token_wm"
    workspace_status = "secondary"
    workspace_family = "world_model"


class VLASFTWorkspace(_PretokenizeVLAWorkspace):
    workspace_name = "vla_sft"
    workspace_status = "current"
    workspace_family = "vla"


class JointDreamerVLAWorkspace(_DreamerVLAWorkspace):
    workspace_name = "joint_dreamer_vla"
    workspace_status = "follow_up"
    workspace_family = "actor"


class LiberoEvalWorkspace(_EvalLiberoVLAWorkspace):
    workspace_name = "libero_eval"
    workspace_status = "current"
    workspace_family = "eval"


class ChameleonLatentWMWorkspace(_ChameleonLatentActionWMWorkspace):
    workspace_name = "chameleon_latent_wm"
    workspace_status = "secondary"
    workspace_family = "world_model"


PUBLIC_WORKSPACES = [
    "ActionHiddenWMWorkspace",
    "PixelWMWorkspace",
    "TokenWMWorkspace",
    "VLASFTWorkspace",
    "JointDreamerVLAWorkspace",
    "LiberoEvalWorkspace",
    "ChameleonLatentWMWorkspace",
]


__all__ = [
    "BaseWorkspace",
    "PUBLIC_WORKSPACES",
    "ActionHiddenWMWorkspace",
    "PixelWMWorkspace",
    "TokenWMWorkspace",
    "VLASFTWorkspace",
    "JointDreamerVLAWorkspace",
    "LiberoEvalWorkspace",
    "ChameleonLatentWMWorkspace",
]
