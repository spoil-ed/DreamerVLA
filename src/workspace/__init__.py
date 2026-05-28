from __future__ import annotations

from src.workspace.base_workspace import BaseWorkspace
from src.workspace.chameleon_latent_action_wm_workspace import (
    ChameleonLatentActionWMWorkspace as _ChameleonLatentActionWMWorkspace,
)
from src.workspace.dreamer_vla_workspace import (
    DreamerVLAWorkspace as _DreamerVLAWorkspace,
)
from src.workspace.dreamerv3_pixel_workspace import (
    DreamerV3PixelWorkspace as _DreamerV3PixelWorkspace,
)
from src.workspace.dreamerv3_token_workspace import (
    DreamerV3TokenWorkspace as _DreamerV3TokenWorkspace,
)
from src.workspace.eval_libero_vla_workspace import (
    EvalLiberoVLAWorkspace as _EvalLiberoVLAWorkspace,
)
from src.workspace.vla_sft_workspace import VLASFTWorkspace as _VLASFTWorkspace
from src.workspace.openvla_oft_workspace import (
    OpenVLAOFTTrainingWorkspace as _OpenVLAOFTTrainingWorkspace,
)
from src.workspace.rynn_backbone_dreamerv3_wm_workspace import (
    RynnBackboneDreamerV3WMWorkspace as _RynnBackboneDreamerV3WMWorkspace,
)
from src.workspace.rynn_dino_wm_workspace import (
    RynnDinoWMTrainingWorkspace as _RynnDinoWMTrainingWorkspace,
)
from src.workspace.latent_classifier_workspace import (
    LatentClassifierWorkspace as _LatentClassifierWorkspace,
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


class VLASFTWorkspace(_VLASFTWorkspace):
    workspace_name = "vla_sft"
    workspace_status = "current"
    workspace_family = "vla"


class OpenVLAOFTWorkspace(_OpenVLAOFTTrainingWorkspace):
    workspace_name = "openvla_oft"
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


class RynnDinoWMWorkspace(_RynnDinoWMTrainingWorkspace):
    workspace_name = "rynn_dino_wm"
    workspace_status = "secondary"
    workspace_family = "world_model"


class OFTDinoWMWorkspace(_RynnDinoWMTrainingWorkspace):
    workspace_name = "oft_dino_wm"
    workspace_status = "current"
    workspace_family = "world_model"


class LatentClassifierWorkspace(_LatentClassifierWorkspace):
    workspace_name = "latent_classifier"
    workspace_status = "current"
    workspace_family = "reward"


PUBLIC_WORKSPACES = [
    "ActionHiddenWMWorkspace",
    "PixelWMWorkspace",
    "TokenWMWorkspace",
    "VLASFTWorkspace",
    "OpenVLAOFTWorkspace",
    "JointDreamerVLAWorkspace",
    "LiberoEvalWorkspace",
    "ChameleonLatentWMWorkspace",
    "RynnDinoWMWorkspace",
    "OFTDinoWMWorkspace",
    "LatentClassifierWorkspace",
]


__all__ = [
    "BaseWorkspace",
    "PUBLIC_WORKSPACES",
    "ActionHiddenWMWorkspace",
    "PixelWMWorkspace",
    "TokenWMWorkspace",
    "VLASFTWorkspace",
    "OpenVLAOFTWorkspace",
    "JointDreamerVLAWorkspace",
    "LiberoEvalWorkspace",
    "ChameleonLatentWMWorkspace",
    "RynnDinoWMWorkspace",
    "OFTDinoWMWorkspace",
    "LatentClassifierWorkspace",
]
