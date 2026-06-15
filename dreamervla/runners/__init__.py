from __future__ import annotations

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.chameleon_latent_action_wm_runner import (
    ChameleonLatentActionWMRunner as _ChameleonLatentActionWMRunner,
)
from dreamervla.runners.dreamervla_runner import (
    DreamerVLARunner as _DreamerVLARunner,
)
from dreamervla.runners.dreamerv3_pixel_runner import (
    DreamerV3PixelRunner as _DreamerV3PixelRunner,
)
from dreamervla.runners.dreamerv3_token_runner import (
    DreamerV3TokenRunner as _DreamerV3TokenRunner,
)
from dreamervla.runners.eval_libero_vla_runner import (
    EvalLiberoVLARunner as _EvalLiberoVLARunner,
)
from dreamervla.runners.latent_classifier_runner import (
    LatentClassifierRunner as _LatentClassifierRunner,
)
from dreamervla.runners.openvla_oft_runner import (
    OpenVLAOFTTrainingRunner as _OpenVLAOFTTrainingRunner,
)
from dreamervla.runners.rynn_backbone_dreamerv3_wm_runner import (
    RynnBackboneDreamerV3WMRunner as _RynnBackboneDreamerV3WMRunner,
)
from dreamervla.runners.rynn_dino_wm_runner import (
    RynnDinoWMTrainingRunner as _RynnDinoWMTrainingRunner,
)
from dreamervla.runners.vla_sft_runner import VLASFTRunner as _VLASFTRunner


class ActionHiddenWMRunner(_RynnBackboneDreamerV3WMRunner):
    runner_name = "action_hidden_wm"
    runner_status = "current"
    runner_family = "world_model"


class PixelWMRunner(_DreamerV3PixelRunner):
    runner_name = "pixel_wm"
    runner_status = "secondary"
    runner_family = "world_model"


class TokenWMRunner(_DreamerV3TokenRunner):
    runner_name = "token_wm"
    runner_status = "secondary"
    runner_family = "world_model"


class VLASFTRunner(_VLASFTRunner):
    runner_name = "vla_sft"
    runner_status = "current"
    runner_family = "vla"


class OpenVLAOFTRunner(_OpenVLAOFTTrainingRunner):
    runner_name = "openvla_oft"
    runner_status = "current"
    runner_family = "vla"


class JointDreamerVLARunner(_DreamerVLARunner):
    runner_name = "joint_dreamervla"
    runner_status = "follow_up"
    runner_family = "actor"


class LiberoEvalRunner(_EvalLiberoVLARunner):
    runner_name = "libero_eval"
    runner_status = "current"
    runner_family = "eval"


class ChameleonLatentWMRunner(_ChameleonLatentActionWMRunner):
    runner_name = "chameleon_latent_wm"
    runner_status = "secondary"
    runner_family = "world_model"


class RynnDinoWMRunner(_RynnDinoWMTrainingRunner):
    runner_name = "rynn_dino_wm"
    runner_status = "secondary"
    runner_family = "world_model"


class OFTDinoWMRunner(_RynnDinoWMTrainingRunner):
    runner_name = "oft_dino_wm"
    runner_status = "current"
    runner_family = "world_model"


class LatentClassifierRunner(_LatentClassifierRunner):
    runner_name = "latent_classifier"
    runner_status = "current"
    runner_family = "reward"


PUBLIC_RUNNERS = [
    "ActionHiddenWMRunner",
    "PixelWMRunner",
    "TokenWMRunner",
    "VLASFTRunner",
    "OpenVLAOFTRunner",
    "JointDreamerVLARunner",
    "LiberoEvalRunner",
    "ChameleonLatentWMRunner",
    "RynnDinoWMRunner",
    "OFTDinoWMRunner",
    "LatentClassifierRunner",
]


__all__ = [
    "BaseRunner",
    "PUBLIC_RUNNERS",
    "ActionHiddenWMRunner",
    "PixelWMRunner",
    "TokenWMRunner",
    "VLASFTRunner",
    "OpenVLAOFTRunner",
    "JointDreamerVLARunner",
    "LiberoEvalRunner",
    "ChameleonLatentWMRunner",
    "RynnDinoWMRunner",
    "OFTDinoWMRunner",
    "LatentClassifierRunner",
]
