from __future__ import annotations

from dreamervla.runners.backbone_dreamerv3_wm_runner import (
    BackboneDreamerV3WMRunner as _BackboneDreamerV3WMRunner,
)
from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.chameleon_latent_action_wm_runner import (
    ChameleonLatentActionWMRunner as _ChameleonLatentActionWMRunner,
)
from dreamervla.runners.collect_rollouts_runner import (
    CollectRolloutsRunner as _CollectRolloutsRunner,
)
from dreamervla.runners.dreamerv3_pixel_runner import (
    DreamerV3PixelRunner as _DreamerV3PixelRunner,
)
from dreamervla.runners.dreamerv3_token_runner import (
    DreamerV3TokenRunner as _DreamerV3TokenRunner,
)
from dreamervla.runners.dreamervla_runner import (
    DreamerVLARunner as _DreamerVLARunner,
)
from dreamervla.runners.embodied_eval_runner import (
    EmbodiedEvalRunner as _EmbodiedEvalRunner,
)
from dreamervla.runners.latent_classifier_runner import (
    LatentClassifierRunner as _LatentClassifierRunner,
)
from dreamervla.runners.latent_wm_runner import (
    LatentWMTrainingRunner as _LatentWMTrainingRunner,
)
from dreamervla.runners.online_cotrain_pipeline_runner import (
    OnlineCotrainPipelineRunner as _OnlineCotrainPipelineRunner,
)
from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner
from dreamervla.runners.openvla_oft_runner import (
    OpenVLAOFTTrainingRunner as _OpenVLAOFTTrainingRunner,
)
from dreamervla.runners.vla_sft_runner import VLASFTRunner as _VLASFTRunner


class ActionHiddenWMRunner(_BackboneDreamerV3WMRunner):
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


class EmbodiedEvalRunner(_EmbodiedEvalRunner):
    runner_name = "embodied_eval"
    runner_status = "current"
    runner_family = "eval"


class ChameleonLatentWMRunner(_ChameleonLatentActionWMRunner):
    runner_name = "chameleon_latent_wm"
    runner_status = "secondary"
    runner_family = "world_model"


class LatentWMRunner(_LatentWMTrainingRunner):
    runner_name = "latent_wm"
    runner_status = "current"
    runner_family = "world_model"


class LatentClassifierRunner(_LatentClassifierRunner):
    runner_name = "latent_classifier"
    runner_status = "current"
    runner_family = "reward"


class CollectRolloutsRunner(_CollectRolloutsRunner):
    runner_name = "collect_rollouts"
    runner_status = "current"
    runner_family = "rollout"


class OnlineCotrainPipelineRunner(_OnlineCotrainPipelineRunner):
    runner_name = "online_cotrain_pipeline"
    runner_status = "current"
    runner_family = "actor"


PUBLIC_RUNNERS = [
    "ActionHiddenWMRunner",
    "PixelWMRunner",
    "TokenWMRunner",
    "VLASFTRunner",
    "OpenVLAOFTRunner",
    "JointDreamerVLARunner",
    "EmbodiedEvalRunner",
    "ChameleonLatentWMRunner",
    "LatentWMRunner",
    "LatentClassifierRunner",
    "OnlineCotrainRunner",
    "OnlineCotrainPipelineRunner",
    "OnlineCotrainRayRunner",
    "ColdStartRayCollectRunner",
    "CollectRolloutsRunner",
]


__all__ = [
    "OnlineCotrainPipelineRunner",
    "BaseRunner",
    "PUBLIC_RUNNERS",
    "ActionHiddenWMRunner",
    "PixelWMRunner",
    "TokenWMRunner",
    "VLASFTRunner",
    "OpenVLAOFTRunner",
    "JointDreamerVLARunner",
    "EmbodiedEvalRunner",
    "ChameleonLatentWMRunner",
    "LatentWMRunner",
    "LatentClassifierRunner",
    "OnlineCotrainRunner",
    "OnlineCotrainRayRunner",
    "ColdStartRayCollectRunner",
    "CollectRolloutsRunner",
]


def __getattr__(name: str) -> object:
    if name == "OnlineCotrainRayRunner":
        from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

        return OnlineCotrainRayRunner
    if name == "ColdStartRayCollectRunner":
        from dreamervla.runners.cold_start_ray_collect_runner import (
            ColdStartRayCollectRunner,
        )

        return ColdStartRayCollectRunner
    raise AttributeError(name)
