import logging

from .chameleon import ChameleonConfig

logger = logging.getLogger(__name__)


class ChameleonXLLMXConfig(ChameleonConfig):
    def __init__(
        self,
        z_loss_weight: float = 0.0,
        action_dim: int = 7,
        time_horizon: int = 5,
        action_head_type: str = "legacy",
        **kwargs,
    ):
        self.z_loss_weight = z_loss_weight
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.action_head_type = action_head_type
        super().__init__(
            **kwargs,
        )
